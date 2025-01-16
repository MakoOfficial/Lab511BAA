import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        attn_ca = self.ca(x)
        out = x * attn_ca
        attn_sa = self.sa(out)
        result = out * attn_sa
        return result, attn_sa


class GatingBlock(nn.Module):
    def __init__(self, in_planes):
        super(GatingBlock, self).__init__()

        self.W_v = nn.Sequential(
            nn.Linear(in_planes, in_planes, bias=True),
            nn.BatchNorm1d(in_planes),
            nn.ReLU(),
            nn.Linear(in_planes, 1, bias=True)
        )
        self.W_g = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2, bias=True),
            nn.BatchNorm1d(in_planes // 2),
            nn.ReLU(),
            nn.Linear(in_planes // 2, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        v = self.W_v(x)
        g = self.W_g(x)
        return v * g


class GatingBlock_Class(nn.Module):
    def __init__(self, in_planes):
        super(GatingBlock_Class, self).__init__()

        self.W_v = nn.Sequential(
            nn.Linear(in_planes, in_planes, bias=True),
            nn.BatchNorm1d(in_planes),
            nn.ReLU(),
            nn.Linear(in_planes, 228, bias=True)
        )
        self.W_g = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2, bias=True),
            nn.BatchNorm1d(in_planes // 2),
            nn.ReLU(),
            nn.Linear(in_planes // 2, 228, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        v = self.W_v(x)
        g = self.W_g(x)
        return v * g


class PatchEmbed(nn.Module):
    def __init__(self, in_size=16, patch_size=2, in_dim=2048, embed_dim=768):
        super().__init__()
        self.in_size = in_size
        self.patch_size = patch_size

        self.grid_size = in_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 如果不传入norm_layer就不做任何操作
        # self.norm = nn.LayerNorm(embed_dim)
        self.norm = nn.Identity()

        # self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape

        # 若输入图片尺寸不满足要求则会报错
        assert H == self.in_size and W == self.in_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.in_size}*{self.in_size})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]

        # flatten从第二个维度开始,也就是H开始;再交换1,2两个维度
        x = self.proj(x).flatten(2).transpose(1, 2)

        # x += self.pos_encoding

        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=2.):
        super(Block, self).__init__()
        act_layer = nn.GELU
        norm_layer = nn.LayerNorm
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class ViTEncoder(nn.Module):
    def __init__(self, in_size=16, patch_size=1, depth=4, in_dim=2048, embed_dim=768, mlp_ratio=2):
        super().__init__()
        self.embedding_layer = PatchEmbed(in_size=in_size, patch_size=patch_size, in_dim=in_dim, embed_dim=embed_dim)

        blocks = []
        for _ in range(depth):
            blocks.append(Block(dim=embed_dim,mlp_ratio=mlp_ratio))

        self.vit_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        patch_embedding = self.embedding_layer(x)   # [B, C, H, W] -> [B, (h*w), C_embed]
        feature = self.vit_blocks(patch_embedding)
        feature = feature.mean(dim=1)

        return feature


if __name__ == '__main__':
    feature = torch.ones((32, 2048, 16, 16))
    vit = ViTEncoder(in_size=16, patch_size=2, depth=2, in_dim=2048, embed_dim=768, mlp_ratio=2)
    print(f"vit Model: {sum(p.nelement() for p in vit.parameters() if p.requires_grad == True) / 1e6}M")
    output = vit(feature)
    print(output.shape)