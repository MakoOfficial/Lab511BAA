import torch
from torch import nn
from einops import rearrange

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
    def __init__(self, dim=1024, attn_dim=1024):
        super().__init__()
        self.scale = attn_dim ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, attn_dim * 3, bias=False)
        self.to_out = nn.Linear(attn_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        return self.to_out(out)

    def evaluation(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        return self.to_out(out), attn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Blocks(nn.Module):
    def __init__(self, dim, depth, attn_dim, mlp_ratio):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.mlp_dim = int(mlp_ratio * dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, attn_dim=attn_dim),
                FeedForward(dim, self.mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

    def evaluation(self, x):
        attn_list = []
        for attn, ff in self.layers:
            attn_x, attn_map = attn.evaluation(x)
            attn_list.append(attn_map)
            x = attn_x + x
            x = ff(x) + x
        return self.norm(x), attn_list


class ViTEncoder(nn.Module):
    def __init__(self, in_size=16, patch_size=1, depth=4, in_dim=2048, embed_dim=1024, attn_dim=1024, mlp_ratio=2):
        super().__init__()
        self.embedding_layer = PatchEmbed(in_size=in_size, patch_size=patch_size, in_dim=in_dim, embed_dim=embed_dim)

        self.vit_blocks = Blocks(dim=embed_dim, depth=depth, attn_dim=attn_dim, mlp_ratio=mlp_ratio)

    def forward(self, x):
        patch_embedding = self.embedding_layer(x)   # [B, C, H, W] -> [B, (h*w), C_embed]
        feature = self.vit_blocks(patch_embedding)
        feature = feature.mean(dim=1)

        return feature

    def evaluation(self, x):
        feature = self.embedding_layer(x)   # [B, C, H, W] -> [B, (h*w), C_embed]
        feature, attn_list = self.vit_blocks.evaluation(feature)

        feature = feature.mean(dim=1)

        return feature, attn_list


class Self_Attention_Adj(nn.Module):
    def __init__(self, feature_size, attention_size):
        super(Self_Attention_Adj, self).__init__()
        # self.queue = nn.Parameter(torch.empty(feature_size, attention_size))
        # nn.init.kaiming_uniform_(self.queue)
        self.queue = nn.Linear(in_features=feature_size, out_features=attention_size)

        # self.key = nn.Parameter(torch.empty(feature_size, attention_size))
        # nn.init.kaiming_uniform_(self.key)
        self.key = nn.Linear(in_features=feature_size, out_features=attention_size)

        self.leak_relu = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x): # B C (HW)
        x = x.transpose(1, 2)    # B (HW) C
        # Q = self.leak_relu(torch.matmul(x, self.queue)) # B (HW) C_attn
        # K = self.leak_relu(torch.matmul(x, self.key)) # B (HW) C_attn
        Q = self.leak_relu(self.queue(x))  # B (HW) C_attn
        K = self.leak_relu(self.key(x))  # B (HW) C_attn
        attn = self.softmax(torch.matmul(Q, K.transpose(1, 2))) # B (HW) (HW)
        return attn


class Graph_GCN(nn.Module):
    def __init__(self, node_size, feature_size, output_size):
        super(Graph_GCN, self).__init__()
        self.node_size = node_size
        self.feature_size = feature_size
        self.output_size = output_size
        # self.weight = nn.Parameter(torch.empty(feature_size, output_size))
        # nn.init.kaiming_uniform_(self.weight)
        self.weight = nn.Linear(feature_size, output_size)

    def forward(self, x, A):
        x = torch.matmul(A, x.transpose(1, 2))  # B (HW) C
        # return (torch.matmul(x, self.weight)).transpose(1, 2)
        x = self.weight(x).transpose(1, 2)  # B C (HW)
        x_output = rearrange(x, 'b d (h w) -> b d h w', h=self.node_size, w=self.node_size)   # B C H W
        return x_output


class Graph_GCN_V2(nn.Module):
    """输入为resnet的第3层和第4层输出，[B, 1024, 32, 32]，[B, 2048, 16, 16]
        将cls_tokem的获取改为平均池化
    """
    def __init__(self, in_channels, attn_dim, in_size) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.attn_dim = attn_dim
        self.scale = attn_dim ** -0.5
        self.in_size = in_size

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.q = nn.Linear(in_channels, attn_dim, bias=False)
        self.k = nn.Linear(in_channels, attn_dim, bias=False)
        self.v = nn.Linear(in_channels, attn_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(attn_dim, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(attn_dim)
        self.softmax = nn.Softmax(dim=-1)

        # self.ffn = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, 1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU()
        # )
        self.ffn = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.shape
        feature_vector = rearrange(x, 'b d h w -> b (h w) d')
        q = self.norm(self.relu(self.q(feature_vector)))   # B (HxW) attn_dim
        k = self.norm(self.relu(self.k(feature_vector)))  # B (HxW) attn_dim
        v = self.norm(self.relu(self.v(feature_vector)))  # B (HxW) attn_dim

        attn = torch.matmul(q, k.transpose(-1, -2))  # B (HxW) (HxW)
        attn = self.softmax(attn * self.scale)  # B (HxW) (HxW)

        feature_out = torch.matmul(attn, v) # B (HxW) attn_dim

        feature_out = feature_out.reshape(B, -1, self.in_size, self.in_size)   # B, attn_dim, 1, 1

        feature_out = self.fc2(feature_out) # B, in_channel, H, W

        return self.ffn(x + feature_out)


class Graph_GCN_V3(nn.Module):
    """输入为resnet的第3层和第4层输出，[B, 1024, 32, 32]，[B, 2048, 16, 16]
        将cls_tokem的获取改为平均池化
    """
    def __init__(self, in_channels, attn_dim, in_size) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.attn_dim = attn_dim
        self.scale = attn_dim ** -0.5
        self.in_size = in_size

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.q = nn.Linear(in_channels, attn_dim, bias=False)
        self.k = nn.Linear(in_channels, attn_dim, bias=False)
        self.v = nn.Linear(in_channels, attn_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(attn_dim, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(attn_dim)
        self.softmax = nn.Softmax(dim=-1)

        # self.ffn = CNNFeedForward(in_channels, 2*in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        feature_vector = rearrange(x, 'b d h w -> b (h w) d')
        q = self.norm(self.relu(self.q(feature_vector)))   # B (HxW) attn_dim
        k = self.norm(self.relu(self.k(feature_vector)))  # B (HxW) attn_dim
        v = self.norm(self.relu(self.v(feature_vector)))  # B (HxW) attn_dim

        attn = torch.matmul(q, k.transpose(-1, -2))  # B (HxW) (HxW)
        attn = self.softmax(attn * self.scale)  # B (HxW) (HxW)

        feature_out = torch.matmul(attn, v) # B (HxW) attn_dim

        feature_out = feature_out.reshape(B, -1, self.in_size, self.in_size)   # B, attn_dim, 1, 1

        feature_out = self.fc2(feature_out) # B, in_channel, H, W

        return feature_out


if __name__ == '__main__':
    feature = torch.ones((32, 2048, 16, 16))
    vit = ViTEncoder(in_size=16, patch_size=2, depth=2, in_dim=2048, embed_dim=1024, attn_dim=2048, mlp_ratio=2)
    print(f"vit Model: {sum(p.nelement() for p in vit.parameters() if p.requires_grad == True) / 1e6}M")
    # output, attn_list = vit.evaluation(feature)
    # print(attn_list[0].shape)
    output = vit(feature)
    print(output.shape)
