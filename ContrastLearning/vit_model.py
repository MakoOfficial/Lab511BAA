from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Linearize(nn.Module):
    """
    实现对维度的降低
    应用的两个层: [B, 512, 64, 64], [B, 1024, 32, 32]
    输出      : [B, 1024, 256], [B, 256, 512]
    """
    def __init__(self, img_size, in_c=512, ratio=2, norm_layer=None):
        super().__init__()
        self.nums_patch = (img_size // 2)**2
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_ch = in_c // ratio

        self.proj = nn.Conv2d(in_c, self.down_ch, kernel_size=1, stride=1, bias=False)
        self.norm = norm_layer(self.down_ch) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.maxpool(x)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """删除多头机制"""
    def __init__(self,
                 dim,   # 输入token的dim
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()

        self.dim = dim
        self.scale = self.dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        attn_size = int((N-1) ** 0.5)
        # print(f"N:{N}, attn_size:{attn_size}")

        # qkv(): -> [batch_size, num_patches + 1, 3 * embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, embed_dim]
        # permute: -> [3, batch_size, num_patches + 1, embed_dim]
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        # [batch_size, num_patches + 1, embed_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        print(attn[:, 0, 1:].shape)

        token_attn = attn[:, 0, 1:].reshape(B, 1, attn_size, attn_size)

        return x, token_attn


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x, token_attn = self.attn(self.norm1(x))
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, token_attn


class ViT(nn.Module):
    def __init__(self, img_size=64, in_c=512, linear_ratio=2, representation_ratio=2,
                 depth=12, mlp_ratio=4.0, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=Linearize):

        super(ViT, self).__init__()
        self.embed_dim = in_c // linear_ratio
        self.representation_dim = in_c * representation_ratio
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, in_c=in_c, ratio=linear_ratio, norm_layer=None)
        self.num_patches = self.patch_embed.nums_patch

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.cat_embed_dim = self.embed_dim + 32

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.block1 =  Block(dim=self.cat_embed_dim, mlp_ratio=mlp_ratio,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[0],
                  norm_layer=norm_layer, act_layer=act_layer)
        self.block2 = Block(dim=self.cat_embed_dim, mlp_ratio=mlp_ratio,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[1],
                  norm_layer=norm_layer, act_layer=act_layer)
        # self.blocks = nn.Sequential(*[
        #     Block(dim=self.cat_embed_dim, mlp_ratio=mlp_ratio,
        #           drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #           norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)
        # ])
        self.norm = norm_layer(self.cat_embed_dim)

        self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(self.cat_embed_dim, self.representation_dim)),
                ("act", nn.Tanh())
            ]))

        # Weight init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x, gender_encode):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        # [1, 1, embed_dim] -> [B, 1, embed_dim]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, num_patches+1, embed_dim]

        gender_encode = gender_encode.clone()
        gender_encode = gender_encode.unsqueeze(1).expand(-1, self.num_patches+1, -1)
        x = torch.cat([x, gender_encode], dim=2)  # [B, num_patches+1, embed_dim+32]

        x = self.pos_drop(x)
        x1, token_attn1 = self.block1(x)
        x2, token_attn2 = self.block1(x1)
        # x = self.blocks(x)
        x = self.norm(x2)

        return self.pre_logits(x[:, 0]), token_attn1, token_attn2


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def getViTBlock(img_size, in_c, linear_ratio, representation_ratio, depth):
    model = ViT(img_size=img_size,
                in_c=in_c,
                linear_ratio=linear_ratio,
                representation_ratio=representation_ratio,
                depth=depth)
    return model


if __name__ == '__main__':
    vit1 = getViTBlock(64, 512, 2, 2, 2).cuda()
    vit2 = getViTBlock(32, 1024, 2, 2, 2).cuda()
    data = torch.rand((2, 512, 64, 64), dtype=torch.float32).cuda()
    gender = torch.rand((2, 32), dtype=torch.float32).cuda()

    output, token_attn1, token_attn2 = vit1(data, gender)

    print(f"vit1 Model: {sum(p.nelement() for p in vit1.parameters() if p.requires_grad == True) / 1e6}M")
    print(f"vit2 Model: {sum(p.nelement() for p in vit2.parameters() if p.requires_grad == True) / 1e6}M")
    # print(f"output: {output.shape}, token_attn1: {token_attn1.shape}, token_attn2: {token_attn2.shape}")
