import torch
from torch import nn
from functools import partial


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob

    # Here, shape is of type tuple, the first element is x.shape[0], followed by (dimension of x - 1) 1s ->: (n,1,1,1,...)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets

    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)


    random_tensor.floor_()  # binarize

    # ???
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self,
                 dim,   # token dim
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.scale = qk_scale or dim ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3) # [3, B, 1025, 800]

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_cls = attn[:, 0, 1:].softmax(dim=-1)
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)  #再经过投影
        # x = self.proj_drop(x)
        return attn_cls


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    # hidden_features一般是in_features的四倍(看图)
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,  # 每个token的dimention
                 mlp_ratio=4.,  # 第一个全连接层的输出个数是输入的四倍
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,  # Multi_head self_attention中最后的全连接层使用的drop_ratio
                 attn_drop_ratio=0.,  # q乘k.T之后通过的softmax之后有一个dropout层的drop_ratio(图上好像看不到,代码里面有)
                 drop_path_ratio=0.,  # 模型图中Encoder_Block中对应的那两层drop_out层对应的drop_ratio
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 #  norm_layer = nn.Identity
                 ):
        super(Block, self).__init__()

        # 调用刚刚定义的attention创建一个Multi_head self_attention模块
        self.attn = Attention(dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        # 使不用使用DropPath方法
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        # MLP中第一个全连接层的输出神经元的倍数(4倍)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):   # B, 1025, 800
        # 前向传播，很清晰
        attn_map = self.attn(x)
        # x = x + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(x))

        return attn_map


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=32, patch_size=1, in_c=1024, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

        # If norm_layer is not passed in, no operation will be performed
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]

        x = self.proj(x).flatten(2).transpose(1, 2)

        x = self.norm(x)
        return x


class Vit_block(nn.Module):

    def __init__(self, img_size, patch_size, input_dim, output_dim, embed_dim) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = int(img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.cat_embed_dim = self.embed_dim + 32

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=1, in_c=input_dim, embed_dim=self.embed_dim)
        self.block = Block(dim=800, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                        drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU)

        # self.out_layer = nn.Sequential(
        #     nn.Linear(self.cat_embed_dim, output_dim),
        #     nn.BatchNorm1d(output_dim),
        #     nn.ReLU(),
        #     nn.Linear(output_dim, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1)
        # )

    def forward(self, x, gender_encode): #  X: B, 1024, 32, 32
        patch_embed = self.patch_embed(x)   # patch_embed: B, 1024, 768
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)   # 1, 1, 768 -> B, 1, 768
        x = torch.cat((cls_token, patch_embed), dim=1)  # [B, num_patches+1, embed_dim] B, 1025, 768
        gender_encode = gender_encode.clone()
        gender_encode = gender_encode.unsqueeze(1).expand(-1, x.shape[1], -1)   # [B, 1025, 32]
        patch_embed = torch.cat([x, gender_encode], dim=2)  # [B, num_patches+1, embed_dim+32] [B, 1025, 800]

        attn = self.block(patch_embed)

        attn_size = int(attn.shape[-1] ** 0.5)

        return attn.reshape(x.shape[0], 1, attn_size, attn_size)





if __name__ == '__main__':
    feature3 = torch.rand((2, 1024, 32, 32), dtype=torch.float32)
    gender3 = torch.rand((2, 32), dtype=torch.float32)
    feature4 = torch.rand((2, 2048, 16, 16), dtype=torch.float32)
    #
    # patch_embed3 = PatchEmbed(img_size=32, patch_size=1, in_c=1024, embed_dim=768)
    # patch_embed3_output = patch_embed3(feature3)
    # print(f"patch_embed3_output shape is {patch_embed3_output.shape}")
    #
    #
    # block3 = Block(dim=768, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
    #                     drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0,
    #                     norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU)
    # block3_output, attn_map3 = block3(patch_embed3_output)
    # print(f"block3_output shape is {block3_output.shape}")
    # print(f"attn_map3 shape is {attn_map3.shape}")
    #
    # patch_embed4 = PatchEmbed(img_size=16, patch_size=1, in_c=2048, embed_dim=768)
    # patch_embed4_output = patch_embed4(feature4)
    # print(f"patch_embed4_output shape is {patch_embed4_output.shape}")
    #
    # block4 = Block(dim=768, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
    #                     drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0,
    #                     norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU)
    # block4_output, attn_map4 = block4(patch_embed4_output)
    # print(f"block4_output shape is {block4_output.shape}")
    # print(f"attn_map4 shape is {attn_map4.shape}")
    # print(patch_embed3)
    # print(block3)
    # print(patch_embed4)
    # print(block4)

    vit_block = Vit_block(32, 1, 1024, 1024, 768)
    out, attn =vit_block(feature3, gender3)
    print(out.shape)
    print(attn.shape)

    for name, param in vit_block.named_parameters():
        print(name)