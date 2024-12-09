import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from Student.student_model import get_student
from torchvision.models import resnet50, resnet18
from ContrastLearning.vit_model_old import getViTBlock
from Unet.UNets import Attention_block

def get_pretrained_resnet50(pretrained=True):
    model = resnet50(pretrained=pretrained)
    output_channels = model.fc.in_features
    model = list(model.children())[:-2]
    return model, output_channels


def get_pretrained_resnet18(pretrained=True):
    model = resnet18(pretrained=pretrained)
    output_channels = model.fc.in_features
    model = list(model.children())[:-2]
    return model, output_channels


class Contrast_Model(nn.Module):

    def __init__(self, backbone) -> None:
        super(Contrast_Model, self).__init__()
        self.backbone0 = backbone[0]
        self.attn0 = backbone[1]
        self.backbone1 = backbone[2]
        self.attn1 = backbone[3]
        self.freeze_params()

        self.vit_block0 = getViTBlock(64, 512, 2, 2, 2)
        self.backbone2 = backbone[4]
        self.vit_block1 = getViTBlock(32, 1024, 2, 2, 2)
        self.backbone3 = backbone[6]

        self.gender_encoder = backbone[8]
        self.gender_bn = backbone[9]

        self.fc = backbone[10]

    def freeze_params(self):
        for _, param in self.backbone0.named_parameters():
            param.requires_grad = False
        for _, param in self.attn0.named_parameters():
            param.requires_grad = False
        for _, param in self.backbone1.named_parameters():
            param.requires_grad = False
        for _, param in self.attn1.named_parameters():
            param.requires_grad = False

    def forward(self, image, gender):
        gender_encode = F.relu(self.gender_bn(self.gender_encoder(gender)))
        x0, attn0 = self.attn0(self.backbone0(image))
        x1, attn1 = self.attn1(self.backbone1(x0))

        cls_token0, token0_attn1, token0_attn2 = self.vit_block0(x1, gender_encode)
        x1 = x1 * F.interpolate(token0_attn2, size=(x1.shape[-2], x1.shape[-1]), mode='nearest')

        x2 = self.backbone2(x1)
        cls_token1, token1_attn1, token1_attn2 = self.vit_block1(x2, gender_encode)
        x2 = x2 * F.interpolate(token1_attn2, size=(x2.shape[-2], x2.shape[-1]), mode='nearest')

        x3 = self.backbone3(x2)


        x = F.adaptive_avg_pool2d(x3, 1)
        x = torch.flatten(x, 1)

        gender_encode = F.relu(self.gender_bn(self.gender_encoder(gender)))

        x = torch.cat([x, gender_encode], dim=1)

        x = self.fc(x)

        return x, cls_token0, token0_attn1, token0_attn2, cls_token1, token1_attn1, token1_attn2, attn0, attn1


class CNNAttention(nn.Module):
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

        self.fc1 = nn.Conv2d(in_channels, attn_dim, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.norm = nn.LayerNorm(attn_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mode="train"):
        avg_vector = self.relu1(self.fc1(self.avg_pool(x)))
        # max_vector = self.relu1(self.fc1(self.max_pool(x)))
        # cls_token = avg_vector + max_vector   # B C 1 1
        cls_token = avg_vector  # B C 1 1

        feature_vector = self.relu1(self.fc1(x))    # 将原特征映射到度量空间

        feature_vector = self.norm(rearrange(feature_vector, 'b d h w -> b (h w) d'))
        cls_token = self.norm(rearrange(cls_token, 'b d h w -> b (h w) d'))

        attn = torch.matmul(cls_token, feature_vector.transpose(-1, -2))  # b 1 n
        attn = self.softmax(attn * self.scale)  # b 1 n
        attn = rearrange(attn, 'b d (h w) -> b d h w', h=self.in_size, w=self.in_size)

        feature_out = attn * x
        # feature_out = self.to_out(feature_out)

        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        if mode == "train":
            return feature_out
        else:
            return feature_out, attn


class CNNFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class CNNViT(nn.Module):

    def __init__(self, in_channels, attn_dim, in_size, mlp_dim, depth) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CNNAttention(in_channels, attn_dim, in_size),
                CNNFeedForward(in_channels, mlp_dim)
            ]))

    def forward(self, x, mode="train"):
        if mode == "train":
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
            return x
        else:
            attmaps = []
            for attn, ff in self.layers:
                ax, amap = attn(x, mode="record")
                x = ax + x
                x = ff(x) + x
                attmaps.append(amap)
            return x, attmaps


class Student_GCN_Model(nn.Module):
    """仅限resnet的stage1和stage2、CBAM1、CBAM2的模块传入预训练参数并固定"""
    def __init__(self, backbone, backbone_res):
        super(Student_GCN_Model, self).__init__()
        # self.out_channels = out_channels
        self.backbone0 = backbone.backbone0
        self.attn0 = backbone.attn0
        self.backbone1 = backbone.backbone1
        self.attn1 = backbone.attn1
        self.freeze_params()

        self.backbone2 = backbone_res[6]
        # self.adj_learning0 = CNNAttention(1024, 768, 32)
        self.adj_learning0 = CNNViT(1024, 768, 32, 2048, depth=2)
        self.backbone3 = backbone_res[7]
        # self.adj_learning1 = CNNAttention(2048, 768, 16)
        self.adj_learning1 = CNNViT(2048, 768, 16, 2048, depth=2)

        self.gender_encoder = nn.Linear(1, 32)
        self.gender_bn = nn.BatchNorm1d(32)

        self.fc = nn.Sequential(
            nn.Linear(2048 + 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, image, gender):
        gender_encode = F.relu(self.gender_bn(self.gender_encoder(gender))) # B * 32
        x0, attn0 = self.attn0(self.backbone0(image))
        x1, attn1 = self.attn1(self.backbone1(x0))
        x2 = self.adj_learning0(self.backbone2(x1))
        x3 = self.adj_learning1(self.backbone3(x2))
        x = F.adaptive_avg_pool2d(x3, 1)
        x = torch.flatten(x, 1)

        x = torch.cat([x, gender_encode], dim=1)

        x = self.fc(x)

        return x, attn0, attn1

    def infer(self, image, gender):
        gender_encode = F.relu(self.gender_bn(self.gender_encoder(gender))) # B * 32
        x0, attn0 = self.attn0(self.backbone0(image))
        x1, attn1 = self.attn1(self.backbone1(x0))
        x2, attn2 = self.adj_learning0(self.backbone2(x1), mode="record")
        x3, attn3 = self.adj_learning1(self.backbone3(x2), mode="record")
        x = F.adaptive_avg_pool2d(x3, 1)
        x = torch.flatten(x, 1)

        x = torch.cat([x, gender_encode], dim=1)

        x = self.fc(x)

        return x, attn0, attn1, attn2, attn3

    def freeze_params(self):
        for _, param in self.backbone0.named_parameters():
            param.requires_grad = False
        for _, param in self.attn0.named_parameters():
            param.requires_grad = False
        for _, param in self.backbone1.named_parameters():
            param.requires_grad = False
        for _, param in self.attn1.named_parameters():
            param.requires_grad = False


def getContrastModel(student_path):
    student_model = get_student()
    student_model.load_state_dict(torch.load(student_path), strict=True)
    contrast_model = Contrast_Model(list(student_model.children()))

    return contrast_model


def get_student_GCN(backbone_path):
    backbone = get_student()
    if backbone_path is not None:
        backbone.load_state_dict(torch.load(backbone_path))

    resnet, output_channels = get_pretrained_resnet50(True)

    return Student_GCN_Model(backbone, resnet)

import matplotlib.pyplot as plt

if __name__ == '__main__':
    contrast_model = getContrastModel(
        "../KD_All_Output/KD_modify_firstConv_RandomCrop/KD_modify_firstConv_RandomCrop.bin").cuda()
    print(f"Contrast Model: {sum(p.nelement() for p in contrast_model.parameters() if p.requires_grad == True) / 1e6}M")
    # print(contrast_model)

    student_GCN = get_student_GCN("../KD_All_Output/KD_modify_firstConv_RandomCrop/KD_modify_firstConv_RandomCrop.bin").cuda()
    print(f"student_GCN Model: {sum(p.nelement() for p in student_GCN.parameters() if p.requires_grad == True) / 1e6}M")
    print(student_GCN)

    data = torch.rand(2, 1, 256, 256).cuda()
    gender = torch.ones((2, 1)).cuda()

    # output, cls_token0, token0_attn1, token0_attn2, cls_token1, token1_attn1, token1_attn2, attn0, attn1 = contrast_model(data, gender)
    # print(f"x: {output.shape}\ncls_token0: {cls_token0.shape}\ncls_token1: {cls_token1.shape}")
    with torch.no_grad():
        output, attn0, attn1, attn2, attn3 = student_GCN.infer(data, gender)
        print(f"x: {output.shape}\nattn0: {attn0.shape}\nattn1: {attn1.shape}\n")
        s31 = attn2[0][0]
        s32 = attn2[1][0]
        s41 = attn3[0][0]
        s42 = attn3[1][0]
        fig, axes = plt.subplots(2, 2, figsize=(15, 5))

        axes[0][0].imshow(s31.squeeze().cpu().numpy(), cmap='viridis')
        axes[0][0].set_title('s31')
        axes[0][0].axis('off')

        axes[0][1].imshow(s32.squeeze().cpu().numpy(), cmap='viridis')
        axes[0][1].set_title('s32')
        axes[0][1].axis('off')

        axes[1][0].imshow(s41.squeeze().cpu().numpy(), cmap='viridis')
        axes[1][0].set_title('s41')
        axes[1][0].axis('off')

        axes[1][1].imshow(s42.squeeze().cpu().numpy(), cmap='viridis')
        axes[1][1].set_title('s42')
        axes[1][1].axis('off')

        plt.tight_layout()
        plt.show()

