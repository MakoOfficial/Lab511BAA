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
        self.fc2 = nn.Conv2d(attn_dim, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(attn_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
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

        avg_out = self.fc2(avg_vector)

        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return attn * x, torch.flatten(avg_out, 1), attn


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
        self.attn1 = CNNAttention(in_channels, attn_dim, in_size)
        self.attn2 = CNNAttention(in_channels, attn_dim, in_size)

    def forward(self, x):
        x, attn1 = self.attn1(x)
        x, attn2 = self.attn2(x)
        return x, attn1, attn2


class Student_GCN_Model(nn.Module):
    def __init__(self, backbone, backbone_res):
        super(Student_GCN_Model, self).__init__()
        # self.out_channels = out_channels
        self.backbone0 = backbone.backbone0
        self.attn0 = backbone.attn0
        self.backbone1 = backbone.backbone1
        self.attn1 = backbone.attn1
        self.freeze_params()

        self.backbone2 = backbone_res[6]
        self.adj_learning0 = CNNAttention(1024, 768, 32)
        # self.adj_learning0 = CNNViT(1024, 768, 32, 1024, 2)
        self.backbone3 = backbone_res[7]
        self.adj_learning1 = CNNAttention(2048, 768, 16)
        # self.adj_learning1 = CNNViT(2048, 768, 16, 2048, 2)

        self.gender_encoder = nn.Linear(1, 32)
        self.gender_bn = nn.BatchNorm1d(32)

        self.fc = nn.Sequential(
            nn.Linear(2048 + 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

        self.fc_cls2 = nn.Sequential(
            nn.Linear(1024 + 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

        self.fc_cls3 = nn.Sequential(
            nn.Linear(2048 + 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, image, gender):
        gender_encode = F.relu(self.gender_bn(self.gender_encoder(gender))) # B * 32
        x0, attn0 = self.attn0(self.backbone0(image))
        x1, attn1 = self.attn1(self.backbone1(x0))
        x2, cls_token2, attn2 = self.adj_learning0(self.backbone2(x1))
        x3, cls_token3, attn3 = self.adj_learning1(self.backbone3(x2))

        x = F.adaptive_avg_pool2d(x3, 1)
        x = torch.flatten(x, 1)

        x = torch.cat([x, gender_encode], dim=1)
        cls_token2 = torch.cat([cls_token2, gender_encode], dim=1)
        cls_token3 = torch.cat([cls_token3, gender_encode], dim=1)

        x = self.fc(x)
        cls_token2 = self.fc_cls2(cls_token2)
        cls_token3 = self.fc_cls3(cls_token3)

        return x, cls_token2, cls_token3, attn0, attn1, attn2, attn3

    def freeze_params(self):
        for _, param in self.backbone0.named_parameters():
            param.requires_grad = False
        for _, param in self.attn0.named_parameters():
            param.requires_grad = False
        for _, param in self.backbone1.named_parameters():
            param.requires_grad = False
        for _, param in self.attn1.named_parameters():
            param.requires_grad = False


class Student_Contrast_Model(nn.Module):
    def __init__(self, backbone, backbone_res):
        super(Student_Contrast_Model, self).__init__()
        # self.out_channels = out_channels
        self.backbone0 = backbone.backbone0
        self.attn0 = backbone.attn0
        self.backbone1 = backbone.backbone1
        self.attn1 = backbone.attn1
        self.freeze_params()

        self.backbone2 = backbone_res[6]
        self.adj_learning0 = CNNAttention(1024, 768, 32)
        self.backbone3 = backbone_res[7]
        self.adj_learning1 = CNNAttention(2048, 768, 16)

        self.gender_encoder = nn.Linear(1, 32)
        self.gender_bn = nn.BatchNorm1d(32)

        self.fc = nn.Sequential(
            nn.Linear(2048 + 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

        self.cls_Embedding_0 = nn.Sequential(
            nn.Linear(1024 + 32, 512),
            nn.ReLU(),
            # nn.BatchNorm1d(512),
            nn.Linear(512, 1024)
        )

        self.cls_Embedding_1 = nn.Sequential(
            nn.Linear(2048 + 32, 512),
            nn.ReLU(),
            # nn.BatchNorm1d(512),
            nn.Linear(512, 1024)
        )

    def forward(self, image, gender):
        gender_encode = F.relu(self.gender_bn(self.gender_encoder(gender))) # B * 32
        x0, attn0 = self.attn0(self.backbone0(image))
        x1, attn1 = self.attn1(self.backbone1(x0))
        x2, cls_token2, attn2 = self.adj_learning0(self.backbone2(x1))
        x3, cls_token3, attn3 = self.adj_learning1(self.backbone3(x2))

        x = F.adaptive_avg_pool2d(x3, 1)
        x = torch.flatten(x, 1)

        x = torch.cat([x, gender_encode], dim=1)
        cls_token2 = torch.cat([cls_token2, gender_encode], dim=1)
        cls_token3 = torch.cat([cls_token3, gender_encode], dim=1)

        cls_token2 = F.normalize(self.cls_Embedding_0(cls_token2), dim=1)
        cls_token3 = F.normalize(self.cls_Embedding_1(cls_token3), dim=1)

        x = self.fc(x)

        return x, cls_token2, cls_token3, attn0, attn1, attn2, attn3

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


def get_student_contrast_model(student_path):
    backbone = get_student()
    if student_path is not None:
        backbone.load_state_dict(torch.load(student_path))

    resnet, output_channels = get_pretrained_resnet50(True)

    return Student_Contrast_Model(backbone, resnet)


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

