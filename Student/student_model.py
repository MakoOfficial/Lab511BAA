import torch
import torch.nn as nn
import torch.nn.functional as F
from CBAM_block import CBAM
from torchvision.models import resnet50, resnet18
from torch import einsum

from einops import rearrange
from einops.layers.torch import Rearrange

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


class Student_Model(nn.Module):
    def __init__(self, gender_encode_length, backbone, out_channels):
        super(Student_Model, self).__init__()
        self.out_channels = out_channels
        self.backbone0 = nn.Sequential(*backbone[0:5])
        self.backbone0[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.attn0 = CBAM(in_planes=256, ratio=8, kernel_size=3)
        self.backbone1 = backbone[5]
        self.attn1 = CBAM(in_planes=512, ratio=8, kernel_size=3)
        self.backbone2 = backbone[6]
        self.attn2 = CBAM(in_planes=1024, ratio=16, kernel_size=3)
        self.backbone3 = backbone[7]
        self.attn3 = CBAM(in_planes=2048, ratio=16, kernel_size=3)

        self.gender_encoder = nn.Linear(1, gender_encode_length)
        self.gender_bn = nn.BatchNorm1d(gender_encode_length)

        self.fc = nn.Sequential(
            nn.Linear(out_channels + gender_encode_length, 1024),
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
        x0, attn0 = self.attn0(self.backbone0(image))
        x1, attn1 = self.attn1(self.backbone1(x0))
        x2, attn2 = self.attn2(self.backbone2(x1))
        x3, attn3 = self.attn3(self.backbone3(x2))

        x = F.adaptive_avg_pool2d(x3, 1)
        x = torch.squeeze(x)
        x = x.view(-1, self.out_channels)

        gender_encode = F.relu(self.gender_bn(self.gender_encoder(gender)))

        x = torch.cat([x, gender_encode], dim=1)

        x = self.fc(x)

        return x, attn0, attn1, attn2, attn3

    def count_params(self):
        num_params = sum(p.nelement() for p in self.backbone0.parameters() if p.requires_grad == True)
        num_params += sum(p.nelement() for p in self.attn0.parameters() if p.requires_grad == True)
        num_params += sum(p.nelement() for p in self.backbone1.parameters() if p.requires_grad == True)
        num_params += sum(p.nelement() for p in self.attn1.parameters() if p.requires_grad == True)

        return num_params


class Student_Model_Res18(nn.Module):
    def __init__(self, gender_encode_length, backbone, out_channels):
        super(Student_Model_Res18, self).__init__()
        self.out_channels = out_channels
        self.backbone0 = nn.Sequential(*backbone[0:5])
        self.backbone0[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.attn0 = CBAM(in_planes=64, ratio=2, kernel_size=3)
        self.backbone1 = backbone[5]
        self.attn1 = CBAM(in_planes=128, ratio=2, kernel_size=3)
        self.backbone2 = backbone[6]
        self.attn2 = CBAM(in_planes=256, ratio=4, kernel_size=3)
        self.backbone3 = backbone[7]
        self.attn3 = CBAM(in_planes=512, ratio=4, kernel_size=3)

        self.gender_encoder = nn.Linear(1, gender_encode_length)
        self.gender_bn = nn.BatchNorm1d(gender_encode_length)

        self.fc = nn.Sequential(
            nn.Linear(out_channels + gender_encode_length, 1024),
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
        x0, attn0 = self.attn0(self.backbone0(image))
        x1, attn1 = self.attn1(self.backbone1(x0))
        x2, attn2 = self.attn2(self.backbone2(x1))
        x3, attn3 = self.attn3(self.backbone3(x2))

        x = F.adaptive_avg_pool2d(x3, 1)
        x = torch.squeeze(x)
        x = x.view(-1, self.out_channels)

        gender_encode = F.relu(self.gender_bn(self.gender_encoder(gender)))

        x = torch.cat([x, gender_encode], dim=1)

        x = self.fc(x)

        return x, attn0, attn1, attn2, attn3

    def count_params(self):
        num_params = sum(p.nelement() for p in self.backbone0.parameters() if p.requires_grad == True)
        num_params += sum(p.nelement() for p in self.attn0.parameters() if p.requires_grad == True)
        num_params += sum(p.nelement() for p in self.backbone1.parameters() if p.requires_grad == True)
        num_params += sum(p.nelement() for p in self.attn1.parameters() if p.requires_grad == True)

        return num_params


# 邻接矩阵的自注意力机制 ，所需参数：特征尺寸，注意力尺寸
class Self_Attention_Adj(nn.Module):
    def __init__(self, feature_size, attention_size, output_size):
        super(Self_Attention_Adj, self).__init__()
        self.scale = attention_size ** -0.5
        self.feature_size = feature_size
        #   初始化Q，先创建一个长为feature_size，宽为attention_size的随机参数，再放入参数组
        #   注:torch.empty() 创建任意数据类型的张量
        self.queue = nn.Parameter(torch.empty(feature_size, attention_size))
        nn.init.kaiming_uniform_(self.queue)  # 凯明初始化参数

        #   初始化K
        self.key = nn.Parameter(torch.empty(feature_size, attention_size))
        nn.init.kaiming_uniform_(self.key)

        #   初始化V
        # self.weight = nn.Parameter(torch.empty(feature_size, output_size))
        # nn.init.kaiming_uniform_(self.weight)
        self.to_out = nn.Sequential(
            nn.Linear(feature_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU()
        )

        #   激活函数采用leak_relu
        self.leak_relu = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        node_feature = x.flatten(start_dim=2)
        node_feature = node_feature.transpose(1, 2)  # 将x的第二个维度和第三个维度转置
        Q = self.leak_relu(torch.matmul(node_feature, self.queue))
        K = self.leak_relu(torch.matmul(node_feature, self.key))

        A = self.softmax(torch.matmul(Q, K.transpose(1, 2)) * self.scale)

        x = torch.matmul(A, node_feature)
        # x = F.leaky_relu((torch.matmul(x, self.weight)).transpose(1, 2))
        x = self.to_out(x).transpose(1, 2)
        return x.view(B, C, H, W), A


class Attention(nn.Module):
    def __init__(self, dim, inner_dim, output_size):
        super().__init__()
        self.scale = inner_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        node_feature = x.flatten(start_dim=2)
        node_feature = node_feature.transpose(1, 2)  # 将x的第二个维度和第三个维度转置
        q, k = self.to_qk(node_feature).chunk(2, dim=-1)  # (b, n(65), dim*2) ---> 2 * (b, n, dim)

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)
        print(f"attn.shape: {attn.shape}")
        print(f"node_feature.shape: {node_feature.shape}")
        out = einsum('b i j, b j d -> b i d', attn, node_feature)
        return self.to_out(out).view(B, C, H, W), attn


class Student_GCN_Model(nn.Module):
    def __init__(self, gender_encode_length, backbone, out_channels):
        super(Student_GCN_Model, self).__init__()
        self.out_channels = out_channels
        self.backbone0 = nn.Sequential(*backbone[0:5])
        self.backbone0[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.attn0 = CBAM(in_planes=256, ratio=8, kernel_size=3)
        self.backbone1 = backbone[5]
        self.attn1 = CBAM(in_planes=512, ratio=8, kernel_size=3)

        self.backbone2 = backbone[6]
        self.adj_learning0 = Self_Attention_Adj(1024, 256, 1024)
        # self.adj_learning0 = Attention(1024, 256, 1024)
        self.backbone3 = backbone[7]
        self.adj_learning1 = Self_Attention_Adj(2048, 512, 2048)
        # self.adj_learning1 = Attention(2048, 512, 2048)

        self.gender_encoder = nn.Linear(1, gender_encode_length)
        self.gender_bn = nn.BatchNorm1d(gender_encode_length)

        self.fc = nn.Sequential(
            nn.Linear(out_channels + gender_encode_length, 1024),
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
        x0, attn0 = self.attn0(self.backbone0(image))
        x1, attn1 = self.attn1(self.backbone1(x0))
        x2, adj0 = self.adj_learning0(self.backbone2(x1))
        x3, adj1 = self.adj_learning1(self.backbone3(x2))

        x = F.adaptive_avg_pool2d(x3, 1)
        x = torch.squeeze(x)
        x = x.view(-1, self.out_channels)

        gender_encode = F.relu(self.gender_bn(self.gender_encoder(gender)))

        x = torch.cat([x, gender_encode], dim=1)

        x = self.fc(x)

        return x, attn0, attn1, adj0, adj1


def get_student(pretrained=True):
    return Student_Model(32, *get_pretrained_resnet50(pretrained=pretrained))


def get_student_res18(pretrained=True):
    return Student_Model_Res18(32, *get_pretrained_resnet18(pretrained=pretrained))


def get_student_GCN(pretrained=True):
    return Student_GCN_Model(32, *get_pretrained_resnet50(pretrained=pretrained))


if __name__ == '__main__':
    data = torch.rand(2, 1, 256, 256).cuda()
    gender = torch.ones((2, 1)).cuda()

    student_res50 = get_student().cuda()
    student_res18 = get_student_res18().cuda()

    # student = Student_Model(32, *get_pretrained_resnet50(pretrained=True)).cuda()
    # print(f"Student: {sum(p.nelement() for p in student.parameters() if p.requires_grad == True) / 1e6}M")
    res18 = resnet18(pretrained=True).cuda()
    print(f"res18: {sum(p.nelement() for p in res18.parameters() if p.requires_grad == True) / 1e6}M")
    res50 = resnet50(pretrained=True).cuda()
    print(f"res50: {sum(p.nelement() for p in res50.parameters() if p.requires_grad == True) / 1e6}M")
    # print(res50)
    # print(student_res18)
    print(f"student_res18: {sum(p.nelement() for p in student_res18.parameters() if p.requires_grad == True) / 1e6}M")
    print(f"student_res50: {sum(p.nelement() for p in student_res50.parameters() if p.requires_grad == True) / 1e6}M")
    print(f"the first two blocks of student_res18: {student_res18.count_params() / 1e6}M")
    print(f"the first two blocks of student_res50: {student_res50.count_params() / 1e6}M")
    # x, attn0, attn1, attn2, attn3 = student_res18(data, gender)
    # print(f"x: {x.shape}\nattn0: {attn0.shape}\nattn1: {attn1.shape}\nattn2: {attn2.shape}\nattn3: {attn3.shape}\n")

    student_GCN_Model = Student_GCN_Model(32, *get_pretrained_resnet50()).cuda()
    print(f"student_GCN_Model: {sum(p.nelement() for p in student_GCN_Model.parameters() if p.requires_grad == True) / 1e6}M")
    x, attn0, attn1, attn2, attn3 = student_GCN_Model(data, gender)
    print(f"x: {x.shape}\nattn0: {attn0.shape}\nattn1: {attn1.shape}\nattn2: {attn2.shape}\nattn3: {attn3.shape}\n")

