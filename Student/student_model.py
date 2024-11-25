import torch
import torch.nn as nn
import torch.nn.functional as F
from CBAM_block import CBAM
from torchvision.models import resnet50


def get_pretrained_resnet50(pretrained=True):
    model = resnet50(pretrained=pretrained)
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
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
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


def get_student(pretrained=True):
    return Student_Model(32, *get_pretrained_resnet50(pretrained=pretrained))


if __name__ == '__main__':
    data = torch.rand(2, 1, 256, 256).cuda()
    gender = torch.ones((2, 1)).cuda()

    student = Student_Model(32, *get_pretrained_resnet50(pretrained=True)).cuda()
    print(f"Student: {sum(p.nelement() for p in student.parameters() if p.requires_grad == True) / 1e6}M")
    print(student)
    x, attn0, attn1, attn2, attn3 = student(data, gender)
    print(f"x: {x.shape}\nattn0: {attn0.shape}\nattn1: {attn1.shape}\nattn2: {attn2.shape}\nattn3: {attn3.shape}\n")
