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
