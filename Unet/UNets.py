import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

    def forward_attention(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi, psi


class Attn_UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(Attn_UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

    def forward_attention(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        _x4, attn4 = self.Att5.forward_attention(g=d5, x=x4)
        d5 = torch.cat((_x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        _x3, attn3 = self.Att4.forward_attention(g=d4, x=x3)
        d4 = torch.cat((_x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        _x2, attn2 = self.Att3.forward_attention(g=d3, x=x2)
        d3 = torch.cat((_x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        _x1, attn1 = self.Att2.forward_attention(g=d2, x=x1)
        d2 = torch.cat((_x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1, x1, x2, x3, x4, x5, attn1, attn2, attn3, attn4

    def forward_classifier(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        return x1, x2, x3, x4, x5


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    def __init__(
            self,
            in_c: int,
            inner_c: int,
            stride: int = 2,
            expansion: int = 4,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_c, inner_c)
        self.bn1 = norm_layer(inner_c)
        self.conv2 = conv3x3(inner_c, inner_c, stride)
        self.bn2 = norm_layer(inner_c)
        self.conv3 = conv1x1(inner_c, inner_c * expansion)
        self.bn3 = norm_layer(inner_c * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
                conv1x1(in_c, inner_c * expansion, stride),
                norm_layer(inner_c * expansion),
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Attn_Unet_Classifier(nn.Module):
    def __init__(self, backbone):
        super(Attn_Unet_Classifier, self).__init__()
        self.backbone = backbone
        for _, param in self.backbone.named_parameters():
            param.requires_grad = False

    def get_features(self, x):
        return self.backbone.forward_classifier(x)


class Attn_UNet_classifier_First(Attn_Unet_Classifier):
    def __init__(self, backbone):
        super(Attn_UNet_classifier_First, self).__init__(backbone)

        self.down_block0 = nn.Sequential(
            Bottleneck(128, 128, 2, 2),
            Bottleneck(256, 256, 2, 2),
            Bottleneck(512, 512, 2, 2),
        )
        self.down_block1 = nn.Sequential(
            Bottleneck(256, 256, 2, 2),
            Bottleneck(512, 512, 2, 2),
        )
        self.down_block2 = Bottleneck(512, 512, 2, 2)

        self.gender_encoder = nn.Linear(1, 32)
        self.gender_bn = nn.BatchNorm1d(32)

        self.fc = nn.Sequential(
            nn.Linear(4096 + 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x, gender):
        # encoding path
        x2, x3, x4, x5, x6 = self.get_features(x)
        x2 = self.down_block0(x2)
        x3 = self.down_block1(x3)
        x4 = self.down_block2(x4)

        x2 = F.adaptive_avg_pool2d(x2, 1)
        x2 = torch.flatten(x2, 1)

        x3 = F.adaptive_avg_pool2d(x3, 1)
        x3 = torch.flatten(x3, 1)

        x4 = F.adaptive_avg_pool2d(x4, 1)
        x4 = torch.flatten(x4, 1)

        x5 = F.adaptive_avg_pool2d(x5, 1)
        x5 = torch.flatten(x5, 1)

        gender_encode = F.relu(self.gender_bn(self.gender_encoder(gender)))

        x_total = torch.cat((x2, x3, x4, x5, gender_encode), dim=1)

        class_vector = self.fc(x_total)

        return class_vector


def get_Attn_Unet(img_ch=1, output_ch=1):
    return Attn_UNet(img_ch=img_ch, output_ch=output_ch)


def get_Attn_Unet_classifier(unet_path):
    unet = Attn_UNet(img_ch=1, output_ch=1)
    unet.load_state_dict(torch.load(unet_path), strict=True)
    classifier = Attn_Unet_Classifier(unet)

    return classifier


if __name__ == '__main__':
    # unet_classifier = get_Attn_Unet_classifier("../ckp/Unet/unet_segmentation_Attn_UNet.pth")
    # print(f"unet_classifier Unet: {sum(p.nelement() for p in unet_classifier.parameters() if p.requires_grad == True) / 1e6}M")
    #
    # data = torch.rand((2, 1, 256, 256), dtype=torch.float32)
    # gender = torch.rand((2, 1), dtype=torch.float32)
    # class_vector = unet_classifier(data, gender)
    #
    # print(f"class_vector.shape: {class_vector.shape}")
    #
    # unet_backbone = Attn_Unet_Classifier(Attn_UNet(img_ch=1, output_ch=1))
    # x1, x2, x3, x4, x5 = unet_backbone.get_features(data)
    #
    # print(f"x1.shape: {x1.shape}, x2.shape: {x2.shape}, x3.shape: {x3.shape}, "
    #       f"x4.shape: {x4.shape}, x5.shape: {x5.shape}")

    unet = get_Attn_Unet()
    print(
        f"Unet: {sum(p.nelement() for p in unet.parameters() if p.requires_grad == True) / 1e6}M")