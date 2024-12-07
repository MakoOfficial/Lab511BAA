import torch
import torch.nn as nn
import torch.nn.functional as F
from Student.student_model import get_student
from vit_model_old import getViTBlock
from Unet.UNets import Attention_block


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


def getContrastModel(student_path):
    student_model = get_student()
    student_model.load_state_dict(torch.load(student_path), strict=True)
    contrast_model = Contrast_Model(list(student_model.children()))

    return contrast_model


if __name__ == '__main__':
    contrast_model = getContrastModel(
        "../KD_All_Output/KD_modify_firstConv_RandomCrop/KD_modify_firstConv_RandomCrop.bin").cuda()
    print(f"Contrast Model: {sum(p.nelement() for p in contrast_model.parameters() if p.requires_grad == True) / 1e6}M")
    print(contrast_model)
    # loss_fn = nn.L1Loss()
    # optimizer = optim.Adam(filter(lambda p : p.requires_grad, contrast_model.parameters()), lr=1e-2)  # 优化器只传入fc2的参数
    # print("contrast_model.backbone0.weight", contrast_model.backbone0[0].weight)
    # print("contrast_model.backbone2.weight", contrast_model.backbone2[0].conv1.weight)
    # for epoch in range(2):
    #     contrast_model.train()
    #     data = torch.rand(2, 1, 256, 256).cuda()
    #     gender = torch.ones((2, 1)).cuda()
    #     label = torch.randint(0, 2, [2]).long().cuda()
    #
    #     output, attn0, attn1, attn2, attn3 = contrast_model(data, gender)
    #
    #     loss = loss_fn(output.squeeze(), label)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    # print("contrast_model.backbone0.weight", contrast_model.backbone0[0].weight)
    # print("contrast_model.backbone2.weight", contrast_model.backbone2[0].conv1.weight)

    data = torch.rand(2, 1, 256, 256).cuda()
    gender = torch.ones((2, 1)).cuda()

    output, cls_token0, token0_attn1, token0_attn2, cls_token1, token1_attn1, token1_attn2, attn0, attn1 = contrast_model(data, gender)
    # print(f"x: {output.shape}\nattn0: {attn0.shape}\nattn1: {attn1.shape}\nattn2: {attn2.shape}\nattn3: {attn3.shape}\n")
    print(f"x: {output.shape}\ncls_token0: {cls_token0.shape}\ncls_token1: {cls_token1.shape}")

