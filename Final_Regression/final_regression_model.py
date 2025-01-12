from ContrastLearning.contrast_model import Student_Contrast_Model_Pretrain
from Student.student_model import get_student
import torch
from torch import nn
from torch.nn import functional as F


class Final_Regression(nn.Module):
    def __init__(self, backbone):
        super(Final_Regression, self).__init__()

        self.backbone = backbone
        self.freeze_params()
        self.clr_layers()

        self.gender_encoder = nn.Linear(1, 32)
        self.gender_bn = nn.BatchNorm1d(32)

        self.fc = nn.Sequential(
            nn.Linear(2048 + 32, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, image, gender):
        contrast_feature, attn0, attn1, attn2, attn3 = self.backbone.downStream(image, gender)

        gender_encode = F.relu(self.gender_bn(self.gender_encoder(gender)))  # B * 32

        x = F.adaptive_avg_pool2d(contrast_feature, 1)
        x = torch.flatten(x, 1)
        x = torch.cat([x, gender_encode], dim=1)

        x = self.fc(x)

        return x, 0, 0, attn0, attn1, attn2, attn3

    def freeze_params(self):
        for _, param in self.backbone.named_parameters():
            param.requires_grad = False

    def clr_layers(self):
        self.backbone.fc = nn.Sequential()
        self.backbone.cls_Embedding_0 = nn.Sequential()
        self.backbone.cls_Embedding_1 = nn.Sequential()



def get_final_regression(backbone_path):
    backbone = Student_Contrast_Model_Pretrain(get_student())
    if backbone_path is not None:
        backbone.load_state_dict(torch.load(backbone_path))

    return Final_Regression(backbone)


if __name__ == '__main__':
    FinalFantasy = get_final_regression("../Contrast_Output/Contrast_WCL_IN_CBAM_AVGPool_AdaA_GenderPlus_Full_1_11_96_Pretrain/Contrast_WCL_IN_CBAM_AVGPool_AdaA_GenderPlus_Full_1_11_96_Pretrain.bin")
    # print(FinalFantasy)
    print(f"Contrast Model: {sum(p.nelement() for p in FinalFantasy.parameters() if p.requires_grad == True) / 1e6}M")