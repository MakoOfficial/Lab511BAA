import torch
import matplotlib.pyplot as plt
import os
from PIL import Image

from Unet.UNets import get_Attn_Unet
from Student.student_model import get_student
from datasets import RSNA_transform_val


def visualize_attn(s1, s2, s3, s4):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    axes[0].imshow(s1.squeeze().cpu().numpy(), cmap='viridis')
    axes[0].set_title('s1')
    axes[0].axis('off')

    axes[1].imshow(s2.squeeze().cpu().numpy(), cmap='viridis')
    axes[1].set_title('s2')
    axes[1].axis('off')

    axes[2].imshow(s3.squeeze().cpu().numpy(), cmap='viridis')
    axes[2].set_title('s3')
    axes[2].axis('off')

    axes[3].imshow(s4.squeeze().cpu().numpy(), cmap='viridis')
    axes[3].set_title('s4')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    student_model = get_student()
    student_path = './baseline/Res50_All.bin'
    student_model.load_state_dict(torch.load(student_path), strict=True)
    for param in student_model.parameters():
        param.requires_grad = False
    student_model.eval()

    age = 46
    gender = torch.zeros((1, 1), dtype=torch.float32)
    image_path = '../../Dataset/RSNA/valid/15504.png'
    # 读取和预处理图片
    img = Image.open(image_path).convert('L')
    img = RSNA_transform_val(img).unsqueeze(0)

    boneage_mean = 127.5305
    boneage_div = 40.50783272412884
    # 进行推理
    with torch.no_grad():
        class_feature, s1, s2, s3, s4 = student_model(img, gender)
        loss = age - (class_feature.cpu().item() * boneage_div + boneage_mean)
        print(loss)
    # 展示结果
    visualize_attn(s1, s2, s3, s4)
    # visualize_attn_mask(t1, t2, t3, t4, s1, s2, s3, s4, img_teacher)
