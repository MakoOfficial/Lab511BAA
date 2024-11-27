import torch
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

from Unet.UNets import get_Attn_Unet
from Student.student_model import get_student
from datasets import RSNA_transform_train, RSNA_transform_val

def visualize_attn_kd(t1, t2, t3, t4, s1, s2, s3, s4, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 5))

    axes[0][0].imshow(t1.squeeze().cpu().numpy(), cmap='viridis')
    axes[0][0].set_title('t1')
    axes[0][0].axis('off')

    axes[0][1].imshow(t2.squeeze().cpu().numpy(), cmap='viridis')
    axes[0][1].set_title('t2')
    axes[0][1].axis('off')

    axes[0][2].imshow(t3.squeeze().cpu().numpy(), cmap='viridis')
    axes[0][2].set_title('t3')
    axes[0][2].axis('off')

    axes[0][3].imshow(t4.squeeze().cpu().numpy(), cmap='viridis')
    axes[0][3].set_title('t4')
    axes[0][3].axis('off')

    axes[1][0].imshow(s1.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][0].set_title('s1')
    axes[1][0].axis('off')

    axes[1][1].imshow(s2.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][1].set_title('s2')
    axes[1][1].axis('off')

    axes[1][2].imshow(s3.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][2].set_title('s3')
    axes[1][2].axis('off')

    axes[1][3].imshow(s4.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][3].set_title('s4')
    axes[1][3].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_path, "attn_ts.png"))


if __name__ == '__main__':
    teacher_path = 'ckp/Unet/unet_segmentation_Attn_UNet.pth'
    teacher = get_Attn_Unet()
    teacher.load_state_dict(torch.load(teacher_path), strict=True)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    student_path = './KD_Output/KD_modify_firstConv/KD_modify_firstConv.bin'
    student_model = get_student()
    student_model.load_state_dict(torch.load(student_path), strict=True)
    for param in student_model.parameters():
        param.requires_grad = False
    student_model.eval()

    age = 120
    gender = torch.zeros((1, 1), dtype=torch.float32)
    # image_path = '../RSNA/train/7038.png'
    image_path = '../RSNA/valid/15504.png'
    # 读取和预处理图片
    img = Image.open(image_path).convert('L')
    img = RSNA_transform_val(img).unsqueeze(0)

    boneage_mean = 127.5305
    boneage_div = 40.50783272412884
    # 进行推理
    with torch.no_grad():
        _, _, _, _, _, _, t1, t2, t3, t4 = teacher.forward_attention(img)
        class_feature, s1, s2, s3, s4 = student_model(img, gender)
        loss = age - (class_feature.cpu().item() * boneage_div + boneage_mean)
        print(loss)
    # 展示结果
    visualize_attn_kd(t1, t2, t3, t4, s1, s2, s3, s4, "./")
    # visualize_attn_mask(t1, t2, t3, t4, s1, s2, s3, s4, img_teacher)
