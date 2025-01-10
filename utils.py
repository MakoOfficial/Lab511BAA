import torch
import os
import csv
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from collections import Counter

stage6_boneage = [14, 12,
                  24, 24,
                  84, 84,
                  156, 156,
                  180, 180,
                  192, 204]

stage6_gender = [0, 1]

ranges_male = [
    (0, 14),
    (15, 36),
    (37, 108),
    (109, 168),
    (169, 192),
    (193, 228)
]

ranges_female = [
    (0, 10),
    (11, 24),
    (25, 84),
    (85, 156),
    (157, 180),
    (181, 228)
]

def iou(input, target, classes=1):
    """  compute the value of iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        iou: float, the value of iou
    """
    input = input.clone().cpu().numpy()
    target = target.clone().cpu().numpy()
    intersection = np.logical_and(target == classes, input == classes)
    union = np.logical_or(target == classes, input == classes)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def preprocess_image(image_path, img_size=None):
    if img_size is None:
        img_size = 256
    # 定义数据变换
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 调整图像大小
        transforms.ToTensor(),          # 转换为Tensor
    ])

    # 读取图像
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)  # 添加批次维度
    print(image.shape)
    return image


def mean_iou(unet, image_path, label_path):
    total_loss = 0
    total_size = 0
    img_list = os.listdir(image_path)
    for img_name in img_list:
        total_size += 1
        img_name = img_name.split('.')[0]
        img = preprocess_image(os.path.join(image_path, img_name) + ".jpg").cuda()
        label = preprocess_image(os.path.join(label_path, img_name) + ".png").cuda()
        with torch.no_grad():
            output = unet(img)
            prediction = torch.nn.functional.sigmoid(output)  # 应用Sigmoid函数
            prediction = (prediction > 0.5).int()  # 将预测值转换为0或1
            iou_loss = iou(prediction, label)
            total_loss += iou_loss.item()
    print(f"mean IoU accuracy: {round(100 * total_loss / total_size,2)}%")


def L1_penalty(net, alpha):
    l1_penalty = torch.nn.L1Loss(size_average=False)
    loss = 0
    for param in net.fc.parameters():
        loss += torch.sum(torch.abs(param))

    return alpha * loss


def L1_penalty_multi(net, alpha):
    l1_penalty = torch.nn.L1Loss(size_average=False)
    loss = 0
    loss2 = 0
    loss3 = 0
    for param in net.fc.parameters():
        loss += torch.sum(torch.abs(param))
    for param in net.fc_cls2.parameters():
        loss2 += torch.sum(torch.abs(param))
    for param in net.fc_cls3.parameters():
        loss3 += torch.sum(torch.abs(param))

    return alpha * (loss + 1/2 * loss2 + 1/2 * loss3)


def log_losses_to_csv_KD(mean_attention_loss, val_attn, cost_time, lr, log_file_path):
    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 如果文件不存在，则创建并写入表头
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Mean_Attention_Loss", "Val_Attn_Loss", "Cost_Time", "LR"])

    # 追加写入损失值
    with open(log_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([round(mean_attention_loss, 5), round(val_attn, 5),
                         round(cost_time, 2), lr])

    # 打印到终端
    print(f"Mean Attention Loss: {mean_attention_loss}, "
          f"Val Attn Loss: {val_attn}, Cost Time: {round(cost_time, 2)}, LR: {lr}")


def log_losses_to_csv(training_loss, mean_attention_loss, val_loss, val_attn, cost_time, lr, log_file_path):
    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 如果文件不存在，则创建并写入表头
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Training_Loss", "Mean_Attention_Loss", "Validation_Loss", "Val_Attn_Loss", "Cost_Time", "LR"])

    # 追加写入损失值
    with open(log_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([round(training_loss, 4), round(mean_attention_loss, 5),
                         round(val_loss, 3), round(val_attn, 5),
                         round(cost_time, 2), lr])

    # 打印到终端
    print(f"Training Loss: {training_loss}, Mean Attention Loss: {mean_attention_loss}, Validation Loss: {val_loss}, "
          f"Val Attn Loss: {val_attn}, Cost Time: {round(cost_time, 2)}, LR: {lr}")


def log_contrast_losses_to_csv(training_loss, mean_triple_loss_0, mean_triple_loss_1, val_loss, cost_time, lr, log_file_path):
    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 如果文件不存在，则创建并写入表头
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Training_Loss", "Mean_Contrast_Loss_0", "Mean_Contrast_Loss_1", "Validation_Loss", "Cost_Time", "LR"])

    # 追加写入损失值
    with open(log_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([round(training_loss, 4), round(mean_triple_loss_0, 5),
                         round(mean_triple_loss_1, 5), round(val_loss, 3),
                         round(cost_time, 2), lr])

    # 打印到终端
    print(f"Training Loss: {training_loss}, Mean Contrast Loss 0: {mean_triple_loss_0}, Mean Contrast Loss 1: "
          f"{mean_triple_loss_1}, Validation Loss: {val_loss}, Cost Time: {round(cost_time, 2)}, LR: {lr}")



def log_contrast_losses_to_csv_End2End(training_loss, mean_attn_loss, mean_triple_loss_0, mean_triple_loss_1, val_loss, cost_time, lr, log_file_path):
    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 如果文件不存在，则创建并写入表头
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Training_Loss", "Mean_Attn_Loss", "Mean_Contrast_Loss_0", "Mean_Contrast_Loss_1", "Validation_Loss", "Cost_Time", "LR"])

    # 追加写入损失值
    with open(log_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([round(training_loss, 4), round(mean_attn_loss, 5), round(mean_triple_loss_0, 5),
                         round(mean_triple_loss_1, 5), round(val_loss, 3),
                         round(cost_time, 2), lr])

    # 打印到终端
    print(f"Training Loss: {training_loss}, Mean_Attn_Loss: {mean_attn_loss}, Mean Contrast Loss 0: {mean_triple_loss_0}, Mean Contrast Loss 1: "
          f"{mean_triple_loss_1}, Validation Loss: {val_loss}, Cost Time: {round(cost_time, 2)}, LR: {lr}")



def log_valid_result_to_csv(id_list, boneage_list, male_list, pred_list, loss_list, log_file_path):
    length = len(id_list)
    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 如果文件不存在，则创建并写入表头
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["id", "boneage", "male", "pred", "loss"])

    # 追加写入损失值
    with open(log_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for i in range(length):
            # if loss_list[i].item() > 10:
            writer.writerow([id_list[i].item(), boneage_list[i].item(),
                             male_list[i].item(), round(pred_list[i].item(), 2),
                             round(loss_list[i].item(), 2)])


def log_valid_result_logits_to_csv(id_list, boneage_list, male_list, pred_list, loss_list, logits_list, log_file_path):
    length = len(id_list)
    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 如果文件不存在，则创建并写入表头
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["id", "boneage", "male", "pred", "loss", "logits"])

    # 追加写入损失值
    with open(log_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for i in range(length):
            # if loss_list[i].item() > 10:
            writer.writerow([id_list[i].item(), boneage_list[i].item(),
                             male_list[i].item(), round(pred_list[i].item(), 2),
                             round(loss_list[i].item(), 2), logits_list[i].item()])


def save_attn_KD(t1, t2, t3, t4, s1, s2, s3, s4, save_path):
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
    plt.savefig(os.path.join(save_path, "attn_ts.png"))
    plt.clf()
    plt.close('all')


def save_attn_Contrast(t1, t2, t3, t4, s1, s2, s3, s4, save_path):
    s31 = s3[0][0]
    s32 = s3[1][0]
    s41 = s4[0][0]
    s42 = s4[1][0]
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))

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

    axes[1][2].imshow(s31.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][2].set_title('s31')
    axes[1][2].axis('off')

    axes[1][3].imshow(s32.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][3].set_title('s32')
    axes[1][3].axis('off')

    axes[0][4].imshow(s41.squeeze().cpu().numpy(), cmap='viridis')
    axes[0][4].set_title('s41')
    axes[0][4].axis('off')

    axes[1][4].imshow(s42.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][4].set_title('s42')
    axes[1][4].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "attn_Contrast.png"))


def save_attn_6Stage(test_loader, model, save_path):
    """输出的图片数量一定得是12张"""
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):

            image, gender = data[0]
            image = image.type(torch.FloatTensor).cuda()
            gender = gender.type(torch.FloatTensor).cuda()

            class_feature, _, _, s1, s2, s3, s4 = model.infer(image, gender)
            img_num = len(image)
            num_cols = 2
            # num_cols = 1
            for i in range(img_num):
                """对于第i张图片"""
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                save_name = f"attn_{i}_Contrast.png"
                """第j张注意力图"""
                attn_map_s3 = s3[i]
                title_s3 = f"attn_{i}_s3"
                attn_map_s4 = s4[i]
                title_s4 = f"attn_{i}_s4"

                axes[0].imshow(attn_map_s3.squeeze().cpu().numpy(), cmap='viridis')
                axes[0].set_title(title_s3)
                axes[0].axis('off')

                axes[1].imshow(attn_map_s4.squeeze().cpu().numpy(), cmap='viridis')
                axes[1].set_title(title_s4)
                axes[1].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_path, save_name))

                plt.clf()


def save_contrast_attn_6Stage(test_loader, model, save_path):
    """输出的图片数量一定得是12张"""
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):

            image, gender = data[0]
            image = image.type(torch.FloatTensor).cuda()
            gender = gender.type(torch.FloatTensor).cuda()

            class_feature, _, _, s1, s2, s3, s4 = model(image, gender)
            img_num = len(image)
            for i in range(img_num):
                """对于第i张图片"""
                age = stage6_boneage[i]
                gender = i % 2
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                save_name = f"attn_{age}_{gender}_Contrast.png"

                axes[0].imshow(s3[i].squeeze().cpu().numpy(), cmap='viridis')
                axes[0].set_title("attn_s3")
                axes[0].axis('off')

                axes[1].imshow(s4[i].squeeze().cpu().numpy(), cmap='viridis')
                axes[1].set_title("attn_s4")
                axes[1].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_path, save_name))

                plt.clf()
                plt.close('all')


def save_attn_all(s3, s4, img_ids, save_path):
    attn_path = os.path.join(save_path, "attn_dir")
    os.makedirs(attn_path, exist_ok=True)
    with torch.no_grad():
        img_num = len(s3)
        num_cols = 2
        for i in range(img_num):
            """对于第i张图片"""
            fig, axes = plt.subplots(1, num_cols, figsize=(15, 5))
            save_name = f"{int(img_ids[i])}.png"

            axes[0].imshow(s3[i].squeeze().cpu().numpy(), cmap='viridis')
            axes[0].set_title("attn_s3")
            axes[0].axis('off')

            axes[1].imshow(s4[i].squeeze().cpu().numpy(), cmap='viridis')
            axes[1].set_title("attn_s4")
            axes[1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(attn_path, save_name))

            plt.clf()
            plt.close('all')


def save_attn_all_KD(s1, s2, s3, s4, img_ids, save_path):
    attn_path = os.path.join(save_path, "attn_dir")
    os.makedirs(attn_path, exist_ok=True)
    with torch.no_grad():
        img_num = len(s1)
        num_cols = 4
        for i in range(img_num):
            """对于第i张图片"""
            fig, axes = plt.subplots(1, num_cols, figsize=(15, 5))
            save_name = f"{int(img_ids[i])}.png"

            axes[0].imshow(s1[i].squeeze().cpu().numpy(), cmap='viridis')
            axes[0].set_title("attn_s1")
            axes[0].axis('off')

            axes[1].imshow(s2[i].squeeze().cpu().numpy(), cmap='viridis')
            axes[1].set_title("attn_s2")
            axes[1].axis('off')

            axes[2].imshow(s3[i].squeeze().cpu().numpy(), cmap='viridis')
            axes[2].set_title("attn_s3")
            axes[2].axis('off')

            axes[3].imshow(s4[i].squeeze().cpu().numpy(), cmap='viridis')
            axes[3].set_title("attn_s4")
            axes[3].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(attn_path, save_name))

            plt.clf()
            plt.close('all')

import pandas as pd
def show_attn_all_KD(s1, s2, s3, s4, img_ids, save_path):
    attn_path = os.path.join(save_path, "attn_dir")
    os.makedirs(attn_path, exist_ok=True)
    with torch.no_grad():
        num_cols = 4
        image_flat = s3.flatten()

        # 对整个图片应用softmax
        softmax_image_flat = F.softmax(image_flat, dim=0)

        # 将结果重新reshape回原来的形状
        softmax_image = softmax_image_flat.view(s3.shape)
        """对于第i张图片"""
        fig, axes = plt.subplots(1, num_cols, figsize=(15, 5))

        axes[0].imshow(s1.squeeze().cpu().numpy(), cmap='viridis')
        axes[0].set_title("attn_s1")
        axes[0].axis('off')

        axes[1].imshow(s2.squeeze().cpu().numpy(), cmap='viridis')
        axes[1].set_title("attn_s2")
        axes[1].axis('off')

        axes[2].imshow(softmax_image.squeeze().cpu().numpy(), cmap='viridis')
        axes[2].set_title("attn_s3")
        axes[2].axis('off')

        axes[3].imshow(s3.squeeze().cpu().numpy(), cmap='viridis')
        axes[3].set_title("attn_s4")
        axes[3].axis('off')
        # 创建一个二维tensor示例

        # 将tensor转换为DataFrame
        df3 = pd.DataFrame(softmax_image.squeeze().cpu().numpy())
        df4 = pd.DataFrame(s3.squeeze().cpu().numpy())

        # 将表格保存为CSV文件
        df3.to_csv(f'{int(img_ids)}s3.csv', index=False, header=False)
        df4.to_csv(f'{int(img_ids)}s4.csv', index=False, header=False)

        plt.tight_layout()
        # plt.savefig(os.path.join(attn_path, save_name))
        plt.show()

        plt.clf()
        plt.close('all')


def KL_loss(p, q):
    p_soft = F.softmax(torch.flatten(p, 1), dim=1) + 1e-3
    q_soft = F.softmax(torch.flatten(q, 1), dim=1) + 1e-3

    return torch.sum(p_soft * (p_soft.log() - q_soft.log()))


def attn_loss(t1, t2, t3, t4, s1, s2, s3, s4):
    """compute the loss between teacher attn and student attn"""

    s1 = F.interpolate(s1, size=(t1.shape[-2], t1.shape[-1]), mode='nearest')
    s2 = F.interpolate(s2, size=(t2.shape[-2], t2.shape[-1]), mode='nearest')
    s3 = F.interpolate(s3, size=(t3.shape[-2], t3.shape[-1]), mode='nearest')
    s4 = F.interpolate(s4, size=(t4.shape[-2], t4.shape[-1]), mode='nearest')

    return F.mse_loss(t1, s1) + F.mse_loss(t2, s2) + F.mse_loss(t3, s3) + F.mse_loss(t4, s4)


def attn_kl_loss_ablation(t1, t2, s1, s2):
    """compute the KD loss between teacher attn and student attn
        t1 -> s1, t2 -> s2
    """
    # s1 = nn.functional.upsample_nearest(s1, scale_factor=4)
    # s1 = F.interpolate(s1, size=(t1.shape[-2], t1.shape[-1]), mode='nearest')
    # s2 = F.interpolate(s2, size=(t2.shape[-2], t2.shape[-1]), mode='nearest')
    t1 = F.interpolate(t1, size=(s1.shape[-2], s1.shape[-1]), mode='nearest')
    t2 = F.interpolate(t2, size=(s2.shape[-2], s2.shape[-1]), mode='nearest')
    # s4 = nn.functional.upsample_nearest(s4, scale_factor=4)

    return KL_loss(t1, s1) + KL_loss(t2, s2)


def attn_kl_loss_singleStage_ablation(t, s):
    """compute the KD loss between teacher attn and student attn
        t -> s
    """
    t = F.interpolate(t, size=(s.shape[-2], s.shape[-1]), mode='nearest')

    return KL_loss(t, s)

def attn_offset_kl_loss(t1, t2, t3, t4, s1, s2, s3, s4):
    """compute the KD loss between teacher attn and student attn
        t2 -> s1,   t3 -> s2
    """
    s1 = F.interpolate(s1, size=(t2.shape[-2], t2.shape[-1]), mode='nearest')
    s2 = F.interpolate(s2, size=(t3.shape[-2], t3.shape[-1]), mode='nearest')
    # s3 = nn.functional.upsample_nearest(s3, scale_factor=4)
    # s4 = nn.functional.upsample_nearest(s4, scale_factor=4)

    return KL_loss(t2, s1) + KL_loss(t3, s2)


def attn_offset_kl_loss_firstStage(t1, t2, t3, t4, s1, s2, s3, s4):
    """compute the KD loss between teacher attn and student attn with first stage changed
        t2 -> s1,   t3 -> s2
    """
    assert s1.shape == t2.shape
    assert s2.shape == t3.shape

    return KL_loss(t2, s1) + KL_loss(t3, s2)

def attn_offset_mse_loss_firstStage(t1, t2, t3, t4, s1, s2, s3, s4):
    """compute the MSE loss between teacher attn and student attn with first stage changed
        t2 -> s1,   t3 -> s2
    """
    assert s1.shape == t2.shape
    assert s2.shape == t3.shape

    return F.mse_loss(t2, s1, reduction="sum") + F.mse_loss(t3, s2, reduction="sum")


def feature_offset_kl_loss_firstStage(t1, t2, t3, t4, t5, s1, s2, s3, s4):
    """compute the KD loss between teacher attn and student attn with first stage changed
        t4 -> s2,   t5 -> s3
    """
    assert s2.shape == t4.shape
    assert s3.shape == t5.shape

    return KL_loss(t4, s2) + KL_loss(t5, s3)

def feature_offset_mse_loss_firstStage(t1, t2, t3, t4, t5, s1, s2, s3, s4):
    """compute the MSE loss between teacher attn and student attn with first stage changed
        t4 -> s2,   t5 -> s3
    """
    assert s2.shape == t4.shape
    assert s3.shape == t5.shape

    return F.mse_loss(t4, s2, reduction="sum") + F.mse_loss(t5, s3, reduction="sum")


def attn_masked_kl_loss(t1, t2, t3, t4, s1, s2, s3, s4, xt):
    """compute the KD loss between teacher attn and student attn with masked
        t2_masked -> s2_masked,   t3_masked -> s3_masked
        注意，该函数必须使用FinalQiu这种背景值为0的数据
    """
    mask = torch.where(xt != 0, torch.tensor(0.0), torch.tensor(-float('inf')))
    # s1 = nn.functional.upsample_nearest(s1, scale_factor=4)
    s2 = F.interpolate(s2, size=(t2.shape[-2], t2.shape[-1]), mode='nearest')
    s3 = F.interpolate(s3, size=(t3.shape[-2], t3.shape[-1]), mode='nearest')
    # s4 = nn.functional.upsample_nearest(s4, scale_factor=4)

    t2_mask = F.interpolate(mask, size=(t2.shape[-2], t2.shape[-1]), mode='nearest')
    t2_masked = t2 + t2_mask
    t3_mask = F.interpolate(mask, size=(t3.shape[-2], t3.shape[-1]), mode='nearest')
    t3_masked = t3 + t3_mask

    s2_masked = s2 + t2_mask
    s3_masked = s3 + t3_mask

    return KL_loss(t2_masked, s2_masked) + KL_loss(t3_masked, s3_masked)


def attn_masked_offset_kl_loss(t1, t2, t3, t4, s1, s2, s3, s4, xt):
    """compute the KD loss between teacher attn and student attn with masked, offset -1
        t2_masked -> s1_masked,   t3_masked -> s2_masked
    """
    mask = torch.where(xt != 0, torch.tensor(0.0), torch.tensor(-float('inf')))

    s1 = F.interpolate(s1, size=(t2.shape[-2], t2.shape[-1]), mode='nearest')
    s2 = F.interpolate(s2, size=(t3.shape[-2], t3.shape[-1]), mode='nearest')

    t2_mask = F.interpolate(mask, size=(t2.shape[-2], t2.shape[-1]), mode='nearest')
    t2_masked = t2 + t2_mask
    t3_mask = F.interpolate(mask, size=(t3.shape[-2], t3.shape[-1]), mode='nearest')
    t3_masked = t3 + t3_mask

    s1_masked = s1 + t2_mask
    s2_masked = s2 + t3_mask

    return KL_loss(t2_masked, s1_masked) + KL_loss(t3_masked, s2_masked)

def label_distribute(df):
    male_data = df[df['male'] == 1]
    female_data = df[df['male'] == 0]

    male_vector = torch.zeros(228, dtype=torch.int)
    female_vector = torch.zeros(228, dtype=torch.int)

    for lower, upper in ranges_male:
        # 统计当前范围的样本数量
        count = male_data[(male_data['boneage'] >= lower) & (male_data['boneage'] <= upper)].shape[0]

        # 将该组的样本数量赋值给属于该组范围的 vector 索引
        male_vector[lower:upper + 1] = count

    for lower, upper in ranges_female:
        # 统计当前范围的样本数量
        count_ = female_data[(female_data['boneage'] >= lower) & (female_data['boneage'] <= upper)].shape[0]

        # 将该组的样本数量赋值给属于该组范围的 vector 索引
        female_vector[lower:upper + 1] = count_
    male_max = torch.max(male_vector)
    female_max = torch.max(female_vector)

    male_vector = male_max / male_vector
    female_vector = female_max / female_vector

    # male_vector, female_vector = F.sigmoid(male_vector), F.sigmoid(female_vector)
    #
    # male_min = torch.min(male_vector)
    # female_min = torch.min(female_vector)
    #
    # male_vector = male_vector / male_min
    # female_vector = female_vector / female_min

    return torch.sqrt(male_vector), torch.sqrt(female_vector)


def scale_loss(label, male, male_distribute, female_distribute):

    label = label.view(-1).type(torch.LongTensor)
    male = male.view(-1)
    param = male * male_distribute[label] + (1 - male) * female_distribute[label]

    return param

# import pandas as pd
# if __name__ == '__main__':
#     train_csv = "E:/code/Dataset/RSNA/train_merge.csv"
#     train_df = pd.read_csv(train_csv)
#     label_male, label_female = label_distribute(train_df)
#     print(label_male)
#     print(label_female)
#     loss = torch.ones((8), dtype=torch.float32)
#     label = torch.tensor([1, 48, 108, 192, 1, 36, 156, 192], dtype=torch.float32)
#     male = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
#     print(scale_loss(loss, label, male, label_male, label_female))

def l1_loss(pred, boneage):
    p = 0.5864
    # p = 0.5954
    new_pred = (pred - boneage) * p + boneage
    return new_pred, torch.abs(boneage - new_pred)