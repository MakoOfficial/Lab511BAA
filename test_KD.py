import os
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets import RSNATestDataset
from utils import log_valid_result_to_csv

from Student.student_model import get_student
from Unet.UNets import get_Attn_Unet

warnings.filterwarnings("ignore")

flags = {}
flags['batch_size'] = 32
flags['num_workers'] = 8
flags['data_dir'] = '../Dataset/RSNA'
flags['teacher_path'] = "./ckp/Unet/unet_segmentation_Attn_UNet.pth"
flags['student_path'] = "./KD_All_Output/KD_modify_firstConv_RandomCrop/KD_modify_firstConv_RandomCrop.bin"
flags['mask_option'] = False


def evaluate_fn(val_loader):
    student_model.eval()

    log_path = os.path.join(ckp_dir, "valid_large.csv")

    mae_loss = 0
    val_total_size = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            batch_size = len(data[1])
            val_total_size += batch_size
            image, gender = data[0]
            image = image.type(torch.FloatTensor).cuda()
            gender = gender.type(torch.FloatTensor).cuda()
            id = data[2]

            label = data[1].cuda()

            _, _, _, _, _, _, t1, t2, t3, t4 = teacher.forward_attention(image)
            class_feature, s1, s2, s3, s4 = student_model(image, gender)
            y_pred = (class_feature * boneage_div) + boneage_mean  # 反归一化为原始标签

            y_pred = y_pred.squeeze()
            label = label.squeeze()
            batch_loss = F.l1_loss(y_pred, label, reduction='none')
            mae_loss += batch_loss.sum().item()

            log_valid_result_to_csv(id, label.cpu(), gender.cpu(), y_pred.cpu(), batch_loss.cpu(), log_path)

    print(f"valid loss: {mae_loss / val_total_size}")


if __name__ == "__main__":
    # set save dir of this train
    ckp_dir = os.path.dirname(flags['student_path'])
    #   prepare teacher model
    teacher_path = flags['teacher_path']
    teacher = get_Attn_Unet().cuda()
    teacher.load_state_dict(torch.load(teacher_path), strict=True)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    #   prepare student model
    student_path = flags['student_path']
    student_model = get_student().cuda()
    student_model.load_state_dict(torch.load(student_path), strict=True)
    for param in student_model.parameters():
        param.requires_grad = False
    student_model.eval()
    #   load data setting
    data_dir = flags['data_dir']

    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")

    train_csv = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_csv)
    valid_csv = os.path.join(data_dir, "valid.csv")
    # valid_df = pd.read_csv("KD_All_Output/KD_modify_firstConv_RandomCrop/valid_loss_2.csv")
    valid_df = pd.read_csv(valid_csv)


    boneage_mean = train_df['boneage'].mean()
    boneage_div = train_df['boneage'].std()
    print(f"boneage_mean is {boneage_mean}")
    print(f"boneage_div is {boneage_div}")
    print(f'valid file save at {ckp_dir}')

    Test_set = RSNATestDataset(valid_df, valid_path, boneage_mean, boneage_div)
    print(f"Test set length: {Test_set.__len__()}")

    valid_loader = torch.utils.data.DataLoader(
        Test_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        pin_memory=True
    )

    evaluate_fn(valid_loader)
