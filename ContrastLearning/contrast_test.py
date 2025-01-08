import os
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets import RSNATestDataset, DHADataset
from utils import log_valid_result_to_csv, save_attn_all, save_attn_all_KD, log_valid_result_logits_to_csv, l1_loss

from Student.student_model import get_student, get_student_res18
from ContrastLearning.contrast_model import get_student_contrast_model
from Unet.UNets import get_Attn_Unet

warnings.filterwarnings("ignore")

flags = {}
flags['batch_size'] = 32
flags['num_workers'] = 8
flags['data_dir'] = '../../Dataset/RSNA'
flags['DHA_dir'] = 'E:/code/Dataset/DHA/Digital Hand Atlas'
flags['student_path'] = "../KD_All_Output/KD_modify_firstConv_RandomCrop/KD_modify_firstConv_RandomCrop.bin"
flags['contrast_path'] = "../Contrast_Output/Contrast_WCL_IN_CBAM_AVGPool_AdaA_GenderPlus_4K_1_7_96/Contrast_WCL_IN_CBAM_AVGPool_AdaA_GenderPlus_4K_1_7_96.bin"

flags['csv_name'] = "Contrast_Gender_96_train.csv"
flags['DHA_option'] = False


def evaluate_fn(val_loader):
    student_model.eval()

    log_path = os.path.join(ckp_dir, flags['csv_name'])

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

            # _, _, _, _, _, _, t1, t2, t3, t4 = teacher.forward_attention(image)
            class_feature, _, _, s1, s2, s3, s4 = student_model(image, gender)    # 对比使用
            # class_feature, s1, s2, s3, s4 = student_model(image, gender)    # 蒸馏使用
            y_pred = (class_feature * boneage_div) + boneage_mean  # 反归一化为原始标签

            y_pred = y_pred.squeeze()
            label = label.squeeze()

            y_pred, batch_loss = l1_loss(y_pred, label)
            mae_loss += batch_loss.sum().item()

            log_valid_result_to_csv(id, label.cpu(), gender.cpu(), y_pred.cpu(), batch_loss.cpu(), log_path)
            # log_valid_result_logits_to_csv(id, label.cpu(), gender.cpu(), y_pred.cpu(), batch_loss.cpu(), logits_list.cpu(), log_path)
            # save_attn_all_KD(s1, s2, s3, s4, id, ckp_dir)
    mae_loss = mae_loss / val_total_size
    print(f"valid loss: {mae_loss}")


if __name__ == "__main__":
    # set save dir of this train

    ckp_dir = os.path.dirname(flags['contrast_path'])
    #   prepare student model

    student_path = flags['student_path']
    student_model = get_student_contrast_model(student_path).cuda()
    contrast_path = flags['contrast_path']
    student_model.load_state_dict(torch.load(contrast_path), strict=True)
    for param in student_model.parameters():
        param.requires_grad = False
    student_model.eval()
    #   load data setting
    data_dir = flags['data_dir']

    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")

    train_csv_ori = os.path.join(data_dir, "train_4K.csv")
    train_df_ori = pd.read_csv(train_csv_ori)
    # train_merge_df = pd.read_csv("E:/code/Dataset/RSNA/train_merge.csv")

    if flags['DHA_option']:
        valid_csv = os.path.join(flags['DHA_dir'], "label.csv")
        valid_df = pd.read_csv(valid_csv)
        valid_path = os.path.join(flags['DHA_dir'], "archive")
        valid_Dataset = DHADataset
        ckp_dir = flags['DHA_dir']
    else:
        valid_csv = os.path.join(data_dir, "valid.csv")
        # valid_df = pd.read_csv("KD_All_Output/KD_modify_firstConv_RandomCrop/valid_loss_2.csv")
        valid_df = pd.read_csv(valid_csv)
        valid_Dataset = RSNATestDataset


    boneage_mean = train_df_ori['boneage'].mean()
    boneage_div = train_df_ori['boneage'].std()


    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))


    print(f"boneage_mean is {boneage_mean}")
    print(f"boneage_div is {boneage_div}")
    print(f'valid file save at {ckp_dir}')

    train_set = valid_Dataset(train_df, train_path, boneage_mean, boneage_div)
    Test_set = valid_Dataset(valid_df, valid_path, boneage_mean, boneage_div)
    print(f"Test set length: {train_set.__len__()}")
    # print(f"Test set length: 1425")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        Test_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        pin_memory=True
    )

    # evaluate_fn(valid_loader)
    evaluate_fn(train_loader)
