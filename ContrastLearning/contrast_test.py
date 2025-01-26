import os
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets import RSNATestDataset, DHADataset
from utils import log_valid_result_to_csv, save_attn_all, save_attn_all_KD, log_valid_result_logits_to_csv, l1_loss, show_attn_all_KD, l1_test_loss, save_s3_attnImg_6Stage, save_s3_attnImg

from Student.student_model import get_student, get_student_res18
from ContrastLearning.contrast_model import get_student_contrast_model, get_student_contrast_model_pretrain, get_student_contrast_model_pretrain_vit, get_student_contrast_model_pretrain_gcn
from Unet.UNets import get_Attn_Unet

warnings.filterwarnings("ignore")

flags = {}
flags['batch_size'] = 32
flags['num_workers'] = 8
flags['data_dir'] = '../../Dataset/RSNA'
flags['DHA_dir'] = 'E:/code/Dataset/DHA/Digital Hand Atlas'
flags['student_path'] = "../KD_All_Output/KD_modify_firstConv_RandomCrop/KD_modify_firstConv_RandomCrop.bin"
flags['contrast_path'] = "../Contrast_Output/Contrast_WCL_IN_CBAM_AVGPool_AdaA_GenderPlus_Full_1_11_96_Pretrain/Contrast_WCL_IN_CBAM_AVGPool_AdaA_GenderPlus_Full_1_11_96_Pretrain.bin"

flags['csv_name'] = "train_output.csv"
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
            # batch_loss = F.l1_loss(y_pred, label, reduction="none")
            # y_pred, batch_loss = l1_test_loss(y_pred, label)
            mae_loss += batch_loss.sum().item()

            # log_valid_result_to_csv(id, label.cpu(), gender.cpu(), y_pred.cpu(), batch_loss.cpu(), log_path)
            # log_valid_result_logits_to_csv(id, label.cpu(), gender.cpu(), y_pred.cpu(), batch_loss.cpu(), logits_list.cpu(), log_path)
            # save_attn_all_KD(s1, s2, s3, s4, id, ckp_dir)
            save_s3_attnImg(s3, label.cpu(), gender.cpu(), id, train_attn, data_path=train_set.file_path)
            # show_attn_all_KD(s1[5], s2[5], s3[5], s4[5], id[5], ckp_dir)
    mae_loss = mae_loss / val_total_size
    print(f"valid loss: {mae_loss}")


if __name__ == "__main__":
    # set save dir of this train

    ckp_dir = os.path.dirname(flags['contrast_path'])
    train_attn = os.path.join(ckp_dir, "train_attn")
    #   prepare student model

    student_path = flags['student_path']
    student_model = get_student_contrast_model_pretrain(student_path).cuda()
    # student_model = get_student_contrast_model_pretrain_vit(student_path).cuda()
    # student_model = get_student_contrast_model_pretrain_gcn(student_path).cuda()

    contrast_path = flags['contrast_path']
    student_model.load_state_dict(torch.load(contrast_path), strict=True)
    for param in student_model.parameters():
        param.requires_grad = False
    student_model.eval()
    #   load data setting
    data_dir = flags['data_dir']

    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")
    test_path = os.path.join(data_dir, "test")

    # train_csv_ori = os.path.join(data_dir, "train_4K.csv")
    # train_df_ori = pd.read_csv(train_csv_ori)
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    valid_df = pd.read_csv(os.path.join(data_dir, "valid.csv"))
    valid_test_df = pd.read_csv(os.path.join(data_dir, "valid_test.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    valid_Dataset = RSNATestDataset


    boneage_mean = train_df['boneage'].mean()
    boneage_div = train_df['boneage'].std()

    print(f"boneage_mean is {boneage_mean}")
    print(f"boneage_div is {boneage_div}")
    print(f'valid file save at {ckp_dir}')

    train_set = valid_Dataset(train_df, train_path, boneage_mean, boneage_div, 256)
    valid_set = valid_Dataset(valid_df, valid_path, boneage_mean, boneage_div, 256)
    valid_test_set = valid_Dataset(valid_test_df, valid_path, boneage_mean, boneage_div, 256)
    test_set = valid_Dataset(test_df, test_path, boneage_mean, boneage_div, 256)
    print(f"Train set length: {train_set.__len__()}")
    print(f"Valid set length: {valid_set.__len__()}")
    print(f"Test set length: {test_set.__len__()}")
    # print(f"Test set length: 1425")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        pin_memory=True
    )

    valid_test_loader = torch.utils.data.DataLoader(
        valid_test_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        pin_memory=True
    )

    # evaluate_fn(valid_loader)
    evaluate_fn(train_loader)
    # evaluate_fn(test_loader)
    # s3_dir = os.path.join(ckp_dir, "s3_dir")
    # os.makedirs(s3_dir, exist_ok=True)
    # save_s3_attnImg_6Stage(test_loader=valid_test_loader, model=student_model, save_path=s3_dir, data_path=valid_test_set.file_path)

