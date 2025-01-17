import os
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets import RSNATestDataset, DHADataset
from utils import log_valid_result_to_csv, save_attn_all, save_attn_all_KD, log_valid_result_logits_to_csv, l1_loss, l1_test_loss

from Final_Regression.final_regression_model import get_final_regression

warnings.filterwarnings("ignore")

flags = {}
flags['batch_size'] = 32
flags['num_workers'] = 4
flags['data_dir'] = '../../Dataset/RSNA'
flags['DHA_dir'] = 'E:/code/Dataset/DHA/Digital Hand Atlas'
flags['backbone_path'] = '../Contrast_Output/Contrast_WCL_IN_CBAM_AVGPool_AdaA_GenderPlus_Full_1_11_96_Pretrain/Contrast_WCL_IN_CBAM_AVGPool_AdaA_GenderPlus_Full_1_11_96_Pretrain.bin'
flags['student_path'] = "./model/Contrast_WCL_IN_CBAM_AVGPool_AdaA_GenderPlus_Full_96_Pretrain_FinalRegression_ViT_192_1_17/Contrast_WCL_IN_CBAM_AVGPool_AdaA_GenderPlus_Full_96_Pretrain_FinalRegression_ViT_192_1_17.bin"
flags['csv_name'] = "FinalR_Full_ViT_test.csv"
flags['mask_option'] = False
flags['DHA_option'] = False


def evaluate_fn(val_loader, l1_fn):
    contrast_model.eval()

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

            # class_feature, _, _, s1, s2, s3, s4 = contrast_model(image, gender)
            class_feature, attn_list = contrast_model(image, gender)
            y_pred = (class_feature * boneage_div) + boneage_mean  # 反归一化为原始标签

            y_pred = y_pred.squeeze()
            label = label.squeeze()
            y_pred, batch_loss = l1_fn(y_pred, label)
            mae_loss += batch_loss.sum().item()

            log_valid_result_to_csv(id, label.cpu(), gender.cpu(), y_pred.cpu(), batch_loss.cpu(), log_path)
            # save_attn_all_KD(s1, s2, s3, s4, id, ckp_dir)
    mae_loss = mae_loss / val_total_size
    print(f"valid loss: {mae_loss}")


if __name__ == "__main__":
    # set save dir of this train
    ckp_dir = os.path.dirname(flags['student_path'])
    #   prepare student model
    student_path = flags['student_path']
    contrast_model = get_final_regression(backbone_path=flags['backbone_path']).cuda()
    contrast_model.load_state_dict(torch.load(student_path), strict=True)
    for param in contrast_model.parameters():
        param.requires_grad = False
    contrast_model.eval()
    #   load data setting
    data_dir = flags['data_dir']

    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")
    test_path = os.path.join(data_dir, "test")

    train_csv = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_csv)
    valid_csv = os.path.join(data_dir, "valid.csv")
    valid_df = pd.read_csv(valid_csv)
    test_csv = os.path.join(data_dir, "test.csv")
    test_df = pd.read_csv(test_csv)
    valid_Dataset = RSNATestDataset

    boneage_mean = train_df['boneage'].mean()
    boneage_div = train_df['boneage'].std()

    print(f"boneage_mean is {boneage_mean}")
    print(f"boneage_div is {boneage_div}")
    print(f'valid file save at {ckp_dir}')

    train_set = valid_Dataset(train_df, train_path, boneage_mean, boneage_div, 256)
    valid_set = valid_Dataset(valid_df, valid_path, boneage_mean, boneage_div, 256)
    test_set = valid_Dataset(test_df, test_path, boneage_mean, boneage_div, 256)
    print(f"Train set length: {train_set.__len__()}")
    print(f"Valid set length: {valid_set.__len__()}")
    print(f"Test set length: {test_df.__len__()}")
    # print(f"Test set length: 1425")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
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

    evaluate_fn(valid_loader, l1_loss)
