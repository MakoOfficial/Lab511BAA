import os
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets import RSNATestDataset, DHADataset
from utils import log_valid_result_to_csv, save_attn_6Stage, save_attn_all, save_attn_all_KD

from Student.student_model import get_student
from ContrastLearning.contrast_model import get_student_GCN, get_student_contrast_model
from Unet.UNets import get_Attn_Unet

warnings.filterwarnings("ignore")

flags = {}
flags['batch_size'] = 32
flags['num_workers'] = 8
flags['data_dir'] = '../../Dataset/RSNA'
flags['DHA_dir'] = 'E:/code/Dataset/DHA/Digital Hand Atlas'
flags['teacher_path'] = "../ckp/Unet/unet_segmentation_Attn_UNet.pth"
flags['backbone_path'] = "../KD_All_Output/KD_modify_firstConv_RandomCrop/KD_modify_firstConv_RandomCrop.bin"
flags['model'] = "../KD_All_Output/Contrast_WCL_IN_CBAM_AVGPool_AdaA_pretrained_DropLast_validNew_12-26/Contrast_WCL_IN_CBAM_AVGPool_AdaA_pretrained_DropLast_12-26.bin"
flags['mask_option'] = False
flags['csv_name'] = "Contrast_Result.csv"
flags['DHA_option'] = False


def expand_and_add_indices(tensor):
    B = tensor.shape[0]
    expanded_tensor = tensor.unsqueeze(1).repeat(1, 13)  # 扩展成 B×13
    indices = torch.arange(13).to(tensor.device)  # 生成列的索引 [0, 1, ..., 12]
    result = expanded_tensor + indices  # 每一列加上对应的索引
    return result


def evaluate_fn(val_loader):
    student_model.eval()

    log_path = os.path.join(ckp_dir, flags['csv_name'])

    mae_loss = 0
    # mae_loss = torch.zeros(13, dtype=torch.float32).cuda()
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
            class_feature, cls_token2, cls_token3, s1, s2, s3, s4 = student_model(image, gender)
            y_pred = (class_feature * boneage_div) + boneage_mean  # 反归一化为原始标签
            cls_token2 = (cls_token2 * boneage_div) + boneage_mean
            cls_token3 = (cls_token3 * boneage_div) + boneage_mean
            batch_pred = torch.cat((y_pred, cls_token2, cls_token3))
            y_pred = y_pred.squeeze()
            cls_token2 = cls_token2.squeeze()
            cls_token3 = cls_token3.squeeze()
            label = label.squeeze()



            # label_expand = expand_and_add_indices(label)
            # y_pred = y_pred.unsqueeze(1).repeat(1, 13)

            y_pred_loss = F.l1_loss(y_pred, label, reduction='none').unsqueeze(1)
            cls_token2_loss = F.l1_loss(cls_token2, label, reduction='none').unsqueeze(1)
            cls_token3_loss = F.l1_loss(cls_token3, label, reduction='none').unsqueeze(1)

            batch_loss = torch.cat((y_pred_loss, cls_token2_loss, cls_token3_loss))
            batch_loss, indices = torch.min(batch_loss, dim=1)
            y_pred = batch_pred[torch.arange(batch_pred.size(0)), indices].squeeze()

            # mae_loss += batch_loss.sum(dim=0)
            mae_loss += batch_loss.squeeze().sum().item()

            log_valid_result_to_csv(id, label.cpu(), gender.cpu(), y_pred.cpu(), batch_loss.cpu(), log_path)
            # save_attn_all(s3, s4, id, save_path=ckp_dir)
            # save_attn_all_KD(s1, s2, s3, s4, id, save_path=ckp_dir)

    mae_loss = mae_loss / val_total_size
    # best_idx = torch.argmin(mae_loss)
    # print(f"valid loss: {mae_loss}, best_idx: {best_idx}")
    print(f"valid loss: {mae_loss}")


if __name__ == "__main__":
    # set save dir of this train
    ckp_dir = os.path.dirname(flags['model'])
    #   prepare teacher model
    teacher_path = flags['teacher_path']
    teacher = get_Attn_Unet().cuda()
    teacher.load_state_dict(torch.load(teacher_path), strict=True)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    #   prepare student model
    student_path = flags['model']
    # student_model = get_student_GCN(backbone_path=flags['backbone_path']).cuda()
    student_model = get_student_contrast_model(student_path=flags['backbone_path']).cuda()
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

    test_csv = os.path.join(data_dir, "valid_test.csv")
    test_df = pd.read_csv(test_csv)


    boneage_mean = train_df['boneage'].mean()
    boneage_div = train_df['boneage'].std()
    print(f"boneage_mean is {boneage_mean}")
    print(f"boneage_div is {boneage_div}")
    print(f'valid file save at {ckp_dir}')

    Test_set = valid_Dataset(valid_df, valid_path, boneage_mean, boneage_div)
    # stage6_set = RSNATestDataset(test_df, valid_path, boneage_mean, boneage_div)

    print(f"Test set length: {Test_set.__len__()}")
    # print(f"Test set length: 1425")

    valid_loader = torch.utils.data.DataLoader(
        Test_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        pin_memory=True
    )

    # test_loader = torch.utils.data.DataLoader(
    #     stage6_set,
    #     batch_size=12,
    #     shuffle=False,
    #     pin_memory=True
    # )
    evaluate_fn(valid_loader)

    # save_attn_6Stage(test_loader, student_model, ckp_dir)

