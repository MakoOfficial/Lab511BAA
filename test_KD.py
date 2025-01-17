import os
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets import RSNATestDataset, DHADataset
from utils import log_valid_result_to_csv, save_attn_all, save_attn_all_KD, log_valid_result_logits_to_csv, l1_loss

from Student.student_model import get_student, get_student_res18, get_student_gate, get_student_squeeze
from ContrastLearning.contrast_model import get_student_contrast_model
from Unet.UNets import get_Attn_Unet

warnings.filterwarnings("ignore")

flags = {}
flags['batch_size'] = 32
flags['num_workers'] = 8
flags['data_dir'] = '../Dataset/RSNA'
flags['DHA_dir'] = 'E:/code/Dataset/DHA/Digital Hand Atlas'
flags['teacher_path'] = "./ckp/Unet/unet_segmentation_Attn_UNet.pth"
# flags['student_path'] = "./KD_All_Output/KD_modify_firstConv_RandomCrop/KD_modify_firstConv_RandomCrop.bin"
# flags['student_path'] = "./Student/baseline/Res50_All.bin"
# flags['student_path'] = "./KD_All_Output/KD_Res18_3090/KD_Res18.bin"
flags['student_path'] = "./Student/baseline/ResNet50_256_Squeeze_4K/ResNet50_256_Squeeze_4K.bin"
flags['csv_name'] = "ResNet50_256_Squeeze_train.csv"
flags['mask_option'] = False
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

    # mae_loss = torch.zeros(13, dtype=torch.float32).cuda()
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
            # class_feature, _, _, s1, s2, s3, s4 = student_model(image, gender)    # 对比使用
            # class_feature, s1, s2, s3, s4 = student_model(image, gender)    # 蒸馏使用
            class_feature, _, s1, s2, s3, s4 = student_model(image, gender)  # Squeeze使用
            y_pred = (class_feature * boneage_div) + boneage_mean  # 反归一化为原始标签

            y_pred = y_pred.squeeze()
            label = label.squeeze()

            # label_expand = expand_and_add_indices(label)
            # y_pred = y_pred.unsqueeze(1).repeat(1, 13)
            y_pred, batch_loss = l1_loss(y_pred, label)
            # batch_loss = F.l1_loss(y_pred, label, reduction='none')
            # mae_loss += batch_loss.sum(dim=0)
            mae_loss += batch_loss.sum().item()
            # logits_list = torch.norm(logits, p=2, dim=1)

            # print(mae_loss)

            log_valid_result_to_csv(id, label.cpu(), gender.cpu(), y_pred.cpu(), batch_loss.cpu(), log_path)
            # log_valid_result_logits_to_csv(id, label.cpu(), gender.cpu(), y_pred.cpu(), batch_loss.cpu(), logits_list.cpu(), log_path)
            # save_attn_all_KD(s1, s2, s3, s4, id, ckp_dir)
    mae_loss = mae_loss / val_total_size
    # best_idx = torch.argmin(mae_loss)
    # print(f"valid loss: {mae_loss}, best_idx: {best_idx}")
    print(f"valid loss: {mae_loss}")


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
    # student_model = get_student().cuda()
    student_model = get_student_squeeze().cuda()
    # student_model = get_student_res18().cuda()
    student_model.load_state_dict(torch.load(student_path), strict=True)

    # student_model = get_student_contrast_model(student_path).cuda()
    # contrast_path = flags['contrast_path']
    # student_model.load_state_dict(torch.load(contrast_path), strict=True)
    # ckp_dir = os.path.dirname(flags['contrast_path'])
    for param in student_model.parameters():
        param.requires_grad = False
    student_model.eval()
    #   load data setting
    data_dir = flags['data_dir']

    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")

    train_csv = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_csv)

    ori_train_csv = os.path.join(data_dir, "train_4K.csv")
    ori_train_df = pd.read_csv(ori_train_csv)

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


    # boneage_mean = train_df['boneage'].mean()
    # boneage_div = train_df['boneage'].std()

    boneage_mean = ori_train_df['boneage'].mean()
    boneage_div = ori_train_df['boneage'].std()


    print(f"boneage_mean is {boneage_mean}")
    print(f"boneage_div is {boneage_div}")
    print(f'valid file save at {ckp_dir}')

    train_set = valid_Dataset(train_df, train_path, boneage_mean, boneage_div, 256)
    Test_set = valid_Dataset(valid_df, valid_path, boneage_mean, boneage_div, 256)
    print(f"Test set length: {Test_set.__len__()}")
    print(f"Train set length: {train_set.__len__()}")
    # print(f"Test set length: 1425")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )


    valid_loader = torch.utils.data.DataLoader(
        Test_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        pin_memory=True
    )

    evaluate_fn(train_loader)
