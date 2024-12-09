import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

from datasets import RSNATrainDataset, RSNAValidDataset
from utils import L1_penalty, log_losses_to_csv, save_attn_KD, \
    attn_offset_kl_loss_firstStage, save_attn_Contrast, save_attn_6Stage

from Student.student_model import get_student, get_student_res18
from ContrastLearning.contrast_model import get_student_GCN
from Unet.UNets import get_Attn_Unet

warnings.filterwarnings("ignore")

flags = {}
flags['lr'] = 5e-4
flags['batch_size'] = 32
flags['num_workers'] = 8
flags['num_epochs'] = 100
flags['data_dir'] = '../archive'
flags['teacher_path'] = "../unet_segmentation_Attn_UNet_RSNA_256.pth"
flags['backbone_path'] = "./KD_All_Output/KD_modify_firstConv_RandomCrop/KD_modify_firstConv_RandomCrop.bin"
flags['save_path'] = './KD_All_Output_3090'
flags['model_name'] = 'KD_Res50_CBAM_BSPC_only_CLS_AVGPool_DelFFN_StackViT_pretrained'
flags['seed'] = 1
flags['lr_decay_step'] = 10
flags['lr_decay_ratio'] = 0.5
flags['weight_decay'] = 0
flags['best_loss'] = 0
flags['mask_option'] = False
flags['attn_loss_ratio'] = 40

seed = flags['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
np.random.seed(seed)  # numpy产生的随机数一致
random.seed(seed)

# CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
# 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
torch.backends.cudnn.deterministic = True

# 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
# 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
torch.backends.cudnn.benchmark = False


def train_fn(train_loader, loss_fn, optimizer):
    '''
    checkpoint is a dict
    '''

    student_model.train()
    training_loss = 0
    attention_loss = 0
    total_size = 0
    for idx, data in enumerate(train_loader):
        image, gender = data[0]
        image = image.type(torch.FloatTensor).cuda()
        gender = gender.type(torch.FloatTensor).cuda()

        batch_size = len(data[1])
        label = data[1].type(torch.FloatTensor).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # firstly, get attention map from teacher model
        _, _, _, _, _, _, t1, t2, t3, t4 = teacher.forward_attention(image)
        class_feature, s1, s2 = student_model(image, gender)
        y_pred = class_feature.squeeze()
        label = label.squeeze()

        loss = loss_fn(y_pred, label)

        # backward,calculate gradients
        penalty_loss = L1_penalty(student_model, 1e-5)
        total_loss = loss + penalty_loss
        total_loss.backward()

        # backward,update parameter
        optimizer.step()
        batch_loss = loss.item()
        print(f"batch_loss: {batch_loss}, penalty_loss: {penalty_loss.item()}")

        training_loss += batch_loss
        total_size += batch_size

    return training_loss, attention_loss, total_size


def evaluate_fn(val_loader):
    student_model.eval()

    mae_loss = 0
    val_total_size = 0
    attn_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            val_total_size += len(data[1])

            image, gender = data[0]
            image = image.type(torch.FloatTensor).cuda()
            gender = gender.type(torch.FloatTensor).cuda()

            label = data[1].cuda()

            _, _, _, _, _, _, t1, t2, t3, t4 = teacher.forward_attention(image)
            class_feature, s1, s2, s3, s4 = student_model.infer(image, gender)
            y_pred = (class_feature * boneage_div) + boneage_mean  # 反归一化为原始标签
            y_pred = y_pred.squeeze()
            label = label.squeeze()
            batch_loss = F.l1_loss(y_pred, label, reduction='sum').item()

            mae_loss += batch_loss

            # if batch_idx == len(val_loader) - 1:
                # save_attn_KD(t1[0], t2[0], t3[0], t4[0], s1[0], s2[0], s3[0], s4[0], save_path)
                # save_attn_Contrast(s3, s4, save_path)

    return mae_loss, attn_loss, val_total_size


def training_start(flags):
    ## Network, optimizer, and loss function creation
    best_loss = float('inf')
    loss_fn = nn.L1Loss(reduction='sum')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=flags['lr'], weight_decay=flags['weight_decay'])
    scheduler = StepLR(optimizer, step_size=flags['lr_decay_step'], gamma=flags['lr_decay_ratio'])

    ## Trains
    for epoch in range(flags['num_epochs']):
        print(f"epoch {epoch + 1}")

        ## Training
        start_time = time.time()
        training_loss, training_attn_loss, total_size = train_fn(train_loader, loss_fn, optimizer)

        ## Evaluation
        # Sets net to eval and no grad context
        valid_mae_loss, valid_attn_loss, val_total_size = evaluate_fn(valid_loader)

        save_attn_6Stage(test_loader=test_loader, model=student_model, save_path=save_path)

        training_mean_loss, training_mean_attn_loss = training_loss / total_size, training_attn_loss / total_size
        valid_mean_mae, valid_mean_attn_loss = valid_mae_loss / val_total_size, valid_attn_loss / val_total_size
        if valid_mean_mae < best_loss:
            best_loss = valid_mean_mae
            torch.save(student_model.state_dict(), '/'.join([save_path, f'{model_name}.bin']))
            flags['best_loss'] = best_loss
            with open(os.path.join(save_path, 'setting.txt'), 'w') as f:
                f.writelines('------------------ start ------------------' + '\n')
                for eachArg, value in flags.items():
                    f.writelines(eachArg + ' : ' + str(value) + '\n')
                f.writelines('------------------- end -------------------')

        log_losses_to_csv(training_mean_loss, training_mean_attn_loss,
                          valid_mean_mae, valid_mean_attn_loss,
                          time.time() - start_time,
                          optimizer.param_groups[0]["lr"], os.path.join(save_path, "KD_loss.csv"))
        scheduler.step()

    print(f'best loss: {best_loss}')


if __name__ == "__main__":
    # set save dir of this train
    model_name = flags['model_name']
    save_path = os.path.join(flags['save_path'], model_name)
    os.makedirs(save_path, exist_ok=True)
    #   prepare teacher model
    teacher_path = flags['teacher_path']
    teacher = get_Attn_Unet().cuda()
    teacher.load_state_dict(torch.load(teacher_path), strict=True)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    #   prepare student model
    # student_model = get_student().cuda()
    # student_model = get_student_res18().cuda()
    student_model = get_student_GCN(backbone_path=flags['backbone_path']).cuda()
    #   load data setting
    data_dir = flags['data_dir']

    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")

    train_csv = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_csv)
    valid_csv = os.path.join(data_dir, "valid_new.csv")
    valid_df = pd.read_csv(valid_csv)

    test_csv = os.path.join(data_dir, "valid_test.csv")
    test_df = pd.read_csv(test_csv)

    boneage_mean = train_df['boneage'].mean()
    boneage_div = train_df['boneage'].std()
    print(f"boneage_mean is {boneage_mean}")
    print(f"boneage_div is {boneage_div}")
    print(f'{save_path} start')

    train_set = RSNATrainDataset(train_df, train_path, boneage_mean, boneage_div)
    valid_set = RSNAValidDataset(valid_df, valid_path, boneage_mean, boneage_div)
    test_set = RSNAValidDataset(test_df, valid_path, boneage_mean, boneage_div)

    print(train_set.__len__())

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=flags['batch_size'],
        shuffle=True,
        num_workers=flags['num_workers'],
        drop_last=False,
        pin_memory=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        num_workers=flags['num_workers'],
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=12,
        shuffle=False,
        pin_memory=True
    )

    training_start(flags)
