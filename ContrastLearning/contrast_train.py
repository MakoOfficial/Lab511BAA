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
from utils import L1_penalty, log_contrast_losses_to_csv, save_attn_KD, save_contrast_attn_6Stage

from ContrastLearning.contrast_model import get_student_contrast_model, get_only_contrast_model
from ContrastLearning.triplet_loss import AdapitiveTripletLoss
from ContrastLearning.WCL import WCL

warnings.filterwarnings("ignore")

flags = {}
flags['lr'] = 5e-4
flags['batch_size'] = 32
flags['num_workers'] = 8
flags['num_epochs'] = 100
flags['data_dir'] = 'C:/BoneAgeAssessment/RSNA'
flags['student_path'] = "./KD_All_Output/KD_modify_firstConv_RandomCrop/KD_modify_firstConv_RandomCrop.bin"
flags['save_path'] = '../KD_All_Output_A5000'
flags['model_name'] = 'Only_Contrast_WCL_IN_CBAM_AVGPool_AdaA_pretrained_1_1'
flags['node'] = '消融蒸馏模块，只使用对比学习'
flags['seed'] = 1
flags['lr_decay_step'] = 10
flags['lr_decay_ratio'] = 0.5
flags['weight_decay'] = 0
flags['best_loss'] = 0
flags['triple_loss_0_lambda'] = 0.4
flags['triple_loss_1_lambda'] = 0.4
flags['WCL_setting'] = dict(p=0.5, tempS=1, thresholdS=0.1, tempW=0.2)


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


def train_fn(train_loader, loss_fn, triple_fn, optimizer):
    '''
    checkpoint is a dict
    '''

    contrast_model.train()
    training_loss = 0
    total_triple_loss_0 = 0
    total_triple_loss_1 = 0
    total_size = 0
    for idx, data in enumerate(train_loader):
        image, gender = data[0]
        image = image.type(torch.FloatTensor).cuda()
        gender = gender.type(torch.FloatTensor).cuda()
        img_gt = data[2].type(torch.FloatTensor).cuda()

        batch_size = len(data[1])
        label = data[1].type(torch.FloatTensor).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # firstly, get attention map from teacher model
        class_feature, cls_token0, cls_token1, _, _, _, _ = contrast_model(image, gender)
        # class_feature, _, _, _, _, _, _ = contrast_model(image, gender)
        y_pred = class_feature.squeeze()
        label = label.squeeze()

        loss = loss_fn(y_pred, label)
        triple_loss_0 = triple_fn(cls_token0, img_gt)
        triple_loss_1 = triple_fn(cls_token1, img_gt)

        # backward,calculate gradients
        penalty_loss = L1_penalty(contrast_model, 1e-5)
        total_loss = loss + penalty_loss
        if triple_loss_0 == 0:
            total_loss = total_loss + triple_loss_0
        else:
            total_loss = total_loss + triple_loss_0.squeeze(0) * flags['triple_loss_0_lambda']
        if triple_loss_1 == 0:
            total_loss = total_loss + triple_loss_1
        else:
            total_loss = total_loss + triple_loss_1.squeeze(0) * flags['triple_loss_1_lambda']
        total_loss.backward()

        # backward,update parameter
        optimizer.step()
        batch_loss = loss.item()
        print(f"batch_loss: {batch_loss}, WCL0: {triple_loss_0.item()}, WCL1: {triple_loss_1.item()}, "
              f"penalty_loss: {penalty_loss.item()}")
        # print(f"batch_loss: {batch_loss}, "
        #       f"penalty_loss: {penalty_loss.item()}")

        training_loss += batch_loss
        total_size += batch_size
        total_triple_loss_0 += triple_loss_0.item()
        total_triple_loss_1 += triple_loss_1.item()

    return training_loss, total_triple_loss_0, total_triple_loss_1, total_size


def evaluate_fn(val_loader):
    contrast_model.eval()

    mae_loss = 0
    val_total_size = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            val_total_size += len(data[1])

            image, gender = data[0]
            image = image.type(torch.FloatTensor).cuda()
            gender = gender.type(torch.FloatTensor).cuda()

            label = data[1].cuda()

            class_feature, cls_token0, cls_token1, s1, s2, s3, s4 = contrast_model(image, gender)
            # class_feature, _, _, s1, s2, s3, s4 = contrast_model(image, gender)
            y_pred = (class_feature * boneage_div) + boneage_mean  # 反归一化为原始标签
            y_pred = y_pred.squeeze()
            label = label.squeeze()
            batch_loss = F.l1_loss(y_pred, label, reduction='sum').item()

            mae_loss += batch_loss

            if batch_idx == len(val_loader) - 1:
                save_attn_KD(s1[0], s2[0], s3[0], s4[0], s1[0], s2[0], s3[0], s4[0], save_path)
                # save_attn_Contrast(t1[0], t2[0], t3[0], t4[0], s1[0], s2[0], s3, s4, save_path)

    return mae_loss, val_total_size


def training_start(flags):
    ## Network, optimizer, and loss function creation
    best_loss = float('inf')
    loss_fn = nn.L1Loss(reduction='sum')
    # triple_fn = AdapitiveTripletLoss()
    wcl_setting = flags['WCL_setting']
    triple_fn = WCL(p=wcl_setting['p'], tempS=wcl_setting['tempS'], thresholdS=wcl_setting['thresholdS'], tempW=wcl_setting['tempW'])


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, contrast_model.parameters()),
                                 lr=flags['lr'], weight_decay=flags['weight_decay'])
    scheduler = StepLR(optimizer, step_size=flags['lr_decay_step'], gamma=flags['lr_decay_ratio'])

    ## Trains
    for epoch in range(flags['num_epochs']):
        print(f"epoch {epoch + 1}")

        ## Training
        start_time = time.time()
        training_loss, training_triple_loss_0, training_triple_loss_1, total_size = train_fn(train_loader, loss_fn, triple_fn, optimizer)

        ## Evaluation
        # Sets net to eval and no grad context
        valid_mae_loss, val_total_size = evaluate_fn(valid_loader)

        save_contrast_attn_6Stage(test_loader=test_loader, model=contrast_model, save_path=save_path)

        training_mean_loss = training_loss / total_size
        mean_triple_loss_0, mean_triple_loss_1 = training_triple_loss_0 / total_size, training_triple_loss_1 / total_size
        valid_mean_mae = valid_mae_loss / val_total_size
        if valid_mean_mae < best_loss:
            best_loss = valid_mean_mae
            torch.save(contrast_model.state_dict(), '/'.join([save_path, f'{model_name}.bin']))
            flags['best_loss'] = best_loss
            with open(os.path.join(save_path, 'setting.txt'), 'w') as f:
                f.writelines('------------------ start ------------------' + '\n')
                for eachArg, value in flags.items():
                    f.writelines(eachArg + ' : ' + str(value) + '\n')
                f.writelines('------------------- end -------------------')

        log_contrast_losses_to_csv(training_mean_loss, mean_triple_loss_0,
                                   mean_triple_loss_1, valid_mean_mae,
                                   time.time() - start_time,
                                   optimizer.param_groups[0]["lr"], os.path.join(save_path, "KD_loss.csv"))
        scheduler.step()

    print(f'best loss: {best_loss}')


if __name__ == "__main__":
    # set save dir of this train
    model_name = flags['model_name']
    save_path = os.path.join(flags['save_path'], model_name)
    os.makedirs(save_path, exist_ok=True)
    #   prepare contrast learning model
    # contrast_model = get_student_contrast_model(student_path=flags['student_path']).cuda()
    contrast_model = get_only_contrast_model(student_path=flags['student_path']).cuda()
    #   load data setting
    data_dir = flags['data_dir']

    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")

    train_csv = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_csv)
    valid_csv = os.path.join(data_dir, "valid.csv")
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
        drop_last=True,
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
        test_set,
        batch_size=12,
        shuffle=False,
        pin_memory=True
    )

    training_start(flags)
