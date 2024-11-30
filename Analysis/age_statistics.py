import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import torch
torch.set_printoptions(sci_mode=False)

data_dir = "E:/code/Dataset/RSNA_large"

#   valid_csv_large + valid_csv_correct = valid_csv_all
# valid_csv_large = pd.read_csv(os.path.join(data_dir, "valid.csv"))
valid_csv_all = pd.read_csv('../KD_All_Output/KD_modify_firstConv_RandomCrop/valid_loss.csv')
# valid_csv_correct = pd.read_csv('../KD_All_Output/KD_modify_firstConv_RandomCrop/valid_loss_2.csv')
#
# valid_csv_large_loss = torch.zeros((2, 229), dtype=torch.float32)
valid_csv_all_loss_male = torch.zeros((3, 229), dtype=torch.float32)
valid_csv_all_loss_female = torch.zeros((3, 229), dtype=torch.float32)
# valid_csv_correct_loss = torch.zeros((3, 229), dtype=torch.float32)


def statistics_loss(df, loss_dict):
    length = len(df)
    for i in range(length):
        row = df.iloc[i]
        boneage = int(row["boneage"])
        loss = row["loss"]
        loss_dict[0][boneage] += loss
        loss_dict[1][boneage] += 1
    for i in range(228):
        if loss_dict[0][i] > 0.:
            month_loss = loss_dict[0][i]
            month_num = loss_dict[1][i]
            loss_dict[2][i] = month_loss / month_num
    return loss_dict


def statistics_loss_by_gender(df, loss_dict_male, loss_dict_female):
    length = len(df)
    for i in range(length):
        row = df.iloc[i]
        boneage = int(row["boneage"])
        loss = row["loss"]
        gender = row["male"]
        if gender > 0.:
            loss_dict_male[0][boneage] += loss
            loss_dict_male[1][boneage] += 1
        else:
            loss_dict_female[0][boneage] += loss
            loss_dict_female[1][boneage] += 1
    for i in range(229):
        if loss_dict_male[0][i] > 0.:
            month_loss = loss_dict_male[0][i]
            month_num = loss_dict_male[1][i]
            loss_dict_male[2][i] = month_loss / month_num
    for i in range(229):
        if loss_dict_female[0][i] > 0.:
            month_loss = loss_dict_female[0][i]
            month_num = loss_dict_female[1][i]
            loss_dict_female[2][i] = month_loss / month_num
    return loss_dict_male, loss_dict_female


# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(36, 12))
#
# ax1, ax2, ax3, ax4 = axes.flatten()
# ax1.bar(range(229), valid_csv_all_loss_male[2])
# ax1.set_xlabel('label')
# ax1.set_ylabel('MAE')
# ax1.set_title('male valid result')
# ax1.set_ylim(0, 40)
# ax1.xaxis.set_major_locator(MultipleLocator(12))
#
# ax3.bar(range(229), valid_csv_all_loss_male[1])
# ax3.set_xlabel('label')
# ax3.set_ylabel('num')
# ax3.set_title('male valid count')
# ax3.xaxis.set_major_locator(MultipleLocator(12))
#
# ax2.bar(range(229), valid_csv_all_loss_female[2])
# ax2.set_xlabel('label')
# ax2.set_ylabel('MAE')
# ax2.set_title('female valid result')
# ax2.set_ylim(0, 40)
# ax2.xaxis.set_major_locator(MultipleLocator(12))
#
# ax4.bar(range(229), valid_csv_all_loss_female[1])
# ax4.set_xlabel('label')
# ax4.set_ylabel('num')
# ax4.set_title('female valid count')
# ax4.xaxis.set_major_locator(MultipleLocator(12))
#
# # 调整布局
# plt.tight_layout()
#
# # 显示图表
# plt.show()

#   ====================================================================================================

valid_csv_all_loss_male_by_month = torch.zeros((3, 20), dtype=torch.float32)
valid_csv_all_loss_female_by_month = torch.zeros((3, 20), dtype=torch.float32)


def statistics_loss_by_month(df, loss_dict_male_by_month, loss_dict_female_by_month):
    length = len(df)
    for i in range(length):
        row = df.iloc[i]
        boneage = int(row["boneage"]) // 12
        loss = row["loss"]
        gender = row["male"]
        if gender > 0.:
            loss_dict_male_by_month[0][boneage] += loss
            loss_dict_male_by_month[1][boneage] += 1
        else:
            loss_dict_female_by_month[0][boneage] += loss
            loss_dict_female_by_month[1][boneage] += 1
    for i in range(20):
        if loss_dict_male_by_month[0][i] > 0.:
            month_loss = loss_dict_male_by_month[0][i]
            month_num = loss_dict_male_by_month[1][i]
            loss_dict_male_by_month[2][i] = month_loss / month_num
    for i in range(20):
        if loss_dict_female_by_month[0][i] > 0.:
            month_loss = loss_dict_female_by_month[0][i]
            month_num = loss_dict_female_by_month[1][i]
            loss_dict_female_by_month[2][i] = month_loss / month_num
    return loss_dict_male_by_month, loss_dict_female_by_month


valid_csv_all_loss_male_by_month, valid_csv_all_loss_female_by_month = statistics_loss_by_month(valid_csv_all,
                                                                                                valid_csv_all_loss_male_by_month,
                                                                                                valid_csv_all_loss_female_by_month)


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(36, 12))

ax1, ax2, ax3, ax4 = axes.flatten()
ax1.bar(range(20), valid_csv_all_loss_male_by_month[2])
ax1.set_xlabel('label')
ax1.set_ylabel('MAE')
ax1.set_title('male valid result')
ax1.set_ylim(0, 40)
ax1.xaxis.set_major_locator(MultipleLocator(1))

ax3.bar(range(20), valid_csv_all_loss_male_by_month[1])
ax3.set_xlabel('label')
ax3.set_ylabel('num')
ax3.set_title('male valid count')
ax3.xaxis.set_major_locator(MultipleLocator(1))

ax2.bar(range(20), valid_csv_all_loss_female_by_month[2])
ax2.set_xlabel('label')
ax2.set_ylabel('MAE')
ax2.set_title('female valid result')
ax2.set_ylim(0, 40)
ax2.xaxis.set_major_locator(MultipleLocator(1))

ax4.bar(range(20), valid_csv_all_loss_female_by_month[1])
ax4.set_xlabel('label')
ax4.set_ylabel('num')
ax4.set_title('female valid count')
ax4.xaxis.set_major_locator(MultipleLocator(1))

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()