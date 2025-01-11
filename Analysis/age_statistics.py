import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import torch

torch.set_printoptions(sci_mode=False)

data_dir = "E:/code/Dataset/RSNA"

# csv_map = {
#     "Origin": "../Student/baseline/未蒸馏2.csv",
#     "KD": "../KD_All_Output/KD_Res18_3090/蒸馏结果4.04_new.csv",
#     "Contrast Learning": "../KD_All_Output/KD_modify_firstConv_RandomCrop/对比效果.csv"
# }

csv_map = {
    "Classify": '../Contrast_Output/Contrast_WCL_IN_CBAM_AVGPool_AdaA_GenderPlus_Full_96_OnlyKD_AddRegression_1_11/Contrast_Gender_Pretrain_valid.csv',
    "Soft Label": '../Contrast_Output/Contrast_WCL_IN_CBAM_AVGPool_AdaA_GenderPlus_Full_1_11_96_Pretrain/Contrast_Gender_Pretrain_valid.csv',
    "Regression": '../Student/baseline/ResNet50_Class_256_Full/Student_256_valid_Class.csv'
}


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


def statistics_loss_by_month(df, loss_dict_male_by_month, loss_dict_female_by_month):
    """
    0:
    """
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


def print_erros_map(male_by_month, female_by_month):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(36, 12))

    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.bar(range(20), male_by_month[2])
    ax1.set_xlabel('label')
    ax1.set_ylabel('MAE')
    ax1.set_title('male valid result')
    ax1.set_ylim(0, 40)
    ax1.xaxis.set_major_locator(MultipleLocator(1))

    ax3.bar(range(20), male_by_month[1])
    ax3.set_xlabel('label')
    ax3.set_ylabel('num')
    ax3.set_title('male valid count')
    ax3.xaxis.set_major_locator(MultipleLocator(1))

    ax2.bar(range(20), female_by_month[2])
    ax2.set_xlabel('label')
    ax2.set_ylabel('MAE')
    ax2.set_title('female valid result')
    ax2.set_ylim(0, 40)
    ax2.xaxis.set_major_locator(MultipleLocator(1))

    ax4.bar(range(20), female_by_month[1])
    ax4.set_xlabel('label')
    ax4.set_ylabel('num')
    ax4.set_title('female valid count')
    ax4.xaxis.set_major_locator(MultipleLocator(1))

    # 调整布局
    plt.tight_layout()

    # 显示图表
    plt.show()


def print_predict_dot_map(df, title, save_path):
    # 提取标签和预测值
    labels_1 = torch.tensor(df[df['male'] == 1]['boneage'].values, dtype=torch.float32)
    predictions_1 = torch.tensor(df[df['male'] == 1]['pred'].values, dtype=torch.float32)
    labels_2 = torch.tensor(df[df['male'] == 0]['boneage'].values, dtype=torch.float32)
    predictions_2 = torch.tensor(df[df['male'] == 0]['pred'].values, dtype=torch.float32)

    # 创建一个新的图形
    plt.figure(figsize=(8, 8))

    # 绘制 loss_1，使用蓝色三角形标记
    plt.scatter(labels_1, predictions_1, color='blue', marker='^', label='male', s=10)

    # 绘制 loss_2，使用红色圆形标记
    plt.scatter(labels_2, predictions_2, color='red', marker='o', label='female', s=10)

    # 绘制 y=x 绿色直线
    plt.plot([0, 228], [0, 228], color='green', linestyle='-', label='Actual Age')

    plt.xticks(np.arange(0, 229, 50))
    plt.yticks(np.arange(0, 229, 50))

    # 设置坐标轴标签和标题
    plt.xlabel('Grand Truth(Months)')
    plt.ylabel('MAE(Months)')
    plt.title(f'Predicted Age vs Grand Truth for {title}')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f"{title}_Predict.png"), dpi=800)
    # plt.show()
    plt.clf()
    plt.close('all')


def print_deviation_dot_map(df, title, save_path):
    # 提取标签和预测值
    labels_1 = torch.tensor(df[df['male'] == 1]['boneage'].values, dtype=torch.float32)
    predictions_1 = torch.tensor(df[df['male'] == 1]['pred'].values, dtype=torch.float32)
    loss_1 = predictions_1 - labels_1
    labels_2 = torch.tensor(df[df['male'] == 0]['boneage'].values, dtype=torch.float32)
    predictions_2 = torch.tensor(df[df['male'] == 0]['pred'].values, dtype=torch.float32)
    loss_2 = predictions_2 - labels_2

    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    # 绘制 loss_1，使用蓝色三角形标记
    plt.scatter(labels_1, loss_1, color='blue', marker='^', label='male', s=10)

    # 绘制 loss_2，使用红色圆形标记
    plt.scatter(labels_2, loss_2, color='red', marker='o', label='female', s=10)

    # 绘制 y=x 绿色直线
    # plt.plot([0, 228], [0, 228], color='green', linestyle='-', label='Actual Age')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, label='y=0')
    plt.axhline(y=10, color='black', linestyle='--', linewidth=1, label='y=10')
    plt.axhline(y=-10, color='black', linestyle='--', linewidth=1, label='y=-10')

    plt.xticks(np.arange(0, 229, 50))
    plt.yticks(np.arange(-40, 41, 10))
    plt.ylim(-60, 60)
    # 设置坐标轴标签和标题
    plt.xlabel('Grand Truth(Months)')
    plt.ylabel('Deviation(Months)')
    plt.title(f'Correlation between actual age and deviation for {title}')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.grid(True)
    # plt.savefig(os.path.join(save_path, f"{title}_Deviation.png"), dpi=800)
    plt.show()
    plt.clf()
    plt.close('all')


def print_group_loss(df, title, save_path):
    labels_1 = torch.tensor(df[df['male'] == 1]['boneage'].values, dtype=torch.float32).unsqueeze(1)
    predictions_1 = torch.tensor(df[df['male'] == 1]['pred'].values, dtype=torch.float32).unsqueeze(1)

    labels_2 = torch.tensor(df[df['male'] == 0]['boneage'].values, dtype=torch.float32).unsqueeze(1)
    predictions_2 = torch.tensor(df[df['male'] == 0]['pred'].values, dtype=torch.float32).unsqueeze(1)

    loss_1 = torch.cat((labels_1, predictions_1), dim=1)
    loss_2 = torch.cat((labels_2, predictions_2), dim=1)

    # 计算标签与预测值之间的绝对差值
    abs_diff_1 = np.abs(loss_1[:, 0] - loss_1[:, 1])
    abs_diff_2 = np.abs(loss_2[:, 0] - loss_2[:, 1])

    # 定义标签范围
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

    # 创建一个空字典来存储每组的平均绝对差值
    grouped_diff_1 = []
    grouped_diff_2 = []
    grouped_diff_merged = []
    group_labels = ["Infancy", "Toddlers", "Pre-puberty", "Early puberty", "Late Puberty", "Post-puberty"]

    # 对每个范围内的标签进行分组
    for i in range(6):
        group_range_m = np.arange(ranges_male[i][0], ranges_male[i][1] + 1)
        group_range_f = np.arange(ranges_female[i][0], ranges_female[i][1] + 1)

        # 找到对应范围内的样本
        mask_1 = np.isin(loss_1[:, 0], group_range_m)
        mask_2 = np.isin(loss_2[:, 0], group_range_f)

        male_loss = abs_diff_1[mask_1]
        female_loss = abs_diff_2[mask_2]
        male_len = len(male_loss)
        female_len = len(female_loss)

        # 计算每组的平均绝对差值
        avg_diff_1 = male_loss.mean().item()
        avg_diff_2 = female_loss.mean().item()
        avg_diff_merged = ((male_loss.sum() + female_loss.sum()) / (male_len + female_len)).item()

        # 存储每组的标签和差值
        grouped_diff_1.append(avg_diff_1)
        grouped_diff_2.append(avg_diff_2)
        grouped_diff_merged.append(avg_diff_merged)

    # 绘图
    fig, ax = plt.subplots(figsize=(7, 6))

    # 绘制柱状图
    x_pos = np.arange(6)  # 横坐标的位置
    bar_width = 0.25  # 每组柱子的宽度

    # 绘制每组的差值
    ax.bar(x_pos - bar_width, grouped_diff_1, bar_width, label='male', color='blue')
    ax.bar(x_pos, grouped_diff_2, bar_width, label='female', color='orange')
    ax.bar(x_pos + bar_width, grouped_diff_merged, bar_width, label='whole', color='green')

    # 设置标签
    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel('MAE(Months)')
    ax.set_xlabel('Grand Truth(Months)')
    ax.set_title(f'{title} Group MAE')
    ax.set_ylim(0, 15)
    ax.legend()

    # 显示图形
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_path, f"{title}_Group.png"), dpi=400)
    plt.clf()
    plt.close('all')


def analysis_one_csv(csv_path, title, save_dir):
    valid_csv = pd.read_csv(csv_path)

    loss_male = torch.zeros((3, 229), dtype=torch.float32)
    loss_female = torch.zeros((3, 229), dtype=torch.float32)

    loss_male_by_month = torch.zeros((3, 20), dtype=torch.float32)
    loss_female_by_month = torch.zeros((3, 20), dtype=torch.float32)

    loss_male_by_month, loss_female_by_month = statistics_loss_by_month(
        valid_csv,
        loss_male_by_month,
        loss_female_by_month)

    # print_predict_dot_map(valid_csv, title=f"{title} Result", save_path=save_dir)
    # print_group_loss(valid_csv, title=f"{title} Result", save_path=save_dir)
    print_deviation_dot_map(valid_csv, title=f"{title} Result", save_path=save_dir)



if __name__ == '__main__':
    save_dir = './'
    #   对比损失
    for key in csv_map.keys():
        analysis_one_csv(csv_map[key], key, save_dir)
