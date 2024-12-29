import os
import torch
import torch.nn as nn
from torchvision import transforms
import UNets
import numpy as np
import random
import matplotlib.pyplot as plt

from Unet.UnetDataset import SegmentationDataset, SegmentationTripleDataset

seed = 1
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


# 参数初始化
def init_xavier(m):  # 参数初始化
    # if type(m) == nn.Linear or type(m) == nn.Conv2d:
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)


def show_a_image(predict, label):
    fig, axes = plt.subplots(2, figsize=(15, 5))
    axes[0].imshow(predict.cpu().numpy(), cmap='gray')
    axes[0].set_title('predict')
    axes[0].axis('off')

    axes[1].imshow(label.cpu().numpy(), cmap='gray')
    axes[1].set_title('label')
    axes[1].axis('off')

    plt.show()


# 定义训练方法
def eval_pic(net, testDataset, idx, compare=False):
    net.eval()
    with torch.no_grad():
        data, label = testDataset.__getitem__(idx)
        data, label = data.unsqueeze(0).cuda(), label.unsqueeze(0).cuda()
        predict, x1, x2, x3, x4, x5, attn1, attn2, attn3, attn4 = net.forward_attention(data)  # [1, 3, H, W]

        if compare is False:
            predict = torch.argmax(predict, dim=1, keepdim=False).squeeze()
            predict = predict*127
        else:
            predict = torch.nn.functional.sigmoid(predict)  # 应用Sigmoid函数
            predict = (predict > 0.5).float()*255
        label = label.squeeze()

    return data, predict, label, x1, x2, x3, x4, x5, attn1, attn2, attn3, attn4


def visualize_attn_single(image, prediction, label, attn1, attn2, attn3, attn4):
    fig, axes = plt.subplots(2, 4, figsize=(15, 5))

    axes[0][0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
    axes[0][0].set_title('Input')
    axes[0][0].axis('off')

    axes[0][1].imshow(prediction.squeeze().cpu().numpy(), cmap='gray')
    axes[0][1].set_title('Predicted')
    axes[0][1].axis('off')

    axes[0][2].imshow((label * 127).squeeze().cpu().numpy(), cmap='gray')
    axes[0][2].set_title('Label')
    axes[0][2].axis('off')

    axes[1][0].imshow(attn1.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][0].set_title('attn1')
    axes[1][0].axis('off')

    axes[1][1].imshow(attn2.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][1].set_title('attn2')
    axes[1][1].axis('off')

    axes[1][2].imshow(attn3.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][2].set_title('attn3')
    axes[1][2].axis('off')

    axes[1][3].imshow(attn4.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][3].set_title('attn4')
    axes[1][3].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_attn_compare(image, prediction, prediction2,
                           attn1, attn2, attn3, attn4,
                           compare1, compare2, compare3, compare4):
    fig, axes = plt.subplots(3, 4, figsize=(15, 5))

    axes[0][0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
    axes[0][0].set_title('Input')
    axes[0][0].axis('off')

    axes[0][1].imshow(prediction.squeeze().cpu().numpy(), cmap='gray')
    axes[0][1].set_title('Predicted')
    axes[0][1].axis('off')

    axes[0][2].imshow(prediction2.squeeze().cpu().numpy(), cmap='gray')
    axes[0][2].set_title('Compare')
    axes[0][2].axis('off')

    axes[1][0].imshow(attn1.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][0].set_title('attn1')
    axes[1][0].axis('off')

    axes[1][1].imshow(attn2.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][1].set_title('attn2')
    axes[1][1].axis('off')

    axes[1][2].imshow(attn3.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][2].set_title('attn3')
    axes[1][2].axis('off')

    axes[1][3].imshow(attn4.squeeze().cpu().numpy(), cmap='viridis')
    axes[1][3].set_title('attn4')
    axes[1][3].axis('off')

    axes[2][0].imshow(compare1.squeeze().cpu().numpy(), cmap='viridis')
    axes[2][0].set_title('compare1')
    axes[2][0].axis('off')

    axes[2][1].imshow(compare2.squeeze().cpu().numpy(), cmap='viridis')
    axes[2][1].set_title('compare2')
    axes[2][1].axis('off')

    axes[2][2].imshow(compare3.squeeze().cpu().numpy(), cmap='viridis')
    axes[2][2].set_title('compare3')
    axes[2][2].axis('off')

    axes[2][3].imshow(compare4.squeeze().cpu().numpy(), cmap='viridis')
    axes[2][3].set_title('compare4')
    axes[2][3].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # 如果需要的话，可以在这里添加归一化
    ])
    idx = 1
    data_dir = "D:/BoneAgeAssessment/ARAA/TSRS_RSNA-Articular-Surface"

    test_image_dir = os.path.join(data_dir, 'val')
    test_label_dir = os.path.join(data_dir, 'val_labels_gray')
    test_mask_dir = os.path.join(data_dir, 'valid_mask_resize')

    testDataset = SegmentationDataset(test_image_dir, test_label_dir, transform=transform_val)

    compare_model = UNets.Attn_UNet(img_ch=1, output_ch=1).cuda()
    compare_path = '../Unet/ckp/Unet/unet_segmentation_Attn_Unet_RSNA_256_Full_50epoch_3.pth'
    compare_model.load_state_dict(torch.load(compare_path), strict=True)

    data, predict, label, _, _, _, _, _, compare1, compare2, compare3, compare4 = eval_pic(compare_model, testDataset,
                                                                                          idx=idx, compare=True)
    # show_a_image(predict, label)
    visualize_attn_single(data, predict, label, compare1, compare2, compare3, compare4)
    # visualize_attn_compare(data, predict, predict2, attn1, attn2, attn3, attn4, compare1, compare2, compare3, compare4)
