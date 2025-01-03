import torch
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image
# from Train_Unet import SegmentationDataset
# from torch.utils.data import Dataset, DataLoader

import UNets as net
import numpy as np
# from utils import preprocess_image


def preprocess_image(image_path, img_size):
    # 定义数据变换
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 调整图像大小
        transforms.ToTensor(),          # 转换为Tensor
    ])

    # 读取图像
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)  # 添加批次维度
    print(image.shape)
    return image



def visualize_results_all(images, labels, predictions):
    fig, axes = plt.subplots(len(images), 3, figsize=(15, 5 * len(images)))
    for i in range(len(images)):
        ax = axes[i] if len(images) > 1 else axes
        ax[0].imshow(images[i].cpu().numpy().transpose(1, 2, 0))
        ax[0].set_title('Input Image')
        ax[0].axis('off')

        ax[1].imshow(labels[i].cpu().numpy().squeeze(), cmap='gray')
        ax[1].set_title('Ground Truth Label')
        ax[1].axis('off')

        ax[2].imshow(predictions[i].cpu().numpy().squeeze(), cmap='gray')
        ax[2].set_title('Predicted Label')
        ax[2].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_prediction_single(image, prediction):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(prediction.squeeze().cpu().numpy(), cmap='gray')
    axes[1].set_title('Predicted Label')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_attn_single(image, prediction, x1, x2, x3, x4, x5, attn1, attn2, attn3, attn4, save_name):
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    x1 = squeeze_feature_map_sum(x1)
    x2 = squeeze_feature_map_sum(x2)
    x3 = squeeze_feature_map_sum(x3)
    x4 = squeeze_feature_map_sum(x4)
    x5 = squeeze_feature_map_sum(x5)

    axes[0][0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
    axes[0][0].set_title('Input')
    axes[0][0].axis('off')

    axes[0][1].imshow(prediction.squeeze().cpu().numpy(), cmap='gray')
    axes[0][1].set_title('Predicted')
    axes[0][1].axis('off')

    axes[0][2].imshow(attn1.detach().squeeze().cpu().numpy(), cmap='viridis')
    axes[0][2].set_title('attn1')
    axes[0][2].axis('off')

    axes[0][3].imshow(attn2.detach().squeeze().cpu().numpy(), cmap='viridis')
    axes[0][3].set_title('attn2')
    axes[0][3].axis('off')

    axes[0][4].imshow(attn3.detach().squeeze().cpu().numpy(), cmap='viridis')
    axes[0][4].set_title('attn3')
    axes[0][4].axis('off')

    axes[0][5].imshow(attn4.detach().squeeze().cpu().numpy(), cmap='viridis')
    axes[0][5].set_title('attn4')
    axes[0][5].axis('off')

    axes[1][0].imshow(x1.detach().squeeze().cpu().numpy(), cmap='viridis')
    axes[1][0].set_title('x1')
    axes[1][0].axis('off')

    axes[1][1].imshow(x2.detach().squeeze().cpu().numpy(), cmap='viridis')
    axes[1][1].set_title('x2')
    axes[1][1].axis('off')

    axes[1][2].imshow(x3.detach().squeeze().cpu().numpy(), cmap='viridis')
    axes[1][2].set_title('x3')
    axes[1][2].axis('off')

    axes[1][3].imshow(x4.detach().squeeze().cpu().numpy(), cmap='viridis')
    axes[1][3].set_title('x4')
    axes[1][3].axis('off')

    axes[1][4].imshow(x5.detach().squeeze().cpu().numpy(), cmap='viridis')
    axes[1][4].set_title('x5')
    axes[1][4].axis('off')

    plt.tight_layout()
    plt.savefig(save_name)


def squeeze_feature_map_sum(feature_batch):
    '''
    将每张子图进行相加
    :param feature_batch:
    :return:
    '''
    # feature_batch = nn.functional.sigmoid(feature_batch)
    # feature_batch = torch.sum(feature_batch, dim=0)
    feature_map = np.squeeze(feature_batch)
    # print(feature_map.shape)
    feature_map_combination = []


    # 取出 featurn map 的数量
    num_pic = feature_map.shape[0]

    # 将 每一层卷积的特征图，拼接层 5 × 5
    for i in range(0, num_pic):
        feature_map_split = feature_map[i, :, :]
        feature_map_combination.append(feature_map_split)

    feature_map_sum = sum(one for one in feature_map_combination) / num_pic
    # feature_map_sum = feature_map

    return feature_map_sum


def dice_coeff(pred, target):
    smooth = 1e-7
    num = len(pred)
    A = torch.flatten(pred, start_dim=1)  # Flatten
    B = torch.flatten(target, start_dim=1)  # Flatten
    intersection = (A * B).sum()
    return (2. * intersection + smooth) / (A.sum() + B.sum() + smooth) # smooth防止除数为0


if __name__ == '__main__':
    unet = net.Attn_UNet(img_ch=1, output_ch=1).cuda()
    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # 如果需要的话，可以在这里添加归一化
    ])
    save_name = "pic_15504.png"
    # save_name = "pic_15504_16_2.png"
    # 加载保存的模型参数
    # model_path = '../../../autodl-tmp/UnetCheckPoint/600size/unet_segmentation_Attn_Unet_RSNA_256_600_2750iter_3090_NoRC_16_NoDL.pth'
    model_path = '../ckp/Unet/unet_segmentation_Attn_UNet.pth'
    # model_path = '../ckp/Unet/unet_segmentation_Attn_Unet_RSNA_512_0.0310.pth'
    unet.load_state_dict(torch.load(model_path))
    unet.eval()  # 设置为评估模式
    # image_path = 'E:/code/Dataset/oderByMonth/512/46/7038.png'
    # image_path = 'E:/code/Dataset/oderByMonth/256/train/46/7038.png'
    # image_path = 'E:/code/Dataset/archiveMask/FinalQiu/train/7038.png'
    image_path = '../../Dataset/RSNA/valid/15504.png'

    # data_dir = 'C:/BoneAgeAssessment/ARAA/TSRS_RSNA-Articular-Surface/val'
    # label_dir = 'C:/BoneAgeAssessment/ARAA/TSRS_RSNA-Articular-Surface/val_labels_gray'
    # batch_size = 24
    #
    # testDataset = SegmentationDataset(data_dir, label_dir, transform=transform_val)
    # testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=4)

    img1 = preprocess_image(image_path, img_size=256).cuda()
    with torch.no_grad():
        d1, x1, x2, x3, x4, x5, attn1, attn2, attn3, attn4 = unet.forward_attention(img1)
        prediction = torch.nn.functional.sigmoid(d1)  # 应用Sigmoid函数
        prediction = (prediction > 0.5).float()*255
    visualize_attn_single(img1, prediction, x1, x2, x3, x4, x5, attn1, attn2, attn3, attn4, save_name)





