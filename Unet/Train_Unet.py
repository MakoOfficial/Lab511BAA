import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import UNets
import numpy as np
import random

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


class PairedTransform:
    def __init__(self, transform):
        self.transform = transform
        # 可以添加其他的变换

    def __call__(self, image, label):
        # 随机设置变换参数
        seed = torch.random.seed()
        # 图像应用变换
        torch.manual_seed(seed)
        image = self.transform(image)
        # 掩码应用相同的变换
        torch.manual_seed(seed)
        label = self.transform(label)
        return image, label


# 定义预处理，加载器
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = PairedTransform(transform=transform)
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)
        self.images.sort()
        self.labels.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        image = Image.open(img_path).convert('L')
        label = Image.open(label_path).convert('L')

        image, label = self.transform(image, label)

        return image, label


# 定义训练方法
def train(net, train_dataloader, valid_dataloader, device, num_epoch, lr, init=True):
    if init:
        net.apply(init_xavier)
    print('training on:', device)
    net.to(device)

    criterion = nn.BCELoss()  # 二分类交叉熵损失,注意output后要sigmoid
    optimizer = optim.Adam(net.parameters(), lr=lr)
    best_loss = 1000
    for epoch in range(num_epoch):
        print("——————第 {} 轮训练开始——————".format(epoch + 1))
        net.train()

        train_loss = 0.0
        for data, label in tqdm(train_dataloader):
            data, label = data.to(device), label.to(device)
            predict = net(data)
            predict = torch.nn.functional.sigmoid(predict)
            # 这里展平为了方便计算损失
            predict_flat = predict.view(predict.size(0), -1)
            label_flat = label.view(label.size(0), -1)

            loss = criterion(predict_flat, label_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epoch}], Train Loss: {train_loss:.4f}')

        # 测试步骤开始
        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, label in valid_dataloader:
                data, label = data.to(device), label.to(device)
                predict = net(data)

                predict = torch.nn.functional.sigmoid(predict)
                # 这里展平为了方便计算损失
                predict_flat = predict.view(predict.size(0), -1)
                label_flat = label.view(label.size(0), -1)
                loss = criterion(predict_flat, label_flat)
                test_loss += loss.item()

            test_loss = test_loss / len(valid_dataloader)
            print(f'Epoch [{epoch + 1}/{num_epoch}],Test Loss: {test_loss:.4f}')
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(net.state_dict(), os.path.join(model_save_path, model_save_name))
    print(f"best loss: {best_loss:.4f}")



if __name__ == '__main__':
    # 定义图像和标签的变换, 新增随机裁剪和旋转，效果更佳
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((800, 800), scale=(0.5, 1.0)),
        transforms.RandomAffine(degrees=(10, 20), translate=(0.1, 0.2)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # 如果需要的话，可以在这里添加归一化
    ])
    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # 如果需要的话，可以在这里添加归一化
    ])
    # 定义数据集路径
    train_image_dir = '../../../ARAA/TSRS_RSNA-Articular-Surface/train/'
    train_label_dir = '../../../ARAA/TSRS_RSNA-Articular-Surface/train_labels_gray/'

    test_image_dir = '../../../ARAA/TSRS_RSNA-Articular-Surface/val'
    test_label_dir = '../../../ARAA/TSRS_RSNA-Articular-Surface/val_labels_gray'

    # 装载数据，设定模型
    trainDataset = SegmentationDataset(train_image_dir, train_label_dir, transform=transform_train)
    trainLoader = DataLoader(trainDataset, batch_size=16, shuffle=True, num_workers=2)

    testDataset = SegmentationDataset(test_image_dir, test_label_dir, transform=transform_val)
    testLoader = DataLoader(testDataset, batch_size=16, shuffle=False, num_workers=2)

    segment_model = UNets.Attn_UNet(img_ch=1, output_ch=1).cuda()
    model_save_path = 'ckp/Unet/'
    os.makedirs(model_save_path, exist_ok=True)
    model_save_name = "unet_segmentation_Attn_Unet_RSNA_256.pth"

    train(segment_model, trainLoader, testLoader, device=torch.device('cuda:0'), num_epoch=50, lr=1e-4)
