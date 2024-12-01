import torch
from torch.utils.data import Dataset
import os
from PIL import Image


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
        # ROI区域应用相同的变换
        torch.manual_seed(seed)
        label = self.transform(label)
        return image, label


class PairedTripleTransform:
    def __init__(self, transform):
        self.transform = transform
        # 可以添加其他的变换

    def __call__(self, image, label, mask):
        # 随机设置变换参数
        seed = torch.random.seed()
        # 图像应用变换
        torch.manual_seed(seed)
        image = self.transform(image)
        # ROI区域应用相同的变换
        torch.manual_seed(seed)
        label = self.transform(label)
        # 掩码应用相同的变换
        torch.manual_seed(seed)
        mask = self.transform(mask)
        return image, label, mask


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


class SegmentationTripleDataset(SegmentationDataset):
    """
    输出二元组，经过数据增强后的图片以及
    """

    def __init__(self, image_dir, label_dir, mask_dir, transform=None):
        super().__init__(image_dir, label_dir, transform)
        self.mask_dir = mask_dir
        self.masks = os.listdir(mask_dir)
        self.transform = PairedTripleTransform(transform)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert('L')
        label = Image.open(label_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image, label, mask = self.transform(image, label, mask)
        mask = (mask*2).type(torch.LongTensor)
        label = label.type(torch.LongTensor)

        merged_mask = torch.where(label == 1, 1, mask)

        return image, merged_mask.type(torch.LongTensor)


from torchvision import transforms
import matplotlib.pyplot as plt

if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((800, 800), scale=(0.5, 1.0)),
        transforms.RandomAffine(degrees=(10, 20), translate=(0.1, 0.2)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # 如果需要的话，可以在这里添加归一化
    ])
    train_image_dir = 'E:/code/EmperorQiu/data/TSRS_RSNA-Articular-Surface/train/'
    train_label_dir = 'E:/code/EmperorQiu/data/TSRS_RSNA-Articular-Surface/train_labels_gray/'
    train_mask_dir = 'E:/code/EmperorQiu/data/TSRS_RSNA-Articular-Surface/train_mask_resize/'
    dataset = SegmentationTripleDataset(train_image_dir, train_label_dir, train_mask_dir, transform_train)

    image, new_label = dataset.__getitem__(0)
    image = image * 255
    new_label = new_label * 127

    fig, axes = plt.subplots(2, figsize=(15, 5))

    axes[0].imshow(image.squeeze().numpy(), cmap='gray')
    axes[0].set_title('image')
    axes[0].axis('off')

    axes[1].imshow(new_label.squeeze().numpy(), cmap='gray')
    axes[1].set_title('new_label')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
