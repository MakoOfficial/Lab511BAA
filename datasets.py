from torch.utils.data import Dataset
from torch import Tensor
from torchvision import transforms
from PIL import Image

RSNA_transform_train = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.75, 1.0)),
    transforms.RandomAffine(
        degrees=20,
        translate=(0.2, 0.2),
        scale=(0.8, 1.2),
        fill=0),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # 转换为张量，并将像素值归一化到 [0, 1]
])

RSNA_transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


class BaseDataset(Dataset):
    def __init__(self, df, file_path, age_mean, age_div):
        def preprocess_df(df):
            # nomalize boneage distribution
            df['zscore'] = df['boneage'].map(lambda x: (x - age_mean) / age_div)
            # change the type of gender, change bool variable to float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path

    def get_image_path(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        file_path = f"{self.file_path}/{num}.png"
        return row, file_path

    def __len__(self):
        return len(self.df)


class RSNATrainDataset(BaseDataset):
    def __init__(self, df, file_path, age_mean, age_div):
        super().__init__(df, file_path, age_mean, age_div)

    def __getitem__(self, index):
        row, image_path = self.get_image_path(index)
        image = Image.open(image_path).convert('L')

        return (RSNA_transform_train(image), Tensor([row['male']])), row['zscore'], row['boneage']


class RSNAValidDataset(BaseDataset):
    def __init__(self, df, file_path, age_mean, age_div):
        super().__init__(df, file_path, age_mean, age_div)

    def __getitem__(self, index):
        row, image_path = self.get_image_path(index)
        image = Image.open(image_path).convert('L')

        return (RSNA_transform_val(image), Tensor([row['male']])), row['boneage']


class RSNATestDataset(BaseDataset):
    def __init__(self, df, file_path, age_mean, age_div):
        super().__init__(df, file_path, age_mean, age_div)
        df['id_int'] = df['id'].astype('float32')

    def __getitem__(self, index):
        row, image_path = self.get_image_path(index)
        image = Image.open(image_path).convert('L')

        return (RSNA_transform_val(image), Tensor([row['male']])), row['boneage'], row['id_int']


if __name__ == '__main__':
    pic = Image.open('data/TSRS_RSNA-Articular-Surface/train/1377.jpg').convert('L')
