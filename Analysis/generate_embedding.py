import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import ContrastLearning.contrast_model
import Student.student_model
import datasets
import warnings

warnings.filterwarnings("ignore")


def evaluate_fn(valid_loader):
    with torch.no_grad():
        mymodel.eval()
        linear_out_valid = torch.zeros((1, 512))
        cls_token2_valid = torch.zeros((1, 1024))
        cls_token3_valid = torch.zeros((1, 1024))
        # cls_token2_valid_before = torch.zeros((1, 1024))
        # cls_token3_valid_before = torch.zeros((1, 2048))
        labels_valid = torch.zeros((1), dtype=torch.int)
        male_valid = torch.zeros((1), dtype=torch.int)
        for idx, data in enumerate(valid_loader):
            image, gender = data[0]
            image = image.type(torch.FloatTensor).cuda()
            gender = gender.type(torch.FloatTensor).cuda()
            id = data[2]

            label = data[1].cuda()

            linear_out, cls_token2, cls_token3, cls_token2_before, cls_token3_before = mymodel.manifold(image, gender)
            label = torch.squeeze(label)

            linear_out_valid = torch.cat((linear_out_valid, linear_out.cpu()), dim=0)
            cls_token2_valid = torch.cat((cls_token2_valid, cls_token2.cpu()), dim=0)
            cls_token3_valid = torch.cat((cls_token3_valid, cls_token3.cpu()), dim=0)
            # cls_token2_valid_before = torch.cat((cls_token2_valid_before, cls_token2_before.cpu()), dim=0)
            # cls_token3_valid_before = torch.cat((cls_token3_valid_before, cls_token3_before.cpu()), dim=0)
            labels_valid = torch.cat((labels_valid, label.cpu().type(torch.IntTensor)), dim=0)
            male_valid = torch.cat((male_valid, gender.squeeze().cpu().type(torch.IntTensor)), dim=0)

        linear_out_valid = linear_out_valid[1:]
        cls_token2_valid = cls_token2_valid[1:]
        cls_token3_valid = cls_token3_valid[1:]
        # cls_token2_valid_before = cls_token2_valid_before[1:]
        # cls_token3_valid_before = cls_token3_valid_before[1:]
        labels_valid = labels_valid[1:]
        male_valid = male_valid[1:]

        torch.save(linear_out_valid, os.path.join(save_path, "linear_out_valid.pt"))
        torch.save(cls_token2_valid, os.path.join(save_path, "cls_token2_valid.pt"))
        torch.save(cls_token3_valid, os.path.join(save_path, "cls_token3_valid.pt"))
        # torch.save(cls_token2_valid_before, os.path.join(save_path, "cls_token2_valid_before.pt"))
        # torch.save(cls_token3_valid_before, os.path.join(save_path, "cls_token3_valid_before.pt"))
        torch.save(labels_valid, os.path.join(save_path, "labels_valid.pt"))
        torch.save(male_valid, os.path.join(save_path, "male_valid.pt"))


if __name__ == "__main__":
    ckpt_path = "../Contrast_Output/Contrast_WCL_IN_CBAM_AVGPool_AdaA_GenderPlus_4K_1_7_96_200epoch/Contrast_WCL_IN_CBAM_AVGPool_AdaA_GenderPlus_4K_1_7_96_200epoch.bin"
    # ckpt_path = "../KD_All_Output/KD_modify_firstConv_RandomCrop/KD_modify_firstConv_RandomCrop.bin"
    ckpt_dir = os.path.dirname(ckpt_path)
    save_path = os.path.join(ckpt_dir, 'manifold')
    os.makedirs(save_path, exist_ok=True)
    student_path = "../KD_All_Output/KD_modify_firstConv_RandomCrop/KD_modify_firstConv_RandomCrop.bin"
    mymodel = ContrastLearning.contrast_model.get_student_contrast_model(student_path).cuda()
    # mymodel = Student.student_model.get_student().cuda()
    mymodel.load_state_dict(torch.load(ckpt_path), strict=True)
    for param in mymodel.parameters():
        param.requires_grad = False
    mymodel.eval()


    flags = {}
    flags['batch_size'] = 32
    flags['seed'] = 1

    data_dir = 'E:/code/Dataset/RSNA'

    # valid_csv = os.path.join(data_dir, "valid.csv")
    valid_csv = os.path.join(os.path.join(ckpt_dir, "Contrast_4K_Gender_96_200e_3.75.csv"))
    valid_df = pd.read_csv(valid_csv)
    valid_path = os.path.join(data_dir, "valid")
    valid_Dataset = datasets.RSNATestDataset(valid_df, valid_path, 0, 1)
    print(f"Test set length: {valid_Dataset.__len__()}")

    valid_loader = torch.utils.data.DataLoader(
        valid_Dataset,
        batch_size=flags['batch_size'],
        shuffle=False,
        pin_memory=True
    )
    print(f'Manifold output start')

    evaluate_fn(valid_loader)