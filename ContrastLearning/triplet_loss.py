import torch
import torch.nn as nn
import random

criterion1 = nn.MSELoss(reduction='sum')
criterion2 = nn.L1Loss(reduction='sum')


class TripletLoss_self(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss_self, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return(x1 - x2).pow(2).sum(0)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, anchor_laber: torch.Tensor, positive_label: torch.Tensor, negative_label: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative_a = self.calc_euclidean(anchor, negative)
        distance_negative_b = self.calc_euclidean(positive, negative)

        label_distance_anchor_to_positive = self.calc_euclidean(anchor_laber, positive_label)
        label_distance_anchor_to_negative = self.calc_euclidean(anchor_laber, negative_label)

        alpha = label_distance_anchor_to_negative - label_distance_anchor_to_positive

        losses = torch.relu(distance_positive - distance_negative_a + self.margin * alpha)

        return losses.mean()


class AdapitiveTripletLoss(nn.Module):
    """传入的标签必须是经过标准化后的标签"""
    def __init__(self, m=0.5):
        super(AdapitiveTripletLoss, self).__init__()

        self.m = m

        self.criterion = nn.MSELoss(reduction='sum')

    def forward(self, minibatch_features, label):

        loss_triplet = 0

        B, _ = minibatch_features.shape

        for i in range(B):
            anchor = minibatch_features[i]

            # 产生两个随机数并按照该随机数从batch中选取一个near和一个far
            index1, index2 = random.sample(range(0,B), 2)

            # 产生的随机数和i不能重复
            while index1 == i or index2 == i:
                index1, index2 = random.sample(range(0,B),2)

            temp1 = minibatch_features[index1]
            temp2 = minibatch_features[index2]

            label_distance1 = criterion2(label[i], label[index1])
            label_distance2 = criterion2(label[i], label[index2])

            if label_distance1 >= label_distance2:
                near = temp2
                far = temp1
                near_label = label[index2]
                far_label = label[index1]
            else:
                near = temp1
                far = temp2
                near_label = label[index1]
                far_label = label[index2]

            tl = TripletLoss_self(margin=0.5)

            loss = tl(anchor, near, far, label[i], near_label, far_label)

            # anchor_to_near = criterion1(anchor.cuda(), near.cuda())
            # anchor_to_far = criterion1(anchor.cuda(), far.cuda())

            # anchor_label = label[i]

            # anchor_to_far_mse = np.abs(anchor_label.cpu() - far_label.cpu()) * np.abs(anchor_label.cpu() - far_label.cpu())
            # anchor_to_near_mse = np.abs(anchor_label.cpu() - near_label.cpu()) * np.abs(anchor_label.cpu() - near_label.cpu())

            # alpha = anchor_to_far_mse - anchor_to_near_mse

            # loss = anchor_to_near - anchor_to_far + alpha.cuda() * self.m

            if loss >= 0:
                loss_triplet += loss

        return loss_triplet


if __name__ == '__main__':
    tl = AdapitiveTripletLoss()
    for name, param in tl.parameters():
        print(name)



