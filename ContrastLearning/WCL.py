import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class WCL(nn.Module):
    """传入的特征必须是经过标准化后的特征"""
    def __init__(self, p=0.5, tempS=0.2, thresholdS=0.02, tempW=0.2):
        super(WCL, self).__init__()

        # 设定计算分数的超参数
        self.p = p
        self.tempS = tempS
        self.thresholdS = thresholdS

        self.tempW = tempW

        self.criterion = nn.MSELoss(reduction='sum')

    def count_score(self, label):
        length = len(label)
        label_clone_1 = label.clone().view(length, 1)
        label_clone_2 = label.clone().view(1, length)

        score_matrix = torch.exp(-(torch.div(torch.abs(label_clone_1 - label_clone_2), self.tempS)).pow(self.p))
        score_matrix = score_matrix * (score_matrix >= self.thresholdS)
        score_matrix_sum = score_matrix.sum(-1, keepdim=True)
        score_matrix = torch.div(score_matrix, score_matrix_sum)
        return score_matrix

    def count_score_in(self, label):
        length = len(label)
        label_clone_1 = label.clone().view(length, 1)
        label_clone_2 = label.clone().view(1, length)

        score_matrix = torch.exp(-(torch.div(torch.abs(label_clone_1 - label_clone_2), self.tempS)).pow(self.p))
        score_matrix = score_matrix * (score_matrix >= self.thresholdS)
        print(score_matrix)
        return score_matrix

    def count_distance_out(self, logit):
        logit_clone = logit.clone()
        dot = torch.exp(torch.matmul(logit, logit_clone.T) / self.tempW)    # BxB
        dot_sum = dot.sum(-1, keepdim=True)
        dot_matrix = torch.log(torch.clamp(dot / dot_sum, min=1e-10))
        print(dot_matrix)
        return dot_matrix

    def count_distance_in(self, logit):
        logit_clone = logit.clone()
        dot = torch.exp(torch.matmul(logit, logit_clone.T) / self.tempW)    # BxB
        print(dot)
        dot_sum = dot.sum(-1, keepdim=True)
        dot_matrix = torch.clamp(torch.div(dot, dot_sum), min=1e-10)
        print(dot_matrix)
        return dot_matrix

    # def forward(self, minibatch_features, label):
    #     """count out"""
    #     # mask = torch.eq(label, label.T).float().cuda()
    #     score_matrix = - self.count_score(label)
    #     dot_matrix = self.count_distance_in(minibatch_features)
    #     loss_triplet = (score_matrix * dot_matrix).sum()
    #     return loss_triplet

    def forward(self, minibatch_features, label):
        """count in"""
        # mask = torch.eq(label, label.T).float().cuda()
        score_matrix = self.count_score_in(label)
        dot_matrix = self.count_distance_in(minibatch_features)
        weight_dot_matrix = (score_matrix * dot_matrix).sum(-1)
        print(weight_dot_matrix)
        weight_dot_matrix = - torch.log(torch.clamp(weight_dot_matrix, min=1e-10))
        print(weight_dot_matrix)
        loss_triplet = weight_dot_matrix.sum()
        return loss_triplet


# class SupConLoss(nn.Module):
#     def __init__(self, temperature=0.07, contrast_mode='all',
#                  base_temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
#         self.base_temperature = base_temperature
#
#     def forward(self, features, labels=None, mask=None):
#         batch_size = features.shape[0]
#         labels = labels.contiguous().view(-1, 1)
#         if labels.shape[0] != batch_size:
#             raise ValueError('Num of labels does not match num of features')
#         mask = torch.eq(labels, labels.T).float().cuda()
#
#         # compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(features, features.T),
#             self.temperature)
#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()
#
#         # tile mask
#         mask = mask.repeat(anchor_count, contrast_count)
#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
#             0
#         )
#         mask = mask * logits_mask
#
#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
#
#         # compute mean of log-likelihood over positive
#         # modified to handle edge cases when there is no positive pair
#         # for an anchor point.
#         # Edge case e.g.:-
#         # features of shape: [4,1,...]
#         # labels:            [0,1,1,2]
#         # loss before mean:  [nan, ..., ..., nan]
#         mask_pos_pairs = mask.sum(1)
#         mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
#
#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#         loss = loss.view(anchor_count, batch_size).mean()
#
#         return loss


if __name__ == '__main__':
    wcl = WCL()
    for name, param in wcl.parameters():
        print(name)
    #
    # label = torch.range(0, 9)
    label = torch.tensor((0, 0, 1, 1, 2, 2), dtype=torch.float32)
    logit = torch.tensor(((1, 1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0),
                          (0, 0, 1, 1, 0, 0), (0, 0, 1, 1, 0, 0),
                          (0, 0, 0, 0, 1, 1), (0, 0, 0, 0, 1, 1)), dtype=torch.float32)
    logit = F.normalize(logit, dim=1)
    print(wcl(logit, label))



