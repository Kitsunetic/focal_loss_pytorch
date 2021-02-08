import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Referenced to
    - https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html
    - https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
    - https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/4
    - https://github.com/Kitsunetic/focal_loss_pytorch
    """

    def __init__(self, gamma=0, eps=1e-6, reduction="mean"):
        assert reduction in ["mean", "sum"], f"reduction should be mean or sum not {reduction}."
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        p = F.softmax(input, dim=-1)
        focal_weight = (1 - p) ** self.gamma
        loss = F.nll_loss(focal_weight * torch.log(p), target)

        return loss
