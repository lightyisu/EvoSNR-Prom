
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss
        Args:
            alpha (float): 平衡因子，通常设置为1。
            gamma (float): 难易样本的权重因子，常见值是2。
            reduction (str): 损失的返回方式，'mean', 'sum', 或 'none'。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型的预测值（logits），形状为 [batch_size, num_classes]。
            targets: 真实标签，形状为 [batch_size]。
        Returns:
            Focal Loss 值
        """
        # Convert targets to one-hot encoding
        targets = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # Compute probabilities with Softmax
        probs = F.softmax(inputs, dim=1)
        pt = (probs * targets).sum(dim=1)  # Get the predicted prob for the true class

        # Compute Focal Loss
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
