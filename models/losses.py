# models/losses.py

import torch
import torch.nn as nn

class ExcitationSpectrumLoss(nn.Module):
    """
    Custom loss: MSE + GFC + Smoothness.
    """

    def __init__(self, alpha=1.0, beta=0.5, lambdad=0.01):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambdad = lambdad

    def smoothness_loss(self, F_pred):
        diff = F_pred[:, :, 1:] - F_pred[:, :, :-1]
        diff2 = diff[:, :, 1:] - diff[:, :, :-1]
        return torch.mean(diff2 ** 2)

    def forward(self, F_pred, F_true):
        mse = torch.mean((F_pred - F_true) ** 2)

        dot = torch.sum(F_pred * F_true, dim=1)
        norm_pred = torch.sqrt(torch.sum(F_pred ** 2, dim=1))
        norm_true = torch.sqrt(torch.sum(F_true ** 2, dim=1))
        gfc = dot / (norm_pred * norm_true + 1e-8)

        weight = torch.max(F_true, dim=1).values
        weight = weight / weight.sum(dim=1, keepdim=True)
        gfc_loss = torch.mean(torch.sum((1 - gfc) * weight, dim=1))

        total_loss = self.alpha * mse + self.beta * gfc_loss + self.lambdad * self.smoothness_loss(F_pred)
        return total_loss