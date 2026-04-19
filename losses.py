import torch
import torch.nn as nn
import torch.nn.functional as F


class UnsupConLoss(nn.Module):
    def __init__(self, tau=0.2):
        super().__init__()
        self.tau = tau

    def forward(self, z):
        # z: (N, D) already L2-normalized
        n = z.size(0)
        sim = torch.mm(z, z.t()) / self.tau
        mask = (~torch.eye(n, dtype=torch.bool, device=z.device)).float()
        sim_exp = torch.exp(sim) * mask
        log_prob = sim - torch.log(sim_exp.sum(dim=1, keepdim=True) + 1e-8)
        loss = -(log_prob * mask).sum() / mask.sum()
        return loss


class SupConLoss(nn.Module):
    def __init__(self, tau=1.4):
        super().__init__()
        self.tau = tau

    def forward(self, z, labels):
        # z: (N, D) l2-normalized, labels: (N,)
        n = z.size(0)
        sim = torch.mm(z, z.t()) / self.tau
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.t()).float()
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        pos_mask[self_mask] = 0

        mask = (~self_mask).float()
        exp_sim = torch.exp(sim) * mask
        denom = exp_sim.sum(dim=1, keepdim=True)

        # avoid divide by zero for samples with no positives
        has_pos = pos_mask.sum(dim=1) > 0
        log_prob = sim - torch.log(denom + 1e-8)
        loss_per = -(log_prob * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        loss = loss_per[has_pos].mean() if has_pos.any() else z.new_tensor(0.0)
        return loss
