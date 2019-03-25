
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class MultiTripletPreselected:

    def __init__(self, n_batch, margin=1, l2_penalty=1e-3):
        self.margin = margin
        self.l2_penalty = l2_penalty
        self.n_batch = n_batch

        self.loss_fun = nn.MarginRankingLoss(margin=margin)
        self.target = torch.FloatTensor(3 * n_batch).fill_(1).cuda()
        
        i = 0
        self.dim0a = []
        self.dim0p = []
        self.dim0n = []
        for i in range(8):
            self.dim0a += 3 * [i]
            self.dim0p += 3 * [i]
            self.dim0n += 3 * [i]
            
        self.dim0a = torch.from_numpy(np.array(self.dim0a, dtype=np.long)).cuda()
        self.dim0p = torch.from_numpy(np.array(self.dim0p, dtype=np.long)).cuda()
        self.dim0n = torch.from_numpy(np.array(self.dim0n, dtype=np.long)).cuda()

    def compute_loss(self, output, a, p, n):

        if len(self.dim0a) != a.shape[0] * a.shape[1] or \
            len(self.dim0p) != p.shape[0] * p.shape[1] * p.shape[2] or \
            len(self.dim0n) != n.shape[0] * n.shape[1] * n.shape[2]:
            
            self.dim0a = []
            self.dim0p = []
            self.dim0n = []
            for i in range(a.shape[0]):
                self.dim0a += a.shape[1] * [i]
                self.dim0p += p.shape[1] * p.shape[2] * [i]
                self.dim0n += n.shape[1] * n.shape[2] * [i]
            
            self.dim0a = torch.from_numpy(np.array(self.dim0a, dtype=np.long)).cuda()
            self.dim0p = torch.from_numpy(np.array(self.dim0p, dtype=np.long)).cuda()
            self.dim0n = torch.from_numpy(np.array(self.dim0n, dtype=np.long)).cuda()
            
            self.target = torch.FloatTensor(a.shape[0] * a.shape[1]).fill_(1).cuda()
        
        
        n_dim = output.shape[1]
        
        out = output.view(output.shape[0], output.shape[1], -1)
        out = torch.transpose(out, 1, 2)
        
        # compute distances
        v_anch = out[self.dim0a, a.view(-1), :].view(-1, 1, n_dim)
        v_pos = out[self.dim0p, p.view(-1), :].view(-1, p.shape[2], n_dim)
        v_neg = out[self.dim0n, n.view(-1), :].view(-1, n.shape[2], n_dim)
        
        l2_norm = v_anch.pow(2).view(-1, n_dim).sum(dim=1).mean()
        
        delta_p = (v_anch - v_pos).pow(2).sum(dim=2)
        delta_n = (v_anch - v_neg).pow(2).sum(dim=2)

        delta_p, _ = delta_p.min(dim=1)
        delta_n, _ = delta_n.min(dim=1)

        loss = self.loss_fun(delta_n, delta_p, self.target) + l2_norm * self.l2_penalty

        return loss