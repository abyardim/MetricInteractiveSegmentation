
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class RandomTripletPreselected:

    def __init__(self, n_batch, n_triplets, margin=1, l2_penalty=1e-3):
        self.margin = margin
        self.l2_penalty = l2_penalty
        self.n_triplets =  n_triplets
        self.n_batch = n_batch

        self.loss_fun = nn.MarginRankingLoss(margin=margin)
        self.target = torch.FloatTensor(n_triplets * n_batch).fill_(1).cuda()
        
        i = 0
        self.dim0 = []
        for i in range(self.n_batch):
            self.dim0 += self.n_triplets * [i]
            
        self.dim0 = torch.from_numpy(np.array(self.dim0, dtype=np.long))

    def compute_loss(self, output, triplets):

        if len(self.dim0) != triplets.shape[0] * triplets.shape[3]:
            self.dim0 = []
            for i in range(triplets.shape[0]):
                self.dim0 += triplets.shape[3] * [i]
            
            self.dim0 = torch.from_numpy(np.array(self.dim0, dtype=np.long))
            
            del self.target
            self.target = torch.FloatTensor(triplets.shape[0] * triplets.shape[3]).fill_(1).cuda()
        
        n_dim = output.shape[1]
        
        #output = output.view(n_dim, -1)
        t = triplets
        
        aix = t[:,0,0,:].contiguous().view(-1)
        aiy = t[:,0,1,:].contiguous().view(-1)
        pix = t[:,1,0,:].contiguous().view(-1)
        piy = t[:,1,1,:].contiguous().view(-1)
        nix = t[:,2,0,:].contiguous().view(-1)
        niy = t[:,2,1,:].contiguous().view(-1)
        
        l2_norm = (output[self.dim0, :, aix, aiy].pow(2).sum(dim=1).mean() + 
                    output[self.dim0, :, pix, piy].pow(2).sum(dim=1).mean() + 
                    output[self.dim0, :, nix, niy].pow(2).sum(dim=1).mean())
                
        distp = (output[self.dim0, :, aix, aiy] - output[self.dim0, :, pix, piy]).pow(2).sum(dim=1)
        distn = (output[self.dim0, :, aix, aiy] - output[self.dim0, :, nix, niy]).pow(2).sum(dim=1)

        loss = self.loss_fun(distn, distp, self.target) + l2_norm * self.l2_penalty

        return loss


