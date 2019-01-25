
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class RandomMinimumTriplet:

    def __init__(self, n_points=300, margin=1, l2_penalty=1e-3):
        self.margin = margin
        self.l2_penalty = l2_penalty
        self.n_points =  n_points

        self.loss_fun = nn.MarginRankingLoss(margin=margin)
        self.target = torch.FloatTensor(n_points).fill_(1).cuda()
        self.diag = torch.arange(self.n_points, dtype=torch.int64, device='cuda:0')

    def similarity_matrix(self, mat):
        # get the product x * y
        # here, y = x.t()
        r = torch.mm(mat, mat.t())
        # get the diagonal elements
        diag = r.diag().unsqueeze(0)
        diag = diag.expand_as(r)
        # compute the distance matrix
        D = diag + diag.t() - 2*r
        return D

    def compute_loss(self, output, labels):

        # first make sure we pick pixels from each classe
        classes, inv_map = np.unique(labels, return_inverse=True)
        n_classes = len(classes)

        inv_map = inv_map.reshape(labels.shape[1], labels.shape[2])
        inv_map_flat = inv_map.reshape(-1)
        
        lx = np.array([], dtype=np.int64)
        for c in range(n_classes):
            indices = np.nonzero(inv_map_flat == c)[0]

            if len(indices) > 0:
                count = np.min((len(indices), 10))
                lx = np.append(lx, np.random.choice(indices, size=(count,), replace=False))
        
        # then pich the rest of the pixels
        
        n_dim = output.shape[1]
        output = output.view(n_dim, -1)
        labels = labels.view(-1)
        
        available = np.nonzero(1 - np.in1d(np.arange(len(inv_map_flat)), lx))[0]
        
        count = np.min((self.n_points, len(available)))
        ix = np.random.choice(available,
                                size=(count,), 
                                replace=False)
        
        ix = np.append(ix, lx)

        # print(len(np.unique(ix)) == len(ix))
        l = labels[ix] # .cuda()
        ix = torch.from_numpy(ix).cuda()

        V = output[:, ix].t()
        dmat = self.similarity_matrix(V)

        inp = []
        inn = []
        for i in range(self.n_points):
            eqmask = (l[i] == l)
            neqmask = 1 - eqmask
            eqmask[i] = 0

            tp = torch.nonzero(eqmask)[0]
            tn = torch.nonzero(neqmask)[0]

            inp.append(tp[torch.argmin(dmat[i, tp])])
            inn.append(tn[torch.argmin(dmat[i, tn])])


        l2_norm = V.t().pow(2).sum(dim=0).mean()

        distp = dmat[self.diag, inp]
        distn = dmat[self.diag, inn]

        loss = self.loss_fun(distn, distp, self.target) + l2_norm * self.l2_penalty

        return loss
