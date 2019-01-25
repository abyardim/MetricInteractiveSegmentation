
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class LocalTriplet:

    def __init__(self, n_triplets=200, *, sigma_pos=10, sigma_neg=80, margin=1, l2_penalty=1e-3, max_size=1599):
        self.margin = margin
        self.l2_penalty = l2_penalty
        self.n_triplets =  n_triplets

        self.loss_fun = nn.MarginRankingLoss(margin=margin)
        self.target = torch.FloatTensor(n_triplets).fill_(1).cuda()

        ox = oy = max_size // 2
        
        X, Y = np.meshgrid(range(max_size), range(max_size))
        p = - ((X - ox) ** 2 + (Y - oy) ** 2) / (sigma_pos ** 2)
        p = np.exp(p)
        
        self.ox = ox
        self.oy = oy
        self.probs_precomputed_pos = np.copy(p)
        self.probs_precomputed_neg = np.exp(- ((X - ox) ** 2 + (Y - oy) ** 2) / (sigma_neg ** 2))
        self.probs_precomputed_pos[ox, oy] = 0.
        

    def generate_triplets(self, output, labels):
        ai_x = np.random.randint(low=0, high=output.shape[2], 
                                size=(self.n_triplets,))
        ai_y = np.random.randint(low=0, high=output.shape[3], 
                                size=(self.n_triplets,))
        
        ai = output.shape[3] * ai_x + ai_y
        
        classes, inv_map = np.unique(labels, return_inverse=True)
        n_classes = len(classes)
        inv_map = inv_map.reshape(labels.shape[1], labels.shape[2])
        inv_map_flat = inv_map.reshape(-1)

        class_lookup = (np.arange(n_classes, dtype=np.int32).reshape((1, 1, n_classes)) !=
                        inv_map.reshape(labels.shape[1], labels.shape[2], 1))
        class_lookup = np.transpose(class_lookup, axes=[2, 0, 1])

        lneg = []
        lpos = []
        for i in range(n_classes):
            ni_temp = np.transpose(class_lookup[i].reshape(-1).nonzero()).reshape((-1))
            pi_temp = np.transpose(np.logical_not(class_lookup[i]).reshape(-1).nonzero()).reshape((-1))
            
            lneg.append(ni_temp)
            lpos.append(pi_temp)

        rs = np.random.rand(2 * self.n_triplets)
            
        ni, pi = [], []
        for i in range(self.n_triplets):
            prangepos = self.probs_precomputed_pos[self.ox - ai_x[i]: self.ox - ai_x[i] + labels.shape[1],
                                                    self.oy - ai_y[i]: self.oy - ai_y[i] + labels.shape[2]]
            prangeneg = self.probs_precomputed_neg[self.ox - ai_x[i]: self.ox - ai_x[i] + labels.shape[1],
                                                    self.oy - ai_y[i]: self.oy - ai_y[i] + labels.shape[2]]
            prangeneg = prangeneg.reshape(-1)
            prangepos = prangepos.reshape(-1)
            
            prob_neg = prangeneg[lneg[inv_map_flat[ai[i]]]]
            prob_pos = prangepos[lpos[inv_map_flat[ai[i]]]]
  
            prob_pos = np.cumsum((prob_pos / prob_pos.sum()))
            prob_neg = np.cumsum((prob_neg / prob_neg.sum()))
            
            ii1 = np.searchsorted(prob_pos, rs[2 * i])
            ii2 = np.searchsorted(prob_neg, rs[2 * i + 1])

            pi.append( lpos[inv_map_flat[ai[i]]][ii1])
            ni.append( lneg[inv_map_flat[ai[i]]][ii2])

        return (torch.tensor(ai, dtype=torch.long),
                torch.tensor(pi, dtype=torch.long),
                torch.tensor(ni, dtype=torch.long))

    def compute_loss(self, output, labels):
        ta, tp, tn = self.generate_triplets(output, labels)
        ta.cuda()
        tp.cuda()
        tn.cuda()

        n_dim = output.shape[1]
        
        output = output.view(n_dim, -1)
        
        l2_norm = (output[:, ta].pow(2).sum(dim=0).mean() +
                    output[:, tp].pow(2).sum(dim=0).mean() +
                    output[:, tn].pow(2).sum(dim=0).mean())
                
        distp = (output[:, ta] - output[:, tp]).pow(2).sum(dim=0)
        distn = (output[:, ta] - output[:, tn]).pow(2).sum(dim=0)

        loss = self.loss_fun(distn, distp, self.target) + l2_norm * self.l2_penalty

        return loss