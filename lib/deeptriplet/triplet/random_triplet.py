
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class RandomTriplet:

    def __init__(self, n_triplets=200, margin=1, l2_penalty=1e-3):
        self.margin = margin
        self.l2_penalty = l2_penalty
        self.n_triplets =  n_triplets

        self.loss_fun = nn.MarginRankingLoss(margin=margin)
        self.target = torch.FloatTensor(n_triplets).fill_(1).cuda()

    def generate_triplets(self, output, labels):
        ai = np.random.randint(low=0, high=output.shape[2]*output.shape[3], 
                                size=(self.n_triplets,))

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
            lneg.append(np.transpose(class_lookup[i].reshape(-1).nonzero()).reshape((-1)))
            lpos.append(np.transpose(np.logical_not(class_lookup[i]).reshape(-1).nonzero()).reshape((-1)))

        ni, pi = [], []
        for i in range(self.n_triplets):
            ni.append( np.random.choice(lneg[inv_map_flat[ai[i]]]))
            pi.append( np.random.choice(lpos[inv_map_flat[ai[i]]]))

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


