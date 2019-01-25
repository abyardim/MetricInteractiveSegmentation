#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
sys.stdout = open('/home/yardima/Python/models/mnist-basic-triplet-ablation/summary2.txt','wt')
import os

def sync_stdout():
    sys.stdout.flush()
    os.fsync(sys.stdout.fileno())

import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets

from sklearn.neighbors import KNeighborsClassifier

# In[3]:


print(torch.cuda.is_available())
print(torch.cuda.device_count())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



# In[ ]:


trainset = datasets.MNIST(root='/scratch/yardima/data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = datasets.MNIST(root='/scratch/yardima/data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                          shuffle=False, num_workers=2)


# ## Pre-training model
class CNN_pretrain(nn.Module):
    
    def __init__(self):
        super(CNN_pretrain, self).__init__()
        
        self.c1 = nn.Conv2d(1, 16, kernel_size=5)
        self.c2 = nn.Conv2d(16, 32, kernel_size=5)
        self.c3 = nn.Conv2d(32, 16, kernel_size=5)
        self.c4 = nn.Conv2d(16, 16, kernel_size=5)
        self.c5 = nn.Conv2d(16, 10, kernel_size=12)
    
    def forward(self, x):
        
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = self.c5(x)
        
        x = x.view(-1, 10)
        
        return x


class CNN(nn.Module):
    
    def __init__(self, *, embed_dim):
        super(CNN, self).__init__()
        
        self.c1 = nn.Conv2d(1, 16, kernel_size=5)
        self.c2 = nn.Conv2d(16, 32, kernel_size=5)
        self.c3 = nn.Conv2d(32, 16, kernel_size=5)
        self.c4 = nn.Conv2d(16, 16, kernel_size=5)
        self.embed_layer = nn.Conv2d(16, embed_dim, kernel_size=12)
        
        self.EMBED_DIM = embed_dim
    
    def forward(self, x):
        
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = self.embed_layer(x)
        
        x = x.view(-1, self.EMBED_DIM)
        
        return x
    
    def load_pretrained(self, cnn_pretrained):   
        params1 = cnn_pretrained.named_parameters()
        params2 = self.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(param1.data)

        self.load_state_dict(dict_params2)
        


cnn_pretrain = CNN_pretrain()
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_pretrain.parameters(), lr = 1e-3, momentum=0.9)
lr_update = lambda epoch: 0.95 ** epoch
cnn_pretrain = cnn_pretrain.to(device)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_update)


print_step = len(trainset) // (trainloader.batch_size * 4)

for e in range(5):
    running_loss = 0.
    
    for i, batch in enumerate(trainloader, 1):
        x, label = batch
        x, label = x.to(device), label.to(device)
        
        optimizer.zero_grad()
        out = cnn_pretrain.forward(x)
        loss = loss_fun(out, label)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % print_step == 0:
            print('[%d, %5d] loss: %.3f' %
                  (e + 1, i, running_loss / print_step))
            running_loss = 0.


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = cnn_pretrain(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the pretrained network on the 10000 test images: {}%'.format(
    100 * correct / total))

sync_stdout()


def gen_triplet(dataset, classindices, n_samples):
    X = []
    y = []
    for i in range(n_samples):
        c = np.random.choice(10, size=2, replace=False)
        
        p_samples = np.random.choice(classindices[c[0]], size=2, replace=False)
        n_sample = np.random.choice(classindices[c[1]], size=1)
        
        X.append(dataset[p_samples[0]][0])
        X.append(dataset[p_samples[1]][0])
        X.append(dataset[n_sample[0]][0])
        
        y.append(dataset[p_samples[0]][1])
        y.append(dataset[p_samples[1]][1])
        y.append(dataset[n_sample[0]][1])
        
    return (torch.stack(X), 
            torch.stack(y))



def triplet_loss(loss_fun, output, n_samples, target, reg):
    ix = np.arange(n_samples, dtype=np.int32) * 3
    
    distp = (output[ix] - output[ix + 1]).pow(2).sum(dim=1)
    distn = (output[ix] - output[ix + 2]).pow(2).sum(dim=1)
    
    norm_loss = output.pow(2).sum() / n_samples
    loss_dist = loss_fun(distn, distp, target) / n_samples
    
    return reg * norm_loss + loss_dist





def try_params(trainset, classindices, n_samples,
                cnn_pretrained, reg, margin, embed_dim):

    model_name = "model_ns{}_reg{:.0e}_margin{:.1e}_dim{}".format(n_samples, reg, margin, embed_dim)
    
    print("\n\n\n\n")
    print("**" * 10)
    print("Testing model " + model_name)

    sync_stdout()

    # reg = 1e-2
    # margin = 10
    loss_fun = nn.MarginRankingLoss(margin=margin)
    # n_samples = 3

    tries = 5
    converge = False
    while tries > 0 and not converge:
        tries -= 1

        cnn = CNN(embed_dim=embed_dim)
        cnn.load_pretrained(cnn_pretrain)
        cnn = cnn.to(device)

        optimizer = optim.SGD(cnn.parameters(), lr = 1e-3, momentum=0.9)
        lr_update = lambda epoch: 0.95 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_update)

        losses = []
        for e in range(2):
            running_loss = 0.
            scheduler.step()
            
            for i in range(1, 10000):
                x, label = gen_triplet(trainset, classindices, n_samples)
                x = x.to(device)
                target = torch.FloatTensor(n_samples).fill_(1).to(device)
                
                optimizer.zero_grad()
                out = cnn.forward(x)
                loss = triplet_loss(loss_fun, out, n_samples, target, reg)
                loss.backward()
                
                optimizer.step()
                
                running_loss += loss.item()
                
                if i % 1250 == 0:
                    # print('[%d, %5d] loss: %.3f' %
                    #     (e + 1, i, running_loss / 1250))
                    losses.append(running_loss / 1250)
                    running_loss = 0.

        hasnan = False
        for p in cnn.parameters():
            hasnan = hasnan or torch.isnan(p.data).any().item()

        if hasnan or losses[-1] > losses[0] * 0.7:
            converge = False
        else:
            converge = True

    if not converge:
        print("Model did not converge...")
    else:
        print("Model converged in {} tries.".format(5 - tries))
        for e in range(20):
            running_loss = 0.
            scheduler.step()
            
            for i in range(1, 10000):
                x, label = gen_triplet(trainset, classindices, n_samples)
                x = x.to(device) 
                target = torch.FloatTensor(n_samples).fill_(1).to(device)
                
                optimizer.zero_grad()
                out = cnn.forward(x)
                loss = triplet_loss(loss_fun, out, n_samples, target, reg)
                loss.backward()
                
                optimizer.step()
                
                running_loss += loss.item()
                
                if i % 1250 == 0:
                    # print('[%d, %5d] loss: %.3f' %
                    #     (e + 1, i, running_loss / 1250))
                    if e == 19:
                        print("Final loss: {}".format(running_loss / 1250))
                    running_loss = 0.


        torch.save(cnn, '/home/yardima/Python/models/mnist-basic-triplet-ablation/' + model_name)

        embed_model = cnn

        train_embed = []
        train_label = []
        with torch.autograd.no_grad():
            for x, label in trainset:
                x = x.to(device).view(1, 1, 28, 28)
                y = embed_model.forward(x).cpu().data.numpy()
                train_embed.append(y)
                train_label.append(label.item())
                
        train_embed = np.array(train_embed)
        train_label = np.array(train_label, dtype=np.int32)

        train_embed = train_embed.reshape((-1, embed_model.EMBED_DIM))

        test_embed = []
        test_label = []
        with torch.autograd.no_grad():
            for x, label in testset:
                x = x.to(device).view(1, 1, 28, 28)
                y = embed_model.forward(x).cpu().data.numpy()
                test_embed.append(y)
                test_label.append(label.item())
                
        test_embed = np.array(test_embed)
        test_label = np.array(test_label, dtype=np.int32)

        test_embed = test_embed.reshape((-1, embed_model.EMBED_DIM))

        if np.isnan(test_embed).any() or 
           np.isnan(train_embed).any():
            print("Model output has NaN values.")
        else:
            nn3_classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=8)
            nn3_classifier.fit(train_embed, train_label)
            pred3 = nn3_classifier.predict(test_embed)
            print("k-NN classification on MNIST embeddings (k=3)")
            print(np.mean(pred3 == test_label))

    print("**" * 10)
    print("\n" * 5)

    sync_stdout()



classindices = []
trainy = trainset.train_labels.data.numpy()
for i in range(10):
    classindices.append(np.nonzero(trainy == i)[0])

print("Starting experiment...")

exper = 0
for reg in [3e-3, 1e-3, 3e-4]:
    for n_samples in [5, 3]:
        for margin in [1]:
            for embed_dim in [50, 100]:
                print("Experiment {}...".format(exper))
                exper += 1
                try_params(trainset, classindices, n_samples, cnn_pretrain, reg, margin, embed_dim)

