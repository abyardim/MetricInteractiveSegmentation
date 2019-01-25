import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.utils.data as data
import torch.nn as nn

import deeptriplet
import deeptriplet.models
import deeptriplet.triplet

import ptsemseg
import ptsemseg.loader

# Tensorboard include
from tensorboardX import SummaryWriter

gpu_id = 0

nEpochs = 50  # Number of epochs for training
resume_epoch = 0   # Default is 0, change if want to resume

p = OrderedDict()  # Parameters to include in report
useTest = True  # See evolution of the test set when training
nTestInterval = 5 # Run on test set every nTestInterval epochs
snapshot = 10  # Store a model every snapshot epochs

p['details'] = 'voc12-fcn8-self-pretrained'
p['lr'] = 1e-3  # Learning rate
p['wd'] = 5e-9  # Weight decay
p['momentum'] = 0.9  # Momentum
p['e_dim'] = 30

run_id = 5

save_dir_root = "/srv/glusterfs/yardima/runs/fcn8_embedding"
exp_name = 'fcn8-embedding-voc-random-1'

print('Using GPU: {} '.format(gpu_id))
# Setting parameters


save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))

net = deeptriplet.models.FCN8Embedding(embedding_dim=p['e_dim'])

modelName = 'embedding-fcn8-voc12-random3890852'

triplet_fun = deeptriplet.triplet.RandomTriplet()
triplet_fun_test = deeptriplet.triplet.RandomTriplet(n_triplets=500)

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()

if resume_epoch == 0:
    print("Training embededings from init by FCN8s...")
else:
    print("Initializing weights from: {}...".format(
        os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage)) # Load all tensors onto the CPU

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

pretrained_state = torch.load("/srv/glusterfs/yardima/runs/pretrained-models/fcn8s_pascal_best_model.pkl",
                              map_location=lambda storage, loc: storage)
                              

net = nn.DataParallel(net)
net.load_state_dict(pretrained_state['model_state'], strict=False)
net.cuda()

optimizer = optim.SGD(net.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 42], gamma=0.3)


if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    p['optimizer'] = str(optimizer)

    data_loader = ptsemseg.loader.get_loader("pascal")

    data_aug = None

    t_loader = data_loader("/srv/glusterfs/yardima/datasets/voc12/VOCdevkit/VOC2012",
                            is_transform=True,
                            split="train_aug",
                            img_size=("same", "same"),
                            augmentations=data_aug)
    v_loader = data_loader("/srv/glusterfs/yardima/datasets/voc12/VOCdevkit/VOC2012",
                            is_transform=True,
                            split="val",
                            img_size=("same", "same"),)

    trainloader = data.DataLoader(t_loader,
                                    batch_size=1,
                                    num_workers=2,
                                    shuffle=True)
    valloader = data.DataLoader(v_loader,
                                    batch_size=1,
                                    num_workers=2)



    generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    num_img_tr = len(trainloader)
    num_img_ts = len(valloader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    global_step = 0
    print("Training Network")

    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()

        scheduler.step()

        net.train()
        for ii, sample_batched in enumerate(trainloader):

            inputs, labels = sample_batched
            # Forward-Backward of the mini-batch
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
            global_step += inputs.data.shape[0]

            if gpu_id >= 0:
                inputs = inputs.cuda()

            outputs = net.forward(inputs)
            loss = triplet_fun.compute_loss(outputs, labels)
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == (num_img_tr - 1):
                running_loss_tr = running_loss_tr / num_img_tr
                writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                # writer.add_scalar('data/lr', scheduler.get_lr(), epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii + inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")

            # Backward the averaged gradient
            loss.backward()

            # Update the weights once in p['nAveGrad'] forward passes
            writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
            optimizer.step()
            optimizer.zero_grad()
            del loss


        # Save the model
        if (epoch % snapshot) == snapshot - 1:
            torch.save(net.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth')))

        # One testing epoch
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            total_iou = 0.0
            net.eval()
            for ii, sample_batched in enumerate(valloader):
                inputs, labels = sample_batched

                # Forward pass of the mini-batch
                inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                if gpu_id >= 0:
                    inputs = inputs.cuda()

                with torch.no_grad():
                    outputs = net.forward(inputs)

                    loss = triplet_fun_test.compute_loss(outputs, labels)

                running_loss_ts += loss.item()

                # Print stuff
                if ii % num_img_ts == num_img_ts - 1:

                    running_loss_ts = running_loss_ts / num_img_ts

                    print('Validation:')
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii + inputs.data.shape[0]))
                    writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
                    print('Loss: %f' % running_loss_ts)
                    running_loss_ts = 0


    writer.close()