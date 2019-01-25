import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import collections
import os
import socket

from datetime import datetime
import timeit

import deeptriplet.datasets
import deeptriplet.metrics
import deeptriplet.triplet

# Tensorboard include
from tensorboardX import SummaryWriter


import argparse

parser = argparse.ArgumentParser(description='Train spatial-late fusion embedding model')

parser.add_argument('--runid', type=int, help='Run ID', required=True)
parser.add_argument('--saveroot', type=str, help='Save directory root', required=True)
parser.add_argument('--name', type=str, help='Name', required=True)
parser.add_argument('--epochs', type=int, help='Number of trianing epochs', required=True)

parser.add_argument('--dim', type=int, help='Embedding dimension', default=45)
parser.add_argument('--lr', type=float, help='Learning rate', default=3e-3)
parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
parser.add_argument('--poly', type=float, help='Poly policy power', default=0.9)
parser.add_argument('--wd', type=float, help='Weight decay', default=5e-4)
parser.add_argument('--ntripletsval', type=int, help='Triplet count during validation', default=500)
parser.add_argument('--ntripletstrain', type=int, help='Triplet count during training', default=500)


parser.add_argument('--intsnapshot', type=int, help='Snapshot interval', default=10)
parser.add_argument('--inttest', type=int, help='Test interval', default=1)

args = parser.parse_args()

print("Run {}".format(args.runid))

gpu_id = 0

nEpochs = args.epochs  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume

p = collections.OrderedDict()  # Parameters to include in report
useTest = True  # See evolution of the test set when training
nTestInterval = args.inttest  # Run on test set every nTestInterval epochs
snapshot = args.intsnapshot  # Store a model every snapshot epochs

p['lr'] = args.lr  # Learning rate
p['wd'] = args.wd  # Weight decay
p['momentum'] = args.momentum  # Momentum
p['poly_power'] = args.poly
p['dim'] = args.dim

save_dir_root = args.saveroot
modelName = args.name
run_id = args.runid
exp_name = args.name

print('Using GPU: {} '.format(gpu_id))
# Setting parameters


save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))


########################## model

net = deeptriplet.models.DeepLab_VGG_SpatialLate(n_classes=p['dim'])
d = torch.load("/srv/glusterfs/yardima/runs/deeplabv2/lfov-triplet/run_8/models/class-vgg-pascal_epoch-49.pth", map_location=lambda storage, loc: storage)
d['fc6.0.weight'] = torch.cat((d['fc6.0.weight'], net.state_dict()["fc6.0.weight"][:,-2:,:,:]), 1)
net.load_state_dict(d, strict=False)
net = net.eval().cuda()


###########################  optimzers, datasets

init_lr = p['lr']

optimizer = optim.SGD([
                {'params': net.get_parameter_group_v2(bias=False, final=False), 'lr': init_lr},
                {'params': net.get_parameter_group_v2(bias=True, final=False), 'lr': init_lr*2, 'weight_decay':0},
                {'params': net.get_parameter_group_v2(bias=False, final=True), 'lr': init_lr*10},
                {'params': net.get_parameter_group_v2(bias=True, final=True), 'lr': init_lr*20, 'weight_decay':0}
            ], lr=init_lr, momentum=p['momentum'], weight_decay=p['wd'])

# loss_fn = nn.CrossEntropyLoss(ignore_index=255)
loss_fn = deeptriplet.triplet.RandomTripletPreselected(n_batch=10, n_triplets=args.ntripletstrain)
loss_fn_val = deeptriplet.triplet.RandomTripletPreselected(n_batch=1, n_triplets=args.ntripletsval)

valset = deeptriplet.datasets.PascalDatasetRandomTriplet(pascal_root="/scratch/yardima/datasets/voc12/VOCdevkit/VOC2012",
                                            split_file="/home/yardima/Python/experiments/pascal_split/val_obj.txt",
                                            n_triplets=args.ntripletsval,
                                            normalize_imagenet=True,
                                            augment=False,
                                            pad_zeros=True,
                                            downsample_label=8)

valloader = data.DataLoader(valset,
                                batch_size=1,
                                num_workers=2,
                                shuffle=False)

trainset = deeptriplet.datasets.PascalDatasetRandomTriplet(pascal_root="/scratch/yardima/datasets/voc12/VOCdevkit/VOC2012",
                        split_file="/home/yardima/Python/experiments/pascal_split/train_obj.txt",
                        n_triplets=args.ntripletstrain,
                        normalize_imagenet=True,
                        augment=True,
                        pad_zeros=True,
                        downsample_label=8,
                        scale_low=0.8,
                        scale_high=1.2)

trainloader = data.DataLoader(trainset,
                                batch_size=10,
                                num_workers=4,
                                shuffle=True)


############################ custom functions

def update_lr_poly(optimizer, init_lr, step, max_step, power):
    lr = init_lr * ((1 - step / float(max_step)) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = 2 * lr
    optimizer.param_groups[2]['lr'] = 10 * lr
    optimizer.param_groups[3]['lr'] = 20 * lr

    return optimizer





################## main script


def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()


print("Training from init...")

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net = net.cuda()


if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    p['optimizer'] = str(optimizer)

    generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    num_img_tr = len(trainloader)
    num_img_ts = len(valloader)
    running_loss_tr = 0.0
    global_step = 0
    max_steps = nEpochs * num_img_tr
    print("Training Network")

    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()

        net = net.eval()
        for ii, sample_batched in enumerate(trainloader):

            inputs, labels = sample_batched
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer = update_lr_poly(optimizer, init_lr, global_step, max_steps, p['poly_power'])

            optimizer.zero_grad()

            outputs = net.forward(inputs)
            loss = loss_fn.compute_loss(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss_tr += loss.item()


            writer.add_scalar('data/total_loss_iter', loss.item(), global_step)
            writer.add_scalar('data/lr', optimizer.param_groups[0]['lr'], global_step)

            global_step += 1

            del loss, outputs


        running_loss_tr = running_loss_tr / num_img_tr
        writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)

        print('[Epoch: %d]' % (epoch,))
        print('Loss: %f' % running_loss_tr)
        running_loss_tr = 0.
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")

        # Save the model
        if (epoch % snapshot) == snapshot - 1:
            torch.save(net.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))
            print("Save model at {}\n".format(
                os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth')))

        # One testing epoch
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            net = net.eval()
            count_val = 0
            loss_val = 0.0
            with torch.no_grad():
                for ii, sample_batched in enumerate(valloader):
                    inputs, labels = sample_batched
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    
                    outputs = net.forward(inputs)
                    loss = loss_fn_val.compute_loss(outputs, labels)
                    
                    loss_val += loss.item()
                    
                    del outputs, loss
                    
                    count_val += 1
            
            loss_val /= count_val
            
            writer.add_scalar('data/val_loss', loss_val, epoch)
            
            ## net = net.train()

    writer.close()
