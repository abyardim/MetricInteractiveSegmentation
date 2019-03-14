
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets
import torch.utils.data as data

from sklearn.neighbors import KNeighborsClassifier
import scipy.ndimage.morphology as morph

import skimage
import skimage.segmentation
from scipy import ndimage

import multiprocessing as mp

def run_clicks(embeddings, labels, clicks, valid, flat_indices=True):
    dim1, dim2 = labels.squeeze().shape

    if not flat_indices:
        clicks = np.array([c[0] * dim2 + c[1] for c in clicks], dtype=np.int32)
        
    embeddings = embeddings.reshape(dim1 * dim2, -1)
    labels = labels.reshape(-1)

    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    knn.fit(embeddings[clicks,:], labels[clicks])

    pred = knn.predict(embeddings)

    cf_matrix = _fast_hist(labels[valid], pred[valid], 2)

    iou = cf_matrix[1,1] / (cf_matrix[0,1] + cf_matrix[1,1] + cf_matrix[1,0])
    acc = np.sum(labels[valid] == pred[valid]) / len(valid)

    return {"pred":pred.reshape(dim1,dim2), 
            "cf":cf_matrix, 
            "iou":iou, 
            "acc":acc}

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def run_click_sim(embeddings, labels, n_clicks, computer_borders_manually, d_margin=15, d_clicks=5):
    label = labels.squeeze()
    embedding = embeddings.squeeze()
    
    ## pick an object at random
    o = np.random.choice(np.sort(np.unique(label[label != 255]))[1:])
    label_obj = (label == o).astype(np.int32)
    
    valid = np.nonzero((label.reshape(-1) != 255))[0]

    if computer_borders_manually:
        border = label == 255
        border = border + morph.binary_dilation(skimage.segmentation.find_boundaries(label_obj), iterations=d_margin)
    else:
        if np.any(label == 255):
            temp = (label == 255) + skimage.segmentation.find_boundaries(label_obj)
            border = morph.binary_dilation(label_obj, iterations=6) & temp
            border = morph.binary_dilation(border, iterations=max(d_margin-3, 1))
        else:
            border = label == 255
            border = border + morph.binary_dilation(skimage.segmentation.find_boundaries(label_obj), iterations=d_margin)

    label_obj_temp = label_obj.copy()
    label_obj_temp[:, 0] = 1
    label_obj_temp[0, :] = 1
    label_obj_temp[-1, :] = 1
    label_obj_temp[:, -1] = 1
    
    inner_part = label_obj
    outer_part = np.logical_not(label_obj_temp)

    ## pick initial points
    dist_inner = ndimage.distance_transform_edt(inner_part)
    dist_outer = ndimage.distance_transform_edt(outer_part)
    
    c1 = np.random.choice(np.nonzero(dist_inner.ravel() == dist_inner.max())[0])
    c2 = np.random.choice(np.nonzero(dist_outer.ravel() == dist_outer.max())[0])

    clicks = [c1, c2]

    results = [run_clicks(embedding, label_obj, clicks, valid, flat_indices=True)]
    last_pred = results[-1]['pred']
    
    click_map = np.zeros_like(label_obj)
    click_map.ravel()[c1] = 1
    click_map.ravel()[c2] = 1

    for k in range(2, n_clicks):
        mislabeled = last_pred != label_obj
        mislabeled_border = mislabeled & ~border

        if mislabeled_border.any():
            mislabeled_positive = mislabeled_border & label_obj
            mislabeled_negative = mislabeled_border & ~label_obj

            dist_mlp = ndimage.distance_transform_edt(mislabeled_positive)
            dist_mln = ndimage.distance_transform_edt(mislabeled_negative)

            if dist_mlp.max() > dist_mln.max():
                new_click = np.random.choice(np.nonzero(dist_mlp.ravel() == dist_mlp.max())[0])
            else:
                new_click = np.random.choice(np.nonzero(dist_mln.ravel() == dist_mln.max())[0]) 
        else:
            valid_clicks = ndimage.distance_transform_edt(np.logical_not(click_map)) > d_clicks
            
            dist = ndimage.distance_transform_edt(~mislabeled)
            dist = dist * (~border) * valid_clicks
            new_click = np.random.choice(np.nonzero((dist == dist[~border * valid_clicks].min()).ravel())[0])

        clicks.append(new_click)
        
        click_map.ravel()[new_click] = 1

        results.append(run_clicks(embedding, label_obj, clicks, valid, flat_indices=True))
        last_pred = results[-1]['pred']
        
    return results


def run_image(i, n_clicks, compute_border):
    embedding = np.load("/scratch-second/yardima/temp/{}.embed.aug.npy".format(i))
    label = np.load("/scratch-second/yardima/temp/{}.label.aug.npy".format(i))

    return run_click_sim(embedding, label, n_clicks, compute_border, d_margin=5, d_clicks=10)


def test_clicks(model, dataset, compute_border, n_clicks=10, n_images=100, tresh=0.85):
    data_loader = data.DataLoader(dataset,
                                                batch_size=1,
                                                num_workers=4,
                                                shuffle=True)
    
    
    if n_images == None:
        n_images = len(dataset)
    
    img_count = 0
    data_iter = iter(data_loader)
    while img_count < n_images:        
        image, label = next(data_iter)

        if image.dim() == 3:
            image = image.unsqueeze(0)

        with torch.no_grad():
            image = image.cuda()
            out = model.forward(image)

            embeddings = out.cpu().data.numpy()
            embeddings = np.transpose(embeddings.squeeze(), axes=[1, 2, 0])

            np.save("/scratch-second/yardima/temp/{}.embed.aug".format(img_count), embeddings)
            np.save("/scratch-second/yardima/temp/{}.label.aug.npy".format(img_count), label)
        
        img_count += 1
    
    
    pool = mp.Pool(processes=16)
    
    results = pool.starmap(run_image, zip(range(0,n_images), [n_clicks] * n_images, [compute_border] * n_images))
    
    pool.close()
    pool.join()
    del pool
    
    mean_acc = np.zeros(n_clicks - 1)
    mean_iou = np.zeros(n_clicks - 1)
    clicks = 0.0
    c = []

    for result in results:
        mean_acc += np.array([r["acc"] for r in result])
        mean_iou += np.array([r["iou"] for r in result])

        iou = np.array([r["iou"] for r in result])
        t = iou > tresh

        if np.any(t):
            clicks += np.argmax(t) + 2.
            c.append( np.argmax(t) + 2.)
        else:
            clicks += 20.0
            c.append(20)

    mean_acc /= len(results)
    mean_iou /= len(results)
    clicks /= len(results)
    
    return mean_iou, mean_acc, clicks

