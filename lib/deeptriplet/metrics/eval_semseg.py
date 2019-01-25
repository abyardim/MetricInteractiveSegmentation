
import torch
import torch.nn as nn

import numpy as np



def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def compute_miou_acc(model, valloader, n_classes):
    n_class = n_classes
    image_count = 0
    pred_count = 0
    acc = 0.
    cf_matrix = np.zeros(shape=(n_class, n_class), dtype=np.float64)

    model = model.eval()

    with torch.no_grad():
        for i, l in valloader:
            i = i.cuda()

            out = model(i)

            fc8_interp_test = nn.UpsamplingBilinear2d(size=(i.shape[2], i.shape[3]))
            out = fc8_interp_test(out).data.cpu().numpy()

            out = out.transpose(0, 2, 3, 1)
            out = np.argmax(out, axis=3)

            out = out.reshape(-1)
            lbl = l.numpy().reshape(-1)

            assert lbl.shape[0] == out.shape[0]

            indices = np.nonzero(lbl != 255)
            lbl = lbl[indices]
            out = out[indices]

            acc += np.sum(out == lbl)
            pred_count += lbl.shape[0]
            #             image_count += 1

            cf_matrix += _fast_hist(lbl, out, n_class)

            del out

    acc /= pred_count
    cf_matrix /= np.sum(cf_matrix)
    hist = cf_matrix
    mean_iou = np.nanmean(np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)))

    return mean_iou, acc
