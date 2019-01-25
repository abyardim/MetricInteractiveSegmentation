# Metric Interactive Segmentation

This repository contains the code for my semester project at the Computer Vision Lab at ETH Zurich.
The project experiments with pixel embedding models in the context of interactive image segmentation.

## Installing the library

Most of the implementation was made as a standalone semantic segmentation library. You can install this with the command:

~~~
pip3 install -e lib
~~~

Then import the library from Python 3 with `import deeptriplet`.

## Training the best model

To train the best performing model, run the script [`train_triplet_aug_spatial_last.py`](script_experiments/deeplabv3p/train_triplet_aug_spatial_last.py) with the following options:

~~~
python script_experiments/deeplabv3p/train_triplet_aug_spatial_last.py --runid=1 --lr=3e-4 --ntripletstrain=7500 --epochs=80 --name="deeplabv3p-triplet-aug-spatial-last" --saveroot=/dir/for/saving/results
~~~

Note that the training script requires a GPU. To learn about other available hyperparameters, you can run the help command

~~~
python ./train_triplet_aug_spatial_last.py -h
~~~

## Testing the best model

The pretrained parameters for the best performing model is made available [here](https://drive.google.com/file/d/1Fsl1HnTnhPbk23Ws0ysDLiKZjLMWdy5M/view?usp=sharing). The minimal code for loading this model is:

```python
import torch
from deeptriplet.models.deeplabv3p.deeplab_spatial_last import DeepLabSpatialLate
d = torch.load("best_model.pth")
net = DeepLabSpatialLate(backbone='resnet', num_classes=50, dynamic_coordinates=True, sync_bn=True, freeze_bn=False)
net.load_state_dict(d)
```

## Requirements

This project requires Python 3.6. It also requires installing the following libraries:

- numpy
- scipy
- matplotlib
- scikit-learn
- jupyter
- tensorboardX
- pytorch
- torchvision
