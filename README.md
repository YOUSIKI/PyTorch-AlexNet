# PyTorch-AlexNet

## Introduction in one line

This repository is a PyTorch implementation of AlexNet.

## Details

- based on **PyTorch 1.5** and **Python 3.7**.
- using **TensorBoardX** to record loss and accuracy.
- pretrained on **imagenette** (a subset of 10 classes from imagenet).
- supports **both Batch Normalization and Local Response Normalization**.
- using **groups** of convolution layers to simulate multi-gpu training, 
  thus the network structure is more familiar to the original one in the paper
  rather than the official implementation of pytorch.
  
## Usage

### Train

#### Requirements

```
python=3.7
tqdm
torch
torchvision
tensorboardx
```

CUDA support is recommended but not essential.

#### Prepare dataset

Please refer to [this page](https://github.com/fastai/imagenette) to download the imagenette dataset.

Extract the `imagenette2` folder to `PyTorch-AlexNet/datasets/`.

### Start training

Run the following command under `PyTorch-AlexNet/`.

```
python train.py --name myalexnet
```

#### Arguments table

- `--normalization` choose which normalization method to use, either `bn` or `lrn`.
- `--activation` choose which activation method to use, either `relu` or `tanh`.
- `--pooling` choose which pooling method to use, either `max` or `avg`.
- `--epochs` how many epochs to train, a possitive integer.
- `--batch_size` how many images a batch contains, a possitive integer.
- `--num_classes` how many classes to classify in this dataset, it can be automatically set if using imagenet or imagenette dataset.
- `--dataset` the name of dataset, either `imagenet` or `imagenette`.
- `--starting_epoch` the starting epoch, default 0, if set to a possitive integer, the starting_epoch-1 checkpoint will be loaded before training.

### Learning rate

I found that 5e-3 is a nice learning rate for both batch normalization and local response normalization network. 
The learning rate will automatically decrease during training. 
Actually, it will be multiplied by 0.1 every 30 epochs.

### TensorboardX

the tensorboard logdir is `log/name`, run the following command to start tensorboard server.

```
tensorboard --logdir log/myalexnet
```

Remember that tensorboard and tensorflow should be installed before this.

## Credit

Krizhevsky, Alex & Sutskever, Ilya & Hinton, Geoffrey. (2012). ImageNet Classification with Deep Convolutional Neural Networks. Neural Information Processing Systems. 25. 10.1145/3065386. 
