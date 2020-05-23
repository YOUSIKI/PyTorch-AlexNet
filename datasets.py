import os
import torch
import torchvision


def imagenet_like(batch_size, train=True, name="imagenet"):
    root = os.path.join("datasets", name, "train" if train else "val")
    mean = [0.485, 0.456, 0.406]
    stdv = [0.229, 0.224, 0.225]
    if train:
        transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize(256),
            # torchvision.transforms.RandomCrop(224),
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, stdv)
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, stdv)
        ])
    dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader


def imagenette(batch_size, train=True):
    return imagenet_like(batch_size, train, "imagenette2")


def imagenet(batch_size, train=True):
    return imagenet_like(batch_size, train, "imagenet")