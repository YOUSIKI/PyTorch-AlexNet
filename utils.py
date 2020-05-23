import os
import torch
import models
import datasets


def save_model(model, filename):
    filename = os.path.join("checkpoints", filename + ".pth")
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    filename = os.path.join("checkpoints", filename + ".pth")
    model.load_state_dict(torch.load(filename))


def auto_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def analyze_arguments(args):
    if hasattr(args, "dataset"):
        if args.dataset == "imagenette":
            args.num_classes = 10
            args.dataset = datasets.imagenette
        elif args.dataset == "imagenet":
            args.num_classes = 1000
            args.dataset = datasets.imagenet
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    if hasattr(args, "activation"):
        if args.activation == "relu":
            args.activation = torch.nn.ReLU
        elif args.activation == "tanh":
            args.activation = torch.nn.Tanh
        elif args.activation == "none":
            args.activation = torch.nn.Identity
        else:
            raise ValueError(f"Unknown activation: {args.activation}")
    if hasattr(args, "normalization"):
        if args.normalization == "bn":
            args.normalization = models.BatchNorm
        elif args.normalization == "lrn":
            args.normalization = models.LocalResponseNorm
        elif args.normalization == "none":
            args.normalization = models.Identity
        else:
            raise ValueError(f"Unknown normalization: {args.normalization}")
    if hasattr(args, "pooling"):
        if args.pooling == "max":
            args.pooling = torch.nn.MaxPool2d
        elif args.pooling == "avg":
            args.pooling = torch.nn.AvgPool2d
        elif args.pooling == "adpmax":
            args.pooling = torch.nn.AdaptiveMaxPool2d
        elif args.pooling == "adpavg":
            args.pooling = torch.nn.AdaptiveAvgPool2d
        else:
            raise ValueError(f"Unknown pooling: {args.pooling}")
    args.device = auto_device()
    return args