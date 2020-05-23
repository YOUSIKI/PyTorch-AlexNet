import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tqdm
import logging
import torchvision.datasets as ds
import torchvision.transforms as transforms
import meters


logging.basicConfig(level=logging.DEBUG)


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Metric(object):

    def __init__(self, func):
        super().__init__()
        self.clear()
        self.func = func
    
    def clear(self):
        self.num = 0
        self.val = 0

    def update(self, *args, **kwargs):
        n, v = self.func(*args, **kwargs)
        self.num += n
        self.val += v
    
    def average(self):
        return self.val / self.num


class CrossEntropyLossMetric(Metric):

    def __init__(self):
        super().__init__(func=cross_entropy_loss_func)


def cross_entropy_loss_func(predict, target):
    n = target.size(0)
    v = F.cross_entropy(predict, target).item()
    return n, v


class AccuracyMetric(Metric):

    def __init__(self):
        super().__init__(accuracy_func)
    

def accuracy_func(predict, target):
    n = target.size(0)
    p = torch.max(predict.data, dim=1)[1]
    v = (target == p).sum().item()
    return n, v


if __name__ == "__main__":
    epochs = 20
    batch_size = 64
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root_dir = os.path.join("datasets", "imagenette2")
    dataset_train_dir = os.path.join(dataset_root_dir, "train")
    dataset_valid_dir = os.path.join(dataset_root_dir, "val")
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalization
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalization
    ])
    train_dataset = ds.ImageFolder(root=dataset_train_dir, transform=train_transform)
    valid_dataset = ds.ImageFolder(root=dataset_valid_dir, transform=valid_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    net = AlexNet(num_classes=10).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    ceriation = nn.CrossEntropyLoss()
    meter = meters.AccuracyMeter((1, 5))
    for epoch in range(epochs):
        logging.info(f"training on epoch {epoch}..")
        net.train()
        train_acc_metric = AccuracyMetric()
        train_cel_metric = CrossEntropyLossMetric()
        meter.reset()
        for x, y in tqdm.tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)
            pred = net(x)
            loss = ceriation(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc_metric.update(pred, y)
            train_cel_metric.update(pred, y)
            meter.update(pred, y)
        logging.info(f"training cel: {train_cel_metric.average()}, acc: {train_acc_metric.average()}, {str(meter)}")
        logging.info(f"testing on epoch {epoch}...")
        net.eval()
        valid_acc_metric = AccuracyMetric()
        valid_cel_metric = CrossEntropyLossMetric()
        meter.reset()
        with torch.no_grad():
            for x, y in tqdm.tqdm(valid_loader):
                x = x.to(device)
                y = y.to(device)
                pred = net(x)
                valid_acc_metric.update(pred, y)
                valid_cel_metric.update(pred, y)
                meter.update(pred, y)
        logging.info(f"testing loss: {valid_cel_metric.average()}, acc: {valid_acc_metric.average()}, {str(meter)}")
