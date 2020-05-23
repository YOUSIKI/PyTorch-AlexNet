import os
import tqdm
import torch
import argparse
import tensorboardX
import utils
import models
import meters
import datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="default")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-s", "--starting_epoch", type=int, default=0)
    parser.add_argument("-d", "--dataset", type=str, default="imagenette")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--normalization", type=str, default="lrn")
    parser.add_argument("--pooling", type=str, default="max")
    parser.add_argument("--num_classes", type=int, default=1000)
    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    args = parse_args()
    args = utils.analyze_arguments(args)
    model = models.AlexNet(
        args.num_classes,
        pooling=args.pooling,
        activation=args.activation,
        normalization=args.normalization
    ).to(args.device)
    if args.starting_epoch > 0:
        utils.load_model(model, "%s-epoch-%03d" % (args.name, args.starting_epoch - 1))
    # optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    accuracy_meter = meters.AccuracyMeter((1, 3, 5))
    writer = tensorboardX.SummaryWriter(os.path.join("log", args.name))
    train_loader = args.dataset(args.batch_size, train=True)
    valid_loader = args.dataset(args.batch_size, train=False)
    training_steps = 0
    for epoch in range(args.starting_epoch, args.starting_epoch + args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        model.train()
        accuracy_meter.reset()
        with tqdm.tqdm(train_loader) as loader:
            loader.set_description(f"training on epoch {epoch}")
            for images, labels in loader:
                images = images.to(args.device)
                labels = labels.to(args.device)
                predict = model(images)
                loss = criterion(predict, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                accuracy_meter.update(predict, labels)
                loader.set_postfix(accuracy_meter.as_dict())
                training_steps += labels.size(0)
                writer.add_scalars("train", {"loss": loss.item()}, training_steps)
                writer.add_scalars("train", accuracy_meter.as_dict(), training_steps)
        print(f"trained on epoch {epoch}, {str(accuracy_meter)}")
        model.eval()
        accuracy_meter.reset()
        with tqdm.tqdm(valid_loader) as loader:
            loader.set_description(f"testing on epoch {epoch}")
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(args.device)
                    labels = labels.to(args.device)
                    predict = model(images)
                    accuracy_meter.update(predict, labels)
                    loader.set_postfix(accuracy_meter.as_dict())
        writer.add_scalars("test", accuracy_meter.as_dict(), training_steps)
        print(f"tested on epoch {epoch}, {str(accuracy_meter)}")
        utils.save_model(model, "%s-epoch-%03d" % (args.name, epoch))
        writer.close()