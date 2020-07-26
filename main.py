import logging
from argparse import ArgumentParser

import torch
import torch.nn as nn
from dataset import *
from model import UNET
from util import *

logger = logging.getLogger(__file__)


def train_UNET(args, train_loader, val_loader):
    logger.info("---------Using device %s--------", args.device)

    model = UNET()
    loss_fun = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    logger.info("---------Initializing Training For UNET!--------")
    for i in range(0, args.n_epochs):
        """Train each epoch"""
        model.train()
        for batch, data in enumerate(train_loader):
            images, labels = data[0], data[1]
            likelihood_map = model(images.to(args.device))
            loss = loss_fun(likelihood_map, labels.to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """Get Loss and Accuracy for each epoch"""
        model.eval()
        total_acc = 0
        total_loss = 0
        for batch, data in enumerate(train_loader):
            images, labels = data[0], data[1]
            with torch.no_grad():
                likelihood_map = model(images.to(args.device))
                loss = loss_fun(likelihood_map, labels.to(args.device))
                pred_class = likelihood_map > 0.5
                acc = accuracy_for_batch(labels.cpu(), pred_class.cpu(), args)
                total_acc += acc
                total_loss += loss.cpu().item()
        train_acc_epoch, train_loss_epoch = total_acc / (batch + 1), total_loss / (batch + 1)
        print('Epoch', str(i + 1), 'Train loss:', train_acc_epoch, "Train acc", train_loss_epoch)

        """Validation for every 5 epochs"""
        if (i + 1) % 5 == 0:
            print('Val loss:', val_loss, "val acc:", val_acc)


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path_train", type=str, default="train_ISBI13/train-volume.tif",
                        help="Path or url of the dataset")
    parser.add_argument("--dataset_path_label", type=str, default="train_ISBI13/train-labels_thin.tif",
                        help="Path or url of the dataset")
    parser.add_argument("--dataset_cache", type=str,
                        default='dataset_cache_ISBI13', help="Path or url of the preprocessed dataset cache")
    parser.add_argument("--train_batch_size", type=int,
                        default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int,
                        default=1, help="Batch size for validation")
    parser.add_argument("--valid_round", type=int,
                        default=5, help="validation part: 1, 2, 3, 4, 5")
    parser.add_argument("--lr", type=float,
                        default=6.25e-4, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
    else "cpu", help="Device (cuda or cpu)")

    args = parser.parse_args()

    logger.info("---------Prepare DataSet--------")
    trainDataset, validDataset = get_dataset(args)
    train_loader = torch.utils.data.DataLoader(dataset=trainDataset, num_workers=6, batch_size=args.train_batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=validDataset, num_workers=6, batch_size=args.valid_batch_size,
                                             shuffle=False)

    train_UNET(args, train_loader, val_loader)


if __name__ == "__main__":
    train()
