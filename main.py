import logging
from argparse import ArgumentParser

import torch
import torch.nn as nn
from dataset import *
from model import UNET
from save_history import *
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
        print('Epoch', str(i + 1), 'Train loss:', train_loss_epoch, "Train acc", train_acc_epoch)

        """Validation for every 5 epochs"""
        if (i + 1) % 5 == 0:
            total_val_loss = 0
            total_val_acc = 0
            for batch, data in enumerate(val_loader):
                images, labels = data[0], data[1]
                with torch.no_grad():
                    likelihood_map = model(images.to(args.device))
                    pred_class = likelihood_map > 0.5
                    loss = loss_fun(likelihood_map, labels.to(args.device))
                    save_prediction(likelihood_map, pred_class, args, batch, i)
                    save_groundTrue(images, labels, args, batch, i)
                    total_val_loss += loss.cpu().item()
                    acc_val = accuracy_check(labels.cpu(), pred_class.cpu())
                    total_val_acc += acc_val
            valid_acc_epoch, valid_loss_epoch = total_val_acc / (batch + 1), total_val_loss / (batch + 1),
            print('Val loss:', valid_loss_epoch, "val acc:", valid_acc_epoch)

            header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
            save_values = [i + 1, train_acc_epoch, train_loss_epoch, valid_acc_epoch, valid_loss_epoch]
            export_history(header, save_values, args)

        if (i + 1) % 10 == 0:
            save_models(i + 1, model, optimizer, args)


def train_LSTM(args):


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path_train", type=str, default="train_ISBI13/train-volume.tif",
                        help="Path or url of the dataset")
    parser.add_argument("--dataset_path_label", type=str, default="train_ISBI13/train-labels_thin.tif",
                        help="Path or url of the dataset")
    parser.add_argument("--dataset_cache", type=str,
                        default='dataset_cache_ISBI13', help="Path or url of the preprocessed dataset cache")
    parser.add_argument("--save_folder", type=str, default="results_unet/ISBI13",
                        help="Path or url of the dataset")
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
    parser.add_argument("--check_point", type=str, default="/model_epoch_100.pwf",
                        help="Path of the pre-trained CNN")
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

    lstmTrainDataset, lstmValidDataset = get_dataset_lstm(args)
    train_LSTM(args)


if __name__ == "__main__":
    train()
