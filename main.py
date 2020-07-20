import numpy as np
import logging
from argparse import ArgumentParser
import torch
from dataset import *
from util import get_dataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__file__)


def train_UNET():
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
                        default=4, help="Batch size for validation")
    parser.add_argument("--lr", type=float,
                        default=6.25e-4, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
    else "cpu", help="Device (cuda or cpu)")

    args = parser.parse_args()

    logger.info("---------Prepare DataSet--------")
    train_loader, val_loader = get_dataset(args)

if __name__ == "__main__":
    train_UNET()