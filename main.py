from argparse import ArgumentParser

from lstm import *
from model import UNET
from save_history import *
from util import *
import numpy as np
import torch.nn as nn
from modules import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# logger = logging.getLogger(__file__).setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG)


def train():
    database = "CREMI"
    parser = ArgumentParser()
    parser.add_argument("--train_type", type=str, default="clstm",
                        help="unet, unet-clstm, or clstm")
    parser.add_argument("--topo_attention", type=bool, default=True,
                        help="Add topo attention loss to train")
    parser.add_argument("--dataset_path_train", type=str, default="database/{0}/train-volume.tif".format(database),
                        help="Path or url of the dataset")
    parser.add_argument("--dataset_path_label", type=str, default="database/{0}/train-labels.tif".format(database),
                        help="Path or url of the dataset")
    parser.add_argument("--dataset_cache", type=str,
                        default='dataset_cache/dataset_cache_{0}'.format(database), help="Path or url of the preprocessed dataset cache")
    parser.add_argument("--topo_dataset_cache", type=str,
                        default='dataset_cache/dataset_cache_cp_{0}'.format(database), help="Path or url of the critical points dataset cache")
    parser.add_argument("--save_folder", type=str, default="results_clstm/{0}".format(database),
                        help="Path or url of the dataset")
    # TODO: batch size enlarge, need fit the total number of input, dividable
    parser.add_argument("--train_batch_size", type=int,
                        default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int,
                        default=1, help="Batch size for validation")
    parser.add_argument("--valid_round", type=int,
                        default=2, help="validation part: 1, 2, 3")
    parser.add_argument("--lr", type=float,
                        default=0.001, help="Learning rate")
    parser.add_argument("--lr_topo", type=float,
                        default=0.0001, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--check_point", type=str, default="/model_epoch_250.pwf",
                        help="Path of the pre-trained CNN")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
    else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--topo_size", type=int, default=30, help="Crop size for topo input")

    parser.add_argument("--step_size", type=int, default=3, help="sequence length for LSTM")

    parser.add_argument("--att_loss_coef", type=float, default=0.2, help="Attention loss rate in total loss")

    args = parser.parse_args()

    if args.train_type == 'unet':
        logging.info("---------Prepare DataSet for UNET--------")
        trainDataset, validDataset = get_dataset(args)
        train_loader = torch.utils.data.DataLoader(dataset=trainDataset, num_workers=8,
                                                   batch_size=args.train_batch_size,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=validDataset, num_workers=8, batch_size=args.valid_batch_size,
                                                 shuffle=False)

        train_UNET(args, train_loader, val_loader)

    if args.train_type == 'unet-clstm':
        # UNEt -> likelihoodMap -> 3 slice (batch, 3, n, n)
        dataset = get_dataset_clstm(args) # (data, label)
        validation_split = .2
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        train_indices, val_indices = indices[:-split], indices[-split:]
        # print(train_indices, val_indices)
        train_sampler = torch.utils.data.Subset(dataset, train_indices)
        valid_sampler = torch.utils.data.Subset(dataset, val_indices)
        train_loader = torch.utils.data.DataLoader(dataset=train_sampler, num_workers=8,
                                                   batch_size=args.train_batch_size,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=valid_sampler, num_workers=8, batch_size=args.valid_batch_size,
                                                 shuffle=False)
        train_LSTM(args, train_loader, val_loader)


    if args.topo_attention:
        logging.info("---------Prepare DataSet for attention CLSTM train--------")
        trainDataset, validDataset = get_dataset_topoClstm(args)
        train_loader = torch.utils.data.DataLoader(dataset=trainDataset, num_workers=8,
                                                   batch_size=args.train_batch_size,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=validDataset, num_workers=8, batch_size=args.valid_batch_size,
                                                 shuffle=False)

        train_LSTM_TopoAttention(train_loader, val_loader, args)
        return

    if args.train_type == 'clstm':
        logging.info("---------Prepare DataSet for CLSTM--------")
        trainDataset, validDataset = get_dataset_topoClstm(args)
        train_loader = torch.utils.data.DataLoader(dataset=trainDataset, num_workers=8,
                                                   batch_size=args.train_batch_size,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=validDataset, num_workers=8, batch_size=args.valid_batch_size,
                                                 shuffle=False)

        train_LSTM_TopoAttention(train_loader, val_loader, args)

if __name__ == "__main__":
    train()
