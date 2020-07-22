import logging
import torch
import os
from dataset import *

logger = logging.getLogger(__file__)

def get_dataset(args):
    train_path = args.dataset_path_train
    label_path = args.dataset_path_label
    if args.dataset_cache and os.path.isfile(args.dataset_cache):
        logger.info("Load enhanced dataset before DataLoader from cache at %s", args.dataset_cache)
        train = torch.load(args.dataset_cache)

    else:
        logger.info("Start Prepare enhanced dataset before DataLoader %s", train_path)
        train = DataTrain(train_path, label_path)
        torch.save(train, args.dataset_cache)
    validSep_beg = int(len(train[0]) / 5 * (args.valid_round-1))
    validSep_end = int(len(train[0]) / 5 * args.valid_round)

    trainData = (torch.cat([train[0][:validSep_beg],train[0][validSep_end:]], dim=0),
                 torch.cat([train[1][:validSep_beg], train[1][validSep_end:]], dim=0))
    validData = (train[0][validSep_beg:validSep_end],
                 train[1][validSep_beg:validSep_end])
    trainDataSet = DataLoaderForUnet(trainData)
    validDataSet = DataLoaderForUnet(validData)
    logger.info("trainDataSet size %s", len(trainData))

    return (trainDataSet, validDataSet)