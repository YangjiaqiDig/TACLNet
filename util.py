import logging
import torch
import os
from dataset import *
from PIL import Image, ImageSequence
import numpy as np

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
    validSep_beg = int(len(train[0]) / 5 * (args.valid_round - 1))
    validSep_end = int(len(train[0]) / 5 * args.valid_round)

    trainData = (torch.cat([train[0][:validSep_beg], train[0][validSep_end:]], dim=0),
                 torch.cat([train[1][:validSep_beg], train[1][validSep_end:]], dim=0))
    validData = (train[0][validSep_beg:validSep_end],
                 train[1][validSep_beg:validSep_end])
    trainDataSet = DataLoaderForUnet(trainData)
    validDataSet = DataLoaderForUnet(validData)
    logger.info("trainDataSet size %s", len(trainData))

    return (trainDataSet, validDataSet)


def accuracy_check(labels, pred_class):
    ims = [labels, pred_class]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)

    return accuracy / len(np_ims[0].flatten())


def accuracy_for_batch(labels, pred_class, args):
    total_acc = 0
    batch_size = labels.size()[0]
    for index in range(batch_size):
        total_acc += accuracy_check(labels, pred_class)
    return total_acc / batch_size
