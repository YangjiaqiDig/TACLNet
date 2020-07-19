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
        saved_data = torch.load(args.dataset_cache)

    else:
        logger.info("Start Prepare enhanced dataset before DataLoader %s", train_path)
        train = DataTrain(train_path, label_path)
        torch.save(train, args.dataset_cache)