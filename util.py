import os, re
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
from dataset import *
from model import UNET
from topo import *
import torch.nn.functional as F
import torchvision
from torchvision import transforms
# logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)


def load_preprocess_dataset(args):
    train_path = args.dataset_path_train
    label_path = args.dataset_path_label
    if args.dataset_cache and os.path.isfile(args.dataset_cache):
        logging.info("Load enhanced dataset before DataLoader from cache at %s", args.dataset_cache)
        train = torch.load(args.dataset_cache)

    else:
        logging.info("Start Prepare enhanced dataset before DataLoader %s", train_path)
        train = DataTrain(train_path, label_path, args)
        torch.save(train, args.dataset_cache)

    return train


def get_vesselDataset_topoClstm(args):
    trainPath, labelPath = 'database/Hepatic/train/', 'database/Hepatic/label/'
    img_as_tensor, msk_as_tensor, folderNames = HepaticDataSet(trainPath, labelPath, args)
    # print(max([f.shape[0] for f in img_as_tensor]), max([f.shape[2] for f in img_as_tensor]), max([f.shape[3] for f in img_as_tensor])) #131 383 263

    trainAll, maskAll, folderNamesAll = [], [], []
 
    for i, img in enumerate(img_as_tensor): # img (n, slices, x , x)
        # endRow = 200 if img.shape[2] > 200 else img.shape[2]
        # endCol = 200 if img.shape[3] > 200 else img.shape[3]
        # imgLimit = img[:, :, :endRow, :endCol]
        
        # labelLimit = msk_as_tensor[i][:, :, :endRow, :endCol]
        # img_batch = [
        #     F.pad(each, [0, 200 - each.size(2), 0, 200 - each.size(1)])
        #     for each in imgLimit
        # ]
        # mask_batch = [
        #     F.pad(mask, [0, 200 - mask.size(2), 0, 200 - mask.size(1)])
        #     for mask in labelLimit
        # ]
        name_batch = [
            folderNames[i] for each in img #imgLimit
        ] * 16
        # trainAll = trainAll + img_batch
        # maskAll = maskAll + mask_batch
        folderNamesAll = folderNamesAll + name_batch

    trainAll = torch.cat(img_as_tensor, 0).unsqueeze(2)
    maskAll = torch.cat(msk_as_tensor, 0)

    if args.crop:
        trainAll = chunkCrop(chunkCrop(trainAll))
        maskAll = chunkCrop(chunkCrop(maskAll))

    # print(trainAll.shape, maskAll.shape)    #[11860, 3, 200, 200]
    print('--------The shape of dataset: {0} ---------------'.format(trainAll.shape))
    sizeOfSamples = len(trainAll)
    indices = list(range(sizeOfSamples))
    split = int(np.floor(sizeOfSamples / 3))
    splitIndex = split * (args.valid_round - 1)
    train_indices, val_indices = indices[:splitIndex] + indices[splitIndex + split:], indices[splitIndex: splitIndex + split]

    dataset = UNETDataSetVessle([trainAll, maskAll, folderNamesAll])

    train_sampler = torch.utils.data.Subset(dataset, train_indices)
    valid_sampler = torch.utils.data.Subset(dataset, val_indices)

    return train_sampler, valid_sampler


def get_dataset_topoClstm(args):
    train = load_preprocess_dataset(args)
    expandChannelDimTrain = train[0].unsqueeze(1)

    seq_train, seq_label = lstmDataTrain([expandChannelDimTrain, train[1]], args)
    print(seq_train.shape, seq_label.shape)
    validation_split = 1 / 3
    dataset_size = len(seq_train)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    splitIndex = split * (args.valid_round - 1)
    train_indices, val_indices = indices[:splitIndex] + indices[splitIndex + split:], indices[
                                                                                      splitIndex: splitIndex + split]

    train_data, valid_data = seq_train[train_indices], seq_train[val_indices]
    train_label, valid_label = seq_label[train_indices], seq_label[val_indices]
    # train_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     # transforms.RandomVerticalFlip(p=0.5),
    #     transforms.RandomCrop(512),
    #     # transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    #     transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=3),
    #     transforms.ToTensor()
    # ])
    
    # images, labels = [], []
    # print(train_data.shape, train_label.shape)
    # for n in range(8):
    #     for i in range(train_data.shape[0]):
    #         seed = np.random.randint(2147483647) # make a seed with numpy generator 
    #         random.seed(seed) # apply this seed to img tranfsorms
    #         torch.manual_seed(seed) # needed for torchvision 0.7
    #         transformImage = train_transform(train_data[i].squeeze(1)).view(3, 1, 512, 512)
    #         images.append(transformImage)
    #         random.seed(seed) # apply this seed to target tranfsorms
    #         torch.manual_seed(seed) # needed for torchvision 0.7
    #         transformTarget = train_transform(train_label[i].float()).view(3, 512,512)
    #         transformTarget = transformTarget.long()
    #         labels.append(transformTarget)
    # images = torch.stack(images,dim=0)
    # labels = torch.stack(labels, dim=0)

    # # train_data = torch.cat([images, chunkCrop(train_data)], dim=0)
    # # train_label = torch.cat([labels, chunkCrop(train_label)], dim=0)
    # train_data = images
    # train_label = labels
    # print(train_data.shape, valid_data.shape)

    # valid_data = chunkCrop(valid_data)
    # valid_label = chunkCrop(valid_label)

    train_sampler = UNETDataSet([train_data, train_label])
    valid_sampler = UNETDataSet([valid_data, valid_label])

    # train_sampler = torch.utils.data.Subset(dataset, train_indices)
    # valid_sampler = torch.utils.data.Subset(dataset, val_indices)

    return train_sampler, valid_sampler


def get_dataset(args):
    train = load_preprocess_dataset(args)
    validSep_beg = int(len(train[0]) / 5 * (args.valid_round - 1))
    validSep_end = int(len(train[0]) / 5 * args.valid_round)

    trainData = (torch.cat([train[0][:validSep_beg], train[0][validSep_end:]], dim=0),
                 torch.cat([train[1][:validSep_beg], train[1][validSep_end:]], dim=0))
    validData = (train[0][validSep_beg:validSep_end],
                 train[1][validSep_beg:validSep_end])
    trainDataSet = UNETDataSet(trainData)
    validDataSet = UNETDataSet(validData)
    logging.info("TrainDataSet shape %s", trainData[0].shape)

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


if __name__ == "__main__":
    x = torch.tensor([[[1, 2]], [[3, 4]], [[3, 4]]])
    y = torch.tensor([[[1, 2]], [[3, 4]], [[3, 4]]])
    z = [x, y]
    print(x.shape, y.shape)
    q = torch.cat(z, dim=0)
    print(q.shape)
    print(q.squeeze(dim=1).shape)
