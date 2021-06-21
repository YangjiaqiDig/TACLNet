from random import randint
import random
from os import listdir
from os.path import isfile, join
from torch.utils.data.dataset import Dataset
import torch
from pre_processing import *
from PIL import Image
from PIL import ImageOps
import PIL
from topo import *
import torchvision
from torchvision import transforms

def chunkCrop(images):
    newData = []
    for image in images:
        test = torch.chunk(image, 2, dim=-2)
        for x in test:
            test2 = torch.chunk(x, 2, dim=-1)
            newData.append(test2[0])
            newData.append(test2[1])

    newData = torch.stack(newData, 0)

    # print(newData.shape)  [batch, 3, 1, size, size]

    return newData #imagesß

def imageTransform(img_as_np):
    # Noise Determine {0: Gaussian_noise, 1: uniform_noise
    if randint(0, 1):
        gaus_sd, gaus_mean = randint(0, 10), 0
        img_as_np = add_gaussian_noise(img_as_np, gaus_mean, gaus_sd)
    else:
        l_bound, u_bound = randint(-10, 0), randint(0, 10)
        img_as_np = add_uniform_noise(img_as_np, l_bound, u_bound)

    # Change brightness
    pix_add = randint(-10, 10)
    img_as_np = change_brightness(img_as_np, pix_add)

    return img_as_np

def HepaticDataSet(train_path, label_path, args):
    if args.dataset_cache and os.path.isfile(args.dataset_cache):
        logging.info("Load enhanced dataset before DataLoader from cache at %s", args.dataset_cache)
        (img_as_tensor, msk_as_tensor, folderNames) = torch.load(args.dataset_cache)

    else:
        trainSamples, labelSamples = [f for f in listdir(train_path)], [f for f in listdir(label_path)]
        trainSamples = sorted(trainSamples, key=lambda x: int(x.replace('sample', '')))[:100]
        labelSamples = sorted(labelSamples, key=lambda x: int(x.replace('sample', '')))[:100]
        allSamplesSliceTrain, allSamplesSliceLabel, folderNames = [], [], []
        for i, train in enumerate(trainSamples):
            key = lambda x: int(x.replace('image', '').replace('.png', ''))
            filesInTrainSample = sorted([f for f in listdir(train_path+train) if isfile(join(train_path+train, f)) and 'image' in f and '_' not in f], key=key)
            filesInLabelSample = sorted([f for f in listdir(label_path+train) if isfile(join(label_path+train, f)) and 'image' in f and '_' not in f], key=key)

            trainSampleGroup, labelSampleGroup = [], []
            for each, img in enumerate(filesInTrainSample):
                imgPathTrain = train_path + train + '/' + img
                imgPathLabel = label_path + train + '/' + img

                imageTrain, imageLabel = Image.open(imgPathTrain), Image.open(imgPathLabel)
                imageTrain.load()
                imageLabel.load()
                imageTrain = ImageOps.equalize(imageTrain, mask=None)
                imageTrainArray = np.asarray(imageTrain, dtype="int32").astype(np.uint8)
                imageLabelArray = np.asarray(imageLabel, dtype="int32").astype(np.uint8) # 383, 286
                trainSampleGroup.append(imageTrainArray)
                labelSampleGroup.append(imageLabelArray)
            trainSampleGroup, labelSampleGroup = np.stack(trainSampleGroup, axis=0), np.stack(labelSampleGroup, axis=0)
            trainSampleSlice, labelSampleSlice = [], []
            for s in range(trainSampleGroup.shape[0] - 2):
                trainSampleSlice.append(trainSampleGroup[s: s+3])
                labelSampleSlice.append(labelSampleGroup[s: s+3])
            trainSampleSlice, labelSampleSlice = np.stack(trainSampleSlice, axis=0), np.stack(labelSampleSlice, axis=0)
            allSamplesSliceTrain.append(trainSampleSlice)
            allSamplesSliceLabel.append(labelSampleSlice)
            folderNames.append(train)

        img_as_np = [normalization2(f.astype(float), max=1, min=0) for f in allSamplesSliceTrain]
        msk_as_np = [f / 255 for f in allSamplesSliceLabel]

        img_as_tensor = [torch.from_numpy(f).float() for f in img_as_np]
        msk_as_tensor = [torch.from_numpy(f).long() for f in msk_as_np]

        logging.info("Start saving prepared dataset before DataLoader %s", train_path)
        torch.save((img_as_tensor, msk_as_tensor, folderNames), args.dataset_cache)

    return img_as_tensor, msk_as_tensor, folderNames

def DataTrain(train_path, label_path, args):
    image_arr = Image.open(str(train_path))
    mask_arr = Image.open(str(label_path))

    img_as_np = []
    for i, img_as_img in enumerate(ImageSequence.Iterator(image_arr)):
        img_as_img = ImageOps.equalize(img_as_img, mask=None)
        singleImage_as_np = np.asarray(img_as_img)
        img_as_np.append(singleImage_as_np)

    msk_as_np = []
    for j, label_as_img in enumerate(ImageSequence.Iterator(mask_arr)):
        singleLabel_as_np = np.asarray(label_as_img)
        msk_as_np.append(singleLabel_as_np)

    img_as_np, msk_as_np = np.stack(img_as_np, axis=0), np.stack(msk_as_np, axis=0)

    # TODO: consider better augment
    validation_split = 1 / 3
    dataset_size = len(img_as_np)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    splitIndex = split * (args.valid_round - 1)
    train_indices, val_indices = indices[:splitIndex] + indices[splitIndex + split:], indices[
                                                                                      splitIndex: splitIndex + split]
    # for indx in train_indices:
    #     img_as_np[indx] = imageTransform(np.expand_dims(img_as_np[indx], axis=0))

    img_as_np = normalization2(img_as_np.astype(float), max=1, min=0)
    msk_as_np = msk_as_np / 255

    img_as_tensor = torch.from_numpy(img_as_np).float()
    msk_as_tensor = torch.from_numpy(msk_as_np).long()

    return (img_as_tensor, msk_as_tensor)


def lstmDataTrain(train, args):
    images = train[0]
    labels = train[1]
    padding_size = int(args.step_size / 2)
    padded_images = torch.cat((images[:padding_size], images, images[-padding_size:]), dim=0)
    padded_labels = torch.cat((labels[:padding_size], labels, labels[-padding_size:]), dim=0)
    seq_train, seq_label = [], []
    for i in range(len(labels)):
        seq_train.append(padded_images[i:i + args.step_size])  # (3, 2, size, size)
        seq_label.append(padded_labels[i:i + args.step_size]) # (size, size)
    seq_train, seq_label = torch.stack(seq_train, 0), torch.stack(seq_label, 0)
    seq_train = downsampling(seq_train.squeeze(2), args, times=1).unsqueeze(2)
    seq_label = downsampling(seq_label, args, times=1)
    if args.crop:
        seq_train = chunkCrop(chunkCrop(seq_train))
        seq_label = chunkCrop(chunkCrop(seq_label))

    return seq_train, seq_label


class UNETDataSet(Dataset):
    def __init__(self, dataset):
        self.image = dataset[0]
        self.mask = dataset[1]
        self.data_len = len(self.image)

    def __getitem__(self, index):

        return (self.image[index], self.mask[index])

    def __len__(self):
        return self.data_len

class UNETDataSetVessle(Dataset):
    def __init__(self, dataset):
        self.image = dataset[0]
        self.mask = dataset[1]
        self.name = dataset[2]
        self.data_len = len(self.image)

    def __getitem__(self, index):
        return (self.image[index], self.mask[index], self.name[index])

    def __len__(self):
        return self.data_len

class LSTMDataSet(Dataset):
    def __init__(self, dataset, args):
        self.lh = dataset[0]  # (n, 2, size, size)
        self.label = dataset[1]
        self.step_size = args.step_size  # 3 (100, 3, 2, size, size) -> (001 012 123 345 .. 989999)
        self.padding_size = int(self.step_size / 2)
        self.padded_lh = torch.cat((self.lh[:self.padding_size], self.lh, self.lh[-self.padding_size:]), dim=0)
        # (102, 2, size, size)
        self.data_len = len(self.lh)

    def __getitem__(self, index):
        seq_lh = self.padded_lh[index:index + self.step_size]  # (3, 2, size, size)
        true_label = self.label[index]  # (size, size)


        return (seq_lh, true_label)

    def __len__(self):
        return self.data_len
