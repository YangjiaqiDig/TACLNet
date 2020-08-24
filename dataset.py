from random import randint

from torch.utils.data.dataset import Dataset
import torch
from pre_processing import *

def chunkCrop(images):
    newData = []
    for image in images:
        test = torch.chunk(image, 2, dim=-2)
        for x in test:
            test2 = torch.chunk(x, 2, dim=-1)
            newData.append(test2[0])
            newData.append(test2[1])
    newData = torch.stack(newData, 0)

    # print(newData.shape)
    return newData

def imageTransform(img_as_np, msk_as_np):
    # Noise Determine {0: Gaussian_noise, 1: uniform_noise
    if randint(0, 1):
        gaus_sd, gaus_mean = randint(0, 20), 0
        img_as_np = add_gaussian_noise(img_as_np, gaus_mean, gaus_sd)
    else:
        l_bound, u_bound = randint(-20, 0), randint(0, 20)
        img_as_np = add_uniform_noise(img_as_np, l_bound, u_bound)

    # Change brightness
    pix_add = randint(-20, 20)
    img_as_np = change_brightness(img_as_np, pix_add)

    return img_as_np, msk_as_np


def DataTrain(train_path, label_path):
    image_arr = Image.open(str(train_path))
    mask_arr = Image.open(str(label_path))

    img_as_np = []
    for i, img_as_img in enumerate(ImageSequence.Iterator(image_arr)):
        singleImage_as_np = np.asarray(img_as_img)
        img_as_np.append(singleImage_as_np)

    msk_as_np = []
    for j, label_as_img in enumerate(ImageSequence.Iterator(mask_arr)):
        singleLabel_as_np = np.asarray(label_as_img)
        msk_as_np.append(singleLabel_as_np)

    img_as_np, msk_as_np = np.stack(img_as_np, axis=0), np.stack(msk_as_np, axis=0)

    # TODO: consider better augment
    # img_as_np, msk_as_np = imageTransform(img_as_np, msk_as_np)
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

    if args.crop:
        seq_train = chunkCrop(seq_train)
        seq_label = chunkCrop(seq_label)
    # print(seq_label.shape, seq_train.shape)

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
