import torch
import numpy as np
from pre_processing import *
from random import randint
from PIL import Image, ImageSequence
from torch.utils.data.dataset import Dataset

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

    img_as_np, = normalization2(img_as_np.astype(float), max=1, min=0)
    msk_as_np = msk_as_np / 255

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

    img_as_np, msk_as_np = imageTransform(img_as_np, msk_as_np)

    # img2 = Image.fromarray(msk_as_np)
    # img2.show()

    img_as_tensor = torch.from_numpy(img_as_np).float()
    msk_as_tensor = torch.from_numpy(msk_as_np).long()

    return (img_as_tensor, msk_as_tensor)


class DataLoaderForUnet(Dataset):
    def __init__(self, dataset):
        self.image = dataset[0]
        self.mask = dataset[1]
        self.data_len = len(self.image)

    def __getitem__(self, index):
        return (self.image[index], self.mask[index])

    def __len__(self):
        return self.data_len
