import numpy as np
import numpy as np
import torch
from PIL import Image, ImageSequence
import math
import torch.nn as nn
torch.set_printoptions(edgeitems=60)
np.set_printoptions(edgeitems=60)


def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image


def get_gaussian_kernel(kernel_size=3, sigma=1, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, padding=1, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

def downloadImages(img):
    img = img.squeeze(0).squeeze(0).numpy()
    img[img > 1] = 1
    img[img < 0] = 0
    # print(img)
    img = img * 255
    img = img.astype(np.uint8)
    # print(img)
    img = Image.fromarray(img)
    img.save('downcp16.png')
    # image = (image.numpy() * 255).astype(np.uint8)
    # image = Image.fromarray(image)
    # image.save('downcp16-2.png')

def addGaussianFilter(image, args):
    filter = get_gaussian_kernel().to(args.device)
    img = filter(image.unsqueeze(0).unsqueeze(0))

    return img

def add_gaussian_noise(image, mean=0, std=1):
    """
    Args:
        image : numpy array of image
        mean : pixel mean of image
        standard deviation : pixel standard deviation of image
    Return :
        image : numpy array of image with gaussian noise added
    """
    for i, img in enumerate(image):
        gaus_noise = np.random.normal(mean, std, img.shape)
        img = img.astype("int16")
        image[i] = img + gaus_noise
        image = ceil_floor_image(image)
    return image


def add_uniform_noise(image, low=-10, high=10):
    """
    Args:
        image : numpy array of image
        low : lower boundary of output interval
        high : upper boundary of output interval
    Return :
        image : numpy array of image with uniform noise added
    """
    for i, img in enumerate(image):
        uni_noise = np.random.uniform(low, high, img.shape)
        img = img.astype("int16")
        image[i] = img + uni_noise
        image = ceil_floor_image(image)
    return image


def change_brightness(image, value):
    for i, img in enumerate(image):
        img = img.astype("int16")
        img = img + value
        image[i] = ceil_floor_image(img)
    return image


def normalization(image, mean, std):
    image = image / 255
    image = (image - mean) / std
    return image


def normalization2(image, max, min):
    for i, img in enumerate(image):
        image[i] = (img - np.min(img)) * (max - min) / (np.max(img) - np.min(img)) + min
    return image
