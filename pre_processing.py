import numpy as np


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
