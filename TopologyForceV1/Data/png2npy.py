import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

filename = 'result1'

img = Image.open( filename + '.png' )
data = np.array( img, dtype='uint8' )/255

np.save( filename + '.npy', data)