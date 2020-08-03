import numpy as np
from PIL import Image

matrix = np.load('result1.npy')
matrix = matrix * 255
im = Image.fromarray(matrix)
im.show()