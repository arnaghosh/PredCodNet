import numpy as np
import idx2numpy
from tqdm import tqdm
from scipy import ndimage

mnist_train = '/network/data1/mnist/train-images-idx3-ubyte'
mnist_train_labels = '/network/data1/mnist/train-labels-idx1-ubyte'

train_imgs = idx2numpy.convert_from_file(mnist_train)
train_labels = idx2numpy.convert_from_file(mnist_train_labels)

print(train_imgs.shape)
print(train_labels)

