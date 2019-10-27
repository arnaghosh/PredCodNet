import numpy as np
import idx2numpy
from tqdm import tqdm
from scipy import ndimage

mnist_train = '/network/data1/mnist/train-images-idx3-ubyte'
mnist_train_labels = '/network/data1/mnist/train-labels-idx1-ubyte'

mnist_test = '/network/data1/mnist/t10k-images-idx3-ubyte'
mnist_test_labels = '/network/data1/mnist/t10k-labels-idx1-ubyte'

train_imgs = idx2numpy.convert_from_file(mnist_train)
train_labels = idx2numpy.convert_from_file(mnist_train_labels)

print(train_imgs.shape)
print(train_labels.shape)

max_translation = 6
max_rotation = 45

tfo_dataset = []
translation_arr = np.random.uniform(-max_translation,max_translation,len(train_imgs))
rotation_arr = np.random.uniform(-max_rotation,max_rotation,len(train_imgs))
for idx in tqdm(range(len(train_imgs))):
	img = train_imgs[idx]
	rotation = rotation_arr[idx]
	translation = translation_arr[idx]

	img_tfo = ndimage.rotate(img,rotation,reshape=False)
	img_tfo = ndimage.shift(img_tfo,[0,translation],c_val=0)
	tfo_dataset.append((img,rotation,translation,img_tfo))
