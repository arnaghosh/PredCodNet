import numpy as np
import idx2numpy
from tqdm import tqdm
from scipy import ndimage
import os

mnist_train = '/network/data1/mnist/train-images-idx3-ubyte'
mnist_train_labels = '/network/data1/mnist/train-labels-idx1-ubyte'

mnist_test = '/network/data1/mnist/t10k-images-idx3-ubyte'
mnist_test_labels = '/network/data1/mnist/t10k-labels-idx1-ubyte'

mode = 0 	# 0 for train and 1 for test dataset

if mode==0:
	imgs = idx2numpy.convert_from_file(mnist_train)
	labels = idx2numpy.convert_from_file(mnist_train_labels)
else:
	imgs = idx2numpy.convert_from_file(mnist_test)
	labels = idx2numpy.convert_from_file(mnist_test_labels)

print(imgs.shape)
print(labels.shape)

max_translation = 6
max_rotation = 45

tfo_dataset = []
translation_arr = np.random.uniform(-max_translation,max_translation,len(imgs))
rotation_arr = np.random.uniform(-max_rotation,max_rotation,len(imgs))
for idx in tqdm(range(len(imgs))):
	img = imgs[idx]
	rotation = rotation_arr[idx]
	translation = translation_arr[idx]

	img_tfo = ndimage.rotate(img,rotation,reshape=False)
	img_tfo = ndimage.shift(img_tfo,[0,translation],cval=0)
	tfo_dataset.append((img,rotation,translation,img_tfo))

tfo_dataset = np.array(tfo_dataset)
np.savez(os.path.join(os.environ['SLURM_TMPDIR'],('processed_train_mnist' if mode==0 else 'processed_test_mnist')),data=tfo_dataset,labels=mnist_labels)