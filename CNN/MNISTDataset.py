import numpy as np 
import os, sys
from tqdm import tqdm
import torch

class MNISTDataset(torch.utils.data.Dataset):
	def __init__(self,data_path,transform=None,fraction=1.0):
		'''
		dataset: numpy array of images/stimuli
		transform: transform to be applied to image
		'''
		self.data_file = np.load(data_path,allow_pickle=True)
		self.data = self.data_file['data']
		if fraction<1.0:
			shuffled_indices = np.random.permutation(len(self.data))
			chosen_indices = shuffled_indices[:int(fraction*len(self.data))]
			self.labels = self.data_file['labels'][chosen_indices]
			self.images = self.data[:,0][chosen_indices]	# np array object, index 0 contains original images
		else:
			self.labels = self.data_file['labels']
			self.images = self.data[:,0]	# np array object, index 0 contains original images

		self.len_dataset = len(self.images)
		self.transform = transform

	def __len__(self):
		return self.len_dataset

	def __getitem__(self,index):
		img = self.images[index]
		label = self.labels[index]
		if self.transform:
			img = self.transform(img)
		return (img,label)

class MNISTDataset_processed(torch.utils.data.Dataset):
	def __init__(self,data_path,transform=None,fraction=1.0):
		'''
		dataset: numpy array of images/stimuli
		transform: transform to be applied to image
		'''
		self.data_file = np.load(data_path,allow_pickle=True)
		self.data = self.data_file['data']
		if fraction<1.0:
			shuffled_indices = np.random.permutation(len(self.data))
			chosen_indices = shuffled_indices[:int(fraction*len(self.data))]
			self.labels = self.data_file['labels'][chosen_indices]
			self.images = self.data[:,3][chosen_indices]	# np array object, index 3 contains transformed images
		else:
			self.labels = self.data_file['labels']
			self.images = self.data[:,3]	# np array object, index 3 contains transformed images

		self.len_dataset = len(self.images)
		self.transform = transform

	def __len__(self):
		return self.len_dataset

	def __getitem__(self,index):
		img = self.images[index]
		label = self.labels[index]
		if self.transform:
			img = self.transform(img)
		return (img,label)

class MNISTDataset_pairs(torch.utils.data.Dataset):
	def __init__(self,data_path,transform=None):
		'''
		dataset: numpy array of images/stimuli
		transform: transform to be applied to image
		'''
		self.data_file = np.load(data_path,allow_pickle=True)
		self.data = self.data_file['data']
		self.labels = self.data_file['labels']
		self.images = self.data[:,0]	# np array object
		self.images_tfo = self.data[:,3]	# np array object
		self.rotations = self.data[:,1]
		self.translations = self.data[:,2]
		self.len_dataset = len(self.images)
		self.transform = transform

	def __len__(self):
		return self.len_dataset

	def __getitem__(self,index):
		img = self.images[index]
		img_tfo = self.images_tfo[index]
		rotation = self.rotations[index]
		translation = self.translations[index]
		label = self.labels[index]
		if self.transform:
			img = self.transform(img)
			img_tfo = self.transform(img_tfo)
		return (img,rotation,translation,img_tfo,label)