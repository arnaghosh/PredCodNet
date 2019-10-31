import numpy as np
from tqdm import tqdm
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from MNISTDataset import MNISTDataset
from CNN_model import CNN
if torch.cuda.is_available():
	gpu = True
else:
	gpu = False

fractions_to_try = [0.005,0.01,0.02,0.05,0.1,0.2,0.5,1]
# fractions_to_try = [0.01,0.02]

for f in fractions_to_try:
	train_loader = torch.utils.data.DataLoader(MNISTDataset('../Data Processing/processed_train_mnist.npz',transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))]),fraction=f), batch_size=128, shuffle=True)
	test_loader = torch.utils.data.DataLoader(MNISTDataset('../Data Processing/processed_test_mnist.npz',transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))])), batch_size=500, shuffle=True)

	network = CNN()
	optimizer = optim.Adam(network.parameters(), lr=0.001)
	if gpu:
		network = network.cuda()

	n_epochs = 21
	log_interval = 200
	test_interval = 3
	train_losses = []
	train_counter = []
	test_losses = []
	test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

	def train(epoch):
		network.train()
		for batch_idx, (data,target) in enumerate(train_loader):
			target = target.type(torch.LongTensor)
			optimizer.zero_grad()
			if gpu:
				data = data.cuda()
				target = target.cuda()
			output = network(data)
			loss = F.nll_loss(output, target)
			loss.backward()
			optimizer.step()
			if batch_idx % log_interval == 0:
				tqdm.write("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
				train_losses.append(loss.cpu().item())
				train_counter.append((batch_idx*128)+((epoch-1)*len(train_loader.dataset)))

	def test():
		network.eval()
		test_loss = 0
		correct = 0
		with torch.no_grad():
			for data, target in test_loader:
				target = target.type(torch.LongTensor)
				if gpu:
					data = data.cuda()
					target = target.cuda()
				output = network(data)
				test_loss += F.nll_loss(output, target, reduction='sum').cpu().item()
				pred = output.data.max(1, keepdim=True)[1]
				correct += pred.eq(target.data.view_as(pred)).sum()
			test_loss /= len(test_loader.dataset)
			test_losses.append(test_loss)
			tqdm.write("Test set: Avg loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))
		return 100.*correct/len(test_loader.dataset)

	best_acc = 0
	best_acc_epoch = 0
	for epoch in tqdm(range(1,n_epochs+1)):
		train(epoch)
		if epoch%test_interval == 0:
			accuracy = test()
			if accuracy>=best_acc:
				best_acc = accuracy
				best_acc_epoch = epoch

	print("For {}% of labels ({}) used, best accuracy achieved is {:.2f}% after {} epochs\n=============================================================\n".format(100*f,len(train_loader.dataset),best_acc,best_acc_epoch))