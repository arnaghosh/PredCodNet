import numpy as np
from tqdm import tqdm
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from MNISTDataset import MNISTDataset
from model import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fractions_to_try = [0.005,0.01,0.02,0.05,0.1,0.2,0.5,1]
# fractions_to_try = [0.01,0.02]

learning_rate = 0.004
learning_rate_decay = 0.002
latent_vector_size = 50
# enc_condition = 1
# dec_condition = 0
# network = CVAE_MLP(28*28,[250,100],latent_vector_size,enc_condition=enc_condition,dec_condition=dec_condition).to(device)
# model_path = 'CVAE{}{}_MLP_model.pt'.format(enc_condition,dec_condition)

# network = CVAE_MLP_latent_tfo(28*28,[250,100],latent_vector_size,norm_context=1).to(device)
# model_path = 'CVAE_Ztfo1_MLP_model++.pt'

network = CVAE_MLP_latent_tfo_modif(28*28,[250,100],latent_vector_size,norm_context=1).to(device)
model_path = 'CVAE_Ztfo2_MLP_model++.pt'


def get_latent_feat(network, data):
	if type(network).__name__ == 'CVAE_MLP':
		rot_zero = torch.zeros(data.size(0)).to(device)
		trans_zero = torch.zeros(data.size(0)).to(device)
		with torch.no_grad():
			x_hat, mu1, sigma1, z = network(data,rot_zero,trans_zero,data)
			feat = torch.cat((mu1,sigma1),dim=-1)
			# feat = z

	elif type(network).__name__ == 'CVAE_MLP_latent_tfo':
		rot_zero = torch.zeros(data.size(0)).to(device)
		trans_zero = torch.zeros(data.size(0)).to(device)
		with torch.no_grad():
			x_hat, mu1, sigma1, z, z_hat = network(data,rot_zero,trans_zero,data)
			feat = torch.cat((mu1,sigma1),dim=-1)
			# feat = z

	elif type(network).__name__ == 'CVAE_MLP_latent_tfo_modif':
		rot_zero = torch.zeros(data.size(0)).to(device)
		trans_zero = torch.zeros(data.size(0)).to(device)
		with torch.no_grad():
			x_hat, x_recon, mu1, sigma1, z, z_hat = network(data,rot_zero,trans_zero,data)
			feat = torch.cat((mu1,sigma1),dim=-1)
			
	return feat

def train(network, classifier_layer, train_loader, epoch):
	classifier_layer.train()
	if learning_rate_decay is not None:
		for param_group in optimizer.param_groups:
			param_group['lr'] = (learning_rate)/(1+epoch*learning_rate_decay)
	for batch_idx, (data,target) in enumerate(train_loader):
		target = target.type(torch.LongTensor)
		optimizer.zero_grad()
		data = data.to(device)
		target = target.to(device)
		latent_feats = get_latent_feat(network,data)
		output = F.log_softmax(classifier_layer(latent_feats), dim=1)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			tqdm.write("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
			train_losses.append(loss.cpu().item())
			train_counter.append((batch_idx*128)+((epoch-1)*len(train_loader.dataset)))

def test(network,classifier_layer,test_loader):
	classifier_layer.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			target = target.type(torch.LongTensor)
			data = data.to(device)
			target = target.to(device)
			latent_feats = get_latent_feat(network,data)
			output = F.log_softmax(classifier_layer(latent_feats), dim=1)
			test_loss += F.nll_loss(output, target, reduction='sum').cpu().item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
		test_loss /= len(test_loader.dataset)
		test_losses.append(test_loss)
		tqdm.write("Test set: Avg loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))
	return 100.*correct/len(test_loader.dataset)

for f in fractions_to_try:
	train_loader = torch.utils.data.DataLoader(MNISTDataset('../Data Processing/processed_train_mnist.npz',transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))]),fraction=f), batch_size=128, shuffle=True)
	test_loader = torch.utils.data.DataLoader(MNISTDataset('../Data Processing/processed_test_mnist.npz',transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))])), batch_size=500, shuffle=True)

	network.load_state_dict(torch.load(os.path.join('Results',model_path)))
	ClassificationLayer = nn.Linear(2*latent_vector_size,10).to(device)
	network.eval()
	optimizer = optim.Adam(ClassificationLayer.parameters(), lr=learning_rate)
	network = network.to(device)

	n_epochs = 100
	log_interval = 200
	test_interval = 5
	train_losses = []
	train_counter = []
	test_losses = []
	test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
	best_acc = 0
	best_acc_epoch = 0
	for epoch in tqdm(range(1,n_epochs+1)):
		train(network,ClassificationLayer,train_loader,epoch)
		if epoch%test_interval == 0:
			accuracy = test(network,ClassificationLayer,test_loader)
			if accuracy>=best_acc:
				best_acc = accuracy
				best_acc_epoch = epoch

	print("For {}% of labels ({}) used, best accuracy achieved is {:.2f}% after {} epochs\n=============================================================\n".format(100*f,len(train_loader.dataset),best_acc,best_acc_epoch))