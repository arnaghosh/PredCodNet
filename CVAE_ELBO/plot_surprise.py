import numpy as np
from tqdm import tqdm
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from MNISTDataset import MNISTDataset_pairs
from model import *

import matplotlib.pyplot as plt
from matplotlib import cm

max_translation = 6
max_rotation = 45

def surprise(mean1, sigma1, mean2, sigma2):
	KLD12 = -0.5*torch.sum(1+torch.log(1e-8+sigma1.pow(2))-torch.log(1e-8+sigma2.pow(2))-((mean1-mean2).pow(2)+sigma1.pow(2))/(sigma2.pow(2)+1e-8),axis=1)
	KLD21 = -0.5*torch.sum(1+torch.log(1e-8+sigma2.pow(2))-torch.log(1e-8+sigma1.pow(2))-((mean2-mean1).pow(2)+sigma2.pow(2))/(sigma1.pow(2)+1e-8),axis=1)
	return (KLD12+KLD21)

def surprise_mean_adjusted(mean1, sigma1, mean2, sigma2):
	KLD12 = -0.5*torch.sum(1+torch.log(1e-8+sigma1.pow(2))-torch.log(1e-8+sigma2.pow(2))-(sigma1.pow(2))/(sigma2.pow(2)+1e-8),axis=1)
	KLD21 = -0.5*torch.sum(1+torch.log(1e-8+sigma2.pow(2))-torch.log(1e-8+sigma1.pow(2))-(sigma2.pow(2))/(sigma1.pow(2)+1e-8),axis=1)
	return (KLD12+KLD21)

def get_surprise(network, data, rot, trans, data_tfo):
	if type(network).__name__ == 'CVAE_MLP':
		assert network.enc_condition==1, "Cannot estimate surprise when condition not used while encoding!!"
		rot_zero = rot.new_zeros(rot.size())
		trans_zero = trans.new_zeros(trans.size())
		with torch.no_grad():
			x_hat, mu1, sigma1, z = network(data_tfo,rot_zero,trans_zero,data_tfo)
			x_hat, mu2, sigma2, z = network(data,rot,trans,data_tfo)

	elif type(network).__name__ == 'CVAE_MLP_latent_tfo':
		rot_zero = rot.new_zeros(rot.size())
		trans_zero = trans.new_zeros(trans.size())
		with torch.no_grad():
			x_hat, mu1, sigma1, z, z_hat = network(data_tfo,rot_zero,trans_zero,data_tfo)
			x_hat, mu2, sigma2, z, z_hat = network(data,rot,trans,data_tfo)
			# z = mu + sigma*torch.randn_like(mu)
			rot = rot.view(-1,1)/max_rotation
			trans = trans.view(-1,1)/max_translation
			C = torch.cat((rot,trans),dim=-1)
			# W_z = torch.diag_embed(F.softplus(network.context_layer_latent(C))).to(mu2.device)
			W_z = torch.diag_embed(1.+F.relu(network.context_layer_latent(C))).to(z.device)
			mu2_unsqueezed = mu2.unsqueeze(2)
			sigma2_unsqueezed = sigma2.unsqueeze(2)
			# breakpoint()
			mu2 = torch.bmm(W_z,mu2_unsqueezed).squeeze() + network.context_layer(C)
			sigma2 = torch.bmm(W_z,sigma2_unsqueezed).squeeze()

	elif type(network).__name__ == 'CVAE_MLP_latent_tfo_modif':
		rot_zero = rot.new_zeros(rot.size())
		trans_zero = trans.new_zeros(trans.size())
		with torch.no_grad():
			x_hat, x_recon, mu1, sigma1, z, z_hat = network(data_tfo,rot_zero,trans_zero,data_tfo)
			x_hat, x_recon, mu2, sigma2, z, z_hat = network(data,rot,trans,data_tfo)
			# z = mu + sigma*torch.randn_like(mu)
			rot = rot.view(-1,1)/max_rotation
			trans = trans.view(-1,1)/max_translation
			C = torch.cat((rot,trans),dim=-1)
			W_z = torch.diag_embed(1.+F.relu(network.context_layer_latent(C))).to(mu2.device)
			mu2_unsqueezed = mu2.unsqueeze(2)
			sigma2_unsqueezed = sigma2.unsqueeze(2)
			mu2 = torch.bmm(W_z,mu2_unsqueezed).squeeze() + network.context_layer(C)
			sigma2 = torch.bmm(W_z,sigma2_unsqueezed).squeeze()

	return surprise(mu1,sigma1,mu2,sigma2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# train_loader = torch.utils.data.DataLoader(MNISTDataset_pairs('../Data Processing/processed_train_mnist.npz',transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(MNISTDataset_pairs('../Data Processing/processed_test_mnist.npz',transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=500, shuffle=True)

# enc_condition = 1
# dec_condition = 0
# network = CVAE_MLP(28*28,[250,100],50,enc_condition=enc_condition,dec_condition=dec_condition).to(device)
# model_path = 'CVAE{}{}_MLP_model.pt'.format(enc_condition,dec_condition)

# network = CVAE_MLP_latent_tfo(28*28,[250,100],50).to(device)
# model_path = 'CVAE_Ztfo1_MLP_model.pt'

network = CVAE_MLP_latent_tfo_modif(28*28,[250,100],50).to(device)
model_path = 'CVAE_Ztfo2_MLP_model++.pt'

network.load_state_dict(torch.load(os.path.join('Results',model_path)))
network.eval()

rot_tensor = torch.Tensor(0)
trans_tensor = torch.Tensor(0)
surp_tensor = torch.Tensor(0)
with torch.no_grad():
	for batch_idx, (data,rot,trans,data_tfo,target) in tqdm(enumerate(test_loader)):
		data = data.to(device)
		rot = torch.Tensor(rot).to(device)
		trans = torch.Tensor(trans).to(device)
		data_tfo = data_tfo.to(device)
		surp = get_surprise(network,data,rot,trans,data_tfo)
		# breakpoint()
		rot_tensor = torch.cat((rot_tensor,rot.cpu()),dim=0)
		trans_tensor = torch.cat((trans_tensor,trans.cpu()),dim=0)
		surp_tensor = torch.cat((surp_tensor,surp.cpu()),dim=0)
rot_tensor = rot_tensor.numpy()
trans_tensor = trans_tensor.numpy()
surp_tensor = surp_tensor.numpy()

plt.figure()
plt.scatter(rot_tensor,trans_tensor,s=5,c=surp_tensor,cmap=cm.plasma,alpha=0.8)
plt.colorbar()
plt.xlabel('Rotation',fontsize=12)
plt.ylabel('Translation',fontsize=12)
plt.title('Surprise (color coded) vs rotation and translation',fontsize=14)
# plt.savefig(model_path[:-3]+'_scatter_surp_meanadjust.png',bbox_inches='tight')
plt.savefig(model_path[:-3]+'_scatter_surp.png',bbox_inches='tight')
plt.figure()
plt.scatter(rot_tensor,surp_tensor,s=5,alpha=0.8)
rsf = np.polyfit(rot_tensor,surp_tensor,2)
rsp = np.poly1d(rsf)
plt.plot(np.sort(rot_tensor),rsp(np.sort(rot_tensor)),'r--',label='Quadratic trend')
plt.xlabel('Rotation',fontsize=12)
plt.ylabel('Surprise',fontsize=12)
plt.yscale('log')
plt.title('Variation of surprise with rotation',fontsize=14)
plt.legend()
# plt.savefig(model_path[:-3]+'_scatter_surp_rot_meanadjust.png',bbox_inches='tight')
plt.savefig(model_path[:-3]+'_scatter_surp_rot.png',bbox_inches='tight')
plt.figure()
plt.scatter(trans_tensor,surp_tensor,s=5,alpha=0.8)
tsf = np.polyfit(trans_tensor,surp_tensor,2)
tsp = np.poly1d(tsf)
plt.plot(np.sort(trans_tensor),tsp(np.sort(trans_tensor)),'r--',label='Quadratic trend')
plt.xlabel('Translation',fontsize=12)
plt.ylabel('Surprise',fontsize=12)
plt.yscale('log')
plt.title('Variation of surprise with translation',fontsize=14)
plt.legend()
# plt.savefig(model_path[:-3]+'_scatter_surp_trans_meanadjust.png',bbox_inches='tight')
plt.savefig(model_path[:-3]+'_scatter_surp_trans.png',bbox_inches='tight')