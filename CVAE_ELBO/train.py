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

def VAE_loss(x,x_hat,mean,sigma):
	x_hat = torch.clamp(x_hat,1e-9,1-1e-9)
	BCE_loss = F.binary_cross_entropy(x_hat.view(-1,28*28),x.view(-1,28*28),reduction='sum')
	KLD_loss = -0.5*torch.sum(1+torch.log(1e-8+sigma.pow(2))-mean.pow(2)-sigma.pow(2))

	return (BCE_loss+KLD_loss)/x.size(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader = torch.utils.data.DataLoader(MNISTDataset_pairs('../Data Processing/processed_train_mnist.npz',transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(MNISTDataset_pairs('../Data Processing/processed_test_mnist.npz',transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=500, shuffle=True)
# max_translation = 6
# max_rotation = 45

enc_condition = 0
dec_condition = 1
network = CVAE_MLP(28*28,[250,100],50,enc_condition=enc_condition,dec_condition=dec_condition).to(device)
learning_rate = 0.002
learning_rate_decay = 0.001
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

n_epochs = 50
log_interval = 200
test_interval = 5
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
	network.train()
	if learning_rate_decay is not None:
		for param_group in optimizer.param_groups:
			param_group['lr'] = (learning_rate)/(1+epoch*learning_rate_decay)
	for batch_idx, (data,rot,trans,data_tfo,target) in enumerate(train_loader):
		optimizer.zero_grad()
		data = data.to(device)
		rot = torch.Tensor(rot).to(device)
		trans = torch.Tensor(trans).to(device)
		data_tfo = data_tfo.to(device)
		recon, mu, sigma, latent_z = network(data,rot,trans,data_tfo)
		loss = VAE_loss(data_tfo,recon,mu,sigma)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
				tqdm.write("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
				train_losses.append(loss.cpu().item())
				train_counter.append((batch_idx*128)+((epoch-1)*len(train_loader.dataset)))

def test():
	network.eval()
	test_loss = 0
	test_cntr = 0
	with torch.no_grad():
		for batch_idx, (data,rot,trans,data_tfo,target) in enumerate(test_loader):
			data = data.to(device)
			rot = torch.Tensor(rot).to(device)
			trans = torch.Tensor(trans).to(device)
			data_tfo = data_tfo.to(device)
			recon, mu, sigma, latent_z = network(data,rot,trans,data_tfo)
			test_loss += VAE_loss(data_tfo,recon,mu,sigma).cpu().item()
			test_cntr +=1
	test_loss/=test_cntr
	tqdm.write("Test set: Avg loss: {:.4f}\n".format(test_loss))
	test_losses.append(test_loss)
	return test_loss

def plot_results():
	network.load_state_dict(torch.load(os.path.join(os.environ['SLURM_TMPDIR'],'CVAE{}{}_MLP_model.pt'.format(enc_condition,dec_condition))))
	network.eval()
	# choose 5 random train images
	rand_indices = np.random.randint(0,len(train_loader.dataset)+1,size=(5,))
	train_image_tensor = torch.Tensor(0)
	train_rot_tensor = torch.Tensor(0)
	train_trans_tensor = torch.Tensor(0)
	train_image_tfo_tensor = torch.Tensor(0)
	for rand_i in rand_indices:
		data,rot,trans,data_tfo,label = train_loader.dataset[rand_i]
		train_image_tensor = torch.cat((train_image_tensor,data),dim=0)
		train_rot_tensor = torch.cat((train_rot_tensor,torch.Tensor([rot])),dim=0)
		train_trans_tensor = torch.cat((train_trans_tensor,torch.Tensor([trans])),dim=0)
		train_image_tfo_tensor = torch.cat((train_image_tfo_tensor,data_tfo),dim=0)
	with torch.no_grad():
		train_image_tensor = train_image_tensor.to(device)
		train_rot_tensor = (train_rot_tensor).to(device)
		train_trans_tensor = (train_trans_tensor).to(device)
		train_image_tfo_tensor = train_image_tfo_tensor.to(device)
		recon_train_images, mu,sigma,latent = network(train_image_tensor,train_rot_tensor,train_trans_tensor,train_image_tfo_tensor)
		recon_train_images = recon_train_images.view(-1,28,28).cpu().numpy()
	train_image_tensor = train_image_tensor.cpu().numpy()
	train_image_tfo_tensor = train_image_tfo_tensor.cpu().numpy()
	# choose 5 random test images
	rand_indices = np.random.randint(0,len(test_loader.dataset)+1,size=(5,))
	test_image_tensor = torch.Tensor(0)
	test_rot_tensor = torch.Tensor(0)
	test_trans_tensor = torch.Tensor(0)
	test_image_tfo_tensor = torch.Tensor(0)
	for rand_i in rand_indices:
		data,rot,trans,data_tfo,label = train_loader.dataset[rand_i]
		test_image_tensor = torch.cat((test_image_tensor,data),dim=0)
		test_rot_tensor = torch.cat((test_rot_tensor,torch.Tensor([rot])),dim=0)
		test_trans_tensor = torch.cat((test_trans_tensor,torch.Tensor([trans])),dim=0)
		test_image_tfo_tensor = torch.cat((test_image_tfo_tensor,data_tfo),dim=0)
	with torch.no_grad():
		test_image_tensor = test_image_tensor.to(device)
		test_rot_tensor = (test_rot_tensor).to(device)
		test_trans_tensor = (test_trans_tensor).to(device)
		test_image_tfo_tensor = test_image_tfo_tensor.to(device)
		recon_test_images, mu,sigma,latent = network(test_image_tensor,test_rot_tensor,test_trans_tensor,test_image_tfo_tensor)
		recon_test_images = recon_test_images.view(-1,28,28).cpu().numpy()
	test_image_tensor = test_image_tensor.cpu().numpy()
	test_image_tfo_tensor = test_image_tfo_tensor.cpu().numpy()
	import matplotlib.pyplot as plt
	plt.figure()
	for i in range(5):
		plt.subplot(5,6,6*i+1); plt.imshow(train_image_tensor[i])
		plt.subplot(5,6,6*i+2); plt.imshow(train_image_tfo_tensor[i])
		plt.subplot(5,6,6*i+3); plt.imshow(recon_train_images[i])
		plt.subplot(5,6,6*i+4); plt.imshow(test_image_tensor[i])
		plt.subplot(5,6,6*i+5); plt.imshow(test_image_tfo_tensor[i])
		plt.subplot(5,6,6*i+6); plt.imshow(recon_test_images[i])

	plt.savefig(os.path.join(os.environ['SLURM_TMPDIR'],'test_CVAE{}{}_recon.png'.format(enc_condition,dec_condition)))

	z_vec = torch.Tensor(0)
	label_vec = torch.Tensor(0).type(torch.LongTensor)
	with torch.no_grad():
		for batch_idx, (data, rot, trans, data_tfo, target) in enumerate(test_loader):
			data = data.to(device)
			rot = (rot).to(device)
			trans = (trans).to(device)
			data_tfo = data_tfo.to(device)
			recon, mu, sigma, latent_z = network(data,rot,trans,data_tfo)
			z_vec = torch.cat((z_vec,latent_z.cpu()),dim=0)
			label_vec = torch.cat((label_vec,target.type(torch.LongTensor)),dim=0)
	from sklearn.decomposition import PCA
	pca = PCA(n_components=2)
	z_pca = pca.fit_transform(z_vec)
	plt.figure()
	plt.scatter(z_pca[:,0],z_pca[:,1],c=label_vec)
	plt.xlabel('Principal Component #1')
	plt.ylabel('Principal Component #2')
	plt.savefig(os.path.join(os.environ['SLURM_TMPDIR'],'test_CVAE{}{}_latent.png'.format(enc_condition,dec_condition)))
	plt.figure()
	plt.hist2d(z_pca[:,0],z_pca[:,1],(50,50),cmap=plt.cm.plasma)
	plt.colorbar()
	plt.xlabel('Principal Component #1')
	plt.ylabel('Principal Component #2')
	plt.savefig(os.path.join(os.environ['SLURM_TMPDIR'],'test_CVAE{}{}_latent_dist.png'.format(enc_condition,dec_condition)))


best_test_loss = 10000
for epoch in tqdm(range(1,n_epochs+1)):
	train(epoch)
	if epoch%test_interval == 0:
		test_loss = test()
		if test_loss <=best_test_loss:
			best_test_loss = test_loss
			tqdm.write("Saving model at epoch# {}\n".format(epoch))
			torch.save(network.cpu().state_dict(),os.path.join(os.environ['SLURM_TMPDIR'],'CVAE{}{}_MLP_model.pt'.format(enc_condition,dec_condition)))
			network = network.to(device)
plot_results()