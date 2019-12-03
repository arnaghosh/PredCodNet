import numpy as np
from tqdm import tqdm
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_MLP(nn.Module):
	def __init__(self,input_dim,hidden_dims,output_dim):
		super(Encoder_MLP, self).__init__()
		self.input_dim = input_dim
		self.hidden_dims = hidden_dims
		self.output_dim = output_dim
		self.linear1 = nn.Linear(self.input_dim,self.hidden_dims[0])
		self.hidden = [nn.Linear(self.hidden_dims[i-1],self.hidden_dims[i]) for i in range(1,len(self.hidden_dims))]
		self.hidden = nn.Sequential(*self.hidden)
		self.readout = nn.Linear(self.hidden_dims[len(self.hidden_dims)-1],2*self.output_dim)

	def forward(self, X):
		X = X.view(-1,28*28)
		X = F.relu(self.linear1(X))
		for hidden_layer in self.hidden.children():
			X = F.relu(hidden_layer(X))
		X = self.readout(X)
		mean = X[:,:self.output_dim]
		# stdev = 1e-6 + F.softplus(X[:,self.output_dim:])
		stdev = 1e-6 + torch.exp(0.5*X[:,self.output_dim:])
		return mean, stdev

class Encoder_MLP_conditional(nn.Module):
	def __init__(self,input_dim,hidden_dims,output_dim):
		super(Encoder_MLP_conditional, self).__init__()
		self.input_dim = input_dim+2	# adding 2 for rotation and translation info
		self.hidden_dims = hidden_dims
		self.output_dim = output_dim
		self.linear1 = nn.Linear(self.input_dim,self.hidden_dims[0])
		self.hidden = [nn.Linear(self.hidden_dims[i-1],self.hidden_dims[i]) for i in range(1,len(self.hidden_dims))]
		self.hidden = nn.Sequential(*self.hidden)
		self.readout = nn.Linear(self.hidden_dims[len(self.hidden_dims)-1],2*self.output_dim)

	def forward(self, X, c1, c2):
		X = X.view(-1,28*28)
		c1 = c1.view(-1,1)
		c2 = c2.view(-1,1)
		X = torch.cat((X,c1,c2),dim=-1)
		X = F.relu(self.linear1(X))
		for hidden_layer in self.hidden.children():
			X = F.relu(hidden_layer(X))
		X = self.readout(X)
		mean = X[:,:self.output_dim]
		# stdev = 1e-6 + F.softplus(X[:,self.output_dim:])
		stdev = 1e-6 + torch.exp(0.5*X[:,self.output_dim:])
		return mean, stdev

class Decoder_MLP(nn.Module):
	def __init__(self,input_dim,hidden_dims,output_dim):
		super(Decoder_MLP, self).__init__()
		self.input_dim = input_dim
		self.hidden_dims = hidden_dims
		self.output_dim = output_dim
		self.linear1 = nn.Linear(self.input_dim,self.hidden_dims[0])
		self.hidden = [nn.Linear(self.hidden_dims[i-1],self.hidden_dims[i]) for i in range(1,len(self.hidden_dims))]
		self.hidden = nn.Sequential(*self.hidden)
		self.readout = nn.Linear(self.hidden_dims[len(self.hidden_dims)-1],self.output_dim)

	def forward(self, X):
		X = F.relu(self.linear1(X))
		for hidden_layer in self.hidden.children():
			X = F.relu(hidden_layer(X))
		X = torch.sigmoid(self.readout(X))
		return X

class Decoder_MLP_conditional(nn.Module):
	def __init__(self,input_dim,hidden_dims,output_dim):
		super(Decoder_MLP_conditional, self).__init__()
		self.input_dim = input_dim+2 	# adding 2 for rotation and translation info
		self.hidden_dims = hidden_dims
		self.output_dim = output_dim
		self.linear1 = nn.Linear(self.input_dim,self.hidden_dims[0])
		self.hidden = [nn.Linear(self.hidden_dims[i-1],self.hidden_dims[i]) for i in range(1,len(self.hidden_dims))]
		self.hidden = nn.Sequential(*self.hidden)
		self.readout = nn.Linear(self.hidden_dims[len(self.hidden_dims)-1],self.output_dim)

	def forward(self, X, c1, c2):
		c1 = c1.view(-1,1)
		c2 = c2.view(-1,1)
		X = torch.cat((X,c1,c2),dim=-1)
		X = F.relu(self.linear1(X))
		for hidden_layer in self.hidden.children():
			X = F.relu(hidden_layer(X))
		X = torch.sigmoid(self.readout(X))
		return X

class VAE_MLP(nn.Module):
	def __init__(self, input_dim, hidden_dims, output_dim):
		super(VAE_MLP, self).__init__()
		self.input_dim = input_dim
		self.hidden_dims = hidden_dims
		self.output_dim = output_dim
		self.encoder = Encoder_MLP(input_dim,hidden_dims,output_dim)
		self.decoder = Decoder_MLP(output_dim,hidden_dims[::-1],input_dim)

	def forward(self, X):
		mu,sigma = self.encoder(X)
		z = mu + sigma*torch.randn_like(mu)
		x_hat = self.decoder(z)
		return x_hat, mu, sigma, z

class CVAE_MLP(nn.Module):
	def __init__(self, input_dim, hidden_dims, output_dim, enc_condition=1, dec_condition=1):
		super(CVAE_MLP, self).__init__()
		# enc_condition = decides if condition is to be added in encoder
		# dec_condition = decides if condition is to be added in decoder
		# if both enc_condition and dec_condition are zero, latent vector tfo is done
		self.input_dim = input_dim
		self.hidden_dims = hidden_dims
		self.output_dim = output_dim
		self.enc_condition = enc_condition
		self.dec_condition = dec_condition
		if self.enc_condition:
			self.encoder = Encoder_MLP_conditional(input_dim,hidden_dims,output_dim)
		else:
			self.encoder = Encoder_MLP(input_dim,hidden_dims,output_dim)
		if self.dec_condition:
			self.decoder = Decoder_MLP_conditional(output_dim,hidden_dims[::-1],input_dim)
		else:
			self.decoder = Decoder_MLP(output_dim,hidden_dims[::-1],input_dim)

	def forward(self, X, c1, c2):
		if self.enc_condition:
			mu,sigma = self.encoder(X,c1,c2)
		else:
			mu,sigma = self.encoder(X)
		z = mu + sigma*torch.randn_like(mu)
		if self.dec_condition:
			x_hat = self.decoder(z,c1,c2)
		else:
			x_hat = self.decoder(z)
		return x_hat, mu, sigma, z

class CVAE_MLP_latent_tfo(nn.Module):
	def __init__(self, input_dim, hidden_dims, output_dim):
		super(CVAE_MLP_latent_tfo,self).__init__()
		# apply tfo in latent space
		self.input_dim = input_dim
		self.hidden_dims = hidden_dims
		self.output_dim = output_dim
		self.encoder = Encoder_MLP(input_dim,hidden_dims,output_dim)
		self.decoder = Decoder_MLP(output_dim,hidden_dims[::-1],input_dim)
		self.context_layer = nn.Linear(2,self.output_dim)
		self.context_layer_latent = nn.Linear(2,self.output_dim)

	def forward(self, X, c1, c2):
		c1 = c1.view(-1,1)
		c2 = c2.view(-1,1)
		C = torch.cat((c1,c2),dim=-1)
		mu,sigma = self.encoder(X)
		z = mu + sigma*torch.randn_like(mu)
		W_z = torch.diag_embed(F.softplus(self.context_layer_latent(C))).to(z.device)
		z_unsqueezed = z.unsqueeze(2)
		z_hat = torch.bmm(W_z,z_unsqueezed).squeeze() + self.context_layer(C)
		x_hat = self.decoder(z_hat)
		return x_hat, mu, sigma, z,z_hat

class CVAE_MLP_latent_tfo_modif(nn.Module):
	def __init__(self, input_dim, hidden_dims, output_dim):
		super(CVAE_MLP_latent_tfo_modif,self).__init__()
		# apply tfo in latent space and use decoder to reconstruct both 
		self.input_dim = input_dim
		self.hidden_dims = hidden_dims
		self.output_dim = output_dim
		self.encoder = Encoder_MLP(input_dim,hidden_dims,output_dim)
		self.decoder = Decoder_MLP(output_dim,hidden_dims[::-1],input_dim)
		self.context_layer = nn.Linear(2,self.output_dim)
		self.context_layer_latent = nn.Linear(2,self.output_dim)

	def forward(self, X, c1, c2):
		c1 = c1.view(-1,1)
		c2 = c2.view(-1,1)
		C = torch.cat((c1,c2),dim=-1)
		mu,sigma = self.encoder(X)
		z = mu + sigma*torch.randn_like(mu)
		x_recon = self.decoder(z)
		W_z = torch.diag_embed(F.softplus(self.context_layer_latent(C))).to(z.device)
		z_unsqueezed = z.unsqueeze(2)
		z_hat = torch.bmm(W_z,z_unsqueezed).squeeze() + self.context_layer(C)
		x_hat = self.decoder(z_hat)
		return x_hat, x_recon, mu, sigma, z,z_hat