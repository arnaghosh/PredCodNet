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
