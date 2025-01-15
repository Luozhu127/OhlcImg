import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np

from func import config




class CNN_20(nn.Module):
	def __init__(self):
		super(CNN_20, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=1,
				out_channels=64,
				kernel_size=(5, 3),
				stride=1,
				padding=2,
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(
				in_channels=64,
				out_channels=128,
				kernel_size=(5, 3),
				stride=1,
				padding=2,
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(
				in_channels=128,
				out_channels=256,
				kernel_size=(5, 3),
				stride=1,
				padding=2,
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.out = nn.Linear(18432, 2)

	def forward(self, x):

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = x.view(x.size(0), -1)
		# output = self.out(x)
		output = torch.tanh(self.out(x))

		return output




class CNN_5(nn.Module):
	def __init__(self):
		super(CNN_5, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=1,
				out_channels=64,
				kernel_size=(5, 3),
				stride=1,
				padding=2,
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(
				in_channels=64,
				out_channels=128,
				kernel_size=(5, 3),
				stride=1,
				padding=2,
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.out = nn.Linear(5120, 2)

	def forward(self, x):

		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)
		# output = self.out(x)
		output = torch.tanh(self.out(x))

		return output



class CNN_60(nn.Module):
	def __init__(self):
		super(CNN_60, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=1,
				out_channels=64,
				kernel_size=(5, 3),
				stride=1,
				padding=2,
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(
				in_channels=64,
				out_channels=128,
				kernel_size=(5, 3),
				stride=1,
				padding=2,
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(
				in_channels=128,
				out_channels=256,
				kernel_size=(5, 3),
				stride=1,
				padding=2,
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(
				in_channels=256,
				out_channels=512,
				kernel_size=(5, 3),
				stride=1,
				padding=2,
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.out = nn.Linear(512*23*12, 2)

	def forward(self, x):

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = x.view(x.size(0), -1)
		# output = self.out(x)
		output = torch.tanh(self.out(x))

		return output













