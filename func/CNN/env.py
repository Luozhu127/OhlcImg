import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from func import config
from func.CNN.cnn import *
from func.others import *




def load_net():
	if config.length_dataset == 20:
		net = CNN_20()
	elif config.length_dataset == 5:
		net = CNN_5()
	elif config.length_dataset == 60:
		net = CNN_60()

	model_path = config.model_path
	if os.path.exists(model_path):
		net.load_state_dict(torch.load(model_path, weights_only=True))
		print(model_path)

	return net




class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('L')  # 转换为灰度图
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


image_paths=[];labels=[]

# 贴上标签

path_image = config.path_image
for filename in os.listdir(path_image):
	image_paths.append(f'{path_image}/{filename}')

	if filename[-5] == '1':
		image_label = 1
	else:
		image_label = 0
	labels.append(image_label)

# 设置数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

'''
define the dataset for train and test
'''
[sample_list_train, sample_list_test, label_train, label_test] = split_list_to_slices(image_paths, labels)

train_dataset = ImageDataset(image_paths=sample_list_train, labels=label_train, transform=transform )
test_dataset = ImageDataset(image_paths=sample_list_test, labels=label_test, transform=transform )

'''
define the a batch of data for train and test
'''
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_sizes, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_sizes, shuffle=True)
