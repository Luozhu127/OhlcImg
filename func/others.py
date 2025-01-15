import os
import torch
import random

def path_for_use(path):
	if not os.path.exists(path):
		print(f'make path {path}')
		os.makedirs(path)
	return None


def place_range(high,low,num,range):
	return min(range, int((num-low)/(high-low)*range))


def accuracy(predictions, labels):
	pred = torch.max(predictions.data,1)[1]
	rights = pred.eq(labels.data.view_as(pred)).sum()
	return rights, len(labels)

def split_list_to_slices(sample_list, label, percentage=20):
	k=random.sample(range(len(sample_list)), int(len(label)*0.2))
	sample_list_train = []
	sample_list_test = []
	label_train = []
	label_test = []
	for l in range(len(sample_list)):
		if l in k:
			sample_list_test.append(sample_list[l])
			label_test.append(label[l])
		else:
			sample_list_train.append(sample_list[l])
			label_train.append(label[l])

	return [sample_list_train, sample_list_test, label_train, label_test]