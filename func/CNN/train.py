import os
import time
import tqdm

from func.CNN.cnn import *
import func.CNN.env as env
from func.CNN.test import test_model_cnn
from func.others import *
from func import config


'''
define network,loss function,optimizer
'''


def train_CNN_stock():

	net = env.load_net()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

	loss_plot = []
	for epoch in (range(config.num_epochs)):
		train_rights = []


		for batch_idx, (data,target) in (enumerate(env.train_loader)):
			# print(f'batch_idx:{batch_idx}')
			net.train()
			output = net(data)
			loss = criterion(output, target)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			right = accuracy(output, target)
			train_rights.append(right)

			if float(loss.data)<2:loss_plot.append(float(loss.data))
			plt.plot(loss_plot);


			if batch_idx % 50\
					== 20:
				train_r = (sum(tup[0] for tup in train_rights)),(sum(tup[1] for tup in train_rights))

				print('Current epoch: {:02d} [{:05d}/{:05d} ({:2d}%)]\t\tloss: {:.6f}\t\t the accuracy in train:{:.2f}%'.format(
					epoch+1, batch_idx*config.batch_sizes, len(env.train_loader.dataset), int(100*batch_idx/len(env.train_loader)),
					loss.data,100. * train_r[0].numpy() / train_r[1]), end='\n')
				# val_r = test_model_cnn(net)


		plt.savefig(f'data/model/loss_image/{config.length_dataset}_{config.prediction_dataset}_{epoch}_{batch_idx}_{int(time.time())}.png')
		test_model_cnn(net)
		torch.save(net.state_dict(), config.model_path)

	return net








