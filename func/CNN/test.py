import func.CNN.env as env
from func.others import *

def test_model_cnn(net= env.load_net()):

	net.eval()
	val_rights = []

	for (data, target) in env.test_loader:
		output = net(data)
		right = accuracy(output, target)
		val_rights.append(right)

		val_r = (sum(tup[0] for tup in val_rights)), (sum(tup[1] for tup in val_rights))

	print('the accuracy in test:{:.2f}%'.format(100. * val_r[0].numpy() / val_r[1]), end='\n')


	return val_r