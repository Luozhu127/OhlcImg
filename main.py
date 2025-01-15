import torch
import time

from func.Image import generate_image
from func.CNN.train import *
from func.CNN.test import test_model_cnn


# 获取当前时间的结构化表示（本地时间）
struct_time = time.localtime()
# 提取年、月、日
year = struct_time.tm_year;month = struct_time.tm_mon;day = struct_time.tm_mday


net = train_CNN_stock()

torch.save(net.state_dict(), f'data/model/cnn_model_state_dict_{year}{month:02d}{day:02d}.pth')

test_model_cnn(net)












