import os
from datetime import datetime,timedelta
import time
import random
import pandas as pd

from func.Image import OHLC

length_dataset = 20
prediction_dataset = 5
model_path = f'data/model/cnn_model_state_dict -{length_dataset}-{prediction_dataset}.pth'
path_image = f'data/image/{length_dataset}_{prediction_dataset}'

num_epochs = 12
batch_sizes = 128
learning_rate = 1e-3