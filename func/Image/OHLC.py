import os.path

import pandas as pd
import numpy as np
from PIL import Image
from func.others import *

def Image_OHLC(data_stock=pd.DataFrame(), data_path=str, data_name=str, length=int, pred_length=int, date_start=str,label=int):

	'''image settings'''
	path_for_use(f'{data_path}/{length}_{pred_length}')
	image_name = f'{data_path}/{length}_{pred_length}/{data_name}_{length}_{pred_length}_{date_start}_{label}.png'
	if os.path.exists(image_name):
		print(f'{image_name} already exists')
		return None

	price_high = data_stock[['移动平均','开盘', '收盘', '最高', '最低']].max().max()
	price_low = data_stock[['移动平均','开盘', '收盘', '最高', '最低']].min().min()

	if (price_high-price_low) < 0.2:
		print(f'price issues: \thigh:{price_high}, \tlow:{price_low}')
		return None

	'''generate image'''
	if length == 5:
		image_array = np.zeros((32, 3*length))
		picre_range = 22
		volumn_range = 8
	elif length == 20:
		image_array = np.zeros((64, 3*length))
		picre_range = 46
		volumn_range = 16
	elif length == 60:
		image_array = np.zeros((96, 3*length))
		picre_range = 70
		volumn_range = 24
	else:
		print('length issues')
		return None

	'''OHLC'''
	for day in range(length):
		image_array[place_range(price_high, price_low, data_stock['开盘'].iloc[day], picre_range)+1 + volumn_range][
			day * 3] = 255
		image_array[(place_range(price_high, price_low, data_stock['最低'].iloc[day], picre_range)+1 + volumn_range):
					(place_range(price_high, price_low, data_stock['最高'].iloc[day], picre_range)+2 + volumn_range),
		day * 3 + 1] = 255
		image_array[place_range(price_high, price_low, data_stock['收盘'].iloc[day], picre_range)+1 + volumn_range][
			day * 3 + 2] = 255

	'''Moving average'''
	for day in range(length):
		image_array[place_range(price_high, price_low, data_stock['移动平均'].iloc[day], picre_range) - 1 + volumn_range][
		day * 3:day * 3 + 3] = 255

	'''volumn'''
	for day in range(length):
		image_array[:(place_range(3, 0, data_stock['换手率'].iloc[day], volumn_range)), day * 3 + 1] = 255



	'''Saving Images'''
	image_array=np.flip(image_array)
	image_array = np.array(image_array, dtype=np.uint8)
	img = Image.fromarray(image_array, 'L')
	# 保存图像
	img.save(f'{image_name}')

	return None