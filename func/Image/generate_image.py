import os
import time
import random
import pandas as pd

from func.Image import OHLC


def generate_image(length=int, pred_length=int):
    data_path = './stock_data/us/us_hfq'
    start_time = time.time()
    count = 1
    while (time.time() - start_time) < 60 * 60 and count < 600:

        '''select file and read'''
        file_name = random.choice(os.listdir(data_path))
        data_stock = pd.read_pickle(f'{data_path}/{file_name}')

        '''data process'''
        data_stock['收盘'] = data_stock['收盘'].ffill().bfill()
        data_stock['移动平均'] = data_stock['收盘'].rolling(window=length).mean()
        data_stock['移动平均'] = data_stock['移动平均'].ffill().bfill()

        '''select the data for imaging'''
        if len(data_stock.index) < length + 10: continue
        index_start = random.choice(data_stock.index[:-length - 1])
        data_stock_image = data_stock[['移动平均', '开盘', '收盘', '最高', '最低', '换手率']][
                           index_start:index_start + length]
        if (data_stock_image['换手率'] == 0).sum() > int(0.6 * length): continue

        '''set image parameters and perform imaging operations'''
        data_name = file_name[:-7]
        date_start = (data_stock['日期'][index_start]).strftime('%Y%m%d')
        if int(date_start) < 20100101 or int(date_start) > 20200101: continue
        label = 1 if data_stock['收盘'][min(index_start + length + pred_length, data_stock.index[-1])] > \
                     data_stock['收盘'][index_start + length] else 0

        OHLC.Image_OHLC(data_stock=data_stock_image, data_path='data/image', data_name=data_name,
                        length=length, pred_length=pred_length, date_start=date_start, label=label)

        print(f'{count}:\t{file_name}\t\t{data_stock["日期"][index_start]}')
        count += 1
    print(f'{count} images generated')
    return None
