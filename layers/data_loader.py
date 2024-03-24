import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
warnings.filterwarnings('ignore')
import pywt

from scipy import stats
# def waveletSmooth_swt(x, wavelet, DecLvl=None, level=1, rtn_kind = None):
#
#     if DecLvl is None:        DecLvl = pywt.swt_max_level(len(x))
#     # print(f'DecLvl: {DecLvl}, len(x): {len(x)}')
#
#     # calculate the wavelet coefficients
#     # thresh_coe = 0.6745 * 10
#     # coeffs = pywt.swt(data=x, wavelet=wavelet, level=level)  ## level=pywt.swt_max_level(len(data))
#     # coeffs_rec = []
#     # for i in range(len(coeffs)):
#     #     a_i = coeffs[i][0]
#     #     mad = stats.median_abs_deviation(coeffs[i][1])
#     #     d_i = pywt.threshold(coeffs[i][1], thresh_coe * mad, 'hard')
#     #     coeffs_rec.append((a_i, d_i))
#     # data_rec = pywt.iswt(coeffs_rec, wavelet)
#     ##
#     coeffs = pywt.swt(data=x, wavelet=wavelet, level=level)
#
#     if rtn_kind is None:
#         coeffs_rec = []
#         for i in range(len(coeffs)):
#             a_i = coeffs[i][0]
#             sigma = stats.median_abs_deviation(coeffs[i][1])
#             uthresh = sigma * np.sqrt(2 * np.log(len(x)))
#             d_i = pywt.threshold(coeffs[i][1], value=uthresh, mode="soft")
#             # d_i = pywt.threshold(coeffs[i][1], value=uthresh, mode="hard")
#             coeffs_rec.append((a_i, d_i))
#
#         data_rec = pywt.iswt(coeffs_rec, wavelet)
#         return data_rec
#
#     elif rtn_kind == 'low':
#         return coeffs[-1][0]
#
#     elif rtn_kind == 'high':
#         return coeffs[-1][1]


def data_provider(args, flag):
    Data = BTCloader

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    elif flag == 'val':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(args, flag=flag)

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_loader


class BTCloader(Dataset):
    def __init__(self, configs, flag='train'):

        self.fiat = configs.fiat
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # init
        self.flag = flag
        self.scaler = MinMaxScaler() # StandardScaler()
        self.root_path = configs.root_path
        self.data_path = configs.data_path
        self.__read_data__()

        configs.enc_in = self.data_x.shape[1]
        configs.c_out = self.data_y.shape[1]

    def __read_data__(self):

        # period_dict = {3: '3h', 12: '12h', 24: '1d'}
        period_dict = {12: '12h', 24: '1d'}

        ## original data set
        fiat = self.fiat
        df = pd.read_csv(self.root_path + fiat + '_tech1h.csv').drop_duplicates().reset_index(drop=True)

        ## technical index data set
        df.rename(columns={'tradedate':'date'}, inplace=True)
        df['date'] = df['date'].astype('datetime64[ns]')
        # split_date = datetime.strptime('2021-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        # split_idx = df.loc[df['date'] == split_date].index[0]
        # df = df.iloc[split_idx:].reset_index(drop=True)

        col_list = [fiat + c for c in
                    ['_openprice', '_highprice', '_lowprice', '_closeprice', '_tradevolume', '_obv', '_macd',
                     '_macdsignal', '_macdhist']]
                    # ['_openprice']]
        col_list_sub = ['_rsi', '_Srsi_K', '_Srsi_D', '_mfi', '_cci']
        periodLst = list(period_dict.values())
        for c in col_list_sub:
            for p in periodLst:
                col_list.append(fiat + c + p)

        col_list.insert(0, 'date')
        df = df[col_list]

        df[fiat+'_openprice'] = df[fiat+'_openprice'].pct_change()
        df[fiat+'_highprice'] = df[fiat+'_highprice'].pct_change()
        df[fiat+'_lowprice'] = df[fiat+'_lowprice'].pct_change()
        df[fiat+'_closeprice'] = df[fiat+'_closeprice'].pct_change()
        df[fiat+'_tradevolume'] = df[fiat+'_tradevolume'].pct_change()
        df = df.iloc[1:].reset_index(drop=True)
        df_stamp = df[['date']].copy()
        df = df.drop(['date'], axis=1)
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp = time_features(pd.to_datetime(df_stamp['date'].values))
        df_stamp = pd.DataFrame(df_stamp.transpose(1, 0))
        df = pd.concat([df_stamp, df], axis = 1)

        df_target = df[[fiat+'_closeprice']].copy()
        df_target.loc[df_target[fiat+'_closeprice'] > 0, 'target'] = 1
        df_target.loc[df_target[fiat+'_closeprice'] <= 0, 'target'] = 0
        del df_target[fiat+'_closeprice']
        # print(f'col list: {df.columns.to_list()}')

        begin_i = end_i = 0
        if self.flag == 'train':
            begin_i, end_i = 0, int(len(df) * 0.7)
        elif self.flag == 'val':
            begin_i, end_i = int(len(df) * 0.7) - self.seq_len, int(len(df) * 0.9)
        else:
            begin_i, end_i = int(len(df) * 0.9) - self.seq_len, len(df)

        df_data = df.iloc[begin_i:end_i].copy().reset_index(drop=True)
        df_target = df_target.iloc[begin_i:end_i].copy().reset_index(drop=True)

        # if len(df_data)%2!=0:
        #     df_data = df_data.iloc[:-1]
        #     df_stamp = df_stamp.iloc[:-1]
        #
        # ## wavelet
        # for c in df_data.columns.values:
        #     df_data.loc[:, c] = waveletSmooth_swt(x=df_data.loc[:, c], wavelet='haar', DecLvl=1)
        # ##

        self.scaler.fit(df.values)
        data = self.scaler.transform(df_data.values)

        self.data_x = data
        self.data_y = df_target.values
        # print('data_x : %s, data_y : %s' %(self.data_x.shape, self.data_y.shape))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)