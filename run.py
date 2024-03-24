import argparse
import torch
from layers.exp_main import Exp_Main
import random
import numpy as np
import os

def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Global High Frequency + Global Trend(DeNoise)')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='1: train, 0: test')

    # supplementary config for FED-Lstm model
    parser.add_argument('--mode_select', type=str, default='random',
                        help='frequency domain 변환 후, 추출 모드: [random, low]')
    parser.add_argument('--modes', type=int, default=8, help='추출할 주파수 갯수')

    # data loader
    parser.add_argument('--root_path', type=str, default='./dataset/BTC/', help='데이터 경로')
    parser.add_argument('--data_path', type=str, default='ETH_tech1h.csv', help='데이터 파일명')
    parser.add_argument('--fiat', type=str, default='ETH', help='fiat')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='모델 저장 위치')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=24, help='과거 사이즈(History Length)')
    parser.add_argument('--pred_len', type=int, default=1, help='')

    # model define
    parser.add_argument('--enc_in', type=int, default=10, help='feature 수')
    parser.add_argument('--c_out', type=int, default=5, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--trend_kernels', default=[6, 12, 24], help='window size of moving average')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments iterations')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    path = os.path.join(args.checkpoints, 'fed_lstm')
    if not os.path.exists(path):
        os.makedirs(path)

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments

            exp = Exp(args)  # set experiments
            exp.train(path)

            exp.test(path, 'fed_lstm', 1)
            torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        exp.test(path, 'fed_lstm', 1)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
