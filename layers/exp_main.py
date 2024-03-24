import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from layers.data_loader import data_provider
import FED_Lstm
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

warnings.filterwarnings('ignore')

class Exp_Main:
    def __init__(self, args):
        # super(Exp_Main, self).__init__(args)

        self.args = args
        if self.args.use_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            self.device = torch.device('cpu')
            print('Use CPU')

        ## load data
        self.train_loader = data_provider(self.args, 'train')
        self.val_loader = data_provider(self.args, 'val')
        self.test_loader = data_provider(self.args, 'test')

        ## build model
        self.model = FED_Lstm.Model(self.args).float().to(self.device)
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.BCELoss()

    def _get_Acc(self, pred, true):
        pred = torch.Tensor(pred)
        true = torch.Tensor(true)
        prediction = pred >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == true
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        return accuracy

    def train(self, path):
        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_pred, train_y = [], []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(self.train_loader):
                iter_count += 1
                self.model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x)

                batch_y = batch_y.to(self.device)
                loss = self.criterion(outputs, batch_y)
                train_loss.append(loss.item())

                train_pred.append(outputs.detach().cpu().numpy())
                train_y.append(batch_y.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_Acc = self.vali(self.val_loader, self.criterion)
            test_loss, test_Acc = self.vali(self.test_loader, self.criterion)

            train_pred = np.array(train_pred)
            train_pred = train_pred.reshape(-1, 1)
            train_y = np.array(train_y)
            train_y = train_y.reshape(-1, 1)
            train_Acc = self._get_Acc(train_pred, train_y)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Train Acc: {5:.5f} Vali Loss: {3:.7f} Vali Acc: {5:.5f} Test Loss: {4:.7f} Test Acc: {5:.5f}".format(
                epoch + 1, train_steps, train_loss, train_Acc, vali_loss, val_Acc, test_loss, test_Acc))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []
        val_pred, val_y = [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                outputs = self.model(batch_x)
                batch_y = batch_y.to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

                val_pred.append(pred.numpy())
                val_y.append(true.numpy())

        total_loss = np.average(total_loss)

        val_pred = np.array(val_pred)
        val_pred = val_pred.reshape(-1, 1)
        val_y = np.array(val_y)
        val_y = val_y.reshape(-1, 1)
        val_Acc = self._get_Acc(val_pred, val_y)
        self.model.train()
        return total_loss, val_Acc

    def test(self, path, setting, test=0):
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x)

                batch_y = batch_y.to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)

        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("../result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        preds = preds.reshape(-1, 1)
        trues = trues.reshape(-1, 1)

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return