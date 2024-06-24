from tkinter import E
from sklearn import datasets
import torch 
from torch.utils import data 
from train_utils import accuracy
from train_utils import seed_torch
import torch.optim as optim


import time
import numpy as np
import pandas as pd 
from init_utils import weights_init2

from torch.utils.data import DataLoader
import os
from init_utils import get_file , get_data_csv,normalization,sample_label_shuffle
import torch.nn as nn

import torch.nn.functional as F
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')


class MetaDataset(torch.utils.data.Dataset):
    def __init__(self,file_path_train,file_path_Test,file_path_fine_tune,mode):
        if mode == 'train':
            self.data = pd.read_csv(file_path_train,index_col=False )
        if mode == 'valid':
            self.data = pd.read_csv(file_path_train,index_col=False)
        if mode =='fine_tune':
            self.data = pd.read_csv(file_path_fine_tune,index_col = False)
        if mode == 'test':
            self.data = pd.read_csv(file_path_Test,index_col=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        label = self.data.iloc[idx, 0]  # 获取标签列
        name = self.data.iloc[idx, 1]  # 获取数据名称列
        sample = self.data.iloc[idx, 2:].values.astype(float)  # 获取数值列

        return sample,label,name


class ConvNet(nn.Module):
    def __init__(self,ways = 4):
        super(ConvNet, self).__init__()
        self.kernel_size = 2
        self.conv1 = nn.Conv1d(1, 64, kernel_size=self.kernel_size, stride=2, padding=1)
        self.normalize1 = nn.BatchNorm1d(64,affine=True)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=self.kernel_size, stride=2, padding=1)
        self.normalize2 = nn.BatchNorm1d(64,affine=True)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=self.kernel_size, stride=2, padding=1)
        self.normalize3 = nn.BatchNorm1d(64,affine=True)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=self.kernel_size, stride=2, padding=1)
        self.normalize4 = nn.BatchNorm1d(64,affine=True)

        self.max_pool = nn.MaxPool1d(kernel_size= self.kernel_size, stride = 2,ceil_mode=False)

        self.fc = nn.Linear(64, ways)  # 假设输出类别数量为4

    def forward(self, x):
        x = x.unsqueeze(1)
        x=x.type(torch.cuda.FloatTensor)  # 增加维度，使其变为(Batch, Channels, Sequence Length)
        x = self.conv1(x)
        x = self.normalize1(x)
        x = F.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.normalize2(x)
        x = F.relu(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.normalize3(x)
        x = F.relu(x)
        x = self.max_pool(x)

        x = self.conv4(x)
        x = self.normalize4(x)
        x = F.relu(x)
        x = self.max_pool(x)


        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

batch_size = 2
learning_rate = 0.001
num_epochs = 100

model = ConvNet()

class CNN_learner(object):
    def __init__(self,ways):
        self.model = ConvNet(ways).to(device)


    def build_data(self,mode = 'train',batch_size = 10 ):
        dataset = MetaDataset(csv_file_path_Train,csv_file_path_Test,csv_file_path_fine_tune,mode = mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader

    def fast_adapt(self,batch,learner,loss_fun):
        data,labels,name = batch
        data,labels = data.to(device),labels.to(device)

        prediction = learner(data)

        acc  = accuracy(prediction,labels)
        error = loss_fun(prediction,labels)


        return error,acc


    def model_save(self, path):
        filename = path+'(1)' if os.path.exists(path) else path
        torch.save(self.model.state_dict(), filename)
        print(f'Save model at: {filename}')         


    def train(self,save_path):
        train_data  = self.build_data(mode = 'train',batch_size=10)
        valid_data = self.build_data(mode = 'valid',batch_size=10)

        self.model.apply(weights_init2)
        optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.95)  #查一下
        loss_fun = torch.nn.CrossEntropyLoss()

        Epochs = 150

        counter = 0

        for ep in range(Epochs):
            t0 = time.time()
            self.model.train()

            train_error = 0.0
            train_accuracy = 0.0
            
            for train in train_data:
                batch = train
                loss,acc = self.fast_adapt(batch,self.model,loss_fun)

                train_error += loss.item()
                train_accuracy += acc.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            t1 = time.time()

            print(f'***Time/epoch {t1 - t0:.4f} ***')
            print(f'epoch{ep + 1},train,loss:{train_error/len(train_data):.4f},'
            f'acc:{train_accuracy/len(train_data):.4f}')

            #valid 
            self.model.eval()
            valid_error = 0.0
            valid_accuracy = 0.0
            for i,batch in enumerate(valid_data):
                with torch.no_grad():
                    loss ,acc = self.fast_adapt(batch,self.model,loss_fun)
                valid_error      += loss.item()
                valid_accuracy   +=acc.item()

            print(f'epoch {ep + 1}, validation, loss: {valid_error / len(valid_data):.4f}, '
                f'acc: {valid_accuracy / len(valid_data):.4f}\n')
            counter +=1

            if (ep+1) >=150 and (ep+1)%2 == 0:
                if input('\n == Stop training? ==(y/n)\n').lower() == 'y':
                    new_save_path  = save_path +rf'_ep{ep + 1}'
                    self.model_save(new_save_path)
                    break
                elif input('\n == Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'


    def fine_tune_test(self,load_path,save_path):

        self.model.load_state_dict(torch.load(load_path))
        print(f'Load Model successfully from [{load_path}]...')

        fine_tune_data = self.build_data(mode='fine_tune',batch_size=2)
        test_data = self.build_data(mode='test', batch_size = 10)

        self.model.eval()
        optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.95)  #查一下
        loss_fun = torch.nn.CrossEntropyLoss()

        Epochs = 100

        for ep in range(Epochs):
            t0 = time.time()
            self.model.train()

            train_error = 0.0
            train_accuracy = 0.0
            
            for train in fine_tune_data:
                batch = train
                loss,acc = self.fast_adapt(batch,self.model,loss_fun)

                train_error += loss.item()
                train_accuracy += acc.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            t1 = time.time()

            print(f'***Time/epoch {t1 - t0:.4f} ***')
            print(f'epoch{ep + 1},train,loss:{train_error/len(fine_tune_data):.4f},'
            f'acc:{train_accuracy/len(fine_tune_data):.4f}')

            if (ep+1) >=100 and (ep+1)%2 == 0:
                if input('\n == Stop training? ==(y/n)\n').lower() == 'y':
                    new_save_path  = save_path +rf'_ep{ep + 1}_fine'
                    self.model_save(new_save_path)
                    break
                elif input('\n == Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}_fine'

        self.model.eval()
        test_error = 0.0
        test_accuracy =0.0
        t0 = time.time()

        for i,batch in enumerate(test_data):
            with torch.no_grad():
                loss,acc = self.fast_adapt(batch,self.model,loss_fun)
            test_error += loss.item()
            test_accuracy += acc.item()
            
            t1 = time.time()

        print(f"*** Time for {len(fine_tune_data)} tasks {t1 - t0:.4f}(s)")
        print(f'Testing, loss: {test_error / len(test_data):.4f}, '
            f'acc: {test_accuracy / len(test_data):.4f}')
                       
    def test(self,load_path):
        self.model.load_state_dict(torch.load(load_path))
        print(f'Load Model successfully from [{load_path}]...')
        test_data = self.build_data(mode='test', batch_size = 10)
        loss_fun = torch.nn.CrossEntropyLoss()
        self.model.eval()
        
        t0 = time.time()
        valid_error = 0.0
        valid_accuracy = 0.0
        for i,batch in enumerate(test_data):
            with torch.no_grad():
                loss ,acc = self.fast_adapt(batch,self.model,loss_fun)
            valid_error      += loss.item()
            valid_accuracy   +=acc.item()
        
        t1 = time.time()

        print(f'validation, loss: {valid_error / len(test_data):.4f}, '
            f'acc: {valid_accuracy / len(test_data):.4f}\n')
        print(f"*** time takes  {t1 - t0:.4f}(s)")



if __name__ == "__main__":
    from train_utils import seed_torch
    seed_torch(2021)
    Net = CNN_learner(ways = 11)
    FT_num = "FT_20"
    FT_num1 = "FT20"
    # csv_file_path_Train = 'E:\\Milk\\averspectra\\1-D\\niu.csv'
    # csv_file_path_Test = 'E:\\Milk\\averspectra\\1-D\\fine_tuning\\'+ FT_num +'\\test.csv'
    # csv_file_path_fine_tune = 'E:\\Milk\\averspectra\\1-D\\fine_tuning\\'+ FT_num +'\\train.csv'

    # csv_file_path_Train = 'E:\\Milk\\averspectra\\1-D\\fine_tuning\\'+ FT_num +'\\train.csv'
    # csv_file_path_Test = 'E:\\Milk\\averspectra\\1-D\\fine_tuning\\'+ FT_num +'\\test.csv'

    # csv_file_path_Train = 'F:\\data\\汇总\\'+FT_num+'\\drought\\0.csv'
    # # csv_file_path_Test = 'F:\\data\\汇总\\'+FT_num+'\\freeze\\0.csv'
    # csv_file_path_Test = 'F:\\data\\汇总\\dataset\\freeze\\0.csv'
    # csv_file_path_fine_tune = 'F:\\data\\汇总\\'+FT_num+'\\FT\\0.csv'

    # path = r"E:\Paper\savepath\5shot\CNN_milk_"+ FT_num1+""
    # # Net.train(save_path = path)
    # Net.test(load_path=r"E:\Paper\savepath\5shot\CNN\CNN_"+FT_num1+"_ep150")
    # Net.fine_tune_test(load_path=r"E:\Paper\savepath\5shot\CNN\CNN_"+FT_num1+"_ep150")     

    csv_file_path_Train = 'E:\\Milk\\averspectra\\processing\\right\\SNV\\niu_tuo.csv'
    # csv_file_path_Train = 'F:\\data\\汇总\\FT_0\\freeze\\0.csv'
    csv_file_path_Test = 'E:\\Milk\\averspectra\\processing\\right\\SNV\\yang_tuo.csv'
    csv_file_path_fine_tune = 'E:\\Milk\\averspectra\\processing\\right\\SNV\\FT.csv'

    path = r"E:\模型保存\milk\CNN\CNN_milk"
    # Net.train(save_path = path)
    # Net.fine_tune_test(load_path=r"E:\模型保存\milk\CNN\CNN_milk_ep150",save_path=path) 
    # Net.fine_tune_test(load_path=r"E:\模型保存\milk\CNN\CNN_milk_ep100_fine(1)",save_path=path) 
    Net.test(load_path=r"E:\模型保存\milk\CNN\CNN_milk_ep100_fine(1)") 
    # Net.fine_tune_test(load_path=r"E:\Paper\savepath\5shot\MAML\ProtoNet_FT0_shot2_ep200",shots=1)    


