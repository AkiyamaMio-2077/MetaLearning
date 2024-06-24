from statistics import mode
from sklearn import datasets
import torch 
from torch.utils import data
from MAML_Moudel import Net4CNN 
from train_utils import accuracy
from train_utils import seed_torch
import torch.nn as nn

import learn2learn as l2l

import time
import numpy as np
import pandas as pd 
import copy
from train_utils import seed_torch
import os
from init_utils import get_file , get_data_csv,normalization,sample_label_shuffle

import torch.nn.functional as F

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')





class MetaDataset(torch.utils.data.Dataset):
    def __init__(self,file_path_train,file_path_Test,file_path_fine_tune,mode):
        if mode == 'train':
            self.data = pd.read_csv(file_path_train,index_col=False )
        if mode == 'valid':
            self.data = pd.read_csv(file_path_train,index_col=False)
        if mode == 'test':
            self.data = pd.read_csv(file_path_Test,index_col=False)
        if mode =='fine_tune':
            self.data = pd.read_csv(file_path_fine_tune,index_col= False)

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

        self.fc = nn.Linear(64, 3)  # 假设输出类别数量为4

    def forward(self, x):

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




class MAML_learner(object):
    def __init__(self,ways):
        h_size = 64
        layers = 4
        samplen_len = 261
        temp = samplen_len//2**layers
        feat_size = (samplen_len//2**layers)*h_size
  
        # self.model = Net4CNN (output_size= ways, hidden_size = h_size,layers= layers, channels= 1 ,embedding_size= feat_size).to(device)
        self.model = ConvNet().to(device)

        self.ways = ways 

    def build_tasks(self,mode = 'train',ways = 3,shots = 5,num_tasks = 60 ,filter_labels = None):
        meta_dataset = l2l.data.MetaDataset(MetaDataset(csv_file_path_Train,csv_file_path_Test,csv_file_path_fine_tune,mode = mode))

        new_ways= len(filter_labels) if filter_labels is not None else ways

        # assert shots*2*new_ways <= meta_dataset.__len__()//ways*new_ways,"Reduce the number of shots"    #dataset.len()//ways 不就是每个类的个数，要求的是 

        tasks = meta_train_dataloader = l2l.data.TaskDataset(meta_dataset, task_transforms=[

            l2l.data.transforms.LoadData(meta_dataset),
            l2l.data.transforms.NWays(meta_dataset, ways), 
            l2l.data.transforms.KShots(meta_dataset, shots+shots),
            l2l.data.transforms.RemapLabels(meta_dataset,shuffle = True),
            l2l.data.transforms.ConsecutiveLabels(meta_dataset),
            ], 
            num_tasks=num_tasks)
        # meta_train_dataloader = torch.utils.data.DataLoader(meta_train_dataloader, batch_size=meta_batch_size, shuffle=True, num_workers=4)
        return tasks

    def model_save(self,path):
        filename = path + '(1)' if os.path.exists(path) else path
        torch.save(self.model.state_dict(),filename)
        print(f'save mode at :{filename}')

    def fast_adapt(self,batch,learner,adapt_opt,loss,adaptation_steps,shots,ways,batch_size =10):
        data,labels,name = batch
        data,labels = data.to(device).unsqueeze(1),labels.to(device)

        # Separate data into adaptation/evaluation sets
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[np.arange(shots * ways) * 2] = True

        evaluation_indices = torch.from_numpy(~adaptation_indices)  # 偶数序号为True, 奇数序号为False
        adaptation_indices = torch.from_numpy(adaptation_indices)  # 偶数序号为False, 奇数序号为True

        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

        # Adapt the model
        for step in range(adaptation_steps):
            idx = torch.randint(adaptation_data.shape[0], size=(batch_size,))
            adapt_x = adaptation_data[idx]
            adapt_y = adaptation_labels[idx]

            adapt_opt.zero_grad()
            A = learner(adapt_x)
            error = loss(A, adapt_y)
            error.backward()
            adapt_opt.step()




        # Evaluate the adapted model
        predictions = learner(evaluation_data)
        valid_error = loss(predictions, evaluation_labels)
        valid_accuracy = accuracy(predictions, evaluation_labels)



        # predictions = learner(data)
        # valid_error = loss(predictions, labels)
        # valid_accuracy = accuracy(predictions, labels)

        return valid_error, valid_accuracy


    def train(self,save_path,shots):
        meta_lr = 0.005
        fast_lr = 0.05

        opt = torch.optim.SGD(self.model.parameters(), meta_lr)
        adapt_opt = torch.optim.Adam(self.model.parameters(), lr=fast_lr, betas=(0, 0.999)) 

        init_adapt_opt_state = adapt_opt.state_dict()
        adapt_opt_state = None
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        train_ways = valid_ways = self.ways
        print(f"{train_ways}-ways, {shots}-shots for training ...")


        meta_batch_size = 10
        train_steps = 40
        valid_steps = 10
        train_batch_size = 10
        valid_batch_size = 10

        Epoch = 100


        train_tasks = self.build_tasks(mode = 'train',ways = valid_ways,num_tasks=50,filter_labels = None)
        valid_tasks = self.build_tasks(mode = 'valid',ways = valid_ways,num_tasks=50,filter_labels = None)


        for ep in range(Epoch):
            t0 = time.time()

            if ep == 0:
                adapt_opt_state  = init_adapt_opt_state
            opt.zero_grad()
        
            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0

            frac_done = float(ep) / 100000
            new_lr = frac_done * meta_lr + (1 - frac_done) * meta_lr
            for pg in opt.param_groups:
                pg['lr'] = new_lr

            # zero-grad the parameters
            for p in self.model.parameters():
                p.grad = torch.zeros_like(p.data)

            for task in range(meta_batch_size):
                # Compute meta-training loss

                learner = copy.deepcopy(self.model)
                adapt_opt = torch.optim.Adam(learner.parameters(), lr=fast_lr, betas=(0, 0.999))
                adapt_opt.load_state_dict(adapt_opt_state)
                batch = train_tasks.sample()

                evaluation_error, evaluation_accuracy = self.fast_adapt(batch, learner, adapt_opt, loss,
                                                                        train_steps, shots, train_ways, train_batch_size)
                adapt_opt_state = adapt_opt.state_dict()
                for p, l in zip(self.model.parameters(), learner.parameters()):
                    p.grad.data.add_(l.data, alpha=-1.0)

                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                # Compute meta-validation loss
                learner = copy.deepcopy(self.model)
                adapt_opt = torch.optim.Adam(learner.parameters(), lr=fast_lr, betas=(0, 0.999))


                # adapt_opt.load_state_dict(adapt_opt_state)
                adapt_opt.load_state_dict(init_adapt_opt_state)
                batch = valid_tasks.sample()
                evaluation_error, evaluation_accuracy = self.fast_adapt(batch, learner, adapt_opt, loss,
                                                                        valid_steps, shots, train_ways, valid_batch_size)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            t1 = time.time()
            print(f'Time /epoch: {t1-t0:.3f} s')
            print('\n')
            print('Iteration', ep + 1)
            print(f'Meta Train Error: {meta_train_error / meta_batch_size: .4f}')
            print(f'Meta Train Accuracy: {meta_train_accuracy / meta_batch_size: .4f}')
            print(f'Meta Valid Error: {meta_valid_error / meta_batch_size: .4f}')
            print(f'Meta Valid Accuracy: {meta_valid_accuracy / meta_batch_size: .4f}')


            for p in self.model.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size).add_(p.data)
            opt.step()



            if (ep + 1) >= 700 and (ep + 1) % 2 == 0:
                if input('\n== Stop training? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)
                    break
                elif input('\n== Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)

            




if __name__ =="__main__":

    seed_torch(2021) 

    Net = MAML_learner(ways = 3)  

    # csv_file_path_Train = 'F:\\data\\汇总\\FT_20\\drought\\0.csv'
    # csv_file_path_Test = 'F:\\data\\汇总\\FT_20\\freeze\\0.csv'

    csv_file_path_fine_tune = 'F:\\data\\汇总\\FT_20\\freeze\\0.csv'
    csv_file_path_Train = 'F:\\data\\汇总\\dataset\\drought\\0.csv'
    csv_file_path_Test = 'F:\\data\\汇总\\dataset\\freeze\\0.csv'

    load_path = r"E:\Paper\savepath\5shot\MAML\MAML_Re_ep300"
    path = r"E:\Paper\savepath\5shot\MAML\MAML_Re_FT20"


    Net.train(save_path=path,shots = 2)

    # Net.fine_tune(load_path,inner_steps=10,shots=1)  


    # Net.test(load_path,inner_steps=10 ,shots=1)

       


