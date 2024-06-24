
from sklearn import datasets
import torch 
from torch.utils import data 
from train_utils import accuracy
from train_utils import seed_torch

import learn2learn as l2l
from MAML_Moudel import Net4CNN
import time
import numpy as np
import pandas as pd 

import os
from init_utils import get_file , get_data_csv,normalization,sample_label_shuffle

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





class MAML_learner(object):
    def __init__(self,ways):
        h_size = 64
        layers = 4
        samplen_len = 261
        temp = samplen_len//2**layers
        feat_size = (samplen_len//2**layers)*h_size
  
        self.model = Net4CNN (output_size= ways, hidden_size = h_size,layers= layers, channels= 1 ,embedding_size= feat_size).to(device)

        self.ways = ways 

    def build_tasks(self,mode = 'train',ways = 3,shots = 5,num_tasks = 60 ,meta_batch_size = 100,filter_labels = None):
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

    def fast_adapt(self,batch,learner,loss,adaptation_steps,shots,ways):
        data,label,name = batch
        data,label = data.to(device).unsqueeze(1),label.to(device)
        # data,label = data.to(device),label.to(device)

        adaptation_indices = np.zeros(data.size(0),dtype=bool)
        selection = np.arange(ways)*(shots + shots)           # 0, shot+q, 2*(shot+q), 3*(), ...

        for offset in range(shots):                             
            adaptation_indices[selection + offset]  = True           # 0:shots, (shot+q):(shot+q+shots), ...

        evalutation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)

        adaptation_data  ,adaptation_label   = data[adaptation_indices],label[adaptation_indices]
        evalutation_data ,evalutation_labels = data[evalutation_indices],label[evalutation_indices]

        for step in range(adaptation_steps):
            train_error = loss(learner(adaptation_data),adaptation_label)
            learner.adapt(train_error)

        predictions = learner(evalutation_data)
        valid_error = loss(predictions,evalutation_labels)
        valid_accuracy = accuracy(predictions,evalutation_labels)


        return valid_error,valid_accuracy



    def model_save(self,path):
        filename = path + '(1)' if os.path.exists(path) else path
        torch.save(self.model.state_dict(),filename)
        print(f'save mode at :{filename}')

    def train(self,save_path,shots = 1 ):
        meta_lr = 0.05
        fast_lr = 0.005

        maml = l2l.algorithms.MAML(self.model,lr = fast_lr)
        opt = torch.optim.Adam(maml.parameters(),meta_lr)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        train_ways = valid_ways = self.ways
        print(f"{train_ways}--ways,{shots}--shots for training...")

        meta_batch_size = 30
        adaptation_steps = 20 


        train_task = self.build_tasks(mode = 'train',ways = valid_ways,num_tasks=60,meta_batch_size =  meta_batch_size)
        valid_task = self.build_tasks(mode = 'valid',ways = valid_ways,num_tasks=60,meta_batch_size =  meta_batch_size)

        counter = 0
        Epochs = 300

        for ep in range(Epochs):
            t0 = time.time()

            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0 
            meta_valid_accuracy = 0.0

            opt.zero_grad()

            for _ in range(meta_batch_size):
                learner = maml.clone()
                Train_task  = train_task.sample()

                train_error ,train_accuracy = self.fast_adapt( Train_task,learner,loss,adaptation_steps,shots,train_ways)
                train_error.backward()

                meta_train_error    += train_error.item()
                meta_train_accuracy += train_accuracy.item()

                learner = maml.clone()
                Test_task = valid_task.sample()
                valid_error,valid_accuracy = self.fast_adapt(Test_task,learner,loss,adaptation_steps,shots,valid_ways)

                meta_valid_error += valid_error.item()
                meta_valid_accuracy += valid_accuracy.item()

                t1 = time.time()


            print(f'Time / epoch:{t1 - t0 :.4f} s')
            print('\n')
            print('Iteration',ep+1)
            print(f'Meta Train Error: {meta_train_error/meta_batch_size: .4f}')
            print(f'Meta Train Accuracy: {meta_train_accuracy / meta_batch_size: .4f}')
            print(f'Meta Valid Error: {meta_valid_error / meta_batch_size: .4f}')
            print(f'Meta Valid Accuracy :{meta_valid_accuracy / meta_batch_size: .4f}')

            for p in maml.parameters():
                p.grad.data.mul_(1.0/meta_batch_size)

            opt.step()
            counter += 1
            if (ep +1 ) >= 300 and (ep + 1)%2 == 0 :
                if input('\n == Stop training? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep+1}'
                    self.model_save(new_save_path)
                    break

                elif input('\n == Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep+1}'
                    self.model_save(new_save_path)


    def test(self,load_path,inner_steps=10,shots = 1):
        self.model.load_state_dict(torch.load(load_path))
        print('Load Model successfully from [%s]'% load_path )

        test_ways = self.ways
        shots = shots 
        print(f"{test_ways} - ways, {shots} -shots for test ...")

        fast_lr = 0.05


        test_task = self.build_tasks('test',test_ways,shots,100,None)
        maml = l2l.algorithms.MAML(self.model ,lr = fast_lr)

        loss = torch.nn.CrossEntropyLoss(reduction ='mean')

        meta_batch_size =100
        adaptation_steps = inner_steps    #这是test 后面可以改为1 在def里面改
        
        meta_test_error = 0.0
        meta_test_accuracy = 0.0

        t0 = time.time()

        for _ in range(meta_batch_size):
            # Compute meta-testing loss
            learner = maml.clone()
            task = test_task.sample()
            evaluation_error, evaluation_accuracy = self.fast_adapt(task, learner, loss,
                                                                    adaptation_steps, shots, test_ways)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        t1 = time.time()

        print(f"----------Time for {meta_batch_size*shots} samples: {t1 - t0:.4f}(s) . ---------")
        print(f'Meta Test Error : {meta_test_error /meta_batch_size: .4f}')
        print(f'Meta Test Accuracy: {meta_test_accuracy / meta_batch_size : .4f} \n ')
    




    def fine_tune(self,load_path,inner_steps = 10,shots = 10):

        self.model.load_state_dict(torch.load(load_path))
        print('Load Model successfully from [%s]'% load_path )

        test_ways  = self.ways
        shots = shots 
        print(f"{test_ways} - ways, {shots} -shots for test ...")

        fast_lr = 0.05
        meta_lr = 0.05


        fine_tune_task = self.build_tasks('fine_tune',test_ways,shots,100,None)
        test_task = self.build_tasks('fine_tune',test_ways,shots,100,None)
        
        
        maml = l2l.algorithms.MAML(self.model ,lr = fast_lr)
        loss = torch.nn.CrossEntropyLoss(reduction ='mean')
        opt = torch.optim.Adam(maml.parameters(),meta_lr)

        meta_batch_size =50
        adaptation_steps = inner_steps    #这是test 后面可以改为1 在def里面改
        
        counter = 0
        Epochs = 150

        for ep in range(Epochs):
            t0 = time.time()

            meta_train_error = 0.0
            meta_train_accuracy = 0.0

            opt.zero_grad()

            for _ in range(meta_batch_size):
                learner = maml.clone()
                task  = fine_tune_task.sample()

                evaluation_error ,evaluation_accuracy = self.fast_adapt(task,learner,loss,adaptation_steps,shots,test_ways)
                evaluation_error.backward()

                meta_train_error    += evaluation_error.item()
                meta_train_accuracy +=evaluation_accuracy.item()

            print('Iteration',ep+1)
            print(f'Meta Train Error: {meta_train_error/meta_batch_size: .4f}')
            print(f'Meta Train Accuracy: {meta_train_accuracy / meta_batch_size: .4f}')


        meta_test_error = 0.0
        meta_test_accuracy = 0.0

        for _ in range(meta_batch_size):
            # Compute meta-testing loss
            learner = maml.clone()
            task = test_task.sample()
            evaluation_error, evaluation_accuracy = self.fast_adapt(task, learner, loss,
                                                                    adaptation_steps, shots, test_ways)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        t1 = time.time()
        print(f"----------Time for {meta_batch_size*shots} samples: {t1 - t0:.4f}(s) . ---------")
        print(f'Meta Test Error : {meta_test_error /meta_batch_size: .4f}')
        print(f'Meta Test Accuracy: {meta_test_accuracy / meta_batch_size : .4f} \n ')



if __name__ == "__main__":

    seed_torch(2021)
    Net = MAML_learner(ways = 3)  

    # csv_file_path_Train = 'F:\\data\\汇总\\FT_20\\drought\\0.csv'
    # csv_file_path_Test = 'F:\\data\\汇总\\FT_20\\freeze\\0.csv'
    csv_file_path_fine_tune = 'F:\\data\\汇总\\FT_20\\freeze\\0.csv'
    csv_file_path_Train = 'F:\\data\\汇总\\dataset\\drought\\0.csv'
    csv_file_path_Test = 'F:\\data\\汇总\\dataset\\freeze\\0.csv'

    load_path = r"E:\Paper\savepath\5shot\MAML\MAML_FT20_ep300"
    path = r"E:\Paper\savepath\5shot\MAML\MAML_FT20"


    Net.train(save_path=path,shots = 2)

    # Net.fine_tune(load_path,inner_steps=10,shots=1)  


    Net.test(load_path,inner_steps=10 ,shots=1)
    



    







        






