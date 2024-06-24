from calendar import weekday
from tkinter import FALSE
from sklearn import datasets
import torch 
from torch.utils import data 
from train_utils import accuracy
from train_utils import seed_torch
import torch.nn.functional as F
import learn2learn as l2l
import time
import numpy as np
import pandas as pd 
from ProtoNet_Moudel import Net4CNN
import os
from init_utils import get_file , get_data_csv,normalization,sample_label_shuffle
from init_utils import weights_init2
from sklearn.metrics import precision_recall_fscore_support

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')


class MetaDataset(torch.utils.data.Dataset):
    def __init__(self,file_path_train,file_path_Test,file_path_fine_tune,mode):
        if mode == 'train':
            self.data = pd.read_csv(file_path_train,index_col=False )
        if mode == 'valid':
            self.data = pd.read_csv(file_path_train,index_col=False)
        if mode == 'test':
            self.data = pd.read_csv(file_path_Test,index_col=False)
        if mode == 'fine_tune':
            self.data = pd.read_csv(file_path_fine_tune,index_col=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        label = self.data.iloc[idx, 0]  # 获取标签列
        name = self.data.iloc[idx, 1]  # 获取数据名称列
        sample = self.data.iloc[idx, 2:].values.astype(float)  # 获取数值列

        return sample,label,name



# meta_dataset = MetaDataset(csv_file_path_Train,csv_file_path_Test,mode = 'train')

class ProtoNet_learner(object):
    def __init__(self,ways):
        super().__init__()
        self.model = Net4CNN(hidden_size = 64,layers = 4,channels = 1).to(device)
        self.ways = ways

    def bulid_tasks(self,mode='train',ways = 10,shots = 5,num_tasks = 100,filter_labels = None ):
        meta_dataset = l2l.data.MetaDataset(MetaDataset(csv_file_path_Train,csv_file_path_Test,csv_file_path_fine_tune,mode = mode))
        new_ways = len(filter_labels) if filter_labels is not None else ways

        assert shots*new_ways <= meta_dataset.__len__()//ways*new_ways , "Reduce the number of shots!"

        tasks  = l2l.data.TaskDataset(meta_dataset,task_transforms = [
            l2l.data.transforms.FusedNWaysKShots(meta_dataset,new_ways,shots+1,filter_labels = filter_labels),
            l2l.data.transforms.LoadData(meta_dataset),
            l2l.data.transforms.RemapLabels(meta_dataset,shuffle = True),
            l2l.data.transforms.ConsecutiveLabels(meta_dataset),
        ],num_tasks = num_tasks)

        return tasks
    def euclidean_metric(self,query_x, proto_x):
        """
        :param query_x: (n, d), N-example, P-dimension for each example; zq
        :param proto_x: (nc, d), M-Way, P-dimension for each example, but 1 example for each Way; z_proto
        :return: [n, nc]
        """

        query_x = query_x.unsqueeze(dim=1)  # [n, d]==>[n, 1, d]
        proto_x = proto_x.unsqueeze(dim=0)  # [nc, d]==>[1, nc, d]
        similarity_map = F.cosine_similarity(proto_x,query_x,dim = -1)
        logits = -torch.pow(query_x - proto_x, 2).mean(dim=2)  # (n, nc)
        return logits


#这个fast_adapt 
    def fast_adapt(self,batch,learner,loss_fun,query_num,shots,ways):
        data ,labels,name = batch
        data,labels = data.to(device).unsqueeze(1),labels.to(device)

         # Separate data into adaptation/evaluation sets
        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)

        support_indices = np.zeros(data.size(0),dtype = bool)
        selection = np.arange(ways)*(shots + query_num)   #0,shot+q ,2*(shot+q),3*(shot+q) ,.....   这部分可能后面要改，原本的代码时全部数据然后抽样的。
        
        for offset in range(shots):
            support_indices [selection + offset] = True    #0:shots ,(shot +q ):(shot+q+shots),....

        query_indices = torch.from_numpy(~support_indices) #shots:2*shots , (shot+q+shots):4*shots,....
        support_indices = torch.from_numpy(support_indices) #0:shots,(shot+q):(shot+q+shots),....
        
        embeddings = learner(data)
        
        support = embeddings[support_indices]
        support1 = support.reshape(ways,shots,-1)
        support = support1.mean(dim = 1) #(ways,dim)    
        query   = embeddings[query_indices]  #(n_query,dim)\

        labels  = labels[query_indices].long()

        logits = self.euclidean_metric(query,support)

        error = loss_fun(logits,labels)
        acc = accuracy(logits,labels)

        return error,acc

    def model_save(self, path):
        filename = path+'(1)' if os.path.exists(path) else path
        torch.save(self.model.state_dict(), filename)
        print(f'Save model at: {filename}')        

    def train(self,save_path,shots):
        train_ways = valid_ways = self.ways
        query_num = 1
        print(f"{train_ways}--ways,{shots}--shots for training...")
        train_task = self.bulid_tasks('train',train_ways,shots,1000,None)
        valid_tasks = self.bulid_tasks('valid',valid_ways,shots,100,None)

        self.model.apply(weights_init2)
        optimizer = torch.optim.Adam(self.model.parameters(),lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.95)  #查一下
        loss_fun = torch.nn.CrossEntropyLoss()

        Epochs = 200
        Episodes = 100
        counter = 0
        
        for ep in range(Epochs):
            
            #training:
            t0 = time.time()
            self.model.train()
            meta_train_error = 0.0
            meta_train_accuracy = 0.0

            for epi in range(Episodes):
                batch = train_task.sample()
                loss,acc = self.fast_adapt(batch,self.model,loss_fun,query_num,shots,train_ways)

                meta_train_error += loss.item()
                meta_train_accuracy += acc.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            t1 = time.time()
            print(f'*** Time /epoch {t1-t0:.4f} ***')
            print(f'epoch {ep+1}, train, loss: {meta_train_error/Episodes:.4f}, '
                  f'acc: {meta_train_accuracy/Episodes:.4f}')

            #validation
            self.model.eval()
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0
            for i,batch in enumerate(valid_tasks):
                with torch.no_grad():
                    loss,acc = self.fast_adapt(batch,self.model,loss_fun,query_num,shots,train_ways)
                meta_valid_error += loss.item()
                meta_valid_accuracy += acc.item()

            print(f'epoch {ep + 1}, validation, loss: {meta_valid_error / len(valid_tasks):.4f}, '
                    f'acc: {meta_valid_accuracy / len(valid_tasks):.4f}\n')

            counter +=1

            if (ep+1) >=100 and (ep+1)%2 == 0:
                if input('\n == Stop training? ==(y/n)\n').lower() == 'y':
                    new_save_path  = save_path +rf'_ep{ep + 1}'
                    self.model_save(new_save_path)
                    break
                elif input('\n == Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'



    def test(self,load_path,shots):
        self.model.load_state_dict(torch.load(load_path))
        print(f'Load Model successfully from [{load_path}]...')

        test_ways = self.ways 
        query_num = 1
        print(f"{test_ways}--ways, {shots}--shots for testing ...") 
        test_tasks = self.bulid_tasks('test',test_ways,shots,1000,None)
        loss_fun = torch.nn.CrossEntropyLoss()
        
        #validation
        self.model.eval()              
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        t0 = time.time()

        for i,batch in enumerate(test_tasks):
            with torch.no_grad():
                loss,acc = self.fast_adapt(batch,self.model,loss_fun,query_num,shots,test_ways)
            meta_valid_error += loss.item()
            meta_valid_accuracy += acc.item()
        
        t1 = time.time()
        print(f"*** Time for {len(test_tasks)} tasks {t1 - t0:.4f}(s)")
        print(f'Testing, loss: {meta_valid_error / len(test_tasks):.4f}, '
              f'acc: {meta_valid_accuracy / len(test_tasks):.4f}')



if __name__ == "__main__":
    from train_utils import seed_torch
    seed_torch(2021)
    way  = 7 # 3 5 7 
    shot  = 20 #1 3 7 15 20 
    Net = ProtoNet_learner(ways = way)



    csv_file_path_Train = 'E:\\Milk\\averspectra\\1-D\\niu_tuo_1.csv'
    # csv_file_path_Train = 'F:\\data\\汇总\\FT_0\\freeze\\0.csv'
    csv_file_path_Test = 'E:\\Milk\\averspectra\\1-D\\test.csv'
    csv_file_path_fine_tune = 'E:\\Milk\\averspectra\\processing\\Train\\Fine_tuning\\FT_1.csv'

    # path = r"E:\模型保存\milk\protonet\ProtoNet_tomato_ways"+str(way)+"_shot "+ str(shot) +" "
    # Net.train(save_path = path ,shots = shot)

    for i in range(10):
        print(i,'\n')
        Net.test(load_path=r"E:\模型保存\milk\protonet\ProtoNet_milk_ways"+str(way)+"_shot "+ str(shot) +" _ep200",shots=i+1) 

    # Net.fine_tune_test(load_path=r"E:\Paper\savepath\5shot\MAML\ProtoNet_FT0_shot2_ep200",shots=1)        

