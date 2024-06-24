from sklearn import datasets
import torch 
from torch.utils import data 
from train_utils import accuracy
from train_utils import seed_torch

import learn2learn as l2l
import time
import numpy as np
import pandas as pd 
from init_utils import weights_init2
from RelationNet_Moudel import encoder_net, relation_net

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import recall_score,f1_score,accuracy_score,precision_score

import os
from init_utils import get_file,get_data_csv,normalization,sample_label_shuffle

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
            self.data = pd.read_csv(file_path_fine_tune,index_col = False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        label = self.data.iloc[idx, 0]  # 获取标签列
        name = self.data.iloc[idx, 1]  # 获取数据名称列
        sample = self.data.iloc[idx, 2:].values.astype(float)  # 获取数值列

        return sample,label,name




class RelationNet_leaner(object):
    def __init__(self,ways):
        self.feature = encoder_net(in_chn = 1, hidden_chn = 64,cb_num = 4).to(device)      #后续的hidden_channels和其他的参数在正是训练的时候要进行调参，这个64感觉肯定是不行的
        embed_size = (261//2**4)//2**2*64
        self.relation = relation_net(hidden_chn=64,embed_size=embed_size,h_size=256).to(device)
        self.ways = ways

    def bulid_tasks(self,mode = 'train',ways = 10,shots = 5,num_tasks = 100,filter_labels = None):   #这里面有用于循环的test1
        meta_dataset = l2l.data.MetaDataset(MetaDataset(csv_file_path_Train,csv_file_path_Test,csv_file_path_fine_tune,mode = mode))
        new_ways = len(filter_labels) if filter_labels is not None else ways


        assert shots* new_ways <= meta_dataset.__len__() // ways * new_ways, "Reduce the number of shots!"

        tasks = l2l.data.TaskDataset(meta_dataset,task_transforms = [
            l2l.data.transforms.FusedNWaysKShots(meta_dataset,new_ways,shots+4,filter_labels = filter_labels), #如果这里2*shots改成shots
            # l2l.data.transforms.FusedNWaysKShots(dataset,new_ways,shots,filter_labels = filter_labels),
            l2l.data.transforms.LoadData(meta_dataset),
            # l2l.data.transforms.RemapLabels(meta_dataset,shuffle = True),
            # do not keep the original labels, use (0 ,..., n-1);
            l2l.data.transforms.ConsecutiveLabels(meta_dataset),
            ],num_tasks = num_tasks)

        return tasks


    #将数据分为adaptation set 和evaluation set
    def fast_adapt(self,batch,loss_fun,query_num,shots,ways):
        data,labels,name = batch

        forclusterlist = labels
        le = LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        labels = torch.from_numpy(labels)


        data,labels = data.to(device).unsqueeze(1),labels.to(device)

        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)

        support_indices = np.zeros(data.size(0),dtype = bool)

        selection = np.arange(ways)*(shots + query_num)           # 0, shot+q, 2*(shot+q), 3*(), ...
        for offset in range(shots):                             
            support_indices[selection + offset]  = True           # 0:shots, (shot+q):(shot+q+shots), ...
        
        query_indices = torch.from_numpy(~support_indices)        # shots:2*shots, (shot+q+shots):4*shots, ...
        support_indices = torch.from_numpy(support_indices)       # 0:shots, (shot+q):(shot+q+shots), ...
        
        embeddings = self.feature(data)                            
        support = embeddings[support_indices]                     # (n_support, chn, length)
        query  = embeddings[query_indices]                        # (n_query, chn, length)
        labels = labels[query_indices].long()                     # (n_query)

        query1 = query
        
        support  = support.reshape(ways,shots,*support.shape[-2:])
        support  = support.mean(dim = 1) 

        support  = support.unsqueeze(0).repeat(query.shape[0],1,1,1)   # (n_q, ways, chn, length)
        query = query.unsqueeze(1).repeat(1,ways,1,1)             # (n_q, ways, chn, length)
        

        relation_pairs = torch.cat((support,query),2).reshape(query.shape[0]*ways,-1,query.shape[-1])
        scores  = self.relation(relation_pairs).reshape(-1,ways)   # (n_q, ways)
        error = loss_fun(scores,labels)
        acc = accuracy(scores,labels)

        #下面为做混淆矩阵做的改动，直接把accuracy 里面结构拿出来，得到预测值
        logits = scores

        tragets = labels
        predictions = logits
        predictions = predictions.argmax(dim =-1).view(tragets.shape)
        acc = (predictions == tragets).sum().float()/ tragets.shape[0]

        predictions = predictions.data.cpu()
        tragets = tragets.data.cpu().numpy()






        predictions = predictions.cpu().detach().numpy()

        predictions = le.inverse_transform(predictions)
        labels = le.inverse_transform(labels.cpu().detach().numpy())

        tragets = labels

        recall = recall_score(tragets,predictions,average='weighted')
        F1 = f1_score(tragets,predictions,average='macro')
        precision = precision_score(tragets,predictions,average='weighted')

        query1 = query1.cpu().detach().numpy().reshape(-1,1024)

        return error,acc,recall,F1,precision,predictions,labels,query1
    


    def model_save(self, path):
        filename = path+'(1)' if os.path.exists(path) else path
        state_dict = {
            'feature': self.feature.state_dict(),
            'relation': self.relation.state_dict(),
        }
        torch.save(state_dict, filename)
        print(f'Save model at: {filename}')

    def train(self,save_path,shots):
        train_ways = valid_ways  = self.ways
        query_num = 4
        print(f"{train_ways}--ways,{shots}--shots for training...")
        train_tasks = self.bulid_tasks('train',train_ways,shots,50,None)
        valid_tasks = self.bulid_tasks('valid',valid_ways,shots,50,None)

        self.feature.apply(weights_init2)
        self.relation.apply(weights_init2)

        optimizer_f = torch.optim.Adam(self.feature.parameters(),lr = 0.0001,weight_decay=2e-4)
        optimizer_r = torch.optim.Adam(self.relation.parameters(),lr = 0.001,weight_decay=2e-4)

        lr_scheduler_r = torch.optim.lr_scheduler.ExponentialLR(optimizer_r,gamma = 0.99)
        loss_fun = torch.nn.CrossEntropyLoss()

        Epochs = 92
        Episodes = 150
        counter  = 0


        for ep in range(Epochs):
            #(1)Training
            t0 = time.time()
            self.feature.train(),self.relation.train()

            meta_train_error = 0.0
            meta_train_accuracy = 0.0

            for epi in range(Episodes):
                batch = train_tasks.sample()
                loss,acc,predications,Labels,embeddings = self.fast_adapt(batch,loss_fun,query_num,shots,train_ways)
                meta_train_error +=loss.item()
                meta_train_accuracy += acc.item()

                optimizer_f.zero_grad()
                optimizer_r.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.feature.parameters(),0.5)
                torch.nn.utils.clip_grad_norm_(self.relation.parameters(),0.5)

                optimizer_f.step()
                optimizer_r.step()    
            
            lr_scheduler_r.step()
            
            t1 = time.time() 
            print(f'*** Time /epoch {t1-t0:.3f} ***')
            print(f'epoch {ep+1}, train, loss: {meta_train_error/Episodes:.3f}, '
                  f'acc: {meta_train_accuracy/Episodes:.3f}')


            self.feature.eval(),self.relation.eval()
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0

            for i ,batch in enumerate(valid_tasks):
                with torch.no_grad():
                    loss,acc,predications,Labels,embeddings = self.fast_adapt(batch,loss_fun,query_num,shots,train_ways)
                meta_valid_error += loss.item()
                meta_valid_accuracy += acc.item()

            print(f'epoch {ep + 1}, validation, loss: {meta_valid_error / len(valid_tasks):.4f}, '
                  f'acc: {meta_valid_accuracy / len(valid_tasks):.4f}\n')
            counter +=1

            if (ep+1) >=92 and (ep+1)%2==0:  
                if input('\n== Stop training? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)
                    break
                elif input('\n== Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)


    def test(self,load_path,shots): #这里面有用于循环的test1
        state_dict = torch.load(load_path)

        self.feature.load_state_dict(state_dict['feature'])
        self.relation.load_state_dict(state_dict['relation'])
        print(f'Load Model successfully from [{load_path}]....')

        test_ways = self.ways
        query_num = 4

        print(f"{test_ways}--ways,{shots}--shots for testing... ")
        test_tasks = self.bulid_tasks('test',test_ways,shots,1000,None) #这里面有用于循环的test1
        loss_fun = torch.nn.CrossEntropyLoss()

        #验证
        self.feature.eval(),self.relation.eval()
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_F1 = 0.0
        meta_recall = 0.0
        meta_precision = 0.0
        t0 = time.time()

        y_pred =[]
        y_label = []
        y_embeddings = []

        for i,batch in enumerate(test_tasks):  
            with torch.no_grad():
                
                loss,acc,recall,F1,precision,predications,Labels,embeddings = self.fast_adapt(batch,loss_fun,query_num,shots,test_ways)

                y_pred1 = predications
                y_pred.extend(y_pred1)

                y_label1 = Labels
                y_label.extend(y_label1)


                y_embeddings1 = embeddings
                y_embeddings.extend(y_embeddings1)

                meta_valid_error +=loss.item()
                meta_valid_accuracy += acc.item()
                meta_F1 += F1.item()
                meta_recall += recall.item()
                meta_precision += precision.item()


        t1 = time.time()

        print(f"*** Time for {len(test_tasks)} tasks :{t1 - t0:.4f}(s)")
        print(f'Testing,loss:{meta_valid_error / len(test_tasks):.4f},\n'
              f'acc:{meta_valid_accuracy / len(test_tasks):.4f}\n'
              f'f1:{meta_F1 / len(test_tasks):.4f}\n'
              f'recall:{meta_recall / len(test_tasks):.4f}\n'
              f'precision:{meta_precision / len(test_tasks):.4f}')
        a = 1


    
        #         #此处是构建混淆
        #这段代码是重新定义标签的代码
        # for i in range(len(y_label)):
        #     if y_label[i] != 0:
        #         y_label[i]= 1

        # for i in range(len(y_pred)):
        #     if y_pred[i] != 0:
        #         y_pred[i]= 1
        # total_samples = len(y_pred)
        # correct_predictions = sum(1 for y_pred, y_label in zip(y_pred,y_label ) if y_pred == y_label)
        # accuracy = correct_predictions / total_samples * 100
        # print('下面这个是2分类时的精度等信息\n')
        # print(f"Accuracy: {accuracy}%")

        
        cm = confusion_matrix(y_label, y_pred)

        # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #这个是百分比的
        cm_normalized = cm.astype('float')  #这个是单纯数字的
        cm_normalized = np.round(cm_normalized,2)
        cm_xtick = ['0','1','2','3','4','5']
        cm_ytick = ['0','1','2','3','4','5']

        sns.heatmap(cm_normalized,fmt='g', cmap='Blues',annot=True,cbar=False,xticklabels=cm_xtick, yticklabels=cm_ytick)
        h=sns.heatmap(cm_normalized,fmt='g', cmap='Blues',annot=True,cbar=False) #画热力图，设置cbar=Falese
        cb=h.figure.colorbar(h.collections[0]) #显示colorbar
        plt.xlabel('True Label')
        plt.ylabel('Predict Label')
        plt.show()




        
        # #此处是Tsne可视化
        # tsne = TSNE(n_components=2,n_iter=2000)
        # X_tsne_2d = tsne.fit_transform(y_embeddings)
        # marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # # classes_list = ['干旱','低温','正常','盐碱']
        # # class_list = ['level_1','level_2','level_3','level_4','level_5','level_6']
        # # class_list = ['0','1','2','3','4','5']
        # class_list = [0,1,2,3]
        # n_class = len(class_list) # 测试集标签类别数
        # palette = sns.hls_palette(n_class) # 配色方案
        # sns.palplot(palette)
        
        # fig = plt.figure(figsize=(14, 14))
        # y_label = np.array(y_label)
        # for idx,fruit in enumerate(class_list):
        #     # 获取颜色和点型
        #     color = palette[idx]
        #     marker = marker_list[idx%len(marker_list)]

        #     # 找到所有标注类别为当前类别的图像索引号
        #     indices = np.where(y_label == fruit)
        #     plt.scatter(X_tsne_2d[indices, 0], X_tsne_2d[indices, 1], color=color, marker=marker, label=fruit, s=150, clip_on=False)
            
        # plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()





if __name__ == '__main__':
    from train_utils import seed_torch
    seed_torch(2021)

    csv_file_path_Train = 'F:\\data\\汇总\\FT_0\\drought\\0.csv'
    csv_file_path_Test = 'F:\\data\\汇总\\FT_0\\freeze\\0.csv'
    csv_file_path_fine_tune = 'F:\\data\\汇总\\FT_20\\FT\\0.csv'


    Net = RelationNet_leaner(ways=3) #             
    path = r"E:\Paper\savepath\1shot\RelationNet\RelationNet_3"
    # Net.train(save_path = path ,shots = 2)

    # # load_path = r"E:\Paper\savepath\5shot\MAML\T3_ep50" #后续上面这个两个path要改，目前这两个路径之哦是为了写程序使用
    # Net.test(load_path=r"E:\Paper\savepath\1shot\RelationNet\RelationNet_3_ep100",shots = 5)  

    for i in range(5):
        Net.test(load_path=r"E:\Paper\savepath\1shot\RelationNet\RelationNet_3_ep100",shots = i+1)  

    # Net.fine_tune_test(load_path=r"E:\Paper\savepath\5shot\MAML\RelationNet_FT20_ep100",shots = 1)
