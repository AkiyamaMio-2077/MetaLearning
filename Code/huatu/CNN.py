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

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from captum.attr import IntegratedGradients 
from openpyxl import load_workbook
from sklearn.metrics import classification_report
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
    def __init__(self,ways = 11):
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

        self.fc = nn.Linear(64, 11)  # 假设输出类别数量为4

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.type(torch.cuda.FloatTensor)  # 增加维度，使其变为(Batch, Channels, Sequence Length)
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
        y = x
        x = self.fc(x)
        return x,y

batch_size = 2
learning_rate = 0.001
num_epochs = 100

model = ConvNet()

class CNN_learner(object):
    def __init__(self):
        self.model = ConvNet().to(device)


    def build_data(self,mode = 'train',batch_size = 10 ):
        dataset = MetaDataset(csv_file_path_Train,csv_file_path_Test,csv_file_path_fine_tune,mode = mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader

    def fast_adapt(self,batch,learner,loss_fun):
        data,labels,name = batch
        data,labels = data.to(device),labels.to(device)

        predication,embeddings = learner(data)

        acc  = accuracy(predication,labels)
        error = loss_fun(predication,labels)

        predication  = predication.argmax(dim =-1).view(labels.shape)
        labels = labels.cpu().detach().numpy()
        predictions = predication
        predications1 = predictions.cpu().detach().numpy()
        embeddings  = embeddings.cpu().detach().numpy()

        return error,acc,predications1,labels,embeddings


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
        y_pred =[]
        y_label = []
        for ep in range(Epochs):
            t0 = time.time()
            self.model.train()

            train_error = 0.0
            train_accuracy = 0.0
            
            for train in train_data:
                batch = train
                loss,acc,predications,Labels = self.fast_adapt(batch,self.model,loss_fun)
                y_pred1 = predications
                y_pred.extend(y_pred1)

                y_label1 = Labels
                y_label.extend(y_label1)


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
                    loss ,acc,predications,Labels = self.fast_adapt(batch,self.model,loss_fun)
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


    def fine_tune_test(self,load_path):

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
                loss,acc,predications,Labels = self.fast_adapt(batch,self.model,loss_fun)

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

        self.model.eval()
        test_error = 0.0
        test_accuracy =0.0
        t0 = time.time()

        y_pred =[]
        y_label = []
        y_embeddings = []

        for i,batch in enumerate(test_data):
            with torch.no_grad():
                loss,acc,predications,Labels = self.fast_adapt(batch,self.model,loss_fun)
                
                y_pred1 = predications
                y_pred.extend(y_pred1)

                y_label1 = Labels
                y_label.extend(y_label1)

            test_error += loss.item()
            test_accuracy += acc.item()
            
            t1 = time.time()
        print("真正的")
        print(f"*** Time for {len(fine_tune_data)} tasks {t1 - t0:.4f}(s)")
        print(f'Testing, loss: {test_error / len(test_data):.4f}, '
            f'acc: {test_accuracy / len(test_data):.4f}')
    #此处是构建混淆矩阵
        for i in range(len(y_label)):
            if y_label[i] != 0:
                y_label[i]= 1

        for i in range(len(y_pred)):
            if y_pred[i] != 0:
                y_pred[i]= 1
        total_samples = len(y_pred)
        correct_predictions = sum(1 for y_pred, y_label in zip(y_pred,y_label ) if y_pred == y_label)
        accuracy = correct_predictions / total_samples * 100
        print(f"Accuracy: {accuracy}%")

        
        # cm = confusion_matrix(y_label, y_pred)

        # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # # cm_normalized = cm.astype('float') 
        # cm_normalized = np.round(cm_normalized,2)
        # cm_xtick = ['0','1','2','3','4','5']
        # cm_ytick = ['0','1','2','3','4','5']

        # sns.heatmap(cm_normalized,fmt='g', cmap='Blues',annot=True,cbar=False,xticklabels=cm_xtick, yticklabels=cm_ytick)
        # h=sns.heatmap(cm_normalized,fmt='g', cmap='Blues',annot=True,cbar=False) #画热力图，设置cbar=Falese
        # cb=h.figure.colorbar(h.collections[0]) #显示colorbar
        # plt.xlabel('True Label')
        # plt.ylabel('Predict Label')
        # plt.show()
        y_embeddings = []
    def test(self,load_path):
        self.model.load_state_dict(torch.load(load_path))
        print(f'Load Model successfully from [{load_path}]...')
        test_data = self.build_data(mode='test', batch_size = 10)
        loss_fun = torch.nn.CrossEntropyLoss()
        self.model.eval()

        valid_error = 0.0
        valid_accuracy = 0.0

        y_pred =[]
        y_label = []
        y_embeddings = []
        for i,batch in enumerate(test_data):
            with torch.no_grad():
                loss ,acc,predications,labels,embeddings = self.fast_adapt(batch,self.model,loss_fun)
                y_pred1 = predications
                y_pred.extend(y_pred1)

                y_label1 = labels
                y_label.extend(y_label1)

                y_embeddings1 = embeddings
                y_embeddings.extend(y_embeddings1)

            valid_error      += loss.item()
            valid_accuracy   +=acc.item()

        print(f'validation, loss: {valid_error / len(test_data):.4f}, '
            f'acc: {valid_accuracy / len(test_data):.4f}\n')

        cm = confusion_matrix(y_label, y_pred)

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # cm_normalized = cm.astype('float') 
        cm_normalized = np.round(cm_normalized,4)
        cm_xtick = ['0','1','2','3','4','5']
        cm_ytick = ['0','1','2','3','4','5']

        sns.heatmap(cm_normalized,fmt='g', cmap='Blues',annot=True,cbar=False,xticklabels=cm_xtick, yticklabels=cm_ytick)
        h=sns.heatmap(cm_normalized,fmt='g', cmap='Blues',annot=True,cbar=False) #画热力图，设置cbar=Falese
        cb=h.figure.colorbar(h.collections[0]) #显示colorbar
        plt.xlabel('True Label')
        plt.ylabel('Predict Label')
        # plt.show()
        print(classification_report(y_label, y_pred,digits = 4))

        diagonal_values = np.diag(cm_normalized)
        print(diagonal_values)

        cm = confusion_matrix(y_label, y_pred)

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # cm_normalized = cm.astype('float') 
        cm_normalized = np.round(cm_normalized,2)
        cm_xtick = ['0','1','2','3','4','5']
        cm_ytick = ['0','1','2','3','4','5']

        sns.heatmap(cm_normalized,fmt='g', cmap='viridis',annot=True,cbar=False,xticklabels=cm_xtick, yticklabels=cm_ytick)
        h=sns.heatmap(cm_normalized,fmt='g', cmap='viridis',annot=True,cbar=False) #画热力图，设置cbar=Falese
        cb=h.figure.colorbar(h.collections[0]) #显示colorbar
        plt.xlabel('True Label')
        plt.ylabel('Predict Label')
        # plt.show()

        tsne = TSNE(n_components=3,n_iter=2000)

        X_tsne_2d = tsne.fit_transform(y_embeddings)
        marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        class_list = [0,1,2,3,4,5,6,7,8,9,10]
        name = ['G10%','G20%','G30%','G40%','G50%','G60%','G70%','G80%','G90%','G100%','Ca100%']

        n_class = len(class_list) # 测试集标签类别数
        palette = sns.hls_palette(n_class) # 配色方案
        sns.palplot(palette)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        y_label = np.array(y_label)
        for idx,fruit in enumerate(class_list):
            # 获取颜色和点型
            color = palette[idx]
            marker = marker_list[idx%len(marker_list)]

            # 找到所有标注类别为当前类别的图像索引号
            indices = np.where(y_label == fruit)
            ax.scatter(X_tsne_2d[indices, 0], X_tsne_2d[indices, 1], X_tsne_2d[indices, 2],color=color, marker=marker, label=name[idx])



        # ax.set_xlabel()
        # ax.set_ylabel()
        # ax.set_zlabel()
        ax.legend(loc='best', title='Classes')
        # ax.set_title('Sparse PCA of Data (3D)')

        plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
        plt.show()





        # plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        # fig.savefig(r'C:\Users\soft\Desktop\论文准备\输出图片.svg',format='svg',dpi=150)


       

    def feature_importance(self,load_path):

        self.model.load_state_dict(torch.load(load_path))
        print(f'Load Model successfully from [{load_path}]...')
        test_data = self.build_data(mode='test', batch_size = 1)
        loss_fun = torch.nn.CrossEntropyLoss()
        self.model.eval()

        X_train  = []
        X_label = []
        # test_data = self.build_data(mode='test', batch_size = 10)


        for batch in test_data:
            inputs, labels, _ = batch
            X_train.append(inputs)
            X_label.append(labels)

        # X_train = np.array(X_train)
        X_train = np.vstack(X_train)

        # X_label = np.vstack(X_label)
        # X_train = X_train.to(device)

        # X_train = MetaDataset(csv_file_path_Train,csv_file_path_Test,csv_file_path_fine_tune,mode = 'test')


        feature_names = ["wave" + str(i) for i in range(21, 274)]
        self.model = self.model.to(device)
        # X_train = X_train.to(device)


        # X_train = torch.from_numpy(X_train).float()
        X_train= inputs.to(device)
        explainer = shap.DeepExplainer(self.model,X_train)
        sample_index = 0  # 选择一个样本
        A = X_train[sample_index]
        print(A)

        shap_values = explainer.shap_values(inputs[0])

        # 可视化特征贡献度
        shap.summary_plot(shap_values, X_train[sample_index], feature_names=feature_names)





    def feature_importance2(self,load_path):

        self.model.load_state_dict(torch.load(load_path))
        print(f'Load Model successfully from [{load_path}]...')
        test_data = self.build_data(mode='test', batch_size = 50)
        loss_fun = torch.nn.CrossEntropyLoss()
        self.model.eval()

        X_train  = []
        X_label = []
        # test_data = self.build_data(mode='test', batch_size = 10)


        for batch in test_data:
            inputs, labels, _ = batch
            X_train.append(inputs)
            X_label.append(labels)

        # X_train = np.array(X_train)
        # X_train = np.vstack(X_train)

        # X_train = torch.from_numpy(X_train).type(torch.FloatTensor)

        ig = IntegratedGradients(self.model)
        inputs = inputs.to(device)
        attr, delta = ig.attribute(inputs,target=1, return_convergence_delta=True)
        attr = attr.cpu().detach().numpy()

        feature_names = ["wave" + str(i) for i in range(21, 274)]

        importance= np.mean(attr,axis=0)
        print(importance)

        # 指定要保存到的Excel文件
        excel_file_path = r'C:\Users\soft\Desktop\123.xlsx'
        sheet_name1 = 'Sheet3'
        importance  = importance.flatten()
        importance = pd.DataFrame({'Diagonal Data': importance})
        # 将DataFrame保存到指定Excel文件
        with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a') as writer:
            writer.book = load_workbook(excel_file_path)
            importance.to_excel(writer, sheet_name = sheet_name1, startrow=1, header=False, index=False)        


        visualize_importances(feature_names, np.mean(attr, axis=0))




def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)











if __name__ == "__main__":
    from train_utils import seed_torch
    seed_torch(2021)
    Net = CNN_learner()

    csv_file_path_Train = 'E:\\Milk\\averspectra\\processing\\right\\SNV\\niu_tuo.csv'
    # csv_file_path_Train = 'F:\\data\\汇总\\FT_0\\freeze\\0.csv'
    csv_file_path_Test = 'E:\\Milk\\averspectra\\processing\\right\\SNV\\yang_tuo.csv'
    csv_file_path_fine_tune = 'E:\\Milk\\averspectra\\processing\\right\\SNV\\FT.csv'

    path = r"E:\Paper\savepath\5shot\CNN\CNN_FT20"
    # Net.train(save_path = path)
    # Net.test(load_path=r"E:\Paper\savepath\5shot\MAML\CNN_ep100")
    # Net.fine_tune_test(load_path=r"E:\Paper\savepath\5shot\CNN\CNN_FT20_ep150")     


    # Net.train(save_path = path)
    # Net.fine_tune_test(load_path=r"E:\模型保存\milk\CNN\CNN_milk_ep100") 
    Net.test(load_path=r"E:\模型保存\milk\CNN\CNN_milk_ep100_fine(1)") 
    # Net.feature_importance2(load_path=r"E:\模型保存\milk\CNN\CNN_milk_ep100_fine(1)") 
    # Net.test(load_path=r"E:\模型保存\milk\CNN\CNN_milk_ep100_fine(1)") 
    # Net.fine_tune_test(load_path=r"E:\Paper\savepath\5shot\MAML\ProtoNet_FT0_shot2_ep200",shots=1)    


