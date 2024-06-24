import numpy
from sklearn import datasets
import sklearn.svm as SVM
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split as ts
import torch 
from sklearn.model_selection import GridSearchCV
import time
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import os
from init_utils import get_file , get_data_csv,normalization,sample_label_shuffle

class SVM_dataset(torch.utils.data.Dataset):
    def __init__(self,file_path_train,file_path_Test,mode):
        if mode == 'train':
            self.data = pd.read_csv(file_path_train,index_col=False )
        if mode == 'valid':
            self.data = pd.read_csv(file_path_train,index_col=False)
        if mode == 'test':
            self.data = pd.read_csv(file_path_Test,index_col=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        label = self.data.iloc[idx, 0]  # 获取标签列
        name = self.data.iloc[idx, 1]  # 获取数据名称列
        sample = self.data.iloc[idx, 2:].values.astype(float)  # 获取数值列

        return sample,label,name


FT_num = "FT_1"
# csv_file_path_Train = 'E:\\Milk\\averspectra\\1-D\\niu.csv'
# csv_file_path_Test = 'E:\\Milk\\averspectra\\1-D\\yang.csv'

# csv_file_path_Train = 'E:\\Milk\\averspectra\\1-D\\fine_tuning\\'+ FT_num +'\\train.csv'
# csv_file_path_Test = 'E:\\Milk\\averspectra\\1-D\\fine_tuning\\'+ FT_num +'\\test.csv'

csv_file_path_Train = 'E:\\Milk\\averspectra\\processing\\right\\SNV\\yang_tuo_all.csv'
csv_file_path_Test = 'E:\\Milk\\averspectra\\processing\\right\\SNV\\yang_tuo.csv'

# csv_file_path_Train = 'C:\\Users\\soft\\Desktop\\论文准备\\奶粉\\说明数据可分离性\\yang_tuo_all.csv'
# csv_file_path_Test = 'C:\\Users\\soft\\Desktop\\论文准备\\奶粉\\说明数据可分离性\\FT.csv'
TrainData  =  SVM_dataset(csv_file_path_Train,csv_file_path_Train,mode = 'train')
TestData   =  SVM_dataset(csv_file_path_Train,csv_file_path_Test,mode = 'test')

train_feature = []
train_label   = []
test_feature  = []
test_label    = []

# 遍历训练集数据加载器，收集数据和标签
for batch in TrainData :
    inputs, labels, _ = batch
    train_feature.append(inputs)
    train_label.append(labels)

for batch in TestData :
    inputs, labels, _ = batch
    test_feature.append(inputs)
    test_label.append(labels)

# 将数据和标签转换为numpy数组
train_feature = np.array(train_feature)
train_label = np.array(train_label)

test_feature = np.array(test_feature)
Test_labels = np.array(test_label)

# 将数据展平为二维数组
train_feature = train_feature.reshape(train_feature.shape[0], -1)
test_feature = test_feature.reshape(test_feature.shape[0], -1)

#下面为测试程序
data = train_feature
labels = train_label
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.34,test_size=0.66, random_state=42)

train_feature = X_train
test_feature = X_test
train_label = y_train
test_label = y_test
t0 = time.time()

for i in range(50):

    scaler = StandardScaler() # 标准化转换

    scaler.fit(test_feature)  # 训练标准化对象
    test_feature = scaler.transform(test_feature)   # 转换数据集

    scaler.fit(train_feature)  # 训练标准化对象
    train_feature = scaler.transform(train_feature)   # 转换数据集

    bp=MLPClassifier(hidden_layer_sizes=(100,50,25), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant')
    # bp=MLPClassifier(hidden_layer_sizes=(180,120,80,50), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant')
    bp.fit(train_feature,train_label)

    y_predict=bp.predict(test_feature)

    # print("BP神经网络准确度\n",classification_report(test_label,y_predict))
    print("BP神经网络准确度:{0:.4f}".format(accuracy_score(test_label,y_predict)))
    print("F-score: {0:.4f}".format(f1_score(test_label,y_predict,average='macro')))
    
    
    result = y_predict


    
    import sklearn.metrics as metrics
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import classification_report
    import warnings
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    warnings.filterwarnings("ignore")
    print("accuracy-score: {0:.4f}".format(accuracy_score(test_label,result)))
    print("F-score: {0:.4f}".format(f1_score(test_label,result,average='macro')))
    # precision_recall_fscore_support(test_label,result,average='macro')
    print(precision_recall_fscore_support(test_label,result,average='macro'))
    print(classification_report(test_label,result,digits = 4))

    cm= metrics.confusion_matrix(test_label,result)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.round(cm_normalized,4)
    diagonal_values = np.diag(cm_normalized)
    print(diagonal_values)
    print('----------------------------------------------------')
    y_label = []
    y_label = test_label
    y_pred = []
    y_pred = y_predict



    # # cm_normalized = cm.astype('float') 
    # cm_normalized = np.round(cm_normalized,2)
    # cm_xtick = ['0','1','2','3','4','5']
    # cm_ytick = ['0','1','2','3','4','5']

    # sns.heatmap(cm_normalized,fmt='g', cmap='viridis',annot=True,cbar=False,xticklabels=cm_xtick, yticklabels=cm_ytick)
    # h=sns.heatmap(cm_normalized,fmt='g', cmap='viridis',annot=True,cbar=False) #画热力图，设置cbar=Falese
    # cb=h.figure.colorbar(h.collections[0]) #显示colorbar
    # plt.xlabel('True Label')
    # plt.ylabel('Predict Label')
    # plt.show()



    # for i in range(len(y_label)):
    #     if y_label[i] != 0:
    #         y_label[i]= 1

    # for i in range(len(y_pred)):
    #     if y_pred[i] != 0:
    #         y_pred[i]= 1

    # total_samples = len(y_pred)
    # correct_predictions = sum(1 for y_pred, y_label in zip(y_pred,y_label ) if y_pred == y_label)
    # accuracy = correct_predictions / total_samples * 100
    # print(f"Accuracy: {accuracy}%\n")

t1 = time.time()
print(f"*** time takes  {t1 - t0:.4f}(s)")