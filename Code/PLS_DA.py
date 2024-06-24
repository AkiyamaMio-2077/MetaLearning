import numpy
from sklearn import datasets
import sklearn.svm as SVM
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split as ts
import torch 
from sklearn.model_selection import GridSearchCV
import time
import numpy as np
import pandas as pd 

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

# csv_file_path_Train = 'C:\\Users\\soft\\Desktop\\论文准备\\奶粉\\说明数据可分离性\\yang_tuo_new.csv'
# csv_file_path_Train = 'C:\\Users\\soft\\Desktop\\论文准备\\奶粉\\说明数据可分离性\\niu_tuo(train).csv'
# csv_file_path_Test = 'C:\\Users\\soft\\Desktop\\论文准备\\奶粉\\说明数据可分离性\\yang_tuo(test).csv'

csv_file_path_Train = 'E:\\Milk\\averspectra\\processing\\right\\SNV\\PCA.csv'
csv_file_path_Test = 'E:\\Milk\\averspectra\\processing\\right\\SNV\\yang_tuo.csv'

TrainData  =  SVM_dataset(csv_file_path_Train,csv_file_path_Test,mode = 'train')
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

# train_label = pd.get_dummies(train_label)

#下面为测试程序
data = train_feature
labels = train_label
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.66,test_size=0.34, random_state=42)

train_feature = X_train
test_feature = X_test
train_label = y_train
test_label = y_test

for x in range(1,50):
    PLS = PLSRegression(n_components=x)             #n_components取多少个主成分
    PLS.fit(train_feature,train_label)


    result = PLS.predict(test_feature)

    result = numpy.array([numpy.argmax(i) for i in result])

    accuracy = accuracy_score(test_label,result)
    print('____________________________________')
    print(x)
    print("accuracy-score: {0:.4f}".format(accuracy_score(test_label,result)))
    print("F-score: {0:.4f}".format(f1_score(result,test_label,average='micro')))

    y_label = []
    y_label = test_label
    y_pred = []
    y_pred = result
    import sklearn.metrics as metrics
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import classification_report
    import warnings
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

    # j = 0
    # for i in range(len(y_label)):
    #     if y_label[i] != j:
    #         y_label[i]= 1

    # for i in range(len(y_pred)):
    #     if y_pred[i] != j:
    #         y_pred[i]= 1

    # total_samples = len(y_pred)
    # correct_predictions = sum(1 for y_pred, y_label in zip(y_pred,y_label ) if y_pred == y_label)
    # accuracy = correct_predictions / total_samples * 100
    # print(f"Accuracy: {accuracy}%\n")