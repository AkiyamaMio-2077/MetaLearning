import numpy
from sklearn import datasets
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split as ts
from sklearn.model_selection import KFold
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

# csv_file_path_Train = 'F:\\data\汇总\\FT_16\\FT\\0.csv'
# csv_file_path_Test = 'F:\\data\\汇总\\FT_16\\freeze\\0.csv'


# csv_file_path_Train = 'C:\\Users\\soft\\Desktop\\论文准备\\奶粉\\说明数据可分离性\\reduced_data_with_labels.csv'
# # csv_file_path_Train = 'C:\\Users\\soft\\Desktop\\论文准备\\奶粉\\说明数据可分离性\\niu_tuo(train).csv'
# csv_file_path_Test = 'C:\\Users\\soft\\Desktop\\论文准备\\奶粉\\说明数据可分离性\\yang_tuo(test).csv'

csv_file_path_Train = 'E:\\Milk\\averspectra\\processing\\right\\SNV\\yang_tuo_all.csv'
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

#下面为测试程序
data = train_feature
labels = train_label
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.34,test_size=0.66, random_state=42)

train_feature = X_train
test_feature = X_test
train_label = y_train
test_label = y_test


kf = KFold(n_splits=2)

LR = LogisticRegressionCV(cv = 5,multi_class="multinomial", penalty='l2', class_weight='balanced',tol = 0.0001,max_iter=10000)
param_grid = {'Cs': [1000],'solver':['lbfgs','newton-cg','sag','saga']} 



grid_search = GridSearchCV(LR, param_grid, n_jobs=6, verbose=1)

grid_search.fit(train_feature,train_label)
best_parameters = grid_search.best_estimator_.get_params()  # 获取最佳模型中的最佳参数
print("best parameters are" % grid_search.best_params_, grid_search.best_params_)  # grid_search.best_params_:已取得最佳结果的参数的组合；
print("best score are" % grid_search.best_params_, grid_search.best_score_)  # grid_search.best_score_:优化过程期间观察到的最好的评分。


best_model = grid_search.best_estimator_

result = best_model.predict(test_feature)

print("accuracy-score: {0:.2f}".format(accuracy_score(test_label,result)))
print("F-score: {0:.2f}".format(f1_score(result,test_label,average='micro')))