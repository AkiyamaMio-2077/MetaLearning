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

FT_num = "FT_20"
# csv_file_path_Train = 'E:\\Milk\\averspectra\\1-D\\niu.csv'
# csv_file_path_Test = 'E:\\Milk\\averspectra\\1-D\\fine_tuning\\'+ FT_num +'\\test.csv'

# csv_file_path_Train = 'E:\\Milk\\averspectra\\1-D\\fine_tuning\\'+ FT_num +'\\train.csv'0
# csv_file_path_Test = 'E:\\Milk\\averspectra\\1-D\\fine_tuning\\'+ FT_num +'\\test.csv'

csv_file_path_Train = 'E:\\Milk\\averspectra\\processing\\right\\SNV\\yang_tuo_all.csv'
csv_file_path_Test = 'E:\\Milk\\averspectra\\processing\\right\\SNV\\FT.csv'

# csv_file_path_Train = 'C:\\Users\\soft\\Desktop\\论文准备\\奶粉\\说明数据可分离性\\yang_tuo_all.csv'
# csv_file_path_Test = 'C:\\Users\\soft\\Desktop\\论文准备\\奶粉\\说明数据可分离性\\FT.csv'

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
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size = 0.34, test_size=0.66, random_state=42)

train_feature = X_train
test_feature = X_test
train_label = y_train
test_label = y_test


# 调参代码
# svc_model = SVC(kernel='rbf')
clf = SVM.SVC(kernel="rbf")
# param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  # param_grid:我们要调参数的列表(带有参数名称作为键的字典)，此处共有14种超参数的组合来进行网格搜索，进而选择一个拟合分数最好的超平面系数。
param_grid = {'C': range(10,2000,5), 'gamma': [0.001,0.01, 0.1,1,5,10]}
grid_search = GridSearchCV(clf, param_grid, n_jobs=6, verbose=1,cv=2)  # n_jobs:并行数，int类型。(-1：跟CPU核数一致；1:默认值)；verbose:日志冗长度。默认为0：不输出训练过程；1：偶尔输出；>1：对每个子模型都输出。

# grid_search.fit(X_train, y_train.astype(int).astype(float).ravel())  # 训练，默认使用5折交叉验证 
grid_search.fit(train_feature,train_label)
best_parameters = grid_search.best_estimator_.get_params()  # 获取最佳模型中的最佳参数
# print("cv results are" % grid_search.best_params_, grid_search.cv_results_)  # grid_search.cv_results_:给出不同参数情况下的评价结果。
print("best parameters are" % grid_search.best_params_, grid_search.best_params_)  # grid_search.best_params_:已取得最佳结果的参数的组合；
print("best score are" % grid_search.best_params_, grid_search.best_score_)  # grid_search.best_score_:优化过程期间观察到的最好的评分。

t0 = time.time()
best_model = grid_search.best_estimator_

result = best_model.predict(test_feature)
t1 = time.time()

print(f"*** time takes  {t1 - t0:.4f}(s)")
# clf = SVM.SVC(kernel = "rbf", C = 3500, gamma = 5)
# clf.fit(train_feature,train_label)

# result = clf.predict(test_feature)
# accuracy = accuracy_score(test_label,result)
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

y_label = []
y_label = test_label
y_pred = []
y_pred = result

# cm = confusion_matrix(y_label, y_pred)

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
plt.show()

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