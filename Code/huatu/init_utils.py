import os
import numpy as np
import pandas as pd 
from sklearn.preprocessing import normalize
import torch.nn as nn 


# 获取路径下所有csv文件的路径，并且最后一个文件夹下只能有一个文件
def get_file(root_path):
    file_list = os.listdir( path = root_path )                        #指定目录下的所有文件夹的名字的列表
    file_list = [os.path.join(root_path,f) for f in file_list] 

    assert len(file_list) == 1, 'there are {} files in [{}] '.format(len(file_list),root_path)       #如果指定路径下的文件不唯一，则存在歧义
    
    return file_list[0]




#按行读取csv里面的数据，从0 开始 0到287是光谱数据

def get_data_csv (file_dir,num):           # 读数据
    data_temp  = pd.read_csv(file_dir,header=None)
    data = []

    index = [i for i in range(num)]
    np.random.shuffle(index)

    # for i in  range(len(index)):
        
    #     data = data_temp[index[i],0:287]
    data =  data_temp.iloc[:,0:288]
 

    data = data.values.reshape(-1)
    

    return data



#把数据转化成  # x = x˙ / ∥x˙∥2

def normalization(x):  # x: (n, length)

    x = normalize(x, norm='l2', axis=1)

    return np.asarray(x)   



#随机化数据顺序

def sample_label_shuffle(data, label):        #有点儿问题


     index = [i for i in range(len(label))]
     np.random.shuffle(index)
     data_ = data[index]
     data = data_

     label_ = label[index]
     label  = label_      

     return data, label


def weights_init2(L):                          #对权重和偏置量进行初始化
    if isinstance(L,nn.Conv1d):
        n = L.kernel_size[0] * L.out_channels
        L.weight.data.normal_(mean=0,std=np.sqrt(2.0/float(n)))
    elif isinstance(L,nn.BatchNorm1d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)
    elif isinstance(L,nn.Linear):
        L.weight.data.normal_(0,0.01)
        if L.bias is not None:
            L.bias.data.fill_(1)