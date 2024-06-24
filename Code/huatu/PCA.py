import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import numpy as np
import pandas as pd 
from sklearn.manifold import TSNE

import seaborn as sns
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



# csv_file_path_Train = 'E:\\Milk\\averspectra\\original\\yang.csv'
csv_file_path_Train = 'E:\\Milk\\averspectra\\processing\\right\\SNV\\yang_tuo_all.csv'
# csv_file_path_Train = 'E:\\Milk\\averspectra\\1-D\\fine_tuning\\FT_10\\train.csv'
# csv_file_path_Train = 'F:\\data\\汇总\\FT_0\\drought\\0.csv'
csv_file_path_Test = 'E:\\Milk\\averspectra\\1-D\\test.csv'


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

data = train_feature
labels = train_label




#下面是保存数据

n_components = 10  # 指定降维到2维
pca = PCA(n_components=n_components)
data_reduced = pca.fit_transform(data)

data_with_labels = pd.DataFrame({'Label': labels})
for i in range(n_components):
    data_with_labels[f'PC{i+1}'] = data_reduced[:, i]

# output_excel_file = r'C:\Users\soft\Desktop\论文准备\奶粉\说明数据可分离性\reduced_data_with_labels.xlsx'
# data_with_labels.to_excel(output_excel_file, index=False)

# 指定输出CSV文件路径
output_csv_path = r'C:\Users\soft\Desktop\论文准备\奶粉\说明数据可分离性\reduced_data_with_labels.csv'

# 保存DataFrame到指定的CSV文件
data_with_labels.to_csv(output_csv_path, index=False)


print(f"降维后的数据已保存到 {output_csv_path}")









# 下面是PCA可视化
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)


plt.figure(figsize=(6, 4))
colors = plt.cm.tab20(np.linspace(0, 1, 11))  # 使用不同颜色来区分不同类别
name = ['C10%','C20%','C30%','C40%','C50%','C60%','C70%','C80%','C90%','C100%','Ca100%']
# name = ['G10%','G20%','G30%','G40%','G50%','G60%','G70%','G80%','G90%','G100%','Ca100%']
# name = ['Ca100%','C100%','G100%']

for i in range(len(name)):
    class_data = data_reduced[labels == i]
    std_dev = np.std(class_data, axis=0)
    plt.scatter(data_reduced[labels == i, 0], data_reduced[labels == i, 1], color=colors[i], alpha=0.8, lw=2, label=f''+name[i]+'')


# 添加置信区间
    # for j, std in enumerate(std_dev):
    #     ell = plt.matplotlib.patches.Ellipse((0, 0), std*2, std*2, angle=0, fill=False, color='black', alpha=0.7)
    #     plt.gca().add_patch(ell)
    #     plt.text(std*2, std*2, f'PC{j+1}', fontsize=12)

plt.legend(loc='best', shadow=False, scatterpoints=1, title='Classes')
font = {'family': 'Times New Roman', 'size': 16}
plt.xlabel('PC1 ({} %)'.format(round(pca.explained_variance_ratio_[0] * 100, 2)), font)
plt.ylabel('PC2 ({} %)'.format(round(pca.explained_variance_ratio_[1] * 100, 2)), font)

plt.title('PCA of Data')
plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import SparsePCA
n_components = 3  # 指定提取前三个主成分
alpha = 0.1  # 控制稀疏度的超参数
sparse_pca = PCA(n_components=n_components)
data_reduced = sparse_pca.fit_transform(data)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.tab20(np.linspace(0, 1, 11))  # 使用不同颜色来区分不同类别

for i in range(len(name)):
    ax.scatter(data_reduced[labels == i, 0], data_reduced[labels == i, 1], data_reduced[labels == i, 2], color=colors[i], label=f''+name[i]+'')
ax.set_xlabel('PC1 ({} %)'.format(round(sparse_pca.explained_variance_ratio_[0] * 100, 2)), font)
ax.set_ylabel('PC1 ({} %)'.format(round(sparse_pca.explained_variance_ratio_[1] * 100, 2)), font)
ax.set_zlabel('PC1 ({} %)'.format(round(sparse_pca.explained_variance_ratio_[2] * 100, 2)), font)
ax.legend(loc='best', title='Classes')
ax.set_title('Sparse PCA of Data (3D)')

plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
plt.show()









# #此处是Tsne可视化
# y_embeddings = data
# y_label = labels


# tsne = TSNE(n_components=2,n_iter=2000)
# X_tsne_2d = tsne.fit_transform(y_embeddings)
# marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# class_list = [0,1,2]
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
#     plt.scatter(X_tsne_2d[indices, 0], X_tsne_2d[indices, 1], color=color, marker=marker, label=fruit, s=150)
    
# plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
# plt.xticks([])
# plt.yticks([])
# plt.show()



# from sklearn.decomposition import SparsePCA

# data = train_feature
# label = train_label
# n_components = 2  # 指定降维到2维
# alpha = 0.1  # 控制稀疏度的超参数
# sparse_pca = SparsePCA(n_components=n_components, alpha=alpha)
# data_reduced = sparse_pca.fit_transform(data)
# n_components = 2  # 指定降维到2维
# alpha = 0.1  # 控制稀疏度的超参数
# sparse_pca = SparsePCA(n_components=n_components, alpha=alpha)
# data_reduced = sparse_pca.fit_transform(data)
# plt.figure(figsize=(8, 6))
# colors = plt.cm.tab20(np.linspace(0, 1, 11))  

# for i in range(11):
#     plt.scatter(data_reduced[labels == i, 0], data_reduced[labels == i, 1], color=colors[i], alpha=0.8, lw=2, label=f'Class {i}')

# plt.legend(loc='best', shadow=False, scatterpoints=1, title='Classes')
# plt.title('Sparse PCA of Data')

# plt.show()