import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import xlwt

class MyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        label = sample.iloc[0]  # 第一列为标签
        features = sample.iloc[1:5].values.astype(float)  # 第二列到第六列为特征

        return torch.Tensor(features), label      


dataset = MyDataset('E:\Practice\HIS\TEST\RE\dataset_drought.csv')

dataloader = DataLoader(dataset,batch_size = 64,shuffle=False)


num_i = 1*4  #输入的是一维向量
num_h1 = 16
num_h2 = 32 
num_o  = 64


class Model(torch.nn.Module):
    
 
    def __init__(self,num_i,num_h1,num_h2,num_o):
        super(Model,self).__init__()
        
        self.linear1=torch.nn.Linear(num_i,num_h1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=num_h1)
        self.relu=torch.nn.ReLU()

        self.linear2=torch.nn.Linear(num_h1,num_h2) #2个隐层
        self.bn2 = torch.nn.BatchNorm1d(num_features=num_h2)
        self.relu2=torch.nn.ReLU()

        self.linear3=torch.nn.Linear(num_h2,num_o) #3个隐层
        self.bn3 = torch.nn.BatchNorm1d(num_features=num_o)
        self.relu2=torch.nn.ReLU()

  
    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        # x = self.relu(x)

        # x = self.linear2(x)
        # x = self.bn2(x)
        # x = self.relu2(x)

        # x = self.linear3(x)
        # x = self.bn3(x)
        # x = self.relu2(x)
        return x

book = xlwt.Workbook() #创建Excel
sheet = book.add_sheet('sheet1') #创建sheet页

model = Model(num_i,num_h1,num_h2,num_o)

result_df = pd.DataFrame()
all_batch_data = torch.tensor([])

# for batch_data, batch_labels in dataloader:
#     outputs = model(batch_data)

#     batch_df = pd.DataFrame(outputs.detach().numpy()) 

#     result_df = result_df.append(batch_df)
    
#     all_batch_data = torch.cat((all_batch_data, outputs), dim=0)
#     pass

for batch_data, batch_labels in dataloader:
    outputs = model(batch_data)
    outputs = batch_data


    batch_df = pd.DataFrame(outputs.detach().numpy()) 

    result_df = result_df.append(batch_df)
    
    all_batch_data = torch.cat((all_batch_data, outputs), dim=0)
    
    pass

##使用余弦相似度进行度量，然后区分
# result_df.to_excel('output_256.xlsx', index=False)
a = all_batch_data.unsqueeze(1)
b = all_batch_data.unsqueeze(0)

# cos = torch.cosine_similarity(a = all_batch_data.unsqueeze(1), b = all_batch_data.unsqueeze(0), dim=2)
cos = torch.cosine_similarity(a ,b, dim=2)
# cos= pd.DataFrame(cos.detach().numpy()) 
# cos.to_excel('cos_256.xlsx', index=False)yua=2
cos = cos.detach().numpy()





##使用升维之后的样本直接进行聚类 聚类方法为：OPTICS
from sklearn.cluster import OPTICS
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

data = np.array(result_df)
# data = np.array(cos)
# data = np.array(all_batch_data)
X = data

# clust = OPTICS(min_samples=2, xi=0.05, min_cluster_size=0.1,metric='minkowski')

# clust.fit(data)

# space = np.arange(len(X))
# reachability = clust.reachability_[clust.ordering_]
# OPTICS_labels = clust.labels_[clust.ordering_]
# labels = clust.labels_[clust.ordering_]

# plt.figure(figsize=(10, 7))
# G = gridspec.GridSpec(2, 3)
# ax1 = plt.subplot(G[0, 0])
# ax2 = plt.subplot(G[1, 0])


# # Reachability plot
# colors = ["g.", "r.", "b.", "y.", "c."]
# for klass, color in zip(range(0, 5), colors):
#     Xk = space[labels == klass]
#     Rk = reachability[labels == klass]
#     ax1.plot(Xk, Rk, color, alpha=0.3)
# ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
# ax1.set_ylabel("Reachability (epsilon distance)")
# ax1.set_title("Reachability Plot")

# # OPTICS
# colors = ["g.", "r.", "b.", "y.", "c."]
# for klass, color in zip(range(0, 5), colors):
#     Xk = X[clust.labels_ == klass]
#     ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
# ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], "k+", alpha=0.1)
# ax2.set_title("Automatic Clustering\nOPTICS")


# plt.tight_layout()
# plt.show()

#K-Means
from sklearn.cluster import KMeans

#Define function:
kmeans = KMeans(n_clusters=2)

#Fit the model:
km = kmeans.fit(X)
km_labels = km.labels_

print(km_labels)
#Print results:
#print(kmeans.labels_)

#Visualise results:
plt.scatter(X[:, 0], X[:, 1], 
            c=kmeans.labels_,      
            s=70, cmap='Paired')

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker='^', s=100, linewidth=2, 
            c=[0, 1])
plt.show()
a = 1 


# # 下面是使用Affinity Propagation

# from sklearn.cluster import AffinityPropagation

# af = AffinityPropagation(preference=-563, random_state=0).fit(X)
# cluster_centers_indices = af.cluster_centers_indices_
# af_labels = af.labels_
# n_clusters_ = len(cluster_centers_indices)

# #Print number of clusters:
# print(n_clusters_)

# import matplotlib.pyplot as plt
# from itertools import cycle

# plt.close("all")
# plt.figure(1)
# plt.clf()

# colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
# for k, col in zip(range(n_clusters_), colors):
#     class_members = af_labels == k
#     cluster_center = X[cluster_centers_indices[k]]
#     plt.plot(X[class_members, 0], X[class_members, 1], col + ".")
#     plt.plot(
#         cluster_center[0],
#         cluster_center[1],
#         "o",
#         markerfacecolor=col,
#         markeredgecolor="k",
#         markersize=14,
#     )
#     for x in X[class_members]:
#         plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

# plt.title("Estimated number of clusters: %d" % n_clusters_)
# plt.show()
# a=1


#Agglomerative Clustering

# from sklearn.cluster import AgglomerativeClustering

# #Fit the model:
# clustering = AgglomerativeClustering(n_clusters=5).fit(X)

# AC_labels= clustering.labels_
# n_clusters = clustering.n_clusters_

# print("number of estimated clusters : %d" % clustering.n_clusters_)

# # Plot clustering results
# colors = ['purple', 'orange', 'green', 'blue', 'red']

# for index, metric in enumerate([#"cosine", 
#                                 "euclidean", 
#  #"cityblock"
#                                 ]):
#     model = AgglomerativeClustering(
#         n_clusters=5, linkage="ward", affinity=metric
#     )
#     model.fit(X)
#     plt.figure()
#     plt.axes([0, 0, 1, 1])
#     for l, c in zip(np.arange(model.n_clusters), colors):
#         plt.plot(X[model.labels_ == l].T, c=c, alpha=0.5)
#     plt.axis("tight")
#     plt.axis("off")
#     plt.suptitle("AgglomerativeClustering(affinity=%s)" % metric, size=20)


# plt.show()
# a=1

# from sklearn.cluster import MeanShift, estimate_bandwidth

# # The following bandwidth can be automatically detected using
# bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)

# #Fit the model:
# ms = MeanShift(bandwidth=bandwidth)
# ms.fit(X)
# MS_labels = ms.labels_
# cluster_centers = ms.cluster_centers_

# labels_unique = np.unique(MS_labels)
# n_clusters_ = len(labels_unique)

# print("number of estimated clusters : %d" % n_clusters_)

# from itertools import cycle

# plt.figure(1)
# plt.clf()

# colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
# for k, col in zip(range(n_clusters_), colors):
#     my_members = MS_labels == k
#     cluster_center = cluster_centers[k]
#     plt.plot(X[my_members, 0], X[my_members, 1], col + ".")
#     plt.plot(
#         cluster_center[0],
#         cluster_center[1],
#         "o",
#         markerfacecolor=col,
#         markeredgecolor="k",
#         markersize=14,
#     )
# plt.title("Estimated number of clusters: %d" % n_clusters_)
# plt.show()


# from sklearn.cluster import DBSCAN

# db = DBSCAN(eps=3, min_samples=10).fit(X)
# DBSCAN_labels = db.labels_
# labels = DBSCAN_labels
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)

# print("Estimated number of clusters: %d" % n_clusters_)
# print("Estimated number of noise points: %d" % n_noise_)

# unique_labels = set(labels)
# core_samples_mask = np.zeros_like(labels, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True

# colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#  if k == -1:
#  # Black used for noise.
#         col = [0, 0, 0, 1]

#         class_member_mask = labels == k

#         xy = X[class_member_mask & core_samples_mask]
#         plt.plot(
#             xy[:, 0],
#             xy[:, 1],
#             "o",
#             markerfacecolor=tuple(col),
#             markeredgecolor="k",
#             markersize=14,
#         )

#         xy = X[class_member_mask & ~core_samples_mask]
#         plt.plot(
#             xy[:, -1],
#             xy[:, 1],
#             "o",
#             markerfacecolor=tuple(col),
#             markeredgecolor="k",
#             markersize=6,
#         )

# plt.title(f"Estimated number of clusters: {n_clusters_}")
# plt.show()


# import matplotlib.colors as colors
# from sklearn.cluster import Birch, MiniBatchKMeans
# from time import time
# from itertools import cycle

# # Use all colors that matplotlib provides by default.
# colors_ = cycle(colors.cnames.keys())

# fig = plt.figure(figsize=(12, 4))
# fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)

# # Compute clustering with BIRCH with and without the final clustering step
# # and plot.
# birch_models = [
#     Birch(threshold=1.7, n_clusters=None),
#     Birch(threshold=1.7, n_clusters=5),
# ]
# final_step = ["without global clustering", "with global clustering"]


# for ind, (birch_model, info) in enumerate(zip(birch_models, final_step)):
#     t = time()
#     birch_model.fit(X)
#     print("BIRCH %s as the final step took %0.2f seconds" % (info, (time() - t)))

#  # Plot result
#     labels = birch_model.labels_
#     print(labels)
#     centroids = birch_model.subcluster_centers_
#     n_clusters = np.unique(labels).size
#     print("n_clusters : %d" % n_clusters)

#     ax = fig.add_subplot(1, 3, ind + 1)
#     for this_centroid, k, col in zip(centroids, range(n_clusters), colors_):
#         mask = labels == k
#         ax.scatter(X[mask, 0], X[mask, 1], c="w", edgecolor=col, marker=".", alpha=0.5)
#     if birch_model.n_clusters is None:
#             ax.scatter(this_centroid[0], this_centroid[1], marker="+", c="k", s=25)
#     ax.set_ylim([-12, 12])
#     ax.set_xlim([-12, 12])
#     ax.set_autoscaley_on(False)
#     ax.set_title("BIRCH %s" % info)

# plt.show()


