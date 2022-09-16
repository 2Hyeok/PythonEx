import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans # k-means
from sklearn.datasets import make_blobs # 버블을 만들어라


# data, label = make_blobs(n_samples=300, centers=3, random_state=0, cluster_std=1)
# data, label = make_blobs(n_samples=300, centers=3, random_state=0, cluster_std=0.5) # cluster_std=3 기본값이1 크기가 커질수록 많이 퍼짐
# fig = plt.figure(figsize=(10,5))
# ax1 = fig.add_subplot(1, 2, 1)
# colors = np.array(["red", "blue", "green"])
# ax1.scatter(data[:,0], data[:,1], color=colors[label], alpha=0.5)

# kmeans = KMeans(n_clusters=3)
# model = kmeans.fit(data)
# ax2 = fig.add_subplot(1, 2, 2)
# ax2.scatter(data[:,0], data[:,1], color=colors[model.labels_], alpha=0.5) # 모델 안에 labels_ 라는 변수가 들어있음, 학습한값
# ax2.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], color="k", marker="^") # 좌표가 3개, 중심점을 잡아줌 ^ 모양으로

# plt.show()


# 엘보우 기법
# from scipy.spatial.distance import cdist
# distortions = []
# ks = np.arange(1, 11)
# for k in ks :
#     kmenas = KMeans(n_clusters=k)
#     model = kmenas.fit(data)
#     # 외곡의 대한 수치
#     distortions.append( 
#         sum( np.min( cdist( data, model.cluster_centers_, "euclidean" ), axis=1 ) ) / 300 ) # 데이터와 중심값의 거리, 열끼리 구해라, 이것들의 합계를 구하고 300으로 나눔
# plt.plot(ks, distortions, "bx-")
# plt.show()


# 실루엣 기법
data, label = make_blobs(n_samples=300, centers=3, random_state=0, cluster_std=0.5) # cluster_std를 낮춰주면 정답률이 올라감
kmeans = KMeans(n_clusters=3)
model = kmeans.fit(data)
predicts = model.predict(data)
# print(predicts)
# clust_labels = np.unique(predicts) # 중복값 제거 [0, 1, 2]
# # print(clust_labels)
# n_cluster = clust_labels.shape[0]
# from sklearn.metrics import silhouette_samples
# from matplotlib import cm # 컬러맵
# silhouette_values = silhouette_samples(data, predicts, metric="euclidean")
#
# # 100 개씩 쌓아 올리기 위한 변수들
# y_lower, y_upper = 0, 0
# yticks = []
# for i, cluster in enumerate(clust_labels) : # [0, 1, 2] 반복문
#     silhouette_cluster = silhouette_values[predicts == cluster] # 그룹별로 출력
#     silhouette_cluster.sort() # 정렬해서 출력
#     y_upper += len(silhouette_cluster) # len 으로 누적
#     colors = cm.jet(float(i) / n_cluster)
#     plt.barh(range(y_lower, y_upper), silhouette_cluster, height=0.1, edgecolor="none", color=colors)
#     yticks.append((y_upper + y_lower) / 2)
#     y_lower += len(silhouette_cluster)
#
# silhouette_avg = np.mean(silhouette_values) # 전체 평균
# plt.axvline(silhouette_avg, color="k", linestyle="--")
# plt.yticks(yticks, clust_labels+1)
# plt.show()



from sklearn.metrics import confusion_matrix
result = confusion_matrix(predicts, label)
# print(result)
# DBSCAN
from sklearn.datasets import make_moons
data, label = make_moons(n_samples=300, noise=0.05, random_state=0)
kmeans = KMeans(n_clusters=2, random_state=0)
model = kmeans.fit(data)

fig = plt.figure(figsize=(10,5))

# 도표 두개
# ax1 = fig.add_subplot(1, 2, 1)
# colors = np.array(["red", "blue"])
# ax1.scatter(data[:,0], data[:,1], color=colors[label], alpha=0.5)

# ax2 = fig.add_subplot(1, 2, 2)
# ax2.scatter(data[:,0], data[:,1], color=colors[model.labels_], alpha=0.5)

# plt.show()


# from sklearn.cluster import DBSCAN
# scan = DBSCAN(eps=0.3, min_samples=4)
# model = scan.fit(data)
# ax1 = fig.add_subplot(1, 2, 1)
# colors = np.array(["red", "blue"])
# ax1.scatter(data[:,0], data[:,1], color=colors[label], alpha=0.5)
#
# ax2 = fig.add_subplot(1, 2, 2)
# ax2.scatter(data[:,0], data[:,1], color=colors[model.labels_], alpha=0.5)
# plt.show()






# 협업필터링
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
ratings_movie = pd.merge(ratings, movies, on="movieId") # ratings를 기준으로 Movie 
# print(ratings_movie.head()) # 헤더 출력
ratings_matrix = ratings_movie.pivot_table("rating", "userId", "title") # 데이터 프레임 새로 생성
# print(ratings_matrix.head())

# NaN값 제거 해야함
ratings_matrix.fillna(0, inplace=True) # NaN 값은 0으로 채워라
print(ratings_matrix.shape) # (610, 9719)

