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
# movies = pd.read_csv("movies.csv")
# ratings = pd.read_csv("ratings.csv")
# ratings_movie = pd.merge(ratings, movies, on="movieId") # ratings를 기준으로 Movie 
# # print(ratings_movie.head()) # 헤더 출력
# ratings_matrix = ratings_movie.pivot_table("rating", "userId", "title") # 데이터 프레임 새로 생성
# # print(ratings_matrix.head())
#
# # NaN값 제거 해야함
# ratings_matrix.fillna(0, inplace=True) # NaN 값은 0으로 채워라
# # print(ratings_matrix.shape) # (610, 9719)
#
# # 유저기반 or 아이템 기반의 협엽필터링
# # 코사인 유사도
# ratings_matrixT = ratings_matrix.T # 행열을 바꿈 사용자 아이디 <-> 영화 아이디
#
# from sklearn.metrics.pairwise import cosine_similarity
# cs = cosine_similarity(ratings_matrixT, ratings_matrixT) # 코사인 유사도
# # print(type(cs)) # 데이터 프레임으로 변경해야함
# similarity = pd.DataFrame(cs, index=ratings_matrixT.index, columns=ratings_matrixT.index) # 같은 데이터를줌
# # print(similarity.head(1)) # [1 rows x 9719 columns]
#
# items = similarity[movies["title"][300]].sort_values(ascending=False)[:5] # 영화 제목을 뽑음 , 영화의 영화제목이라는 칼럼을 300개를 뽑음, 내림차순 정렬
# # print(items) # 평점이 유사항 영화들을 추천, 개인맞춤이라기 보단 평가점수가 높은것들
#
#
# # 함수 생성
# def predict_rating(ratings_array, item_similarity ): # 등급 배열, 아이템 유사도
#     sum = ratings_array @ item_similarity # 행렬곱의 연산자
#     sum_abs = np.array([np.abs(item_similarity).sum(axis=1)])
#     return sum / sum_abs
#
# # 함수호출
# rating_pred = predict_rating(ratings_matrix.values, similarity.values) # 평점의 대한 예측값
# rating_df = pd.DataFrame(data=rating_pred, index=ratings_matrix.index, columns=ratings_matrix.columns)
# # print(rating_df.head())
# # print(ratings_matrix.shape)
# # print(ratings_matrix.index)
# # print(ratings_matrix.columns)
#
# # 사용자별로 안본영화 리스트를 함수로 출력
# def no_watch(ratings_matrix, user):
#     user_rating = ratings_matrix.loc[user, :] # 열을 다 가져와라
#     no_watching = user_rating[user_rating==0].index.tolist() # user가 rating이 0인 영화를 리스트로 만들어라
#     movie_list = ratings_matrix.columns.tolist() # 영화 제목을 뽑음
#     return [ movie for movie in movie_list if movie in no_watching] # 리스트로 만들어서 반복문으로 돌림, 리스트에 있는것중에 noWatch 안에 있는것들만 뽑아라,
#                                                                     # 그것을 반복문으로 돌리고, 그안에 있는것만 리스트로 만들어라
#
#
# # 추천함수
# def recommend(rating_df, userId, no_list, top=10 ): # 누구를 예측할지
#     return rating_df.loc[userId, no_list].sort_values(ascending=False)[:top] # 내림차순, 끝자리가 10개까지만
#
# # 함수 실행
# userId = 15 # 유저 아이디는 15번 유저, 나중에 프로젝트때는 모든 유저들을 반복문 돌려서 해야함
# no_watching = no_watch(ratings_matrix, userId)
# recommend_movie = recommend(rating_df, userId, no_watching, 10) # 기본을 10개루 주었기에 안써도 되지만, 10개 하겠다 써준것
# recommend_df = pd.DataFrame(recommend_movie.values, index=recommend_movie.index, columns=["predict_score"])
# print(recommend_df) # 10개를 뽑아줌, 이부분을 출력을 해주면됨,


# 애니메이션 추천
anime = pd.read_csv("anime.csv")
rating = pd.read_csv("rating.csv")
# print(anime.head())
# print(rating.head()) # -1은 점수가 없다
# print(anime.shape) # (12294, 7)
# print(rating.shape) # (7813737, 3)

# 가공 시작
rating.replace( {-1:np.nan}, regex=True, inplace=True ) # -1 값을 nan 으로 바꿔라
# print(rating.head())

anime_tv = anime[ anime["type"] == "TV" ] # 타입이 tv 인것만 뽑아 리스트를 새로 만들어라
merged = rating.merge( anime_tv, left_on="anime_id", right_on="anime_id", suffixes=["_user", ""] )
# print(merged.shape) # (5283596, 9)
# print(merged.head())
merged.rename(columns={"rating_user":"user_rating"}, inplace=True) # 이름을 바꿔서 저장해라

merged_sub = merged[["user_id", "name", "user_rating"]]
# print(merged_sub.shape) # (5283596, 3)
merged_sub = merged_sub[ merged_sub.user_id <= 30000 ] #  user_id 가 30000 보다 작은 애들
# print(merged_sub.shape) # (2199324, 3)

# 사용자기반 협업필터링 => 데이터 프레임의 행렬 차이  행 : 사용자 열 : 아이템 : 값 : 평점
# 아이템 기반 협업필터링 =>                    헹 : 아이템 열 : 사용자 : 값 : 평점

piv = merged_sub.pivot( index=["user_id"], columns=["name"], values="user_rating" )
# print(piv.shape) # (29802, 3031)
piv_norm = piv.apply( lambda x : ( ( x-np.min(x) ) / (np.max(x) - np.min(x) ) ), axis=1 ) # 함수를 만들어 적용해 주어야함, 열끼리 연산
piv_norm.fillna( 0, inplace=True ) # nan 값을 0으로 바꿔라

# 주로 많이 쓰는것이 아이템기반임
piv_norm = piv_norm.T
pib_norm = piv_norm.loc[:, (piv_norm != 0).any(axis=0)] # 평점이 0인 애들을 행을 다 뽑아라, 희소행렬 => 압축 희소행렬로 바꿔줌

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
piv_sparse = csr_matrix( piv_norm )
item_similarity = cosine_similarity( piv_sparse )
user_similarity = cosine_similarity( piv_sparse.T )

item_df = pd.DataFrame( item_similarity, index=piv_norm.index, columns=piv_norm.index )
user_df = pd.DataFrame( user_similarity, index=piv_norm.columns, columns=piv_norm.columns )
# print( item_df.head() )

# def top_animes( anime_name ) :
#     count = 1
#     print( anime_name )
#     result = item_df.loc[ ~item_df.index.isin( [anime_name] ), anime_name ].sort_values( ascending=False )[:10]  # 10개만 가져와라
#     for item, score in result.items() :
#         print( count, item, score )
#         count += 1
#         # 애니메이션을 주면 애니메이션과 유사한 작품 10개를 프린트함
# top_animes( "Cowboy Bebop" )

print()

# # 사용자기준
# def top_users( user ) :
#     if user not in piv_norm.columns :
#         print( "No User" )
#         return 
#     result = user_df.sort_values( by=user, ascending=False ).loc[:, user][1:6]
#     for user, sim in result.items() :
#         print( user, sim )
# top_users( 3 )


import operator
def user_anime( user ) :
    if user not in piv_norm.columns :
        return
    sim_users = user_df.sort_values( by=user, ascending=False ).index[1:6]
    best = []
    most_common = {} # 딕셔너리로 생성
    for i in sim_users :
        result = piv_norm.loc[:, i][ (piv_norm.loc[:,user]==0 )].sort_values( ascending=False ) # 모든행의 i 열 => 사용자 아이디
        best.append( result.index[:5].tolist() ) # 리스트로 만들어서 가져와라
    for i in range( len(best ) ) : # 내가 추천해주려는? 건 빼야함
        for movie in best[i] :
            print( movie )
            if movie in most_common :
                most_common[movie] += 1
            else :
                most_common[movie] = 1
    return sorted( most_common.items(), key=operator.itemgetter(1), reverse=True ) # 같은것도 많은 순서대로 뽑음, 키가 기준이 되어야함, 몇번째 인덱스인지.
user_anime( 3 )     # 3번이라는 사람에게 추천

    