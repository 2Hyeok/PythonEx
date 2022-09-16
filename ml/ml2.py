# 분류 Classfier

# 붓꽃 분류
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# iris = load_iris()
# print(iris.DESCR)
# data = iris.data
# label = iris.target
# print(data.shape) # (150, 4)
# print(label.shape) # (150,)
# print(label)

# names = iris.target_names # 이름만 가지고있는 리스트
# print(names)
# 학습시작
# train_data, test_data, train_label, test_label = \
#     train_test_split( data, label, test_size=0.3, random_state=0 )
# svc = SVC()
# model = svc.fit( train_data, train_label )
# print( model.score( train_data, train_label ) )     # 0.97
# print( model.score( test_data, test_label ) )       # 0.97

# fig = plt.figure( figsize=( 10, 5 ) ) # 정답
# colors = np.array( ["red", "green", "blue"] )
# ax1 = fig.add_subplot( 1, 2, 1 )
# ax1.scatter( data[:, 2], data[:, 3], color=colors[label], alpha=0.5 )
#
# predicts = model.predict( data ) 
# ax2 = fig.add_subplot( 1, 2, 2 ) # 예측값
# ax2.scatter( data[:, 2], data[:, 3], color=colors[predicts], alpha=0.5 )
# plt.show()

# 검증을 한다고 한다면
# k겹 교차검증
from sklearn.model_selection import cross_val_score
# scores = cross_val_score(svc, data, label, cv=3)
# print(scores) # 스코어3개가 나옴
# print(scores.mean())


# k-fold
from sklearn.model_selection import KFold
# kfold = KFold(n_splits=3, shuffle=True, random_state=0) # 섞어서
# scores = cross_val_score(svc, data, label, cv=kfold) # 출력해라 라는 의미
# print(scores)
# print(scores.mean())

# 모델 최적화
# from sklearn.model_selection import GridSearchCV
# params = [
#     {"C":[1, 10, 100, 1000], "kernel":["liner"]},
#     {"C":[1, 10, 100, 1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]}
#     ]
# # 딕셔너리 생성, 수치 지정
# # 단 학습이 느려짐
#
# gs = GridSearchCV(SVC(), params, n_jobs=1)
# models = gs.fit(data, label)
# # print(models.cv_results_) # 전체 결과 확인
# print(models.best_score_)
# print(models.best_params_)
# model = models.best_estimator_
# predict = model.predict([[2.3, 2.5, 4.7, 5.5]])
# print(predict)
# print(names[predict]) # names = iris.target_names, 품종이 무엇인지


# # RandomSearch
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint
# params = {"C":randint(1, 100)}
# rs = RandomizedSearchCV(SVC(), param_distributions=params, cv=5, n_iter=100, return_train_score=True) #  100번반복해라
# models = rs.fit(data,label)
# # print(models.cv_results_)
# print(models.best_score_)
# print(models.best_params_)

# confusion matrix



# 로지스틱 회귀
# 이차원배열로 만들어야함
# data = [
#         [8,8], [7,5], [6,8], [9,3], [7,2],
#         [5,4], [5,6], [7,3], [9,7], [7,8],
#         [6,6], [6,9], [3,2], [3,6], [5,3],
#         [6,1], [5,4], [5,5], [9,7], [9,9],
#         [9,9], [9,6], [8,4], [8,8], [7,7],
#         [9,3], [3,9], [8,1], [8,7], [7,9]
#         ]

# # np.array로 만들어야함
# label = np.array(
#         [1,0,0,0,0,
#          0,0,0,1,1,
#          0,1,0,0,0,
#          0,0,0,1,1,
#          1,1,0,1,0,
#          0,0,0,1,1
#          ])

# train_data, test_data, train_label, test_label = \
#     train_test_split(data, label, test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression # 로지스틱회귀 임포트
# lr = LogisticRegression()
# model = lr.fit(train_data, train_label)
# print(model.score(train_data, train_label)) # 0.95
# print(model.score(test_data, test_label)) # 0.88

# print(model.predict([[7, 8]]))
# print(model.predict([[6, 8]]))

# # 검증
# from sklearn import metrics # 검증 메소드의 집합
# predicts = model.predict(data)

# score = metrics.accuracy_score(predicts, label)
# print("정답률 : " , score)

# score = metrics.precision_score(predicts, label)
# print("정밀도 : " , score) # 양성 맞춘 개수 / 양성으로 진단한 개수

# score = metrics.recall_score(predicts, label)
# print("재현율 : " , score) # 양성 맞춘 개수 / 전체 양성 개수 , 몇명을 맞추었는가

# score = metrics.f1_score(predicts, label)
# print("f1-score : ", score) # 조화평균

# score = metrics.roc_auc_score(predicts, label)
# print("AUC : ", score)

# result = metrics.classification_report(predicts, label) # 모든 데이터를 한번에 담고 있음
# print(result)

# # 어떤 데이터가 있는데 불합격 7 합격이 3이다
# # 그럼 데이터를 어떻게 뽑아야할까? 
# # 불합격이 7, 합격이 3으로 뽑으면 안된다.
# # 데이터가 많은쪽의 학습은 잘하는데 데이터가 적은쪽의 학습은 불가능하다.
# # 불합격3 합격3 즉 1대1로 데이터를 뽑아야한다.


# # 도표출력
# df = pd.DataFrame(data,columns=["attendance", "hour"]) # 하나로 묶음
# df["pass"] = label # 합격 불합격
# df.plot()
# sns.pairplot(df, hue="pass")
# plt.show()


# 의사 결정 트리 Decision Tree
# 붓꽃 품종 분류
# iris = load_iris() # iris 받아옴
# data = iris.data
# label = iris.target
#
# data = data[:, [2,3] ] # 데이터를 자름, 모든행의 2열 3열
# # print(data)
# from sklearn.preprocessing import StandardScaler
#
# # StandardScaler => 특성들의 평균은 0, 분산은 1로 스케일링 하는것
# sc = StandardScaler() # 객체 생성
# sc.fit(data)
# data = sc.transform(data)
# # print(data) # 정규화 완료
#
# train_data, test_data, train_label, test_label = \
#     train_test_split(data, label, test_size=0.3, random_state=0)
#
# from sklearn.tree import DecisionTreeClassifier
# dtc = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0) # 진위개수, max_depth는 꼭 주어야함
# model = dtc.fit(train_data, train_label)
# print(model.score(train_data, train_label)) # 0.98
# print(model.score(test_data, test_label)) # 0.97
#
# from pydotplus import graph_from_dot_data
# from sklearn.tree import export_graphviz
# dot_data = export_graphviz(dtc,feature_names=["petal length", "petal width"], class_names=iris.target_names, rounded=True, special_characters=True, filled=True) # 칼럼의 이름, 라벨의 이름(정답의 원래 값을 가지고있음)
# graph = graph_from_dot_data(dot_data)
# graph.write_png("iris.png") # png파일로 저장


# # 암검진 데이터        악성 / 양성
# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# # print(cancer.DESCR) # 데이터 확인
# # print(cancer.target_names) # 출력시 ['malignant' 'benign']  =>  ['악성' '양성'] 이라는뜻
# data = cancer.data
# label = cancer.target
# # print(data)
# train_data, test_data, train_label, test_label = \
#     train_test_split(data, label, test_size=0.3, random_state=0)

# from sklearn.tree import DecisionTreeClassifier
# dtc = DecisionTreeClassifier(max_depth=5, random_state=0)
# model = dtc.fit(train_data, train_label)
# print(model.score(train_data, train_label)) # 0.98
# print(model.score(test_data, test_label)) # 0.93
#
# from sklearn.metrics import confusion_matrix
# predicts = model.predict(data)# 예측값
# result = confusion_matrix(predicts, label)
# # print(result)
#
# from sklearn.metrics import classification_report
# result = classification_report(predicts, label)
# print(result)

# malignant = cancer.data[cancer.target==0] # 전체 데이터에서 라벨(전체에서의 타겟)이 0인 데이터만
# benign = cancer.data[cancer.target==1] # 전체 데이터에서 라벨(전체에서의 타겟)이 1인 데이터만
# plt.figure(figsize=(12, 6))
# for i in range(len(cancer.feature_names)) :
#     plt.subplot(8, 4, i+1) # 32개, i는 0부터 시작하기에 i+1을줌
#     plt.hist(malignant[:, i], bins=20, label="malignant", alpha=0.5)
#     plt.hist(benign[:, i], bins=20, label="benign", alpha=0.5)
#     plt.title(cancer.feature_names[i])

# plt.legend()
# # plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(10,5))
# n_features = len(cancer.feature_names)
# plt.xlabel("importance")
# plt.ylabel("features")
# plt.yticks(np.arange(n_features), cancer.feature_names)
# plt.barh(range(n_features), model.feature_importances_) # 옆으로 그리는 바차트, 중요도
# plt.legend()
# # # plt.tight_layout()
# plt.show()


# 랜덤 포레스트
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# data = cancer.data
# label = cancer.target
# train_data, test_data, train_label, test_label = \
#     train_test_split(data, label, test_size=0.3, random_state=0)
#
# rfc = RandomForestClassifier(n_estimators=100, random_state=0) # 몇개를 만들건지 옵션추가
# model = rfc.fit(train_data, train_label)
# print(model.score(train_data, train_label)) # 1.0
# print(model.score(test_data, test_label)) # 0.95
#
# plt.figure(figsize=(10,5))
# n_features = len(cancer.feature_names)
# plt.xlabel("importance")
# plt.ylabel("features")
# plt.yticks(np.arange(n_features), cancer.feature_names)
# plt.barh(range(n_features), model.feature_importances_) # 옆으로 그리는 바차트, 중요도
# plt.legend()
# # # plt.tight_layout()
# plt.show()


# 식용버섯 분류
# import urllib.request as req
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
# req.urlretrieve( url, "mushroom.csv" ) # 파일 다운로드

# mr = pd.read_csv("mushroom.csv", header=None)
# data=[]
# label = []
# for index, row in mr.iterrows() :
#     label.append(row.loc[0])
#     row_data=[]
#     for value in row[1:] :
#         row_data.append(ord(value)) # 아스키코드로 바꿈

#     data.append(row_data)
# data = np.array(data)
# label = np.array(label)
# print(data.shape) # (8124, 22)
# print(label.shape) # (8124,)
#
# train_data, test_data, train_label, test_label = \
#     train_test_split(data, label, test_size=0.3, random_state=0)

# lr = LogisticRegression()
# model = lr.fit(train_data, train_label) # 0.94 0.93
#
# from sklearn.naive_bayes import GaussianNB
# nb = GaussianNB()
# model = nb.fit(train_data, train_label) # 0.90 0.90

# from sklearn.tree import DecisionTreeClassifier
# dtc = DecisionTreeClassifier()
# model = dtc.fit(train_data, train_label) # 1.0 1.0

# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier()
# model = rfc.fit(train_data, train_label) # 1.0 1.0
#
# print(model.score(train_data, train_label))
# print(model.score(test_data, test_label))


# SVM 비교
# from sklearn.datasets import load_iris
# iris = load_iris()
# data = iris.data
# label = iris.target
# train_data, test_data, train_label, test_label = \
#     train_test_split(data, label, test_size=0.3, random_state=0)

# from sklearn.svm import SVC
# svc = SVC()
# model = svc.fit(train_data, train_label) # 0.97, 0.97

# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier()
# model = knn.fit(train_data, train_label) # 0.97, 0.97

# from sklearn.gaussian_process import GaussianProcessClassifier
# gpc = GaussianProcessClassifier()
# model = gpc.fit(train_data, train_label) # 0.97, 0.97

# from sklearn.ensemble import AdaBoostClassifier
# ada = AdaBoostClassifier()
# model = ada.fit(train_data, train_label) # 0.96, 0.91

# from sklearn.naive_bayes import GaussianNB
# nb = GaussianNB()
# model = nb.fit(train_data, train_label) # 0.94, 1.0

# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# qda = QuadraticDiscriminantAnalysis()
# model = qda.fit(train_data, train_label) # 0.99, 0.97

# from sklearn.neural_network import MLPClassifier # 인공신경망, 옵션을 주면 값이 정확하게 나옴
# mlp = MLPClassifier()
# model = mlp.fit(train_data, train_label) # 0.98, 0.93

# print(model.score(train_data, train_label))
# print(model.score(test_data, test_label))


# 종양분석
# 서포트 벡터 머신
# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# data = cancer.data
# label = cancer.target
# train_data, test_data, train_label, test_label = \
#     train_test_split(data, label, test_size=0.3, random_state=0)
#
# from sklearn.svm import SVC
# svc = SVC() # 비선형 분류, 기본 rbf
# model = svc.fit(train_data, train_label) # 0.90, 0.92
#
# # 가공 다시 하기
# from sklearn.preprocessing import MinMaxScaler
# mm = MinMaxScaler()
# mm.fit(data) # fit 후계산을 다시해주어야함
# train_data = mm.transform(train_data)
# test_data = mm.transform(test_data)
# model = svc.fit(train_data, train_label) # 0.98, 0.97
#
# svc = SVC(C=100)
# model = svc.fit(train_data, train_label) # 1.0, 0.97
#
# print(model.score(train_data, train_label))
# print(model.score(test_data, test_label))


# DNN , CNN => 딥러닝, 인공신경망
# K-NN => 다른개념
iris = load_iris()
data = iris.data
label = iris.target

# 정규화 해주자
# 정답률이 올라감
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
mm.fit(data)
data = mm.transform(data)
train_data, test_data, train_label, test_label = \
    train_test_split(data, label, test_size=0.3, random_state=0)

neighbors = range(1, 11) # 리스트 나열
train_score = [] # 리스트 생성
test_score = []

from sklearn.neighbors import KNeighborsClassifier
for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n) # 근접 이웃일 하나만 설정해서 늘려가라
    model = knn.fit(train_data, train_label)
    train_score.append(knn.score(train_data, train_label))
    test_score.append(knn.score(test_data, test_label))

# plt.plot(neighbors, train_score, label="train")
# plt.plot(neighbors, test_score, label="Test")
# plt.legend()
# plt.show()

# 알고리즘 생성
knn = KNeighborsClassifier(n_neighbors=3) # 3으로 값을 줌
model = knn.fit(train_data, train_label)

# 예측 시작
# 정규화 해서 학습을 시켰기에 예측할때도 정규화를 해주어야함
pre = [[1.2, 2.3, 5.5, 2.7]]
pre = mm.transform(pre) # pre 라는 값을 정규화해서 
predict = model.predict(pre) # 예측 해라
# print(predict) # 종류가 무엇인지 [2]
print(iris.target_names[predict]) # 이름이 무엇인지 확인