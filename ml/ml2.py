# 분류 Classfier

# 붓꽃 분류
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.datasets import load_iris
iris = load_iris()
# print(iris.DESCR)
data = iris.data
label = iris.target
# print(data.shape) # (150, 4)
# print(label.shape) # (150,)
# print(label)

names = iris.target_names # 이름만 가지고있는 리스트
# print(names)
# 학습시작
train_data, test_data, train_label, test_label = \
    train_test_split( data, label, test_size=0.3, random_state=0 )
svc = SVC()
model = svc.fit( train_data, train_label )
print( model.score( train_data, train_label ) )     # 0.97
print( model.score( test_data, test_label ) )       # 0.97

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
scores = cross_val_score(svc, data, label, cv=3)
print(scores) # 스코어3개가 나옴
print(scores.mean())


# k-fold
from sklearn.model_selection import KFold
kfold = KFold(n_splits=3, shuffle=True, random_state=0) # 섞어서
scores = cross_val_score(svc, data, label, cv=kfold) # 출력해라 라는 의미
print(scores)
print(scores.mean())

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
#
# # np.array로 만들어야함
# label = np.array(
#         [1,0,0,0,0,
#          0,0,0,1,1,
#          0,1,0,0,0,
#          0,0,0,1,1,
#          1,1,0,1,0,
#          0,0,0,1,1
#          ])
#
# train_data, test_data, train_label, test_label = \
#     train_test_split(data, label, test_size=0.3, random_state=1)
#
# from sklearn.linear_model import LogisticRegression # 로지스틱회귀 임포트
# lr = LogisticRegression()
# model = lr.fit(train_data, train_label)
# print(model.score(train_data, train_label)) # 0.95
# print(model.score(test_data, test_label)) # 0.88
#
# print(model.predict([[7, 8]]))
# print(model.predict([[6, 8]]))
#
#
# # 검증
# from sklearn import metrics # 검증 메소드의 집합
# predicts = model.predict(data)
#
# score = metrics.accuracy_score(predicts, label)
# print("정답률 : " , score)
#
# score = metrics.precision_score(predicts, label)
# print("정밀도 : " , score) # 양성 맞춘 개수 / 양성으로 진단한 개수
#
# score = metrics.recall_score(predicts, label)
# print("재현율 : " , score) # 양성 맞춘 개수 / 전체 양성 개수 , 몇명을 맞추었는가
#
# score = metrics.f1_score(predicts, label)
# print("f1-score : ", score) # 조화평균
#
# score = metrics.roc_auc_score(predicts, label)
# print("AUC : ", score)
#
# result = metrics.classification_report(predicts, label) # 모든 데이터를 한번에 담고 있음
# print(result)
#
# # 어떤 데이터가 있는데 불합격 7 합격이 3이다
# # 그럼 데이터를 어떻게 뽑아야할까? 
# # 불합격이 7, 합격이 3으로 뽑으면 안된다.
# # 데이터가 많은쪽의 학습은 잘하는데 데이터가 적은쪽의 학습은 불가능하다.
# # 불합격3 합격3 즉 1대1로 데이터를 뽑아야한다.
#
#
# # 도표출력
# df = pd.DataFrame(data,columns=["attendance", "hour"]) # 하나로 묶음
# df["pass"] = label # 합격 불합격
# df.plot()
# sns.pairplot(df, hue="pass")
# plt.show()


# 의사 결정 트리 Dicision Tree