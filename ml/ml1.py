# 반증법에 의한 증명

# 대립가설 - 우리가 증명하고 싶은 가설 ex)학생들의 평균키는 175이다. 이것이 증명하고 싶은가설
# 귀무가설 - ex)학생들의 평균키는 175가 아니다

# # 데이터 생성
# import numpy as np
# from scipy import stats # 통계분석 라이브러리
# import pandas as pd

# np.random.seed(1) # 값을 고정
# # heights = [180 + np.random.normal(0,5) for i in range(100)] # 100개 생성, 틀리다를 증명(귀무가설), 표준편차가 5정도 퍼져있어라
# # 귀무 가설이 기각이됨, 대립가설이 체택이 됨
# # 귀무가설이 기각이 되면 안되는 데이터임
# # heights = [180 + np.random.normal(0,5) for i in range(10000)] # pvalue가 달라짐 0.05이하 일경우 귀무가설이 체택이됨, 대립가설이 기각
# # 귀무가설을 참으로 봤을 때 표본에서 실제로 해당되는 통계치가 나올 가능성은 5%미만 이라는 의미

# heights = [175 + np.random.normal(0,5) for i in range(10000)] # 0.05이상 일경우 대립가설이 체택이됨, 귀무가설이 기각됨
# result = stats.ttest_1samp(heights, 175) # 175 를 기준으로 잡는다. 집단 평균 검정, ttest_1samp => 하나의 데이터 집단의 평균과 비교하고자 하는 관측치를 통해 차이를검정하는 방법
# print(result) # 여기서 중요한것은 pvalue 이다.

# 상관관계 분석
# 피어슨의 상관관계 분석
import numpy as np
from scipy import stats # 통계분석 라이브러리
import pandas as pd
import matplotlib.pyplot as plt # 시각화
import seaborn as sns # 히트맵

# df = pd.DataFrame( # 판다스로 생성 (데이터 프레임)
#         { # 딕셔너리로 생성
#             "a" : [i*100 for i in range(100)], # 리스트로생성
#             "b" : [i*-100 for i in range(100)],
#             "c" : [i*np.random.randint(1, 100) for i in range(100)]
#         }
#     )
# # df.plot()
# # plt.show()
#
# corr = df.corr(method="pearson") # 기본값이 피어슨임
# print(corr)
#
# sns.heatmap(corr, annot=True, annot_kws={"size":20}, fmt=".2f") # 히트맵 도표, 키워드는 딕셔너리로 키워드 생성, 출력형식은 소수점 둘째자리까지
# plt.show()


#-------------------------------------------------------------------------------------------------------------------------------------

# 머신러닝 분석
# pip install scikit-learn scikit-image
# 내가 알고리즘을 정하는것이아님, 그냥 다 써보는것임
# 머신러닝을 분석하는 순서
from sklearn import svm
# data = [[0,0],[0,1],[1,0],[1,1]] # 데이터는 반드시 2차원으로 주어야한다.
# label = [0, 1, 1, 0] # 라벨은 반드시 1차원으로 주어야한다., 답은 반드시 이것이다 라며 학습을함
# svc = svm.SVC()
# model = svc.fit (data,label) # 학습을 시킴
# # 모델이 정확한지 모델 검증도 해주어야함, 
# print(model.predict([[0,1]])) # 예측을 시킴, [0,1] 이라는 예측값임
#
# # and
# label = [0,0,0,1]
# model = svc.fit (data,label)
# print(model.predict([[0,1]])) # 정답이 다름
#
# # or
# label = [0,1,1,1]
# model = svc.fit (data,label)
# print(model.predict([[0,1]]))
#
#
# # 똑같은 것을 두번줌 (정답률 )
# data = [[0,0],[0,1],[1,0],[1,1], [0,0],[0,1],[1,0],[1,1]]
# # label = [0,0,0,1,0,0,0,1]
# label = [0,0,0,1,0,0,0,0] # 정답률 이 떨어짐
# svc = svm.SVC()
# model = svc.fit (data,label)
# print(model.score(data,label)) # 학습해서 나온 모델에 다시한번 확인해봐라

#---------------------------------------------------------------------------------------------------------------------------------------

# 회귀 Regress
# 선형회귀 - 선을 찾는것,
# 전기 생산량과 소비량 분석
# x = [2.42, 2.56, 3.31, 4.62, 3.98, 4.29, 4.9, 3.67, 3.78 , 3.45, 3.59, 3.81] # 생산량 (독립변수)
# y = [2.56, 2.67, 2.87, 3.98, 4.65, 3.54, 2.65, 4.78, 4.23, 3.87, 4.59, 3.71] # 소비량 (종속변수)
# result = stats.linregress(x, y) # 가정 적합한 선을찾음
# # print(result)
# # 튜플로 나오기에 데이터를 뽑을 수있음
# slope, intercept, rvalue, pvalue, stderr = result # 필요가 없는 데이터가있더라도 꼭 가져와서 뽑아야함
#
# # xx = np.array(x)
# # plt.scatter(xx, y)
# # plt.plot(xx, slope * xx + intercept,"r") # 기울기 * x + intercept, 이선이 모든 선들과 가까운 선 이다 라는뜻, Wx + b
# # plt.show()
#
# # 예측
# print(slope * 4.0 + intercept) # 생산량이 4.0일때의 소비량의 값을 예측

#----------------------------------------------------------------------------------------------------------------------------------------

# csv 데이터 사용해서 분석
# csv 읽을때는 판다스로 읽는것이 편함
# ozone = pd.read_csv("ozone.csv") # header=None
# # print(ozone.describe())
# # print(ozone.count()) # 카운트 확인
# # print(ozone.shape)
# # print(ozone.head())
#
# ozone = ozone.dropna(axis=0) # 없애다, how=의 기본값은 any,  가로인지 세로인지지정
# # print(ozone.count()) # None값 제거
#
# # 오존과 온도
# # x = ozone["Temp"].values # 알고있는데이터
#
# # 오존과 바람
# # x = ozone["Wind"].values # 알고있는데이터,
#
# # 오존과 태양광
# x = ozone["Solar.R"].values # 알고있는데이터,
#
# # 오존의 양
# y = ozone["Ozone"].values # 알고싶은 데이터
#
# result = stats.linregress(x,y)
# # print(80 * result.slope + result.intercept) # 오존양 예측
# print(15 * result.slope + result.intercept) # 바람의 오존양 예측
# # print(result)
#
# # plt.scatter(x,y)
# # plt.plot(x, result.slope * x + result.intercept, "r")
# # plt.show()
# sns.heatmap(ozone.loc[:,"Ozone":"Temp"].corr(), annot=True)
# plt.show()

#---------------------------------------------------------------------------------------------------

# 다중 선형 회귀
# 보스턴 주택가격 분석
# from sklearn.datasets import load_boston
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# boston = load_boston()

# # print(boston) # 양이 엄청 많음, 딕셔너리임
# # print(boston.DESCR) # 설명서 확인
# data = boston.data
# label = boston.target
# # print(data.shape) # 2차원 (506, 13)
# # print(label.shape) # 1차원 (506,)
#
# train_data, test_data, train_label, test_label = \
#     train_test_split( data, label, test_size=0.3, random_state=0 ) # 한줄짜리 코딩을 두줄로 나눴기에 역슬래수가 들어감
# lr = LinearRegression()
# model = lr.fit(train_data, train_label)
# print(model.score(train_data, train_label)) # 0.76
# print(model.score(test_data, test_label)) # 0.67


# 보스턴 추가
# from sklearn.datasets import load_boston
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# import mglearn

# data,label = mglearn.datasets.load_extended_boston() # 보스턴 데이터를 확장시켜논것, 데이터가 두개가나옴, 튜플로 넘어옴
# # print(data.shape)   # 506, 104 2차원배열
# # print(label.shape)  # 506 1차원배열
#
# # # print(data.count()) # numpyarray 이기에 데이터 프레임으로 변환 해야함
# # print(pd.DataFrame(data).count())
#
# train_data, test_data, train_label, test_label = \
#     train_test_split( data, label, test_size=0.3, random_state=0 )
#
# lr = LinearRegression()
# model = lr.fit(train_data, train_label) # 모델을 만듬
# # print(model)
# print(model.score(train_data, train_label)) # 0.95
# print(model.score(test_data, test_label)) # 0.64
# # 정답률이 올라간 대신 과적합이 더 심해짐
#
#
# # 릿지회귀
# from sklearn.linear_model import Ridge # 릿지회귀, cv-> 교차검증(옵션값이 있는애들은 최적의 옵션 찾기가 힘듬) - 한곳애 때려박고 값을 한곳에 때려박음
# # ridge = Ridge()                 # 기본값 1.0, 0.88, 0.78
# # ridge = Ridge(alpha = 0.1)      # 기본값 0.1, alpha -> 0.92, alpha -> 0.79
# # ridge = Ridge(alpha = 0.01)     # 기본값 0.01, alpha -> 0.94, alpha -> 0.74
# ridge = Ridge(alpha = 10)         # 기본값 10.0, alpha -> 0.77, alpha -> 0.67
# model = ridge.fit(train_data, train_label)
# # 과적합을 줄이는 대신에 정답률이 떨어짐, 기본값 1.0
# print(model.score(train_data, train_label))
# print(model.score(test_data, test_label))


from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mglearn
# 멘하탄 임대료

url = "https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/manhattan.csv"
manhattan = pd.read_csv(url) # url 사용
# print(manhattan.head())
# print(manhattan.shape) # (3539, 18)
corr = manhattan.corr()
# print(corr)
# print(corr["rent"].sort_values(ascending=False)) # "rent"값만을 가져오고, 내림차순 정렬해라
corr_df = pd.DataFrame(corr["rent"].sort_values(ascending=False))

# plt.figure(figsize=(6, 4))
# plt.bar(corr_df.index, corr_df["rent"]) # 바차트
# plt.xticks(rotation=45)
# plt.show()

data = manhattan[["bedrooms","bathrooms","size_sqft","min_to_subway","floor",
                  "building_age_yrs","no_fee","has_roofdeck","has_washer_dryer",
                  "has_doorman","has_elevator","has_dishwasher","has_patio","has_gym"]]
label = manhattan["rent"] # 렌트값만 뺀것이 라벨

# 나누기
train_data, test_data, train_label, test_label = \
    train_test_split( data, label, test_size=0.3, random_state=0)
lr = LinearRegression()
model = lr.fit(train_data, train_label)
print(model.score(train_data, train_label)) # 0.78
print(model.score(test_data, test_label)) # 0.77

# 가장 많은것만 분류
data = manhattan[["bedrooms", "bathrooms", "size_sqft"]]
label = manhattan["rent"]
train_data, test_data, train_label, test_label = \
    train_test_split( data, label, test_size=0.3, random_state=0)
lr = LinearRegression()
model = lr.fit(train_data, train_label) 
print(model.score(train_data, train_label)) # 0.76
print(model.score(test_data, test_label)) # 0.75

# predicts = model.predict(train_data)
# plt.scatter(train_label,predicts)
# plt.xlabel("price")
# plt.ylabel("predict", alpha=0.5)
# plt.show() # 높은 가격대에서 정답률이 낮아짐

# 예측시작, 선형회귀
my = [[1, 1, 600]]
print(model.predict(my)) # [3102.81035166]
