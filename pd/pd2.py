import numpy as np
import pandas as pd
df = pd.DataFrame(
        [["kim", 89, 79, 78],
         ["lee", 46, np.NaN, 65],
         ["park", 79, 85, np.NaN],
         ["hong", np.NaN, 95, 98],
         ["kang", np.NaN, np.NaN, np.NaN]], columns=["name", "kor", "eng", "mat"]
    )
print(df.info())
print(df.describe()) # 표준편차
print(df.count()) # 결측값 구할땐 count 함수를 쓰자, numpy는 없음
print(df.count(axis=1)) # 열끼리 계산
print(df.sum()) # 문자열은 더해버림( 다합쳐짐 )
print(df[["kor","eng","mat"]].sum(axis=1)) # 칼럼이름을 제외한 정수들의 합, 열끼리의 계산
print(df)

# min max
print(df.min())
print(df.min(axis=1)) # 경고가 생김, 숫자중에서 제일 작은거(최소값)

print(df.min())
print(df.max(axis=1)) # 최대값

print(df.mean()) # 평균
print(df.mean(axis=1)) # 열끼리 계산의 평균

# 칼럼추가
# df["tot"] = df.sum(axis=1) # 세로 한줄을 추가해라, 합계값
# df["avg"] = df.mean(axis=1) # 세로 한줄 추가, 평균

# 칼럼 이름 추가해 경고 문구 없음
df["tot"] = df[["kor","eng","mat"]].sum(axis=1) # 2차원이기에 배열기호 추가 및 칼럼이름 추가
df["avg"] = df.loc[:, "kor":"mat"].mean(axis=1) # 이름추가, loc이용 평균 값 구하기

print(df.median()) # 행끼리의 중간값, 
print(df.median(axis=1)) # 열끼리의 중간값

# 편차
print(df.mad()) # 거리의 제곱의 합 -> 편차

# 분산
print(df.var()) # 기본적으로 axis는0, axis=1 줄 수 있음
print(df.var(axis=1))

# 표준편차
print(df.std())

# 누적합
print(df.cumsum())
# print(df.cumprod()) # 누적곱 -> 에러
print(df.loc[:, "kor":"mat"].cumprod()) # NaN은 빼고 계산

# 가장 큰것의 인덱스를돌려줌
print(df.loc[:, "kor":"mat"].idxmax()) # axis 가능
# 가장 작은 것의 인덱스 
print(df.loc[:, "kor":"mat"].idxmin()) # axis 가능

print(df)


# 가설 새울때의 상관관계를 분석해서 같이 늘어야한다
# 상관관계분석
# ex) 국어 점수가 증가하면 영어점수도 증가하느냐
# 1에 가까울수록 양의 상관관계, -1에 가까울수록 음의 상관관계, 0 이면 관계가 낮다.
print(df["kor"].corr(df["eng"])) # 데이터를 거꾸로 주어서 -1 출력
print(df["kor"].corr(df["avg"])) # 평균에 미치는 영향
print(df["mat"].corr(df["avg"]))
print(df["eng"].corr(df["avg"]))
print()
# 공분산
# 수치가 크면 좋다.
print(df["kor"].cov(df["avg"])) # 평균에 미치는 영향
print(df["mat"].cov(df["avg"]))
print(df["eng"].cov(df["avg"]))
print()

print(df.sort_index(ascending=False)) # 인덱스로 정렬해라, 행끼리 할지 열끼리 할지 axis로 결정가능, 내림차순 해라
print(df.sort_index(axis=1))
print(df.sort_values(by="kor")) # 값 정렬, 국어점수를 가지고 정렬, axis 불가능, 칼럼이름이기때문, 인덱스를 잡으면 가능

print(df["kor"].unique()) # 충돌이 나지 않는 값이 무엇인가
print(df["kor"].value_counts()) # 값들이 몇개씩 있는지, NaN 제외
print(df["kor"].isin([46, 50, 60, 70])) # 값들중에 포함되 있는 값이 있는지 판단 True, False

print(df.loc[df["kor"].isin([46,56,66,76]), "kor":"mat"]) # 조건을 줄때 True,False 판단

print()
print(df)

print()



df = pd.DataFrame(np.random.randn(4, 3), columns=["A","B","C"],
                  index=["kim", "lee", "hong", "park"])
print(df)
# 데이터 프레임에 모두 적용하고싶다.
# 함수를 하나 만들어야함
# 람다함수 이용
def func(x): # 데이터를 한줄을받음, 가로든 세로든, 그러면 데이터가 여러개이며 리스트라는 의미임
    return x.max() - x.min() # 최대값에서 최소값을빼 리턴함
print(df.apply(func)) # apply임
print(df.apply(func, axis=1))
    
print(df.apply(lambda x : x.max()-x.min(), axis=1)) # lambda 매개변수 : 리턴값(익명함수)


# Pandas Plot
# 도표로 만들거 생성
np.random.seed(0) # 아무값이나, 랜덤값 고정임
df = pd.DataFrame(np.random.rand(100,3), index=pd.date_range("9/1/2022", periods=100), columns=["A","B","C"]).cumsum() # 날짜를 나열함 월/일/년, 100개, 누적합

print(df.tail()) # 기본 5개를 보여주어라

import matplotlib.pyplot as plt

df.plot() # 메트플롯 라이브러리에 도움을 받아야함 

# plt.xlabel("Time") # 한글쓰면 에러남, 나중에 설정할 예정
# plt.ylabel("Value")
# plt.title("PLOT") # 차트의 이름
# plt.show() # 도표출력

import seaborn as sns
iris = sns.load_dataset("iris")
titanic = sns.load_dataset("titanic")
# print(iris)

# varchar2
# 데이터가 하나일경우
# iris["sepal_length"][:20].plot(kind="bar", rot=0) # 20개만 찍어라
# plt.show()


# 데이터가 여러개일경우
# iris[:5].plot(kind="bar")
# iris[:5].plot.bar(rot=0) # 이런식으로 작성해도 똑같다. rot=0 -> 안주면 누움
# plt.show()

# # 옆으로 그리는
# iris[:5].plot(kind="barh")
# plt.show()

# df = titanic.pclass.value_counts() # 타이타닉의 pclass라는 칼럼을 가져와라
# # print(df)
# df.plot.pie( autopct="%.2f%%") # 퍼센테이지 및 소수점 출력(2자리), %까지 출력 -> %% 두개 써주어야 % 출력
# plt.show() 


# # 수치의 범위안에 갯수
# iris.plot.hist()
# plt.show() # 범위는 임의로 지정됨


# 밀도차트
# iris.plot.kde()
# plt.show()

# # 이상치
# iris.plot.box()
# plt.show()

# 박스 프롯
iris.boxplot(by="species") # 종류라는 칼럼
plt.show()