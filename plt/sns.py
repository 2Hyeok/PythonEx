import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset("iris")
titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
flights = sns.load_dataset("flights")

# 러그 플룻
# print(iris.head())
# x = iris.petal_length.values # 시디즈
# sns.rugplot(x)

# sns.rugplot(data=iris, x="petal_length") # 이런식으로 작성해도 똑같다.

# 커널밀도
# sns.kdeplot(data=iris, x="petal_length") # 곡선이 출력

# 둘다 표시, 디스트플롯
# x = iris["petal_length"].values
# sns.distplot(x, kde=True, rug=True) # x값을 따로 만들어도됨

# 카운트플롯
# plt.hist와 같음
# sns.countplot(data=titanic, x="class")
# sns.countplot(data=tips, x="day")


# 조인트플롯
# sns.jointplot(iris["petal_length"], iris["petal_width"], alpha=0.5)
# plt.suptitle("jointplot")
# sns.jointplot(data=iris, x="petal_length", y="petal_width", kind="scatter")
# sns.jointplot(data=iris, x="petal_length", y="petal_width", kind="kde")



# 페어플롯 데이터들의 분포 확인
# sns.pairplot(iris) # 데이터들의 도표를 다 보여줌
# sns.pairplot(iris, hue="species", markers=["o", "s","D"]) # legend가 자동으로나옴

# 히트맵
# print(titanic.head())
# titanic_size = titanic.pivot_table(index="class", columns="sex", aggfunc="size") # 새로 테이블을 만들고 정의함
# print(titanic_size)
# sns.heatmap(titanic_size, cmap=sns.light_palette("gray", as_cmap=True), annot=True, fmt="d") # 정수로 찍어라


# 바플롯
# sns.barplot(data=tips, x="day", y="total_bill")
# sns.barplot(data=tips, x="day", y="total_bill", hue="sex")

# 박스플롯
# sns.boxplot(data=tips, x="day", y="total_bill")
# sns.boxplot(data=tips, x="day", y="total_bill", hue="sex")

# 바이올린 플롯
# sns.violinplot(data=tips, x="day", y="total_bill")
# sns.violinplot(data=tips, x="day", y="total_bill", hue="sex")
# sns.violinplot(data=tips, x="day", y="total_bill", hue="sex", split=True)


# 스트립플롯
# sns.stripplot(data=tips, x="day", y="total_bill", jitter=True, alpha=0.5)
# sns.stripplot(data=tips, x="day", y="total_bill", jitter=True, alpha=0.5, hue="sex")
sns.stripplot(data=tips, x="day", y="total_bill", jitter=True, alpha=0.5, hue="sex", dodge=True) # 겹치는것들이 있음, 옵션을 통해 겹치는것을 옆으로 좌표를 옮김


# 스웜 플롯
# sns.swarmplot(data=tips, x="day", y="total_bill",size=3)
# sns.swarmplot(data=tips, x="day", y="total_bill",size=3, hue="sex")
# sns.swarmplot(data=tips, x="day", y="total_bill",size=3, hue="sex", dodge=True) # hue라고 주면 기준을잡아서 나눠서 출력해줌




# 캣플롯
data = titanic[titanic.survived.notnull()]
# sns.catplot(data=data, x="age", y="sex", hue="survived")
# sns.catplot(data=data, x="age", y="sex", hue="survived", kind="violin")
# sns.catplot(data=data, x="age", y="sex", hue="survived", kind="swam", split=True)
sns.catplot( data=data, x="age", y="sex", hue="survived", kind="swarm", split=True, height=2, aspect=4 )



# sns.boxplot(data=tips, x="tip", y="day", whis=np.inf)
# sns.stripplot(data=tips, x="tip", y="day", color="0.4", alpha=0.5)
# sns.violinplot(data=tips, x="tip", y="day", inner=None, alpha=0.5)
# sns.swarmplot(data=tips, x="tip", y="day", color="0.9", alpha=0.5)


# 스타일 지정
# sns.set_style( "ticks" )
# sns.set_style( "darkgrid" )
# sns.set_style( "whitegrid" )
# def sinplot( flip=1 ) :
#     x = np.linspace( 0, 14, 100 )
#     for i in range( 1, 7 ) :
#         plt.plot( x, np.sin( x+i*0.5) * (7-i) * flip )
# sinplot()        


plt.show()