import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 파일차트
radio = [32, 34, 16, 18]
fruits = ["apple", "banana", "melon", "grape"]
# plt.plot(radio, labels=fruits, autopct="%.1f%%") # 소수점 하나까지 출력,
# plt.plot(radio, labels=fruits, autopct="%.1f%%", startangle=260, countclock=False) # startangle => 시작위치(각도)를 잡을때 사용, 시계반대방향으로 돌아감

explod = [0, 0.1, 0, 0.1]
# plt.plot(radio, labels=fruits, autopct="%.1f%%", explod=explod, shadow=True) # 차트간격 조정

# colors = ["silver", "gray", "lightgray", "darkgray"]
# plt.pie( ratio, labels=fruits, autopct="%.1f%%", colors=colors ) # 차트 색상 변경

# wedgeprops = {"width":0.7, "edgecolor":"w", "linewidth":5} # 딕셔너리로 생성, 가운대가 뚫린 도표
# plt.pie( radio, labels=fruits, autopct="%.1f%%", wedgeprops=wedgeprops )


# 히트맵
# a = np.random.standard_normal((30,40)) # 정규분포에 맞게 랜덤발생해라
# plt.matshow(a)
# plt.colorbar(shaink=0.8, aspect=10) # 색상 구분을 위해 컬러바를 추가해줌
# plt.clim(-3, 3) # 색상표의 최대 최소

# cmap = plt.get_cmap("PiYG") # 이름이 정해져 있어서 정해진 이름으로 사용
# cmap = plt.get_cmap("BuGn")
# plt.matshow(a, cmap=cmap)
# plt.colorbar()
# plt.show()

# 여러 그래프 그리기
# x1 = np.linspace(0, 5)
# x2 = np.linspace(0, 2)
# y1 = np.cos(2 * np.pi * x1) * np.exp(-x1) # pi는 원주율
# y2 = np.cos(2 * np.pi * x2)
#
#
# plt.subplot(2, 1, 1) # plt 힐때는 subplot, 피규어를 받으면 addsubplot, 행, 열, 위치 라는 의미
# plt.plot(x1, y1, "o-")
# plt.title("graph 1")
# plt.xlabel("x1") # 라벨지정
# plt.ylabel("y1")
#
# plt.subplot(2,1,2) # 그래프 추가
# plt.plot(x2, y2, ".-")
# plt.title("graph 2")
# plt.xlabel("x2") # 라벨지정
# plt.ylabel("y2")
# plt.yticks(visible=False) # y틱을 없앰, 순서가중요, 1번도표면 위애다가 주어야함



# 컬러맵
# from matplotlib import cm # 임포트
# 컬러맵의 종류를 확인할 수있는 함수
# cmaps = plt.colormaps()
# for cmap in cmaps : # 컬러들 모두 출력
#     print(cmap, end=" ")

# np.random.seed(0)
# arr = np.random.standard_normal((8, 100)) # 8행의 100열, 정규분포
# plt.subplot( 2, 2, 1 )
#
# # plt.scatter( arr[0], arr[1], c=arr[1], cmap="spring" )
# plt.scatter( arr[0], arr[1], c=arr[1])
# plt.spring() # 함수를 바로 주어도 색상 적용
# plt.title( "spring" )

# plt.subplot( 2, 2, 2 )
# # plt.scatter( arr[2], arr[3], c=arr[3], cmap="summer" )
# plt.scatter( arr[2], arr[3], c=arr[3])
# plt.viridis()
# plt.title( "summer" )

# plt.subplot( 2, 2, 3 )
# # plt.scatter( arr[4], arr[5], c=arr[5], cmap="autumn" )
# plt.scatter( arr[2], arr[3], c=arr[3])
# plt.plasma()
# plt.title( "autumn" )

# plt.subplot( 2, 2, 4 )
# # plt.scatter( arr[6], arr[7], c=arr[7], cmap="winter" )
# plt.scatter( arr[2], arr[3], c=arr[3])
# plt.gray()
# plt.title( "winter" )


# 컬러맵 만들기
# from matplotlib import cm
# from matplotlib.colors import LinearSegmentedColormap
# colors = ["#FF0000", "#FFB2F5", "#0100FF", "#212121"]
# cmap = LinearSegmentedColormap.from_list("my_cmap", colors, gamma=2) # from.list라는 함수를 주어야함
# plt.scatter([1,2,3,4,5],[1,3,5,7,9], c=[8,5,6,5,3], cmap=cmap)


# 텍스트넣기






















plt.tight_layout()
plt.show()
