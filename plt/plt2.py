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
# a = 2.0 * np.random.randn( 10000 ) + 1.0
# b = np.random.standard_normal( 10000 )
# c = 20.0 * np.random.rand( 5000 ) - 10.0
# plt.hist( a, bins=100, density=True, alpha=0.7, histtype="step" ) # density => 정규화해라
# plt.hist( b, bins=50, density=True, alpha=0.5, histtype="stepfilled" )
# plt.hist( c, bins=100, density=True, alpha=0.9, histtype="step" )
#
#
# font1 = { # 딕셔너리로 생성
#     "family" : "serif",
#     "color" : "darkgray",
#     "weight" : "normal",
#     "size" : 16
#     }
# font2 = { # 딕셔너리로 생성
#     "family" : "Times New Roman",
#     "color" : "blue",
#     "weight" : "bold",
#     "size" : 14,
#     "alpha": 0.7
#     }
# font3 = { # 딕셔너리로 생성
#     "family" : "Arial",
#     "color" : "forestgreen",
#     "size" : 14
#     }

# # 글자 박스 삽입
# box1 = {
#    "boxstyle" : "round",
#    "edgecolor" : (1.0, 0.5, 0.5),
#    "fc" : (1.0, 0.8, 0.8)
#     }

# box2 = {
#    "boxstyle" : "square",
#    "edgecolor" : "black",
#    "fc" : "lightgray", # 글자색 
#    "linestyle" : "--"
#     }
#
# box3 = {
#    "boxstyle" : "square",
#    "linestyle" : "-",
#    "linewidth" : 2
#     }
#
# plt.text(1.0, 0.35, "A", fontdict=font1, rotation=85, bbox=box1) # 폰트변경, 기울기 조정
# plt.text(2.0, 0.25, "B", fontdict=font2, rotation=-50, bbox=box2)
# plt.text(5.0, 0.08, "C", fontdict=font3, bbox=box3)




# 그래프 스타일 설정
print(plt.style.available) # 스타일 출력

# use먼저 사용후 그레프를 그려야함
# plt.style.use("bmh")
# plt.style.use("ggplot")
# plt.style.use("classic")
# plt.style.use("grayscale")
# # plt.style.use("Solarize_Light2")
# plt.style.use("default") # 기본 스타일
#
# # 도화지 설정
# plt.rcParams["figure.figsize"] = (10, 5) # 인치, 도화지의 모양(사이즈) 조정
# plt.rcParams["font.size"] = 12
# plt.rcParams["lines.linewidth"] = 3
# plt.rcParams["lines.linestyle"] = "-."
# plt.rcParams["xtick.top"] = True
# plt.rcParams["ytick.right"] = True
# plt.rcParams["xtick.direction"] = "in" # 안으로 넣어라
# plt.rcParams["ytick.direction"] = "in"
# plt.rcParams["xtick.major.size"] = 7 # 전체 나오는 사이즈# plt.rcParams["ytick.major.size"] = 7
# plt.rcParams["xtick.minor.visible"] = True # 마이너눈금, 보여라
# plt.rcParams["ytick.minor.visible"] = True
#
# plt.plot([1,2,3,4,5],[2,4,6,8,10] )
#
# plt.savefig("save_default.png") # 이미지저장, 기본값이 png파일
# plt.savefig("save_default50.png", dpi=50) # 사이즈 조절
# plt.savefig("save_default100.png", dpi=100)
# plt.savefig("save.png", facecolor="#eeeeee", bbox_inches="tight", pad_inches=0.3) # 테두리색상, 타이트하게




# 객체지향 인터페이스
# fig, ax = plt.subplots(2,2, sharex=True, sharey=True) # 피규어(설정)와 ax(도화지, 도표), 도표를 4개그림, 눈금의 공유
# x = np.arange(10)
# ax[0][0].plot(x, np.sqrt(x), label="graph1")
# ax[0][1].plot(x, x)
# ax[1][0].plot(x, -x+5 )
# ax[1][1].plot(x,np.sqrt(x+5))
#
# ax[0][0].set_title("Graph 1") # 이름을 붙일때에는 set_title사용
# ax[0][0].legend() # 라벨도 지정가능


# 이중 y축
# plt.rcParams["figure.figsize"] = (10, 5)
# plt.rcParams["font.size"] = 12
# x = np.arange(10)
# fig, ax1 = plt.subplots()
# ax1.plot(x, x+1, label="y1")
#
# ax2 = ax1.twinx() # ax1을 복사해서 같은 도표에 뿌려라
# ax2.plot(x, -x-1, label="y2")
#
#
# ax1.set_xlabel("x")
# ax1.set_ylabel("y1")
# ax2.set_ylabel("y2")
#
# ax1.legend() # 레전드도 따로따로 줄 수 있다.
# ax2.legend()

# 겹칠때 먼저 뿌랴라
# fig, ax1 = plt.subplots()
# x = np.arange( 2020, 2030 )
# y1 = np.array( [1, 3, 5, 7, 9, 10, 12, 14, 16, 20] )
# y2 = np.array( [1, 3, 7, 9, 11, 13, 16, 18, 20, 25] )
# ax1.plot( x, y1, color="g", marker="o", linewidth=5, alpha=0.5, label="Price" )
# ax1.legend( loc="upper left" )
#
# ax2 = ax1.twinx()
# ax2.bar( x, y2, color="r", label="Count", alpha=0.7, width=0.7 )
# ax2.legend( loc="upper right" )
#
# ax1.set_zorder(ax2.get_zorder() + 10)
# ax1.patch.set_visible(False)
# ax2.patch.set_visible(False)

# from matplotlib import font_manager, rc

# # 폰트 경로확인
# for font in font_manager.fontManager.ttflist:
#     print(font)

# # 한글폰트 지정
# # from matplotlib import font_manager, rc
# font_name = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\HANDotum.ttf").get_name() # 윈도우의 폰트 c->window->font
# rc("font", family=font_name)

# # 라벨 한글로
# # 한글 출력 불가
# ax1.set_xlabel( "연도", fontproperties=font_name )
# ax1.set_ylabel( "가격", fontproperties=font_name )
# ax2.set_ylabel( "수량", fontproperties=font_name )


# 박스플룻
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 12
np.random.seed(0)
a = np.random.normal(0, 2.0, 1000)
b = np.random.normal(-3.0, 1.5, 500)
c = np.random.normal(1.5, 1.5, 1500)
fig, ax = plt.subplots()
# ax.boxplot([a, b, c]) # 기본 1.5배
# ax.boxplot([a, b, c], whis=2.5, notch=True) # 이상치 사라짐, notch-> 중간에 홈이 생김
box = ax.boxplot([a, b, c], whis=1.5, notch=True)
ax.set_ylim()
ax.set_ylim([-10, 10])
ax.set_xlabel("Data")
ax.set_ylabel("Value")

for flier in box["fliers"] : # 이상치를 알려줌
    print(flier.get_data())

for median in box["medians"]: # 중간값을 알려줌
    print(median.get_data())
    
print(box) # 확인용

plt.tight_layout()
plt.show()
