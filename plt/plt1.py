import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plt.plot([1,2,3,4])
# plt.plot([1,2,3,4], [1,4,9,16]) # x값, y값
# plt.plot([1,2,3,4], [1,4,9,16], "ro-") # x값, y값, row는 선의 모양을 잡음, ro- 는 선도 그어줌

# numpy
# t = np.arange(0,5, 0.2) # 0~5까지 나열, 0.2씩 잘라라
# plt.plot(t, t,"r--", t, t**2, "bs", t, t**3, "g^") #x좌표, y좌표, r-- 점선, bs는 사각형, g^ 는 위로 뾰족한 삼각형


# 딕셔너리
# data_dict = {"data_x" : [1,2,3,4,5], "data_y":[2,4,6,8,10]} # 틱의 범위와 간격도 정할수있음
# plt.plot("data_x","data_y", data=data_dict, label="Price") # 어떤 데이터를 가지고 있는지, legend를 위해 label 추가
# plt.xlabel("data_x")
# plt.ylabel("data_y")
#
# # 폰트변경
# font1 = {
#     "family" : "serif",
#     "color" : "b",
#     "weight" : "bold",
#     "size" : 14
#     }
# font2 = {
#     "family" : "fantasy",
#     "color" : "deeppink",
#     "weight" : "normal",
#     "size" : "xx-large"
#     }
#
# plt.xlabel("data_x", labelpad=15, fontdict=font1, loc="right")# 라벨 간격 조정, 폰트, 위치 조정
# plt.ylabel("data_y", labelpad=20, fontdict=font2, loc="top")

# plot 추가, legend
# plt.plot([1,2,3,4,5], [-1,-2,-3,-4,-5], "r", label="Count") # 이렇게 추가시 선이 2개그려짐, legend를 위해 label 추가
# plt.legend() # 두개가 안나오고 하나만나옴, 각각 선에다 라벨을 붙여주어야함
# plt.legend(loc=(0.5, 0.5)) # 가운데 붙음
# plt.legend(loc="best") # 선이 없는곳으로 알아서 옮김
# plt.legend(loc="upper right") # 상단의 오른쪽
# plt.legend(ncol=2) # 여러줄로 보일지 한줄로 보일지 정함, 기본값이1, 한줄에 하나만 찍어라
# plt.legend(fontsize=14) # 폰트사이즈
# plt.legend(fontsize=14, frameon=False) # 테두리를 없앨것인가
# plt.legend(shadow=True) # 그림자


# x축 y축 값의 범위 지정, limit
# plt.xlim([0, 10]) # 0부터 10까지
# plt.ylim([-10, 20]) # 0부터 20까지, -가 있는것이면 -값 사라짐
# plt.axis([0, 10, -10, 20]) # axis를 이용해 한번에 지정가능
# plt.axis("auto") # 기본이 auto
# plt.axis("equal") # x축과 y축 범위를 중간값으로?
# plt.axis("square") # x축과 y축 범위를 사각형으로?
# plt.axis("scaled") # 세로로 길어짐
# plt.axis("tight") # 꽉차게 그려라

#---------------------------------------------------------------------------------------------------------------------

# 라인스타일
# plt.plot([1,2,3], [4,4,4], linestyle="solid", label="solid") # 라인의 모양 지정
# plt.plot([1,2,3], [3,3,3], linestyle="dashed", label="dashed")
# plt.plot([1,2,3], [2,2,2], linestyle="dotted", label="dotted")
# plt.plot([1,2,3], [1,1,1], linestyle="dashdot", label="dashdot")


# 이런식으로 작성해도 라인스타이이 똑같다
# plt.plot([1,2,3], [4,4,4], "-", label="solid") # 라인의 모양 지정
# plt.plot([1,2,3], [3,3,3], "--", label="dashed")
# plt.plot([1,2,3], [2,2,2], ":", label="dotted")
# plt.plot([1,2,3], [1,1,1], "-.", label="dashdot")


# 튜플로 지정
# plt.plot([1,2,3], [4,4,4], linestyle=(0, (1,1)), label="(0, (1,1))") # 라인의 모양 지정, 튜플로 작성
# plt.plot([1,2,3], [3,3,3], linestyle=(0, (1,5)), label="(0, (1,5))")
# plt.plot([1,2,3], [2,2,2], linestyle=(0, (5,1)), label="(0, (5,1))")
# plt.plot([1,2,3], [1,1,1], linestyle=(0, (3, 5, 1, 5)), label="(0, (3, 5, 1, 5))")


# 라인스타일
# plt.plot([1,2,3], [4,4,4],"-", linewidth=10, solid_capstyle="butt") # 라인의 모양 지정
# plt.plot([1,2,3], [3,3,3],"-", linewidth=10, solid_capstyle="round")
# plt.plot([1,2,3], [2,2,2],"--", linewidth=10, dash_capstyle="butt")
# plt.plot([1,2,3], [1,1,1],"--", linewidth=10, dash_capstyle="round")

# 마커 -> 점의 모양
# plt.plot([1,2,3], [4,4,4],"-", marker="o" )
# plt.plot([1,2,3], [3,3,3],"-", marker="^")
# plt.plot([1,2,3], [2,2,2],"--", marker="v")
# plt.plot([1,2,3], [1,1,1],"--", marker="<")

# 컬러, 마커, 라인스타일을 한번에 지정
# plt.plot([1,2,3], [4,4,4],"bo-") 
# plt.plot([1,2,3], [3,3,3],"g^-")
# plt.plot([1,2,3], [2,2,2],"rv--")
# plt.plot([1,2,3], [1,1,1],"y<--")

# 마커
# plt.plot([1,2,3], [4,4,4], marker="H", color="limegreen") 
# plt.plot([1,2,3], [3,3,3], marker="d", color="violet")
# plt.plot([1,2,3], [2,2,2], marker="x", color="dodgerblue")
# plt.plot([1,2,3], [1,1,1], marker=11)
# plt.plot([1,2,3], [1,1,1], marker="$z$", color="#ff0000") # $사이에 문자를 넣으면 문자로도 넣을 수 있다.

# 색 체우기
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 7, 9, 11]
# plt.plot(x, y)
# plt.fill_between(x[1:4], y[1:4], alpha=0.5) # x, y 의 범위 지정후, alpha(투명도 지정) 
# plt.fill_betweenx(y[1:4], x[1:4], alpha=0.5) # y값을 채움
# z = [ 4, 5, 8, 11, 14]
# plt.plot(x, z)
# plt.fill_between(x[1:4], y[1:4], z[1:4], alpha=0.5, color="lightgray") # 값을 하나 더주어 선과 선 사이의 간격에 색을 채움

# 강제로 체우기
# plt.fill([1.9, 1.9, 3.1, 3.1], [1.0, 4.0, 6.0, 3.0], color="magenta", alpha=0.5) # 강제로 영역을 지정해 색을 체움


# 스케일
# x = np.linspace(-10, 10, 100) 
# y = x**3
# plt.plot(x,y)
# plt.xscale("symlog")
# plt.yscale("log")

# # 그리드
# plt.grid(True, axis="y", color="red", alpha=0.5, linestyle="--") # 배경 변경, axis를 주면 세로 선이 사라짐

# x = np.arange(0, 2, 0.2) # arange 나열해라, linspace 잘라라
# plt.plot(x, x, "bo")
# plt.plot(x, x**2, "r*")

# # 틱 지정
# # plt.xticks([0, 1, 2, 3]) # 강제로 x의 세부적인 틱 지정 
# # plt.yticks(np.arange(0, 5))

# # 대신 숫자는 맞아야한다, 안맞으면 에러가난다.
# plt.xticks(np.arange(0, 2, 0.5), labels=["Jan", "Feb", "Mar","Apr"]) # 0.5씩 증가하면서 0.5마다 이름 지정
# plt.yticks(np.arange(0, 5), ("0GB", "1GB", "2GB", "3GB", "4GB")) # 라벨스라고 리스트로주어도됨, 혹은 튜플로 주어도됨
# plt.tick_params( axis="x", direction="inout", length=10, pad=6, labelsize=12, labelcolor="green", top=True) # in , out, inout, 눈금 색깔, top은 top에도 눈금을 주어라
# plt.tick_params( axis="y", direction="inout", length=10, pad=10, labelsize=12, labelcolor="green", right=True)
# # 기준접잡기, 강제로 선 긋기
# plt.axhline(2, 0.1, 0.9, color="red", linestyle="--", linewidth=2) # 좌표, 선을그림, 가로선
# plt.vlines(0.5, 1, 3, color="red", linestyle=":", linewidth=2) # 세로선
# plt.legend()


# 바차트 ( 막대그래프 )
x = np.arange(3)
y = [100, 300, 700] # 3개의 대한 값
years = ["2020", "2021", "2022"]

# # plt.bar(x,y)
# # plt.xticks(np.arange(3), years) # x틱 변경
# # plt.bar(x, y, color="y")
# # 막대마다 색 다르게
# colors = ["y", "b", "g"]
# # plt.bar(x,y, color=colors, width=0.5) # 폭도 줄임, 기본사이즈는 0.8정도
#
# plt.bar(x, y, align="edge", edgecolor="r", linewidth=5, tick_label=years) # edge => 끝에 붙어라, 테두리컬러 red, 두깨는 5,



# 바차트 옆으로
# plt.barh(x, y)
# plt.yticks(np.arnage(3), years)
# plt.barh(x, y, align="edge", edgecolor="#eee", linewidth=5, tick_label=years)


# 산점도
# np.random.seed(0)
# n = 100
# x = np.random.rand(n)
# y = np.random.rand(n)
# plt.scatter(x, y, alpha=0.5) # 스케터차트
# area = (30 * np.random.rand(n)) ** 2 # 점들의 크기를 랜덤
# plt.scatter(x, y, s=area, alpha=0.5) # 점들의 크기를 랜덤

# 좌표에 점찍기
# plt.plot([1],[1], 'o',markersize=20, c="r")
# plt.scatter([2],[1], s=400, c="b")

# # 해당 좌표에 문자 찍기
# plt.text(1.0, 1.01, "Plot", fontdict={"size":14})
# plt.text(2.0, 1.01, "Scatter", fontdict={"size":14})

# # 리미트 지정
# plt.axis([0.5, 2.5, 0, 2])

# colors = np.random.rand(n) # n개만큼 컬러 지정
# plt.scatter(x, y, c=colors, alpha=0.5)
# plt.colorbar() # 컬러바 출력

# from mpl_toolkits.mplot3d import Axes3D

# n = 100
# xmin, xmax, ymin, ymax, zmin, zmax = 0, 20, 0, 20, 0, 50
# cmin, cmax = 0,2 # 해당범위로 랜덤발생
# x = np.array([(xmax - xmin ) * np.random.random_sample() + xmin  for _ in range(n)]) # _ 는모든범위라는뜻 랜덤값 발생
# y = np.array([(ymax - ymin ) * np.random.random_sample() + ymin  for _ in range(n)])
# z = np.array([(zmax - zmin ) * np.random.random_sample() + zmin  for _ in range(n)])
# color = np.array([(cmax - cmin ) * np.random.random_sample() + cmin  for _ in range(n)])
# # 도표가 하나씩 뿌리기 때문에 창을 잘라서 도표를 여러개 뿌리도록
# fig = plt.figure(figsize=(6, 6)) # 도화지를 가져옴, 인치
# ax = fig.add_subplot(111, projection="3d") # 1행 1열의 1번째
# # ax.scatter(x, y, z, c=color, marker="o", s=15)
# # ax.plot(x, y, z) # 가능
# ax.bar(x, y, z) # 가능





# 히스토그램 - 범위의 갯수표현
np.random.seed(0)
n = 100
x = np.random.rand(n)
y = np.random.rand(n)

weights = np.array([(150-40) * np.random.random_sample() + 40 for _ in range(n)])
# plt.hist(weights)
# plt.hist(weights, bins=20) # 좀더 세밀하게
# plt.hist(weights, bins=20, cumulative=True) # 누적해서 나옴, 기본값은 False

# 두가지 데이터 한번에 표시
# weights1 = np.array([(150-40) * np.random.random_sample() + 40 for _ in range(n)])
# plt.hist((weights, weights1), label=("weights", "weights1"))
# plt.hist((weights, weights1), label=("weights", "weights1"), histtype="bar") # bar 기본값
# plt.hist((weights, weights1), label=("weights", "weights1"), histtype="barstacked") # 쌓아라
# plt.hist((weights, weights1), label=("weights", "weights1"), histtype="step") # 각각의 값들을 봄
# plt.hist((weights, weights1), label=("weights", "weights1"), histtype="stepfilled", alpha=0.5) # 겹쳐서 보여줌

# 정규화, 최대값,최소값 정규화, 
weights1 = np.array([(150-40) * np.random.random_sample() + 40 for _ in range(n*10)]) # 수치를 바꿔주어야함
plt.hist((weights, weights1), label=("weights", "weights1"), histtype="step", density=True)












plt.legend()
plt.tight_layout() # 글씨 잘리는거 사라짐
plt.show()