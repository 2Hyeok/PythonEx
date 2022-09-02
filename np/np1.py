import numpy as np # 임포트 필요

a = [[1,2,3,4], [5,6,7,8]] # 이게 리스트임
print(a)

# nparray
a=np.arange(15) # 나열해라 0부터 14까지
print(a)
print(type(a)) # 리스트랑 다른것임

a=np.arange(15).reshape(3, 5) # 행열을 바꿈 3행 5열로
print(a) # 2차원 배열로 출력, 콤마가 없음
# 출력이 다름
print(a.shape) # 행열 형태
print(a.ndim) # 축의 갯수
print(a.dtype) # 데이터 타입
print(a.itemsize) # 아이템의 사이즈 4바이트
print(a.size) # 전체 갯수가 몇개인지

# numpy 연동
t = (10, 20, 30, 40, 50)
print(type(t))
tt = np.array(t) # numpy로 변경
print(type(tt))

s = {10,20,30,40,50} # set
print(type(s))
ss = np.array(s)
print(type(ss))

m = [10, 20, 30, 40, 50] # list
# print(m.shape) # 리스트라 불가능 numpy만 가능함
mm = np.array(m)
print(mm.shape)

w = [-5, 9, 8, 7, 5, -6, -7, 3, -5, 2] # sort 할것임
print(np.shape(w))
print(np.abs(w)) # 절대값
print(np.square(w)) # 제곱
print(np.sqrt(w)) # 루트, 음수는 루트가 불가.
print()
print(np.sqrt(np.abs(w))) # 루트, 음수는 루트가 불가. 절대값으로 변경해서 출력
# print(help(np)) # numpy의 대한 설명
# print(help(np.shape)) # np.shape의 대한 설명

print(np.isnan(w)) # 숫자가 아니냐, 숫자이기에전부 false
print(np.sum(w)) # 합계
print(np.mean(w)) # 평균
print(np.max(w)) # 최대값
print(np.min(w)) # 최소값
print(np.argmax(w)) # 인덱스값 제일큰것이 몇번 인덱스 인지
print(np.cumsum(w)) # 누적합
# w = sorted(w) # 리스트 정렬
# print(w)
# w.sort() # 원본 정렬
# print(w)
print(np.sort(w)) # numpy 정렬
print(np.sort(w)[::-1]) # 내림차순

# 2차원 리스트
w = [[1, 3, 5],[2, 4, 6]]
print(w)
w = np.array(w) # 2차원리스트 Numpy변경
print(w)

w = [1, 2, 3, 4, 5, 6] 
w = np.array(w).reshape(2,3) # 2차원으로 변경 2행 3열
# w = np.array(w).reshape(3,3) # 이건 불가능, 데이터가 부족하기때문
w = w.reshape(3,2) # 행열 바꾸는것은 가능 3행 2열
print(w)

print(np.sum(w)) # 2차원 summ 전체합계가나옴
print(np.sum(w, axis=0))# 행끼리 계산 1 + 3 + 5
                        # 2 + 4 + 6
print(np.sum(w, axis=1))# 열끼리 계산 1+2
                        #          3+4
                        #          5+6
print(np.mean(w))
print(np.mean(w, axis=0))
print(np.mean(w, axis=1))
print(np.median(w)) # 중간값 구하기, 나중에 쓸것임
print(np.median(w, axis=0)) # axis 적용가능

print(np.var(w)) # 분산 - 거리 ** 2 값의 평균 
print(np.std(w)) # 표준편차 - 분산의 제곱근(루트)
print(np.std(w, axis=0)) # 표준편차의 행끼리 계산


m = [10, 20, 30, 40, 50]
# print(m + 2) # 숫자 연산 안됨, 에러발새
m = np.array(m)
print(m + 2) # 연산가능 엘리먼트 와이드, element wise, 원래라면 반복문을 해주어야함


w = [1, 2, 3, 4, 5]
# print(w + 2) # 에러남
print(m + w) # 얘는 가능

print(np.sin(m)) # 함수로계산
print(np.sin(m)*10)

print(m > 20) # 분립연산 가능 ,true, false 로 표현

# 행열 끼리 연산
a = np.array([[1,1],[1,1]])
b = np.array([[1,2],[3,4]])

# 덧샘
print(np.add(a, b))
print(a+b) # 둘다 연산 가능

# 뺄샘
print(np.subtract(a, b))
print(a-b)

# 곱샘
print(np.multiply(a, b)) # 그냥 곱샘
print(a*b)

# 나눗셈
print(np.divide(a, b))
print(a // b)

# 행열끼리의 곱
print(a @ b)
print(a.dot(b))

print(np.zeros(10)) # 배열이 zero인 배열 10개를 만들어라
print(np.zeros((3,3))) # 2차원, 3행 3d열짜리를 만들어라
print(np.ones((3,3,))) # 1이 이들어가는것

# 랜덤
x = np.random.randn(10) # 정규분포의 맞게 0 ~ -1 사이로 10개출력
print(x)
y = np.random.randn(10)
print(np.maximum(x, y)) # 2개의 숫자를 비교해서 큰거를 돌려줌
print(np.minimum(x, y)) # 2개의 숫자릴 비교해서 작은거를 돌려줌


# 인덱싱 슬라이싱
s = "ABCDEFG"
print(s[1]) # 인덱싱
print(s[1:4]) # end 는 -1

a = [ i for i in range (10 , 20) ] # 반복문으로 리스트 생성
print(a[0]) # 인덱싱
print(a[0:5]) # 슬라이싱
print(a[0:-1])
print(a[:6])
print(a[6:])

b = np.arange(10, 20) # ndarray
print(b[0])
print(b[0:5])
print(b[0:-1])
print(b[:6])
print(b[6:])


# 2차원 리스트
a = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]
# print(a)
print(a[2][2]) # 13
print(a[:][:]) # 모든값
print(a[:]) # 똑같음
print(a[2][:]) # 인덱싱2번
print(a[0:2][0:2]) # 비정상적인 출력 0:2 -> 1,2,3,4,5,6,7,8,9,10 -> 0:2 -> 1,2,3,4,5,6,7,8,9,10
                    # 앞에 나온결과에서 뒤에나온결과를 가져온다.
                    # 하지만 판다스에서는 결과가 정상적으로 출력됨
                    
print(a[0:2][1])
# print(a[0:2, 0:2]) # 리스트는 불가 numpy, 판다스에서는 가능

# numpy로 확인
b = np.arange(1,16).reshape(3, 5)
print(b)
print(b[2][2]) # 13
print(b[:][:]) # 모든것
print(b[:]) # 모든것
print(b[2][:]) # 2행의 모든것
print(b[0:2][0:2])
print(b[0:2][1]) # 야기까지는 똑같음
print(b[0:2, 0:2]) # numpy는 가능
print(b[0:-1, 0:-1]) # 행하나 열 하나 빠짐
print(b[1:2, :]) # 1행 2열의 모든것
print(b[2, 2]) # 2의 2번인덱스


# 인덱스를 이용해 배열의 값 가져오기
# a = np.empty((8,4)) # numpy
a = list(np.empty((8,4))) # 리스트
for i in range(8) :
    a[i][:] = i # i번째 방에 i를 넣어라
print(type(a))
print(a)
# print(a[[5,7,2]]) # 에러남

b = np.arange(32).reshape(8,4) # 8행 4열짜리
print(b)
print()
print(b[[5,7,2]]) # 행의 인덱스 5번째 7번째 2번째
print(b[[5,7,2], [1,2,3]]) # 5행의1열, 7행의2열 2행의3열

# 전체 행열, 행열을 바꾼것
print(b)
print(b.T) # 대문자 T 사용시 행열을 바꿈

print()
# 데이터 쌓기 -> Stacking together different arrays
a = np.floor(np.random.rand(2, 2) * 10) # 그냥 버리면 0 나옴
print(a)
b = np.floor(np.random.rand(2, 2) * 10)
print(b)
print(np.hstack((a, b))) # a, b 를 가로로 쌓아라, 튜플로 주어야함
print(np.vstack((a, b))) # a, b 를 세로로 쌓아라, 튜플로 주어야함

# 스플릿
c = np.floor(np.random.rand(2, 12) * 10) # 옆으로 긴
print(c)
print(np.hsplit(c, 3)) # 가로로 잘라라 3등분 해라
print(np.hsplit(c,(3,5))) # 튜플로 주면 경계를 지정할 수 있음 0~2, 3~4, 5~11

d = np.floor(np.random.rand(12, 2) * 10)
print(d)
print(np.vsplit(d, 3)) # 세로로 잘라라 3등분 해라
print(np.vsplit(d,(3,5))) # 튜플로 경계 지정 3칸, 2칸, 6칸

