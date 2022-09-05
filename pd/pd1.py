import numpy as np

# where
# 3항연산, numpy
x = np.array([1, 2, 3, 4, 5])
y = np.array([6, 7, 8, 9, 10])
z = np.array([True, False, False, True, False]) # true 면 앞에거, False면 뒤에거를 뽑음

# 참일때 결과가 들어가는 자리가 if 앞에 들어감, True일때 a를 쓰고 False일때 b를 써라
print([a if c else b for a,b,c in zip(x, y, z)]) # 리스트를 만들겠다, 다뽑아야하니 반복문으로, zip은 튜플로 묶임
# 단 2차원 이상일경우 이런방법 불가능

# 그렇기에 where를 써야한다.
print(np.where(z,x,y))

# 2차원 배열 생성
a = np.random.randn(4, 4)
# print(a) # 양수 음수 섞임
print(np.where( a>0, 1, 0)) # where에 바로 조건식 써도 됨, 0보다크면 1로 0보다 작으면 0으로
print(np.where( a<0, 0, a)) # 0보다 작으면 0, 0보다 크면 a 를 바로 출력

# 파일 입출력
# 피클링 이라고하는데
# 이거는 리스트를 뽑는것임
a = np.arange(10)
np.save("np1.npy", a) # 파일 저장, a라는 변수를 저장해라, 리스트가 들어있기에 실제파일은 깨져서 나옴

# 저장한 파일을 불러옴
arr = np.load("np1.npy")
print(arr)

# 리스트 여러개
b = np.arange(10, 20) # 10부터 20까지
np.savez("np2.npz",arr1=a, arr2=b) # 파일을 여러개 저장할경우 savez라고 해야함, 다시 꺼내기 위해 이름을 지정해 주어야함

# 꺼낼때에는 이름을 지정해준것으로 꺼내면됨
c = np.load("np2.npz")
print(c["arr1"])
print(c["arr2"])



# pandas
import pandas as pd

# Series    1차원
# 1차원이기에 리스트와 호환이됨, numpy와 호환되서 변환가능
s = pd.Series([1, 0, 8, -3, -6, 9, 3, -2, 6, -7])
print(type(s)) # pandas 의 series
# print(s) # 인덱스도 사용할 수 있고, 이름도 지정해 줄 수 있음
print(s.index) # 인덱스값도 꺼냄, primary의 개념
print(s.values)
print(type(s.values)) # 타입은 numpy 배열로 나옴
print(s.dtype)

s.index = ["a","b","c","d","e","f","g","h","i","j"] # 인덱스의 이름을 바꿈
print(s)
print(s.index) # 인덱스출력, 타입은 오브젝트형


print(s["a"]) # 인덱스를 사용해 데이터꺼내기
print(s.a) # 이런식으로도 가능
print(s[0]) # 의 0번째방
print(s[0:5]) # 슬라이싱 가능, 인덱스까지 같이 잘림
print(s["a":"e"]) # 이름으로도 슬라이싱 가능, 이건 pandas만 가능



# 딕셔너리를 series로 변환
# 키값은 인덱스가 자동으로 들어감
d = {"A":65, "B":66, "C":67, "D":68} # 딕셔너리

# 딕셔너리
print(type(d))
print(d["A"])
# print(d.A) # 이것도 불가능
# print(d[0:3]) # 반복문, 슬라이싱이 불가능하다
# print(d["A":"C"])

for key,value in d.items() :
    print(key, value) # 딕셔너리는 items를 이용해야 사용가능


# seires로 변경
s = pd.Series(d)
print(type(s))
print(s.index)
print(s.values)
print(s[0:3]) # 시리즈로 바뀌었기에 가능
print(s["A":"C"])


print("여기부터 데이터프레임")
# DataFrame
# 딕셔너리 생성
d = { # 리스트로 생성
    "name" : ["kim", "park", "hong", "lee"],
    "age" : [20, 30, 40, 50],
    "tel" : ["1111-1111", "1111-2222", "2222-1111", "2222-2222"]
    }

# 인덱스는 자동으로 들어가고 , 키값이 칼럼의 이름이됨
# 변환
df = pd.DataFrame(d)
print(df)
print(df.index) # 인덱스는 s가 안붙음
print(df.columns)
print(df.values) # numpy임

df.index.name = "Num" # 이름 지정가능, 잘 쓰지는 않음
df.columns.name="user"
print(df)
print(df.index) # 이름을 지정해주면 이름도 출력가능
print(df.columns)

# 2차원 리스트 생성
# numpy array
n = np.array(
    [["kim",20,"1111-2222"],
     ["lee",30,"2222-1111"],
     ["hong",40,"3333-2222"]]
    )

df = pd.DataFrame(n) # 변환
print(df)
print(df.index) # 자동으로 붙었기에 rangeindex 출력
print(df.columns)

# 이름 지정
df.index = ["a","b","c"]
df.columns = ["name", "age", "tel"]
print(df)

df1= pd.DataFrame(n, index=["a", "b", "c"], columns=["name", "age", "tel"]) # index와 columns를 줄 수 있음 
print(df1)


print(df.info) # 정보를 알려줌, 숫자가 있으면 숫자의 대한 기본적인 통계를 알려줌
print(df.describe()) # 통계를 내줌, count를 쓰거나, describe를 사용
# 결과 값도 데이터프레임임
# 이 값도 받아서 사용가능
result = df.describe()
print(result.index)
print(result.columns)

print()
print(df)
print(df["name"]) # name이 나옴, 데이터 프레임에서 한줄을꺼내면 series가됨
print(df["name"]["a"]) # name의 a번째방
print(df["name"].a)
print(df["name"][:2]) # 인덱싱가능
print(df["name"]["b":"c"]) # 이름으로 했기에 -1로 안나옴


# 데이터프레임 슬라이싱
print()
print(df)
# print(df["name","age"]) # 한줄이 아니기에 2차원으로 가야함
print(df[["name","age"]])
print(df.name, df.age) # 2줄을 따로 꺼내라, 따로따로 출력됨
print(df[:][:])
print(df[:2]) # 2행만 짤라라
print(df[:2][:2]) # 뒤에는 열이 되는게 아님, 앞의 처리값의 뒤를 처리해라가됨
# print(df[:2, :2]) # numpy는 가능

print(df.values[:2,:2]) # 예는 numpy임


# 데이터프레임에 추가 삭제
df["address"] = ["서울", "수원", "인천"] # 추가함, 갯수를 맞춰주어야함
print(df)
df["adult"] = df["age"] >= "30" # age가 30 이상이면 True False 로 출력
print(df)
df["age"][0] = 40 # age의 0번방을 40으로 바꿔라
df["adult"][0] = True # adult의 0번방의 값을 True로 바꿈
del(df["adult"]) # adult를 지워라
print(df)

# index를바꿈
df.index= ["A", "B", "C"] # 데이터의 갯수만큼 써야함 3개이기에 3개로
print(df)
index = ["I","II","III"]
df.reindex(index) # reindex도 같은개념임, 인덱스를 바꿈
df.reset_index() # 원래대로 되돌림
print(df)



print()
print()
print("정렬")


# 정렬
s = pd.Series(range(4), index=["b","d","c","a"])
s.sort_index() # 기본출력, 원본을 바꾸지 않음
s = s.sort_index()
s = s.sort_index(ascending=False) # 오름차순을 하지 말아라, 내림차순해라
s = s.sort_values() # 값으로 정렬
s = s.sort_values(ascending=False) # 오름차순을 하지 말아라, 내림차순해라
print(s)

df = pd.DataFrame(np.arange(12).reshape(3,4),
                  index=["b", "c","a"],
                  columns=["B","A","D","C"]) # 3행4열

df = df.sort_index()
print(df)
df = df.sort_index(axis=1) # sort_columns가없고 axis를 이용해 열끼리 sort가됨
print(df)


df = df.sort_values(by="A", ascending=False) # 값이 여러개라 기준값을 잡아주어야함, 내림차순까지
print(df)


df = df.sort_values(by="a", axis=1) # 기준이 a 이며 열끼지 정렬
print(df)

df = df.sort_values(axis=1, by=["a", "b"]) # a를 기준으로잡은 후 b를 기준으로 잡아라
print(df)

print("여기부터")
# loc / iloc
print()
# 2차원 리스트
m = [[1,2,3], [4,5,6], [7,8,9]]
print(m)
print(m[1][2])

# 인덱싱, 슬라이싱이 좀 다름
# 행열의 개념이 아님
print(m[0:2][0:2])
# print(m[0:2, 0:2]) # 에러

n = np.array(m) # numpy
print(n)
print(n[1][2])
print(n[0:2][0:2])
print(n[0:2, 0:2]) # numpy는 가능

# w = pd.DataFrame(m)
w = pd.DataFrame(m, index=["a","b","c"], columns=["A","B","C"]) # pandas
print(w) # 자동인덱스 생성
# print(w[1][2]) # 앞에가 행, 컬럼의 이름을 먼저쓰고 인덱스를씀, w[컬럼의 이름][인덱스], 이름을 지정해 준 상태라면 에러
print(w["A"][2]) # 지정해준 칼럼의 이름을 삽입해주어야함

print(w[0:2][0:2])
# print(w[0:2, 0:2]) # 에러, 이런식으로 자를땐 loc or iloc를 써야함


print()
df = pd.DataFrame(np.arange(10,26).reshape(4,4), columns=["A","B","C","D"])
print(df)
# print(df[0][0]) # 행열의 개념이 아니기에 불가
print(df["A"][0]) # 인덱싱 / 10
# print(df[["A":"C"]]) # 불가, loc사용시가능
print(df[["A", "D"]])
print(df[["A", "D"]][0:2]) # 가져온것을 0:2로 잘라라 [컬럼][인덱스]
# print(df[["A",0:2]]) # 콤마를 주고 슬라이싱을 줄 수 없음

# iloc - 숫자로만 가능
print()

# iloc - 인덱싱
# print(df[1,1]) # 에러
# print(df.loc[1,1]) # 이름이 있다면 이름으로 써야함
print(df.iloc[1,2]) # 1행의 2열, [행, 열]
# print(df.iloc[1, "A"]) # 칼럼의 이름으로는 못씀, 숫자로만가능
# print(df.iloc["A",1])

# iloc - 슬라이싱
print(df.iloc[:2, :2])
print(df.iloc[:2, :-1]) # 열에서 하나가 빠짐
print(df.iloc[:-1, :-1])

print()
print()
print()
print()

# loc
# 인덱싱
print()
print(df[0:2][0:2]) # 슬라이싱을 사용시 열 이름이됨
print(df[["A","C"]])
print(df.iloc[1,1])
print(df.iloc[2:,2:])
# print(df.iloc["A",1]) # iloc는 에러
print(df.loc[1][1])
# print(df.lic[1, 1]) # 에러
# print(df.loc["A",1]) # 에러
# print(df.loc["A"][1]) # 에러
print(df.loc[1]["A"]) # A는 컬럼의 이름임 loc[인덱스][칼럼명], 인덱스가 1번인것의 A, 1번을 먼저 꺼낸 후, A라는 애의 값

print(df["A"][1]) # [칼럼][인덱스]
print(df.loc[1]["A"]) # [인덱스][칼럼]
print(df.iloc[2][1]) # [인덱스][칼럼]
# print(df[1]["A"]) # 에러


# 슬라이싱
print()
print(df.loc[:2, :"B"])
print(df.loc["A":"C"]) # 데이터가 안나옴, 앞에가 인덱스 위치인데 칼럼값을줌
# print(df.loc[:2]["A":"C"]) # 에러
# 앞에 인닥스 값을 주어야함
print(df.loc[:,"A":"C"]) # 이런식으로 작성
print(df.loc[:2])
# print(df.loc[:2].loc["A":"C"]) # 값이 이상하게 나옴
print(df.loc[:2][["A","B"]]) # 2차원으로 뽑아서 씀, 이렇게복잡하게 쓸 일은 없음

# 데이터 추가
print()
print(df)
df["E"] = [30, 31, 32, 33] # 데이터 프레임은 칼럼의 이름이 먼저임
df.loc[:, "F"] = [40, 41, 42, 43] # loc의 데이터 추가, 앞에 인덱스가 먼저오기에 순서가 반대임, 열추가
df.loc[0, "A"] = 99 # 0번인덱스의 A라는 컬럼의 값을 99로 바꿔라
df.loc[3,["A", "C", "E"]] = [50, 51, 52] # 3번인덱스의 A C E를 각각 리스트로 만들어 바꾸어줌

# print(df[:][4]) # 인덱스가 없음
# df[:][4] = [60, 61, 62, 63, 64, 65] # 에러
df.loc[4] = [60, 61, 62, 63, 64, 65]

del(df["F"]) # 삭제
# df.drop("E", axis=0) # drop은 칼럼과 인덱스를 안나눔, E라주고 행을 지우라하면 안됨, 지울행이 아닌 컬럼의 이름임, 세로를지워라 밖에 안됨
df = df.drop("E", axis=1) # 원본에 한번 받아서 출력해야함
# df.drop(1, axis=1) # 옆으로 지우라해야지 세로로 지우라 하면 안됨
df = df.drop(1, axis=0)
print(df)

print()
print()
print()
print()



print(df.loc[::-1, ::-1]) # 인덱스값이 거꾸로 찍힘
print(sorted([4,6,8,2,4])[::-1])
df = pd.DataFrame(
    [["kim", 20, "1111-2222"],
     ["lee", 30, "2222-1111"],
     ["park", 40,"1111-3333"],
     ["hong", 35, "2222-3333"]
     ], columns=["name", "age", "tel"]
    )

# boolean indexing
print()
print(df)
print(df.iloc[:, 0:2]) # 모든 행의 0부터 2까지 출력해라
print(df.loc[:,"name":"age"]) # name부터 age까지
# print(df.loc[:, 0:2]) # 에러
# print(df.iloc[:, "name":"age"]) # 에러


print()
# boolean indexing
print()
print(df)
print(df.loc[df["age"] >= 30, ["name", "age"] ]) # age한줄을 꺼내옴, age가 30보다 크거나 같은 애들의 이름과 나이만을 출력
print(df.loc[df["name"]=="kim", ["name", "tel"]]) # 이름이 kim인 사람의 이름과 번호
print(df.loc[df["tel"]=="1111-2222", "name":"tel"]) # 슬라이싱도 가능
print(df.loc[(df["age"] >= 30) & (df["age"] <=40), ["name","age"]]) # 조건이 여러개 일경우 묶어주어야함, 다중 조건문일경우 비트논리연산자 사용

print()
# 결측값 -> 값이 없는것
df["address"] = "" # address 추가, 빈값을 추가함
df.loc[df["address"] == "", ["address"]] = "서울" # address 빈값에 서울이라고 대입해라

df.loc[::2, ["address"]] = None # 2개를 address 만 걸러내라
# df.loc[df["address"] == None, ["address"]] = "수원" # 자바에서의 Null, 파이썬은 Null이 비교가 되지않아 대입불가
df.loc[df["address"].isnull(), ["address"]] = "수원" # isnull로 사용해야 가능하다.

# 결측값 삭제
# df["income"]
df.loc[:, "income"] = [np.NaN, 5000, np.NaN, 6000]# 전체 인덱스라고 지정해주어야함, 뒤에는 칼럼이름
df.loc[5, :] = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
# df.dropna(how="any") # 유효하지 않은값은 지워라
# df.dropna(how="all") # 전부 nan 인것을 다 지워라
df.dropna(how="all", inplace=True) # 디비에 바로 적용을 해라

# df["income"].fillna(value=0, inplace=True)  # 데이터를 골라냄, 그후 채워라 
# df["income"].fillna(value=np.mean(df["income"]), inplace=True) # 평균으로 채워라

# print(df.isnull()) # true false로 표시됨, null인것만 골라냄
print(df.isnull().loc[:, "income"]) # 인컴이라는 한줄만 뽑음
print(df.drop(df[df.isnull()["income"]].index, axis=0)) # 드랍함

print(df.drop(["income"], axis=1)) # 리스트로 주어야함, 세로로 지워라
print(df.drop(["tel", "income"], axis=1))


print()
print(df) # 원본은 남아있음

# 전치 -> 행열을 바꿈
print(df.T)
print(df.transpose())


