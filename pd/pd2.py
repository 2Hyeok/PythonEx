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
print(df[["kor","eng","mat"]].sum(axis=1)) # 칼럼이름을 제외한 정수들의 합

print(df)