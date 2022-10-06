# Python

---

### 이클립스용 파이썬 설치

[https://www.python.org/downloads/release/python-3810/](https://www.python.org/downloads/release/python-3810/)

텐서플로우 → 딥러닝을 위한 라이브러리    파이썬 3.8에만 적용

3.8버젼 설치

- 이전에 파이썬을 설치 했다면 미리 지우고 설치를 진행
- 버전이 여러개가있으면 패스를 계속 바꿔야함
- pip인스톨시 계속 바꿔주어야함

download → windows

![image](https://user-images.githubusercontent.com/87698248/194306703-1bed5eaa-caf3-40cd-b74a-09d4824c82d4.png)

![image](https://user-images.githubusercontent.com/87698248/194306733-01bb0c59-0c83-41c3-9ab3-a2a167f76b95.png)

![image](https://user-images.githubusercontent.com/87698248/194306755-0c747921-8c54-4f33-9168-1de72c7608c6.png)

Add Python 3.8 to PATH 꼭체크 해줄것

안해주면 패스를 수동으로 설치 해야함

install 클릭

![image](https://user-images.githubusercontent.com/87698248/194306772-d6021ba0-0315-4b59-b086-f3b6be3b6b7b.png)

설치 완료후 close

파이썬 경로 확인

![image](https://user-images.githubusercontent.com/87698248/194306804-ca057a58-d3e1-4462-8fd1-b4e5b6e1874c.png)

cmd → ptyhon —version → 파이썬 설치 확인

![image](https://user-images.githubusercontent.com/87698248/194306836-85f2b622-2db9-45e4-9863-2d06f6f76b5a.png)

cmd 창에서 python 코딩 가능

![image](https://user-images.githubusercontent.com/87698248/194306863-44b0694c-05b9-43cd-95f8-f578e8e1082e.png)

![image](https://user-images.githubusercontent.com/87698248/194306885-fdc32be2-af83-4f54-8591-68bd6bd864da.png)
exit()로 나가기

---

### 추가설치

PyDev 설치

[https://sourceforge.net/projects/pydev/?source=typ_redirect](https://sourceforge.net/projects/pydev/?source=typ_redirect)

![image](https://user-images.githubusercontent.com/87698248/194306928-a715e7ef-5e60-4f81-930d-a694ab49f25f.png)

pydev 클릭

![image](https://user-images.githubusercontent.com/87698248/194306962-9204f8d2-d8cf-4a36-94b9-09089c3aebc9.png)

이클립스도 다시 설치 해주어야함 → 버젼 문제로 적용이 안됨

[이클립스 다운(클릭)](https://www.eclipse.org/downloads/download.php?file=/technology/epp/downloads/release/2022-06/R/eclipse-jee-2022-06-R-win32-x86_64.zip) 22년 6월 버젼

![image](https://user-images.githubusercontent.com/87698248/194307001-5ac29c63-108f-434f-8d0c-7c4141ad1879.png)

pydev도 9.3 으로 풀어줌

![image](https://user-images.githubusercontent.com/87698248/194307028-a2b459bb-8475-4e37-a60a-faadfc0c3511.png)

압축을 푼 후에 덮어쓰기를 해줌

![image](https://user-images.githubusercontent.com/87698248/194307065-973cb0fc-143e-4918-9fd3-42ecea4fb3f7.png)

파이썬 적용 완료

- 기존의 파일은 사라지는것이 아닌 파일이 추가가 되는것임

windows → preferences → PyDev  → interpreters → Python interpreters

![image](https://user-images.githubusercontent.com/87698248/194307091-64098b68-624f-475a-8138-5aa1b878989d.png)

![image](https://user-images.githubusercontent.com/87698248/194307111-b58b06e1-89c9-4a11-bf79-5a70a1c9a082.png)

파이썬 패스의 맨 위의 패스를 쓸것이다 라는 뜻

![image](https://user-images.githubusercontent.com/87698248/194307140-87176b4e-75c9-4877-87a9-edd549c1bd6c.png)

이후 프로젝트 생성후 실행하면 잘 작동함

---

### 프로젝트 생성

file → other → pydev → pydev project

![image](https://user-images.githubusercontent.com/87698248/194307167-6b44d7fa-b8a2-43c2-bc72-677ced2248f9.png)

![image](https://user-images.githubusercontent.com/87698248/194307195-2952f5c0-04c6-484e-b5d7-df544ba7d5e2.png)

이상태로 생성

![image](https://user-images.githubusercontent.com/87698248/194307233-6a679633-89a6-46e9-985a-a0154e9cb7ac.png)

기존프로젝트와 연결 할 것이냐 라는 뜻

연결 안할것 이기에 그냥 finish 해준다.

---

파일 하나를 모듈 이라고 부름

확장자는 .py

한 파일안에 다 들어감

파이썬은 모듈지향

파일 하나가 부품이됨

### 프로젝트 생성

PythonEx → 우클릭 → PyDev Package

![image](https://user-images.githubusercontent.com/87698248/194307273-56c3ff6a-fa10-4939-b6af-b0a9f8dfd14e.png)

![image](https://user-images.githubusercontent.com/87698248/194307295-1056e449-551d-4b36-a8b1-dd748e2639b4.png)

simple → 우클릭 → PyDev Module

![image](https://user-images.githubusercontent.com/87698248/194307309-a84e5ece-93c6-4a8a-be6c-78bd5d7f3b91.png)

empty

![image](https://user-images.githubusercontent.com/87698248/194307322-81c53586-ba90-468b-b7b0-217f91d31240.png)

- 클래스 이름은 첫 글자는 대문자로 만들어야한다!!
- 인터프리터 언어이기에 특징이 다른것이 있음
    - 컴파일 링크언어는 c 계열 이기에 무조건 코딩을 다 하고 전체를 컴파일 해야함 하지만
    - 파이썬은 한줄씩 컴파일을함 한줄을 코딩후 컴파일하면 에러를 확인함
        - 에러가 있어도 실행을 하는데 에러가 있는 부분에서 멈춤
        - 메모리할당부터 실행할때 결정을함
        - 미리 방의 크기를 잡아 주어야함 ex) int a = 10 → a = 10
        - 하지만 파이썬은 메모리의 크기를 잡아줄 필요가 없음
        - 데이터는 따지기는 따짐 자바처럼 세세하게 따지지는 않음
        - 정수, 실수, 숫자, 문자 만 따짐
        

### 실행 시작

```python
a=10
print(a)
```

ctrl + f11

![image](https://user-images.githubusercontent.com/87698248/194307341-f0b0c69a-2adc-4dcb-924f-994a805894c7.png)

한번은 이렇게 실행 해주어야함

![image](https://user-images.githubusercontent.com/87698248/194307368-6b83ac9b-3612-4f33-abfd-5954d7aaae34.png)

but 한줄로 코딩할 때에만 세미콜론을 찍어주어야함

```python
a=10; print(a)
```

![image](https://user-images.githubusercontent.com/87698248/194307392-097f3bf4-72b3-4d92-a55c-6ca54924e3fa.png)

이상없이 잘 나옴

들여쓰기도 주의 해야함

탭을 누르게 되면 하위 영역이기에 잘 해주어야함

영역표시가 없어 들여쓰기 잘못하면 큰일남!
