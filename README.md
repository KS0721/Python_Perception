# Python_Perception

## MNIST

### 1. tensorflow에서 load_data() 함수를 사용해서 MNIST 데이터를 불러온다.
```
from tensorflow.keras.datasets import mnist

# 데이터 나누어 불러오기 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
### 2. matplotlib을 이용해 MNIST 중 800번째의 이미지를 시각화한다.
```
import matplotlib.pyplot as plt

# 800번째 이미지를 흑백으로 출력
plt.imshow(x_train[800], cmap='Greys')
plt.show()
```
## 퍼셉트론과 논리 게이트 (AND, OR)
### 3. 퍼셉트론 기반 AND 게이트 구현 및 실행
```
def AND(x1, x2):
  w1, w2, b = 0.5, 0.5, 0.7  # 가중치(w1, w2)와 임계값(bias)
  result = x1 * w1 + x2 * w2
  if result <= b:
    return 0
  else:
    return 1

# 실행
AND(1,1), AND(1,0), AND(0,1), AND(0,0)
```
### 4. 퍼셉트론 기반 OR 게이트 구현 및 실행
```
def OR(x1, x2):
# 퍼셉트론으로 작성한 OR
  w1, w2, b = 0.4, 0.4, 0.3
  result = x1 * w1 + x2 * w2
  if result <= b:
    return 0
  else:
    return 1

# 실행
OR(1,1), OR(1,0), OR(0,1), OR(0,0)
```
## AND, OR, NAND를 활용한 XOR 구현
### 5. 퍼셉트론 기반 NAND 게이트 구현 및 실행
```
# NAND를 실행하기전 퍼셉트론 기반의 AND 게이트와 OR 게이트를 실행해 주세요.

def NAND(x1, x2):
  w1, w2, b = -0.5, -0.5, -0.7
  result = x1 * w1 + x2 * w2
  if result <= b:
    return 0
  else:
    return 1

# 실행
NAND(0,0), NAND(0,1), NAND(1,0), NAND(1,1)
```
### 6.  XOR 게이트 구현 및 실행
```
def XOR(x1, x2):
  h1 = NAND(x1, x2)
  h2 = OR(x1, x2)
  return AND(h1,h2)

# 실행
XOR(0,0), XOR(0,1), XOR(1,0), XOR(1,1)
```
### 7. 다층 퍼셉트론
```
import numpy as np

def DLPerecetion(X, W1, W2):
  x = np.array(X)
  w1 = np.array(W1)
  w2 = np.array(W2)
  h1 = np.array([0] *len(x))
  y = np.array([0]*len(x))
  for count in range(len(x)):
    h1[count] = np.sum(x * w1[count]) > 0
  for count in range(len(x)):
    y[count] = np.sum(h1 * w2[count]) > 0
  return y
```
```
# 가중치 설정
W1 = [
    [-0.4, -0.4, 0, 0, 0, 0.6],   # NAND 역할
    [0.4,  0.4, 0, 0, 0, -0.3],   # OR 역할
    [0,    0,   0, 0, 0, 0],
    [0,    0,   0, 0, 0, 0],
    [0,    0,   0, 0, 0, 0],
    [0,    0,   0, 0, 0, 1]       # bias 노드
]

# 출력층 가중치
W2 = [
    [0.4, 0.4, 0, 0, 0, -0.7],    # AND 역할
    [0,   0,   0, 0, 0, 0],
    [0,   0,   0, 0, 0, 0],
    [0,   0,   0, 0, 0, 0],
    [0,   0,   0, 0, 0, 0],
    [0,   0,   0, 0, 0, 0]
]
```
```
# 입력값
Ex00 = [0, 0, 0, 0, 0, 1]  # x1 = 0, x2 = 0
Ex01 = [0, 1, 0, 0, 0, 1]  # x1 = 0, x2 = 1
Ex10 = [1, 0, 0, 0, 0, 1]  # x1 = 1, x2 = 0
Ex11 = [1, 1, 0, 0, 0, 1]  # x1 = 1, x2 = 1

# 실행
print(DLPerecetion(Ex00, W1, W2))  # [0 0]
print(DLPerecetion(Ex01, W1, W2))  # [1 1]
print(DLPerecetion(Ex10, W1, W2))  # [1 1]
print(DLPerecetion(Ex11, W1, W2))  # [0 0]
```
## MNIST를 활용한 인식 코드
### 8. 라이브러리 및 데이터 로딩
```
# process = (미분사용 : 1 , 역전파사용 : 2)
process = 2

# MNIST 데이터 가져오기
import numpy as np
import time
from keras.datasets import mnist

(x_train, t_train), (x_test, t_test) = mnist.load_data()
t_trainlbl, t_testlbl = t_train, t_test
```
```
# 28X28 을 784 로 수정
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

# one-hot label
T0 = np.zeros((t_train.size, 10))
T1 = np.zeros((t_test.size, 10))

for idx in range(t_train.size): T0[idx][t_train[idx]] = 1
for idx in range(t_test.size): T1[idx][t_test[idx]] = 1

t_train, t_test = T0, T1

# normalize 0.0 ~ 1.0
x_train = x_train / 255
x_test = x_test / 255
```
### 9. 필요함수 정의
```
# 수치 미분 기반 학습에 사용 
python
def numerical_diff(f, x):
    h = 1e-4
    nd_coef = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        index = it.multi_index
        tmp = float(x[index])
        x[index] = tmp + h
        fxh2 = f()
        x[index] = tmp - h
        fxh1 = f()
        nd_coef[index] = (fxh2 - fxh1) / (2*h)
        x[index] = tmp
        it.iternext()
    return nd_coef
```
```
# 분류용 활성화 함수
def softmax(x):
    if x.ndim == 1:  # 기본 1개 처리과정 , 벡터입력
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
    if x.ndim == 2:  # 배치용 n 개 처리, 행렬입력
        x = x.T - np.max(x.T, axis=0)
        return (np.exp(x) / np.sum(np.exp(x), axis=0)).T
```
```
# 수치적으로 측정하는 손실 함수 (소프트맥스와 함께 사용하여 다중 클래스 분류 문제에 효과적)
def cee(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)  # 크기가 1xN 인 2차원 행렬로 재구성
        y = y.reshape(1, y.size)
    result = -np.sum(t * np.log(y + 1e-7)) / y.shape[0]
    return result
```
### 10. 주요 구성 클래스
```
# 비선형 활성화 함수
class Relu:
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout
```
```
# 완전연결 계층
class Affine:
    def __init__(self, W, b):
        self.W = W    # W0, W1
        self.b = b    # b0, b1
        self.x = None
        self.dW = None    # W0, W1 의 기울기
        self.db = None    # b0, b1 의 기울기

    def forward(self, x):
        self.x = x
        result = np.dot(self.x, self.W) + self.b
        return result

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
```
```
# 확률화 + 손실 계산
class SoftmaxWithLoss:
    def __init__(self):
        self.y = None    # 출력(계산결과)
        self.t = None    # 정답(MNIST레이블)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        result = cee(self.y, self.t)
        return result

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
```
### 10. 전체 네트워크 구성
```
class SimpleNetwork:
    def __init__(self, inputx, hidden, outy, weight):
        # 가중치 초기화
        self.netMat = {}
        self.netMat['W0'] = weight * np.random.randn(inputx, hidden)
        self.netMat['b0'] = np.zeros(hidden)
        self.netMat['W1'] = weight * np.random.randn(hidden, outy)
        self.netMat['b1'] = np.zeros(outy)
```
```
        # 계층 생성
        self.netLayers = {}
        self.netLayers['Affine1'] = Affine(self.netMat['W0'],
                                           self.netMat['b0'])
        self.netLayers['Relu1'] = Relu()
        self.netLayers['Affine2'] = Affine(self.netMat['W1'],
                                           self.netMat['b1'])
        self.netLayers['Softmax'] = SoftmaxWithLoss()

    def predict(self, x):
        x = self.netLayers['Affine1'].forward(x)
        x = self.netLayers['Relu1'].forward(x)
        x = self.netLayers['Affine2'].forward(x)
        return x
```
```
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.netLayers['Softmax'].forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        lossfunc = lambda : self.loss(x, t)
        grads = {}
        grads['W0'] = numerical_diff(lossfunc, self.netMat['W0'])
        grads['b0'] = numerical_diff(lossfunc, self.netMat['b0'])
        grads['W1'] = numerical_diff(lossfunc, self.netMat['W1'])
        grads['b1'] = numerical_diff(lossfunc, self.netMat['b1'])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
```
```
        # backward
        dout = 1
        dout = self.netLayers['Softmax'].backward(dout)
        dout = self.netLayers['Affine2'].backward(dout)
        dout = self.netLayers['Relu1'].backward(dout)
        dout = self.netLayers['Affine1'].backward(dout)

        # 기울기(dW, db) 저장
        grads = {}
        grads['W0'] = self.netLayers['Affine1'].dW
        grads['b0'] = self.netLayers['Affine1'].db
        grads['W1'] = self.netLayers['Affine2'].dW
        grads['b1'] = self.netLayers['Affine2'].db
        return grads
```
## 11. 학습을 위한 설정치 입력
```
train_size = x_train.shape[0]
lr = 0.1
iter = 0

# 미분을 사용할 경우 :: 배치 20, 1000회 반복
# (20개 묶음 데이터로 1000번 학습진행)
if process == 1:
    iters_num = 1000
    batch_size = 20
    iter_per_epoch = 1

# 역전파사용 : 배치 100, 60000회 반복
# 100개 묶음 데이터로 60000 회 학습진행
else :
    iters_num = 60000
    batch_size = 100
    iter_per_epoch = int(train_size / batch_size)    # 600

# MNIST 입력(784), 은닉층(노드 50개), 출력층(노드 10개)
network = SimpleNetwork(inputx=784, hidden=100, outy=10, weight = 0.2)
```
## 12. 학습과 검증
```
# 시간측정 시작
t1 = time.time()
print('loss = _______  time = ________ 초 n = ______ | [TrainAcc] [TestAcc]')

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)    # 60000 개중 100 개
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산

    if process==1:
        grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    else:
        grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)

    # 위에서 만들어진 기울기로 W 와 b 갱신
    for key in ('W0', 'b0', 'W1', 'b1'):
        network.netMat[key] -=  lr * grad[key]

    loss = network.loss(x_batch, t_batch)
    # train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        iter = iter + 1
        print('loss = {:7.4f}  '.format(loss), end='')
        print('time = {:8.4f}  '.format(time.time()-t1), end='')
        print('n = {:06d} |{:8.4f}{:11.4f}'.format(iter, train_acc, test_acc))
```

## 배운 소감
### 이번 프로젝트를 통해 퍼셉트론의 기본 구조를 배우고 MNIST를 활용해서 인식코드를 구현 및 실행을 해보며 딥러닝을 이해하는데 도움이 되었습니다. 
