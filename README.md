# Python_Perception

## MNIST

1. tensorflow에서 load_data() 함수를 사용해서 MNIST 데이터를 불러온다.
```
from tensorflow.keras.datasets import mnist

# 데이터 나누어 불러오기 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
2. matplotlib을 이용해 MNIST 중 800번째의 이미지를 시각화한다.
```
import matplotlib.pyplot as plt

# 800번째 이미지를 흑백으로 출력
plt.imshow(x_train[800], cmap='Greys')
plt.show()
```
## 퍼셉트론과 논리 게이트 (AND, OR)
3. 퍼셉트론 기반 AND 게이트 구현
```
def AND(x1, x2):
  w1, w2, b = 0.5, 0.5, 0.7  # 가중치(w1, w2)와 임계값(bias)
  result = x1 * w1 + x2 * w2
  if result <= b:
    return 0
  else:
    return 1
```
4. 퍼셉트론 기반 OR 게이트 구현
```
def OR(x1, x2):
#퍼셉트론으로 작성한 OR
  w1, w2, b = 0.4, 0.4, 0.3
  result = x1 * w1 + x2 * w2
  if result <= b:
    return 0
  else:
    return 1
```
## AND, OR, NAND를 활용한 XOR 구현
5. 퍼셉트론 기반 NAND 게이트 구현
6. 
```
# NAND를 실행하기전 퍼셉트론 기반의 AND 게이트와 OR 게이트를 실행해 주세요.

def NAND(x1, x2):
  w1, w2, b = -0.5, -0.5, -0.7
  result = x1 * w1 + x2 * w2
  if result <= b:
    return 0
  else:
    return 1




