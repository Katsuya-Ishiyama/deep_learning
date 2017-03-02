
# coding: utf-8

# # ゼロから作るDeep Learning
# # 4章 ニューラルネットワークの学習

# ## 4.1 データから学習する
# #### 4.1.1 データ駆動

# ## 4.2 損失関数
# #### 4.2.1 2乗和誤差
# 損失関数の中で最も有名なものは次の2乗和誤差(Mean Squared Error)である。
# （平均2乗誤差の方が一般的なイメージか？）
# \begin{equation}
# E = \frac{1}{2} \sum_{k} (y_{k} - t_{k})^{2}
# \end{equation}
# ここで、$y_{k}$はニューラルネットワークの出力、$t_{k}$は教師データを表す。  
# $y_{k}$の分布に正規分布を仮定した場合の対数尤度関数の定数項を除いたものである。

# In[1]:

# 必要なモジュールを読み込む
import numpy as np


# In[2]:

# 平均二乗誤差の実装
def mean_squared_error(y, t):
    
    return 0.5 * np.sum((y - t)**2)


# In[3]:

# 実際に試してみる
print('----- Mean Squared Error -----')
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

# 2の確率が最も悪い場合
y1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print('Example 1:', mean_squared_error(y1, t))

# 2の確率が最も高い場合
y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print('Example 2:', mean_squared_error(y2, t))


# #### 4.2.2 交差エントロピー誤差
# 交差エントロピー誤差(Cross Entropy Error)もよく用いられる損失関数で、次式で表される。
# \begin{equation}
# E = -\sum_{k} t_{k} \log y_{k}
# \end{equation}
# これは多項ロジスティック回帰で回帰係数を求める際の対数尤度関数である。
# (Bishop, pp.235を参照)

# In[4]:

# 交差エントロピー誤差の実装
def cross_entropy_error(y, t):
    # yが0だった場合でも計算を進められるようにするため
    delta = 1e-7

    return -np.sum(t * np.log(y + delta))


# In[5]:

# 実際に試してみる
print('----- Cross Entropy Error -----')
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

y1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print('Example 1:', cross_entropy_error(y1, t))

y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print('Example 2:', cross_entropy_error(y2, t))


# #### 4.2.3 ミニバッチ学習
# 先程の交差エントロピー誤差はデータが1つの場合のものである。
# これを$N$個のデータに適用できるように拡張すると次のようになる。
# \begin{equation}
# E = - \frac{1}{N} \sum_{n} \sum_{k} t_{nk} \log y_{nk}
# \end{equation}
# 
# ミニバッチ学習とは数百万、数千万という膨大なデータの中から一部を抜き出し、その抜き出したデータを使って学習を行うこと。
# ミニバッチ学習のために訓練データの中から指定された個数のデータをランダムに選び出すコードを書く。

# In[6]:

# MNISTデータを読み込む
import sys
import os
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)


# In[7]:

# 実際に訓練データを抜き出すコード
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


# #### 4.2.4 ［バッチ対応版］交差エントロピー誤差の実装
# データが1つの場合と、データがバッチとしてまとめられて入力される場合の両方のケースに対応できるようにする。

# In[8]:

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = t.reshape(1, y.size)
    
    batch_size = y.shape[0]
    
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


# In[9]:

# reshapeメソッドを知らなかったので、動作検証してみる
test = np.array([0.1, 2, 0.5, 9])
print('test =', test)
print('test.reshape(1, test.size) =', test.reshape(1, test.size))


# In[10]:

# 複雑な部分の動作を確認する
y = np.array([[0.5, 1, 0.1],
              [0.2, 2, 7],
              [10, 3, 0]])
print('y =', y)

t = np.array([2, 0, 1])
print('t =', t)

print('y[np.arange(3), t] =', y[np.arange(3), t])
print('y[:, t] =', y[:, t])


# ## 4.3 数値微分
# #### 4.3.1 微分

# In[11]:

def numerical_diff(f, x):

    h = 1e-4

    return (f(x + h) - f(x - h)) / (2 * h)


# #### 4.3.2 数値微分の例
# 上記の数値微分で次の関数を試してみる。
# \begin{equation}
# y = 0.01 x^{2} + 0.1 x
# \end{equation}

# In[12]:

def f1(x):

    return 0.01 * x**2 + 0.1 * x

# 数値微分を試す
print('numerical_diff(f1,  5) =', numerical_diff(f1, 5))
print('numerical_diff(f1, 10) =', numerical_diff(f1, 10))


# #### 4.3.3 偏微分
# 2変数関数
# \begin{equation}
# f(x_{0}, x_{1}) = x_{0}^{2} + x_{1}^{2}
# \end{equation}
# について、  
# (1)$x_{0}=3$, $x_{1}=4$のときの$x_{0}$に対する偏微分$\frac{\partial f}{\partial x_{0}}$を求めよ。  
# (2)$x_{0}=3$, $x_{1}=4$のときの$x_{1}$に対する偏微分$\frac{\partial f}{\partial x_{1}}$を求めよ。  

# In[13]:

def f2(x):
    
    return np.sum(x**2)


def f2_tmp1(x0):

    return x0 ** 2 + 4.0 ** 2


def f2_tmp2(x1):
    
    return 3.0 ** 2 + x1 ** 2

print('numerical_diff(f2_tmp1, 3.0) =', numerical_diff(f2_tmp1, 3.0))
print('numerical_diff(f2_tmp2, 4.0) =', numerical_diff(f2_tmp2, 4.0))


# ## 4.4 勾配
# 2変数関数の勾配(gradient)
# \begin{equation}
# \nabla f(x0, x1) = (\frac{\partial f}{\partial x_{0}}, \frac{\partial f}{\partial x_{1}})
# \end{equation}
# を実装する。

# In[14]:

# 全微分の実装
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i, tmp in enumerate(x):
        tmp_x = x.copy()
        
        # f(x + h)の計算
        tmp_x[i] = tmp + h
        fxh1 = f(tmp_x)
        
        # f(x - h)の計算
        tmp_x[i] = tmp - h
        fxh2 = f(tmp_x)
        
        grad[i] = (fxh1 - fxh2) / (2 * h)
    
    return grad


# テスト
print('numerical_gradient(f2, np.array([3, 4])) =', numerical_gradient(f2, np.array([3.0, 4.0])))
print('numerical_gradient(f2, np.array([0, 2])) =', numerical_gradient(f2, np.array([0.0, 2.0])))
print('numerical_gradient(f2, np.array([3, 0])) =', numerical_gradient(f2, np.array([3.0, 0.0])))


# In[15]:

# zeros_likeメソッドの動作確認
x = np.arange(10)
y = np.zeros_like(x)

print('x =', x)
print('y =', y)


# #### 4.4.1 勾配法

# In[16]:

# 勾配法を実装する
def gradient_desent(f, init_x, lr=0.01, step_num=100):
    # init_xを変更させないようにするため
    x = init_x.copy()
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x


# In[17]:

# 実装した勾配法を試す
gradient_desent(
    f=f2,
    init_x=np.array([-3.0, 4.0]),
    lr=0.1,
    step_num=100
)


# In[18]:

init_x = np.array([-3.0, 4.0])

# 学習率が大きすぎる例
print('学習率が大きすぎる例(lr = 10.0): ', gradient_desent(f=f2, init_x=init_x, lr=10.0, step_num=100))
print('学習率が小さすぎる例(lr = 1e-10): ', gradient_desent(f=f2, init_x=init_x, lr=1e-10, step_num=100))


# #### 4.4.2 ニューラルネットワークに対する勾配

# In[19]:

import sys
import os
import numpy as np
from common.functions import (softmax, cross_entropy_error)
from common.gradient import numerical_gradient

class SimpleNet(object):
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss


# In[20]:

# SimpleNetを試す
net = SimpleNet()
print('net.W =', net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print('p =', p)
print('np.argmax(p) =', np.argmax(p))

t = np.array([0, 0, 1])
print('net.loss(x, t) =', net.loss(x, t))


# In[21]:

# なぜダミーでWを引数に設定するのか分からない
def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print('dW =', dW)


# #### 4.5.1 2層ニューラルネットワークのクラス
# 初めに、2層ニューラルネットワークを1つのクラスとして実装することから始める。

# In[22]:

import sys
import os
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}

        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x):
        W1 = self.params['W1']
        b1 = self.params['b1']
        
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        
        return accuracy
    
    def numerical_gradient(self, x, t):
        def loss_W(W):
            return self.loss(x, t)
            
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads


# In[23]:

# TwoLayerNetを試す
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

print('----- 重みとバイアス -----')
print("net.params['W1'].shape =", net.params['W1'].shape)
print("net.params['b1'].shape =", net.params['b1'].shape)
print("net.params['W2'].shape =", net.params['W2'].shape)
print("net.params['b2'].shape =", net.params['b2'].shape)


# In[28]:

# 推論処理
x = np.random.rand(100, 784)
y = net.predict(x)

print('----- 推論処理 -----')
print('y =', y)


# In[25]:

# 勾配の計算
x = np.random.rand(100, 784)
t = np.random.rand(100, 10)

grads = net.numerical_gradient(x, t)

print('----- 勾配の計算 -----')
print("grads['W1'] =", grads['W1'].shape)
print("grads['b1'] =", grads['b1'].shape)
print("grads['W2'] =", grads['W2'].shape)
print("grads['b2'] =", grads['b2'].shape)


# #### 4.5.2 ミニバッチ学習の実装
# TwoLayerNetクラスを対象に、MNISTデータセットを使って学習を行う。

# In[26]:

import sys
import numpy as np
from dataset.mnist import load_mnist
# 下記のモジュールは上で実装したTwoLayerNetと同じため、ここではコメントアウトしておく
# sys.path.append('/home/ishiyama/deep_learning/deep-learning-from-scratch/ch04')
# from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# hyperparameters
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)
    
    # パラメータの更新
    for key in network.params:
        network.params[key] -= learning_rate * grad[key]
    
    # 学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)


# In[27]:

print('x_train.shape =', x_train.shape)
print('t_train.shape =', t_train.shape)


# ## 参考文献
# 斎藤(2016), ゼロから作るDeep Learning, O'reilly Japan  
# Bishop(2006), Pattern Recognition and Machine Learning, Springer-Verlag New York

# In[ ]:



