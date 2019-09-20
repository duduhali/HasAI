import numpy as np

class Conv2D:
    pass

class Dropout:
    pass
class MaxPooling2D:
    pass
class UpSampling2D:
    pass
class Activation:
    pass
class BatchNormalization:
    pass
class Reshape:
    pass

class Relu:
    def init(self):
        print('init Relu')

    def forward(self,signal):
        return np.maximum(0, signal)

    def backward(self,signal):
        return -np.minimum(0, signal)

class Softmax:
    def init(self):
        print('init Softmax')

    def softmax(self,x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def forward(self,input):
        self.input = input
        self.value = self.softmax(input)
        return self.value

    def backward(self):
        for i in range(len(self.value)):
            for j in range(len(self.input)):
                if i == j:
                    self.gradient[i] = self.value[i] * (1 - self.input[i])
                else:
                    self.gradient[i] = -self.value[i] * self.input[j]

from functools import reduce
class Flatten:
    def init(self):
        print('init Flatten')
    def forward(self,x):
        self.shape = x.shape
        return x.reshape(-1, reduce(lambda a, b: a * b, x.shape[1:]))  # 压平  Flatten
    def backward(self,dx):
        dx = dx.reshape(self.shape)
        return dx


class Layer:
    def __init__(self):
        self.trainable = True #默认可被训练

class Dense(Layer):
    def __init__(self, units,input_shape=None,activation=None):
        self.units = units
        self.input_shape = input_shape
        self.activation = activation
        super(Dense, self).__init__()

    def printweight(self):
        print('Dense', self.units)
        print(self.weights)
        print(self.bias)

    def init(self,input_shape=None):
        if input_shape != None:
            input = input_shape
        else:
            print('input:',self.input_shape)
            input = self.input_shape
        self.weights = np.random.randn(self.units, input[0]) / np.sqrt(input[0])
        self.bias = np.random.randn(self.units, 1)
        print('init Dense', self.units)
        return (self.units,)

    def forward(self, x):
        # print('x:',x)
        self.x = x  # 把中间结果保存下来，以备反向传播时使用
        self.y = np.array([np.dot(self.weights, xx) + self.bias for xx in x])  # 计算全连接层的输出
        return self.y  # 将这一层计算的结果向前传递

    def backward(self, d,lr):
        ddw = np.array([np.dot(dd, xx.T) for dd, xx in zip(d, self.x)])
        self.dw = np.sum(ddw, axis=0) / self.x.shape[0]
        self.db = np.sum(d, axis=0) / self.x.shape[0]

        self.dx = np.array([np.dot(self.weights.T, dd) for dd in d])

        self.weights -= lr * self.dw
        self.bias -= lr * self.db
        return self.dx  # 反向传播梯度



