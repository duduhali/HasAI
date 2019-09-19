import numpy as np

class Conv2D:
    pass
class Flatten:
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
    def init(self,input_shape=None):
        print('init Relu')
        return input_shape

    def forward(self,signal):
        return np.maximum(0, signal)

    def backward(self,signal):
        return -np.minimum(0, signal)

class Dense:
    def __init__(self, units,input_shape=None,activation=None):
        self.units = units
        self.input_shape = input_shape
        self.activation = activation

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
        self.x = x  # 把中间结果保存下来，以备反向传播时使用
        self.y = np.array([np.dot(self.weights, xx) + self.bias for xx in x])  # 计算全连接层的输出
        return self.y  # 将这一层计算的结果向前传递

    def backward(self, d,lr):
        ddw = [np.dot(dd, xx.T) for dd, xx in zip(d, self.x)]
        self.dw = np.sum(ddw, axis=0) / self.x.shape[0]
        self.db = np.sum(d, axis=0) / self.x.shape[0]
        self.dx = np.array([np.dot(self.weights.T, dd) for dd in d])

        # 更新参数
        self.weights -= lr * self.dw
        self.bias -= lr * self.db
        return self.dx  # 反向传播梯度



