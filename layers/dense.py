import numpy as np
from layers.base import Layer

class Dense(Layer):
    #第一维永远是批次大小
    #只能传入二维或者三维的数，传入二位的数会被转化为三维
    #input_shape只能是一维或者二维
    def __init__(self, units,input_shape=None):
        self.units = units
        self.input_shape = input_shape
        super(Dense, self).__init__()

    def printweight(self):
        print('Dense', self.units)
        print(self.weights)
        print(self.bias)

    def init(self,input_shape=None):
        # units 只能是一个数，input必须是元组
        input = self.input_shape if input_shape == None else input_shape
        self.weights = np.random.randn(self.units, input[0]) / np.sqrt(input[0])
        if len(input)==2:
            self.bias = np.random.randn(self.units,input[1])
        else:
            self.bias = np.random.randn(self.units,1)
        print('init Dense weights:{0} bias'.format(self.weights.shape),self.bias.shape)
        return (self.units,)

    def forward(self, x):
        if len(x.shape)<3:
            x = np.atleast_3d(x)
        self.x = x  # 把中间结果保存下来，以备反向传播时使用
        self.y = np.array([np.dot(self.weights, xx) + self.bias for xx in x])  # 计算全连接层的输出
        return self.y  # 将这一层计算的结果向前传递

    def backward(self, d,lr):
        if len(d.shape)<3:
            d = np.atleast_3d(d)
        ddw = np.array([np.dot(dd, xx.T) for dd, xx in zip(d, self.x)])
        self.dw = np.sum(ddw, axis=0) / self.x.shape[0]
        self.db = np.sum(d, axis=0) / self.x.shape[0]

        self.dx = np.array([np.dot(self.weights.T, dd) for dd in d])
        self.weights -= lr * self.dw
        self.bias -= lr * self.db
        return self.dx  # 反向传播梯度

if( __name__ == '__main__'):
    # zero = np.zeros((14, 14))
    # print(zero.shape)
    # data = np.full((3,2),1)
    # print(data)
    # arr = np.arange(5 * 5).reshape(5, 5)
    # print(arr)  # (5, 5)

    # data = np.full((32,64, 1),1)
    # layer = Dense(20,input_shape=(8*8,1))
    # layer.init(None)
    # y = layer.forward(data)
    # print('y',y.shape)
    # dx = layer.backward(y,lr=0.001)
    # print('dx',dx.shape)

    data2 = np.full((32, 64,3), 1)
    layer2 = Dense(20)
    layer2.init(input_shape=(8 * 8,3))
    y = layer2.forward(data2)
    print('y', y.shape)
    dx = layer2.backward(y, lr=0.001)
    print('dx', dx.shape)

