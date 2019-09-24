import numpy as np
class Dense():
    def __init__(self, units,input_shape=None,name='Dense'):
        if len(input_shape) == 1:
            self.bias = np.random.randn(units, )
        elif len(input_shape) == 2:
            self.bias = np.random.randn(units, input_shape[1])
        else:
            raise Exception('参数类型错误')
        self.units = units
        self.weights = np.random.randn(self.units, input_shape[0]) / np.sqrt(input_shape[0])
        print('init Dense weights:{0} bias'.format(self.weights.shape), self.bias.shape)
        self.lr = 0.001
    def forward(self, x):
        self.x = x  # 把中间结果保存下来，以备反向传播时使用
        self.y = np.array([np.dot(self.weights, xx) + self.bias for xx in x])  # 计算全连接层的输出
        return self.y  # 将这一层计算的结果向前传递
    def backward(self, d):
        d_shape = d.shape
        x_shape = self.x.shape
        if len(d_shape) == 2 and len(x_shape) == 2:
            d_ = d.reshape((d_shape[0],d_shape[1],1))
            x_ = self.x.reshape((x_shape[0],x_shape[1],1))
            ddw = np.array([np.dot(dd, xx.T) for dd, xx in zip(d_, x_)])
        else:
            ddw = np.array([np.dot(dd, xx.T) for dd, xx in zip(d, self.x)])
        self.dw = np.sum(ddw, axis=0) / self.x.shape[0]
        self.db = np.sum(d, axis=0) / self.x.shape[0]
        self.dx = np.array([np.dot(self.weights.T, dd) for dd in d])
        self.weights -= self.lr * self.dw
        self.bias -= self.lr * self.db
        return self.dx  # 反向传播梯度
if( __name__ == '__main__'):
    with open('../data/train_x_mini.npy', 'rb') as f:
        train_x = np.load(f, allow_pickle=True) #(1024, 289, 1)
    with open('../data/train_y_mini.npy', 'rb') as f:
        train_y = np.load(f, allow_pickle=True) #(1024,)
    # train_x = train_x.reshape(train_x.shape[0:2])
    print(train_x.shape)
    print(train_y.shape)

    d = Dense(289,(289,1))
    # d = Dense(289, (289,))
    y = d.forward(train_x)
    print('y',y.shape)

    dx = d.backward(y)
    print('dx',dx.shape)


