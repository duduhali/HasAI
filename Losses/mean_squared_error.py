import numpy as np

class MeanSquaredError:
    #是能是二维数据，第一个维度是批次大小
    #label 是一维数据，结果用一个编号表示
    def forward(self, x, label):
        self.shape = x.shape
        self.x = x.reshape(label.shape)
        self.label = label
        # loss= sum(  (x-y)*(x-y) )/sum.size
        self.loss = np.sum(np.square(x - self.label)) / self.x.shape[0] / 2  # 只用来展示，不用于反向传播
        return self.loss

    def backward(self):
        self.dx = (self.x - self.label) / self.x.shape[0]
        return self.dx

if( __name__ == '__main__'):
    import numpy as np
    x = np.full((32, 64), 1)
    y = np.full((32, 64), 0)
    mse = MeanSquaredError()
    loss = mse.forward(x,y)
    print('loss',loss)

    dx = mse.backward()
    print(dx)