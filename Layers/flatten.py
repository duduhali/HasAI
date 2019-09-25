class Flatten:
    def setParam(self,shape):
        sum = 1
        for i in shape:
            sum = sum*i
        return (sum,)

    def forward(self,x):
        self.shape = x.shape
        return x.reshape((self.shape[0],-1))  # 压平  Flatten,压成二维
    def backward(self,dx):
        dx = dx.reshape(self.shape)
        return dx

class Reshape:
    def __init__(self,input_shape):
        self.input_shape = input_shape
    def setParam(self):
        return self.input_shape

    def forward(self,x):
        self.shape = x.shape
        return x.reshape((-1,*self.input_shape))
    def backward(self,dx):
        dx = dx.reshape(self.shape)
        return dx

if( __name__ == '__main__'):
    import numpy as np

    # data = np.full((32, 64,3), 1)
    # f = Flatten()
    # print('data',data.shape)
    # y = f.forward(data)
    # print('y',y.shape)
    # dx = f.backward(y)
    # print('dx', dx.shape)

    data = np.full((32, 64,3), 1)
    f = Reshape((64*3,1))
    print('data',data.shape)
    y = f.forward(data)
    print('y',y.shape)
    dx = f.backward(y)
    print('dx', dx.shape)