class Flatten:
    def init(self):
        print('init Flatten')
    def forward(self,x):
        self.shape = x.shape
        print((self.shape[0],-1,1))
        return x.reshape((self.shape[0],-1,1))  # 压平  Flatten,使最后一维为1
    def backward(self,dx):
        dx = dx.reshape(self.shape)
        return dx

if( __name__ == '__main__'):
    import numpy as np

    data = np.full((32, 64,3), 1)
    f = Flatten()
    print(data.shape)
    y = f.forward(data)
    print('y',y.shape)
    dx = f.backward(y)
    print('dx', dx.shape)