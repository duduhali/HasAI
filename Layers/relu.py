import numpy as np

class Relu:
    def init(self):
        print('init Relu')

    def forward(self,signal):
        return np.maximum(0, signal)

    def backward(self,signal):
        return -np.minimum(0, signal)

if( __name__ == '__main__'):
    r = Relu()
    data = np.full((32, 64, 3), 1)
    y = r.forward(data)
    print(y)
    print(r.backward(y))

