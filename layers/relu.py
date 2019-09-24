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
    y = r.forward(np.array([[1, 2]]))
    print(y)
    print(r.backward(y))

    y = r.forward(np.array([[1, -2], [3, 5]]))
    print(y)
    print(r.backward(y))
