import numpy as np

class Softmax:
    def init(self):
        print('init Softmax')

    def softmax(self,x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=1, keepdims=True)  # 如果是列向量，则axis=0

    def forward(self,input):
        self.input = input
        self.shape= input.shape
        return self.softmax(input)

    def _backward(self,value,input):
        gradient = np.zeros_like(value)
        for i in range(len(value)):
            for j in range(len(input)):
                if i == j:
                    gradient[i] = value[i] * (1 - input[i])
                else:
                    gradient[i] = -value[i] * input[j]
        return gradient

    def backward(self,value):
        value = value.reshape(self.shape[0:-1])
        self.input = self.input.reshape(self.shape[0:-1])
        gradient = np.vstack([self._backward(np.atleast_2d(v),np.atleast_2d(i)) for v,i in zip(value,self.input)])

        return gradient.reshape(self.shape)

if( __name__ == '__main__'):
    s = Softmax()
    # y = s.forward(np.array([[1, 2]]))
    # print(y)
    # print(s.backward(y))

    data = np.array([[1, 2], [3, 5]])
    data = data.reshape((2,2,1))
    print('data',data.shape)
    y = s.forward(data)
    print(y)
    print(s.backward(y))
