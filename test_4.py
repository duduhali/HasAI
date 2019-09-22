import numpy as np





class Softmax:
    def softmax(self,x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=1, keepdims=True)  # 如果是列向量，则axis=0
    def forward(self,input):
        self.input = input
        self.value = self.softmax(input)
        return self.value
    def backward(self):
        gradient = np.zeros_like(self.value)
        for i in range(len(self.value)):
            for j in range(len(self.input)):
                if i == j:
                    gradient[i] = self.value[i] * (1 - self.input[i])
                else:
                    gradient[i] = -self.value[i] * self.input[j]
        return gradient

    def backward2(self):
        gradient = np.zeros_like(self.value)
        shape = self.value.shape
        for batch_i  in range(shape[0]):
           self.backward()

        return gradient

s = Softmax()

print(s.forward(np.array([[1,2]])))
print(s.backward())

# print(s.forward(np.array([[1,2],[3,5]])))
# print(s.backward2())


