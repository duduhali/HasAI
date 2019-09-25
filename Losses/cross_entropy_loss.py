import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, x, label):
        self.x = x
        self.label = np.zeros_like(x)
        the_label = np.argmax(label, 1) #label是独热码
        for a, b in zip(self.label, the_label):
            a[b] = 1.0
        self.loss = np.nan_to_num(-self.label * np.log(x) - ((1 - self.label) * np.log(1 - x)))  # np.nan_to_num()避免log(0)得到负无穷的情况
        # 用0代替数组x中的nan元素，使用有限的数字代替inf元素
        self.loss = np.sum(self.loss) / x.shape[0]
        return self.loss

    def backward(self):
        self.dx = (self.x - self.label) / self.x / (1 - self.x)
        return self.dx

if( __name__ == '__main__'):
    import numpy as np
    x = np.full((32, 64), 1)
    y = np.full((32, 1), 2)
    from keras.utils import np_utils
    # labels = np_utils.to_categorical(y, 64)  # 独热码
    # print('labels',labels.shape)
    mse = CrossEntropyLoss()
    loss = mse.forward(x,y)
    print('loss',loss)

    # dx = mse.backward()
    # print(dx)