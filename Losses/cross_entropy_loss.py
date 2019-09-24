import numpy as np

class CrossEntropyLoss:
    # 是能是二维数据，第一个维度是批次大小
    # label 独热码
    def forward(self, x, label):
        self.x = x
        self.label = np.zeros_like(x)
        for a, b in zip(self.label, label):
            a[b] = 1.0
        self.loss = np.nan_to_num(-self.label * np.log(x) - ((1 - self.label) * np.log(1 - x)))  # np.nan_to_num()避免log(0)得到负无穷的情况
        # 用0代替数组x中的nan元素，使用有限的数字代替inf元素
        self.loss = np.sum(self.loss) / x.shape[0]
        return self.loss

    def backward(self):
        self.dx = (self.x - self.label) / self.x.shape[0]
        return self.dx

if( __name__ == '__main__'):
    import numpy as np
    x = np.full((32, 64), 1)
    y = np.full((32, 64), 0)
    mse = CrossEntropyLoss()
    loss = mse.forward(x,y)
    print('loss',loss)

    dx = mse.backward()
    print(dx)