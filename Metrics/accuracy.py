import numpy as np

# 准确率层
class Accuracy:
    def __init__(self):
        pass

    def forward(self, x, label):  # 只需forward
        # x是(10000, 26, 1)维向量
        # label是正确结果的一维向量(数组)
        # argmax返回的是最大数的索引 这里xx是(26,1)维向量，表示每个字母的概率，np.argmax(xx)返回最大概率的下标
        self.accuracy = np.sum([np.argmax(xx) == ll for xx, ll in zip(x, label)])
        self.accuracy = 1.0 * self.accuracy / x.shape[0]
        return self.accuracy
