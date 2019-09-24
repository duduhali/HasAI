import numpy as np

class MeanSquaredError:
    def forward(self, x, label):
        self.shape = x.shape
        self.x = x.reshape(label.shape)
        self.label = label
        # loss= sum(  (x-y)*(x-y) )/sum.size
        self.loss = np.sum(np.square(x - self.label)) / self.x.shape[0] / 2  # 求平均后再除以2是为了表示方便
        return self.loss

    def backward(self):
        self.dx = (self.x - self.label) / self.x.shape[0]
        return self.dx


class CrossEntropyLoss:
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
        self.dx = (self.x - self.label)
        return self.dx

class softmax_cross_entropy_with_logits:
    # softmax函数及其导数
    def forward(self,inputs, label):
        print(inputs.shape)
        print(label.shape)
        self.inputs = inputs
        self.label = label
        temp1 = np.exp(inputs)
        probality = temp1 / (np.tile(np.sum(temp1, 1), (inputs.shape[1], 1))).T
        temp2 = np.argmax(label, 1)
        temp3 = [probality[i, j] for (i, j) in zip(np.arange(label.shape[0]), temp2)]
        loss = -1 * np.mean(np.log(temp3))
        return loss

    def backward(self):
        temp1 = np.exp( self.inputs)
        temp2 = np.sum(temp1, 1)
        probability = temp1 / (np.tile(temp2, ( self.inputs.shape[1], 1))).T
        gradient = probability -  self.label
        return gradient