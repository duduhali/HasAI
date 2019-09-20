import numpy as np

class MeanSquaredError:
    def forward(self, x, label):
        self.x = x
        self.label = label
        # loss= sum(  (x-y)*(x-y) )/sum.size
        self.loss = np.sum(np.square(x - self.label)) / self.x.shape[0] / 2  # 求平均后再除以2是为了表示方便
        return self.loss

    def backward(self):
        self.dx = (self.x - self.label) / self.x.shape[0]
        return self.dx
