import numpy as np

# 数据层
class Data:
    def __init__(self, name, batch_size):  # 数据所在的文件名name和batch中图片的数量batch_size
        print(name)
        with open(name, 'rb') as f:
            data = np.load(f,allow_pickle=True)
        self.x = data[0]  # 输入x
        self.y = data[1]  # 预期正确输出y
        self.len = len(self.x)
        self.batch_size = batch_size
        self.pos = 0  # pos用来记录数据读取的位置

    def forward(self):
        pos = self.pos
        bat = self.batch_size
        len = self.len
        if pos + bat >= len:  # 已经是最后一个batch时，返回剩余的数据，并设置pos为开始位置0
            ret = (self.x[pos:len], self.y[pos:len])
            self.pos = 0
            index = range(len)
            # 将训练数据打乱:先生成一个从0到len-1的列表，再把列表打乱，最后获取数据
            np.random.shuffle(list(index))
            self.x = self.x[index]
            self.y = self.y[index]
        else:  # 不是最后一个batch, pos直接加上batch_size
            ret = (self.x[pos:pos + bat], self.y[pos:pos + bat])
            self.pos += self.batch_size

        return ret, self.pos  # 返回的pos为0时代表一个epoch已经结束

    def backward(self, d):  # 数据层无backward操作
        pass

# 全连接层
class FullyConnect:
    def __init__(self, l_x, l_y):  # 两个参数分别为输入层的长度和输出层的长度
        self.weights = np.random.randn(l_y, l_x) / np.sqrt(l_x)  # 使用随机数初始化参数，请暂时忽略这里为什么多了np.sqrt(l_x)
        self.bias = np.random.randn(l_y, 1)  # 使用随机数初始化参数
        self.lr = 0  # 先将学习速率初始化为0，最后统一设置学习速率

    def forward(self, x):
        self.x = x  # 把中间结果保存下来，以备反向传播时使用
        # print(self.weights.shape)#(26, 289) //17*17=289
        # print(self.bias.shape)#(26, 1)
        self.y = np.array([np.dot(self.weights, xx) + self.bias for xx in x])  # 计算全连接层的输出
        return self.y  # 将这一层计算的结果向前传递

    def backward(self, d):
        ddw = [np.dot(dd, xx.T) for dd, xx in zip(d, self.x)]  # 根据链式法则，将反向传递回来的导数值乘以x，得到对参数的梯度
        self.dw = np.sum(ddw, axis=0) / self.x.shape[0]
        # print('d',d.shape) (1024, 26, 1)
        # print(np.sum(d, axis=0).shape) (26, 1)
        self.db = np.sum(d, axis=0) / self.x.shape[0]
        # print('self.db',self.db.shape) self.db (26, 1)
        self.dx = np.array([np.dot(self.weights.T, dd) for dd in d])

        # 更新参数
        self.weights -= self.lr * self.dw
        self.bias -= self.lr * self.db
        return self.dx  # 反向传播梯度

    # （对矩阵求导可能大部分本科生都不会。但其实也不难，如果你线性代数功底可以，可以尝试推导矩阵求导公式。）

# 激活函数层
class Sigmoid:
    def __init__(self):  # 无参数，不需初始化
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        # print(x.shape) (1024, 26, 1)
        self.y = self.sigmoid(x) #可以处理单个元素，也可以处理np数组
        return self.y

    def backward(self, d):
        sig = self.sigmoid(self.x)
        self.dx = d * sig * (1 - sig)
        return self.dx  # 反向传递梯度

# 损失函数层
class QuadraticLoss:
    def __init__(self):
        pass

    def forward(self, x, label):
        self.x = x
        # print(x.shape)#(1024, 26, 1)
        # print(label.shape) #(1024,)
        # np.zeros_like(x)返回和传入矩阵相同维度的零矩阵
        self.label = np.zeros_like(x)  # 由于我们的label本身只包含一个数字，我们需要将其转换成和模型输出值尺寸相匹配的向量形式
        # print(self.label.shape) #(1024, 26, 1)
        for a, b in zip(self.label, label):
            # a 是(26,1)维向量   b是一个数
            a[b] = 1.0  # 只有正确标签所代表的位置概率为1，其他为0
        # loss= sum(  (x-y)*(x-y) )/sum.size
        self.loss = np.sum(np.square(x - self.label)) / self.x.shape[0] / 2  # 求平均后再除以2是为了表示方便
        return self.loss

    def backward(self):
        self.dx = (self.x - self.label) / self.x.shape[0]  # 2被抵消掉了
        # self.dx.shape = (1024, 26, 1)
        return self.dx

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

# 构建神经网络
def main():
    datalayer1 = Data('data/train.npy', 1024)  # 用于训练，batch_size设置为1024
    datalayer2 = Data('data/validate.npy', 10000)  # 用于验证，所以设置batch_size为10000,一次性计算所有的样例
    inner_layers = []
    inner_layers.append(FullyConnect(17 * 17, 26))
    inner_layers.append(Sigmoid())
    losslayer = QuadraticLoss()
    accuracy = Accuracy()

    for layer in inner_layers:
        layer.lr = 1000.0  # 为所有中间层设置学习速率

    epochs = 1
    for i in range(epochs):
        print('epochs:', i)
        losssum = 0
        iters = 0
        while True:
            data, pos = datalayer1.forward()  # 从数据层取出数据
            x, label = data
            for layer in inner_layers:  # 前向计算
                x = layer.forward(x)

            loss = losslayer.forward(x, label)  # 调用损失层forward函数计算损失函数值
            losssum += loss
            iters += 1
            d = losslayer.backward()  # 调用损失层backward函数层计算将要反向传播的梯度

            for layer in inner_layers[::-1]:  # 反向传播
                d = layer.backward(d)

            if pos == 0:  # 一个epoch完成后进行准确率测试
                data, _ = datalayer2.forward()
                x, label = data
                for layer in inner_layers:
                    x = layer.forward(x)
                accu = accuracy.forward(x, label)  # 调用准确率层forward()函数求出准确率
                print('loss:', losssum / iters)
                print('accuracy:', accu)
                break

if __name__ == '__main__':
    main()
