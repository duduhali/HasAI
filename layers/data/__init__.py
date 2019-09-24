import numpy as np
class Data:
    def __init__(self, x,y, batch_size=None):  # 数据所在的文件名name和batch中图片的数量batch_size
        self.x = x  # 输入x
        self.y = y  # 预期正确输出y
        self.len = len(self.x)
        if batch_size!=None:
            self.batch_size = batch_size
        else:
            self.batch_size = self.len
        self.pos = 0  # pos用来记录数据读取的位置

    def getData(self):
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

if( __name__ == '__main__'):
    with open('../../data/train.npy', 'rb') as f:
        data = np.load(f, allow_pickle=True)
        train_x = data[0]  # 输入x
        train_y = data[1]  # 预期正确输出y
    train = Data(train_x, train_y, 1024)
    (x, label), pos = train.getData()  # 从数据层取出数据
    x = np.array([xx for xx in x])
    label = np.array([y for y in label])