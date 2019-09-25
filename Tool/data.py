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
            index =  np.arange(len)
            # 将训练数据打乱:先生成一个从0到len-1的列表，再把列表打乱，最后获取数据
            np.random.shuffle(index)
            self.x = self.x[index]
            self.y = self.y[index]


        else:  # 不是最后一个batch, pos直接加上batch_size
            ret = (self.x[pos:pos + bat], self.y[pos:pos + bat])
            self.pos += self.batch_size
        return ret, self.pos  # 返回的pos为0时代表一个epoch已经结束

if( __name__ == '__main__'):
    from keras.utils import np_utils

    with open('../data/train.npy', 'rb') as f:
        data = np.load(f, allow_pickle=True)
        train_x = np.array([xx for xx in data[0]])
        train_y = np.array([y for y in data[1]])
        print('train_x', train_x.shape)
        print('train_y', train_y.shape)
        train_labels = np_utils.to_categorical(train_y, 26)
        print('train_labels', train_labels.shape)
    with open('../data/validate.npy', 'rb') as f:
        data = np.load(f, allow_pickle=True)
        validate_x = np.array([xx for xx in data[0]])
        validate_y = np.array([y for y in data[1]])
        print('validate_x', validate_x.shape)
        print('validate_y', validate_y.shape)

    d = Data(train_x,train_labels,1024)
    while True:
        (x, label), pos = d.getData()
        if pos == 1024:
            print(pos)
            print('x',x.shape)
            print('label',label.shape)
            xx = x[10]
            print(xx.shape)
            print(np.nonzero(xx))
        if pos == 0:
            break
    while True:
        (x, label), pos = d.getData()
        if pos == 1024:
            print(pos)
            print('x',x.shape)
            print('label',label.shape)
            xx = x[10]
            print(xx.shape)
            print(np.nonzero(xx))
        if pos == 0:
            break

