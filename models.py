from layers import Layer

class Sequential:
    def __init__(self):
        self.trainable = True #模型是否可训练
        self.layers = []
    def add(self,layer):
        self.layers.append(layer)
    #编译模型
    def compile(self,loss, optimizer,metrics):
        output_shape = None
        for layer in self.layers:
            if isinstance(layer, Layer):
                output_shape = layer.init(output_shape)
            else:
                layer.init()
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    #训练模型
    def fit(self,train_x,train_y,validation_data=None, epochs=1, batch_size=None):
        if batch_size==None:
            batch_size = train_x.shape[0]
        train = Data(train_x,train_y,batch_size)
        validation = None if validation_data==None else Data(validation_data[0], validation_data[1])

        for i in range(epochs):
            print('epochs:',i)
            losssum = 0
            iters = 0
            while True:
                (x,label), pos = train.getData()  # 从数据层取出数据
                for layer in self.layers:
                    x = layer.forward(x) # 前向计算
                loss = self.loss.forward(x, label)  # 调用损失层forward函数计算损失函数值
                losssum += loss
                iters += 1
                dx = self.loss.backward()  # 调用损失层backward函数层计算将要反向传播的梯度
                for layer in self.layers[::-1]:  # 反向传播
                    if isinstance( layer,Layer ):
                        dx = layer.backward(dx,self.optimizer.lr)
                    else:
                        dx = layer.backward(dx)

                if pos == 0:  # 一个epoch完成
                    print('loss:', losssum / iters)

                    if validation!=None:  # 进行准确率测试
                        (x, label), _ = validation.getData()
                        for layer in self.layers:
                            x = layer.forward(x)
                        accu = self.metrics(x, label)  # 调用准确率层forward()函数求出准确率
                        print('accuracy:', accu)
                    break
    def printweight(self):
        print('打印参数：')
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.printweight()

    #评估,返回评分scores
    def evaluate(self):
        pass
    # 预测，返回预测结果
    def predict(self):
        pass
    def save(self):
        pass
    def load(self):
        pass


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

