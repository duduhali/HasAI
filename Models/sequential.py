from Tool.data import Data
from Layers.dense import Dense
from Layers.flatten import Flatten
from Layers.flatten import Reshape

class Sequential:
    def __init__(self):
        self.trainable = True #模型是否可训练
        self.layers = []
    def add(self,layer):
        self.layers.append(layer)
    #编译模型
    def compile(self,loss, lr, metrics):
        self.loss = loss
        self.metrics = metrics

        layer_shape = None
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer_shape = layer.setParam(layer_shape, lr)
            elif isinstance(layer, Flatten):
                layer_shape = layer.setParam(layer_shape)
            elif isinstance(layer, Reshape):
                layer_shape = layer.setParam()


    #训练模型
    def fit(self,train_x,train_y,validation_data=None, epochs=1, batch_size=None,verbose=0):
        if batch_size==None:
            batch_size = train_x.shape[0]
        data_train = Data(train_x, train_y, batch_size)
        data_validate = None if validation_data==None else Data(validation_data[0], validation_data[1], 10000)

        # epochs = 20
        for i in range(epochs):
            print('epochs:', i, end=' ')
            losssum = 0
            iters = 0
            while True:
                (x, label), pos = data_train.getData()
                for layer in self.layers:
                    x = layer.forward(x)

                loss = self.loss.forward(x, label)
                if verbose == 1:
                    print('\n===={0}/{1} loss: {2}'.format(pos,train_x.shape[0],loss), end=' ')
                losssum += loss
                iters += 1

                d = self.loss.backward()
                for layer in self.layers[::-1]:  # 反向传播
                    d = layer.backward(d)

                if pos == 0:  # 批次结束，验证
                    print('        loss: ', losssum / iters, end=' ')
                    if validation_data !=None:
                        (x, label), pos = data_validate.getData()
                        for layer in self.layers:
                            x = layer.forward(x)
                        accu = self.metrics[0].forward(x, label)  # 调用准确率层forward()函数求出准确率
                        print('    accu', accu)
                    break

    # 预测，返回预测结果
    def predict(self,data_x):
        x = data_x
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # def save(self):
    #     pass
    # def load(self):
    #     pass
