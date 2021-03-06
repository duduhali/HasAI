import numpy as np
from keras.utils import np_utils

from Layers.dense import Dense
from Layers.relu import Relu
from Layers.sigmoid import Sigmoid
from Layers.flatten import Flatten
from Losses.mean_squared_error import MeanSquaredError
from Metrics.accuracy import Accuracy
from Tool.data import Data



d1 = Dense(26,input_shape=(17*17,1))
f1 = Flatten()
s1 = Sigmoid()

layers = [d1,f1,s1]
losslayer=MeanSquaredError()
accuracy = Accuracy()

with open('data/train.npy', 'rb') as f:
    data = np.load(f, allow_pickle=True)
    train_x = np.array([xx for xx in data[0]])
    train_y = np.array([y for y in data[1]])
    print('train_x', train_x.shape)
    print('train_y', train_y.shape)
    train_labels = np_utils.to_categorical(train_y, 26)  #独热码
    print('train_labels', train_labels.shape)
with open('data/validate.npy', 'rb') as f:
    data = np.load(f, allow_pickle=True)
    validate_x = np.array([xx for xx in data[0]])
    validate_y = np.array([y for y in data[1]])
    print('validate_x', validate_x.shape)
    print('validate_y', validate_y.shape)
train_data = Data(train_x,train_labels,2048)
validate_data = Data(validate_x,validate_y,10000)



epochs = 100
for i in range(epochs):
    print('epochs:', i)
    losssum = 0
    iters = 0
    while True:
        (x, label), pos = train_data.getData()
        for layer in layers:
            x = layer.forward(x)

        loss = losslayer.forward(x, label)
        # print('===={0}/{1} loss: {2}'.format(pos,train_x.shape[0],loss))
        losssum += loss
        iters += 1

        d = losslayer.backward()
        for layer in layers[::-1]:  # 反向传播
            d = layer.backward(d)

        if pos == 0: #批次结束，验证
            (x, label), pos = validate_data.getData()
            for layer in layers:
                x = layer.forward(x)
            accu = accuracy.forward(x, label)  # 调用准确率层forward()函数求出准确率
            print('all loss: ', losssum / iters, end=' ')
            print('accu', accu)
            break

with open('data/test.npy', 'rb') as f:
    data = np.load(f, allow_pickle=True)
    test_x = np.array([xx for xx in data[0]])
    test_y = np.array([y for y in data[1]])
    print('test_x', test_x.shape)
    print('test_y', test_y.shape)

x = test_x[0:20]
label = test_y[0:20]
for layer in layers:
    x = layer.forward(x)
inferenced_y = np.argmax(x, 1)
print(inferenced_y,'Inferenced')
print(label,'Real')
