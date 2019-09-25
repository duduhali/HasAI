import numpy as np
from keras.utils import np_utils

from Models.sequential import Sequential
from Layers.dense import Dense
from Layers.relu import Relu
from Layers.sigmoid import Sigmoid
from Layers.flatten import Flatten
from Layers.flatten import Reshape
from Losses.mean_squared_error import MeanSquaredError
from Losses.cross_entropy_loss import CrossEntropyLoss
from Metrics.accuracy import Accuracy



model = Sequential()

model.add( Dense(40,input_shape=(17*17,1)) )
model.add(Reshape((40,)))
# model.add(Flatten())
model.add(Sigmoid())
model.add(Dense(26))


# model.add( Dense(26,input_shape=(17*17,1)) )
# model.add(Flatten())

model.add( Sigmoid() )

losslayer=MeanSquaredError()
# losslayer = CrossEntropyLoss()  # 交叉熵损失函数
accuracy = Accuracy()

model.compile( loss=losslayer, lr=2000, metrics=[accuracy])


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

model.fit( train_x,train_labels,validation_data=(validate_x,validate_y), epochs=100, batch_size=2048)



with open('data/test.npy', 'rb') as f:
    data = np.load(f, allow_pickle=True)
    test_x = np.array([xx for xx in data[0]])
    test_y = np.array([y for y in data[1]])
    print('test_x', test_x.shape)
    print('test_y', test_y.shape)


x = model.predict( test_x[0:20] )
inferenced_y = np.argmax(x, 1)
print(inferenced_y,'Inferenced')
print(test_y[0:20],'Real')
