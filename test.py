from models import Sequential
from layers import Dense,Relu,Flatten,Softmax
from optimizers import SGD
from losses import CrossEntropyLoss
import accuracy
import numpy as np


model = Sequential()
model.add( Dense(17*17,input_shape=(17*17,1)) )
model.add( Relu() )
# model.add(Flatten())
model.add( Dense(40) )
model.add( Relu() )
model.add( Dense(26) )
model.add( Softmax())

model.compile( loss=CrossEntropyLoss(), optimizer=SGD(lr=0.05),metrics=accuracy.accuracy)


with open('data/train.npy', 'rb') as f:
    data = np.load(f, allow_pickle=True)
    train_x = data[0]  # 输入x
    train_y = data[1]  # 预期正确输出y
with open('data/train.npy', 'rb') as f:
    data = np.load(f, allow_pickle=True)
    validate_x = data[0]
    validate_y = data[1]

model.fit( train_x,train_y,validation_data=(validate_x,validate_y), epochs=10, batch_size=1024)


# scores = model.evaluate(test_x,test_y,verbose=0)


