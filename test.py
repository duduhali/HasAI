from models import Sequential
from layers import Dense,Relu,Flatten,Softmax
from optimizers import SGD
from losses import MeanSquaredError
import accuracy
import numpy as np


model = Sequential()
model.add( Dense(17*17,input_shape=(17*17,1)) )
model.add( Relu() )
model.add(Flatten())
model.add( Dense(40) )
model.add( Relu() )
model.add( Dense(26) )
model.add( Softmax())



# model.compile( loss='categorical_crossentropy', optimizer='adagrad',metrics=['accuracy'])
# model.fit( train_x,train_y,validation_data=(validate_x,validate_y), epochs=10, batch_size=1024)
# scores = model.evaluate(test_x,test_y,verbose=0)


