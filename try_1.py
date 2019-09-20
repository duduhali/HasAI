from models import Sequential
from layers import Dense,Relu,Flatten
from optimizers import SGD
from losses import MeanSquaredError
import accuracy
import numpy as np


model = Sequential()
model.add( Dense(2,input_shape=(3,1)) )
model.add( Relu() )

model.add( Dense(1) )
model.add( Relu() )
model.add( Flatten() )

model.compile( loss=MeanSquaredError(), optimizer=SGD(lr=0.05),metrics=accuracy.accuracy)



data = [ [1,2,3], [4,0,-5], [0,1,1], [-1,-2,-3], [5,5,6] ]
data = np.array(data) #(5, 3)
train_x = data.reshape((*data.shape,1)) #(5, 3, 1)
lable = [380, 3, 149, 3, 905]
lable = np.array(lable)
train_y = lable.reshape((*lable.shape,1)) #(5, 1)

# model.printweight()
model.fit( train_x,train_y,validation_data=None, epochs=3, batch_size=None)

# model.printweight()