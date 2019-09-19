from models import Sequential
from layers import Dense,Relu
from optimizers import SGD
from losses import MeanSquaredError
import accuracy

model = Sequential()
model.add( Dense(2,input_shape=(3,1)) )
model.add( Relu() )

model.add( Dense(1) )
model.add( Relu() )

model.compile( loss=MeanSquaredError(), optimizer=SGD(lr=0.05),metrics=accuracy.accuracy)

# model.fit( train_x,train_y,validation_data=(validate_x,validate_y), epochs=10, batch_size=1024)



