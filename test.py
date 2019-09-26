import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Flatten, Lambda,Input,Activation

class Data:
    def __init__(self, name, batch_size):  # 数据所在的文件名name和batch中图片的数量batch_size
        print(name)
        with open(name, 'rb') as f:
            data = np.load(f,allow_pickle=True)
        self.x = data[0]  # 输入x
        self.y = data[1]  # 预期正确输出y
        self.len = len(self.x)
        self.batch_size = batch_size
        self.pos = 0  # pos用来记录数据读取的位置

    def forward(self):
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

def tran_y(y):
    y_ohe = np.zeros(26)
    y_ohe[y] = 1
    return y_ohe

data_train = Data('data/train.npy', 40000)  # 用于训练，batch_size设置为1024
data_validate = Data('data/validate.npy', 10000) # 用于验证所以设置batch_size为10000,一次
data_test = Data('data/test.npy', 10000)  # 用于测试

train_d, pos = data_train.forward()  # 从数据层取出数据
# train_d[0][0].shape = (289, 1)
train_x = np.array([one for one in train_d[0]])
train_y = np.array([tran_y(one) for one in train_d[1]])

# 验证数据
validate_d, _ = data_validate.forward()
validate_x = np.array([one for one in validate_d[0]])
validate_y = np.array([tran_y(one) for one in validate_d[1]])
#测试数据
test_d, _ = data_test.forward()
test_x = np.array([one for one in test_d[0]])
test_y = np.array([tran_y(one) for one in test_d[1]])



from keras import backend as K
from keras.engine.topology import Layer
class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        # print('output_dim',output_dim) #output_dim 40
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # print('shape',(input_shape[1], self.output_dim))  #shape (11560, 40)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


inputs = Input(shape=(17*17,1))
x = Dense(40, activation='relu')(inputs)
x = Flatten()(x)

# x = Dense(40, activation='relu')(x)
x = MyLayer(40)(x)
x = Activation('relu')(x)

predictions = Dense(26, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)

model.summary()

# 完成模型的搭建后，我们需要使用.compile()方法来编译模型
# from keras import losses
# model.compile(loss=losses.mean_squared_error, optimizer='sgd')
model.compile( loss='categorical_crossentropy', optimizer='adagrad',metrics=['accuracy'])
model.fit( train_x,train_y,validation_data=(validate_x,validate_y), epochs=2, batch_size=1024)

scores = model.evaluate(test_x,test_y,verbose=0) #对我们的模型进行评估

# 预测数据
test_output = model.predict(test_x[0:20], batch_size=20)
inferenced_y = np.argmax(test_output, 1)
print(inferenced_y, 'Inferenced numbers')  # 推测的数字
print(np.argmax(test_y[:20], 1), 'Real numbers')  # 真实的数字


