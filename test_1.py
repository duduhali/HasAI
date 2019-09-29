#基于Keras的简单的文本情感分类问题
from keras.layers import Dense, Flatten, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
import numpy as np

# 定义训练文档
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
# define class labels
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

#将文本编码成数字格式
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
'''
one_hot编码映射到[1,n]，不包括0，n为上述的vocab_size
每次运行得到的数字可能会不一样，但同一个单词对应相同的数字
'''
print(encoded_docs)
#输出 [[34, 26], [16, 49], [25, 44], [16, 49], [25], [37], [3, 44], [29, 16], [3, 49], [45, 10, 26, 20]]

'''
padding到相同长度
然后padding到最大的词汇长度，用0向后填充，这也是为什么前面one-hot不会映射到0的原因
'''
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

# 定义模型
input = Input(shape=(4, ))
x = Embedding(vocab_size, 8, input_length=max_length)(input)    #这一步对应的参数量为50*8
x = Flatten()(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input, outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()	#输出模型结构

#训练模型
model.fit(padded_docs, labels, epochs=100, verbose=0)
# 评估模型
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))
# 测试
test = one_hot('good',50) #good的编码每次运行都不同，但和上面的训练文档中的相同单词是一致的
print('test',test)
padded_test = pad_sequences([test], maxlen=max_length, padding='post')
print('padded_test',padded_test)
print(model.predict(padded_test))
