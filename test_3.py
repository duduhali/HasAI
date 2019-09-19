import numpy as np
'''
一个两层网络，输入层(不算到层数里面)为3 隐藏层2个节点 输出层1个节点
'''
def fnW(arg): #不用矩阵单独计算一条输入数据
    i1, i2, i3 = arg
    w1, w2, w3, w4, w5, w6, w7, w8 = 1, 2, 3, 4, 5, 6, 7, 8
    b1, b2, b3 = 1, 2, 3

    h1_in = i1 * w1 + i2 * w2 + i3 * w3 + b1
    h2_in = i1 * w4 + i2 * w5 + i3 * w6 + b2
    h1_out = max(0, h1_in)  # relu激活函数
    h2_out = max(0, h2_in)

    y_in = h1_out * w7 + h2_out * w8 + b3
    y_out = max(0, y_in)
    return y_out
# data = [ [1,2,3], [4,0,-5], [0,1,1], [-1,-2,-3], [5,5,6] ]
# result = []
# for i in data:
#     result.append(fnW(i))
# print(result) #[380, 3, 149, 3, 905]

def forward(x,weights1,weights2,bias1,bias2):  #用矩阵，同时计算一批输入数据
    # print(x.shape) #(-1, 3, 1)

    # for xx in x:
    #     #weights1 (2,3): xx (3,1)  => (2,1)
    #     print(np.dot(weights1, xx)+bias1)
    h_in = np.array([np.dot(weights1, xx) + bias1 for xx in x])
    h_out =  np.array([hh * (hh > 0) for hh in h_in]) # relu激活函数
    # [[[15]
    #   [34]]]
    # (1, 2, 1)

    #第二层
    y_in = np.array([np.dot(weights2, hh) + bias2 for hh in h_out])
    y_out = np.array([yy * (yy > 0) for yy in y_in]) # relu激活函数

    return y_out ,x,h_out # x,h_out在反向传播时用

# data = [ [1,2,3], [4,0,-5], [0,1,1], [-1,-2,-3], [5,5,6] ]
# data = np.array(data) #(5, 3)
# data = data.reshape((*data.shape,1)) #(5, 3, 1)
#
# w1, w2, w3, w4, w5, w6, w7, w8 = 1, 2, 3, 4, 5, 6, 7, 8
# b1, b2, b3 = 1, 2, 3
# weights1 = np.array([
#     [w1,w2,w3],
#     [w4,w5,w6]
# ])
# bias1 = np.array([
#     [b1],
#     [b2]
# ])
# weights2 = np.array([
#     [w7, w8],
# ])
# bias2 = np.array([
#     [b3],
# ])
# result = forward(data,weights1,weights2,bias1,bias2) #(5, 1, 1)
# print(result)


######################################################################################
# 训练数据和标签
data = [ [1,2,3], [4,0,-5], [0,1,1], [-1,-2,-3], [5,5,6] ]
data = np.array(data) #(5, 3)
data = data.reshape((*data.shape,1))
lable = [380, 3, 149, 3, 905]
lable = np.array(lable)
lable = lable.reshape((*lable.shape,1))


#初始化参数
w1, w2, w3, w4, w5, w6, w7, w8 = 1, 1, 1, 1, 1, 1, 1, 1
b1, b2, b3 = 0,0,0
weights1 = np.array([
    [w1,w2,w3],
    [w4,w5,w6]
])
bias1 = np.array([
    [b1],
    [b2]
])
weights2 = np.array([
    [w7, w8],
])
bias2 = np.array([
    [b3],
])

# 正向传播
r,x,h_out = forward(data,weights1,weights2,bias1,bias2)
# x,h_out在反向传播时用
from functools import reduce
result = r.reshape(-1,reduce(lambda x,y:x*y,r.shape[1:])) #压平  Flatten

#损失值 平方差
loss = np.sum(np.square(lable - result)) / 2 / lable.shape[0]
print(loss)
lr = 0.001 #学习率

dx = (result - lable)/lable.shape[0]  # 是标签减预测值，还是预测值减标签,关系到权重更新时加增量还是减增量
dx = dx.reshape(r.shape)   #上面压平的反操作，缺少了维度不对下面会出错
#(5, 1, 1)
print('误差数组',dx)

#反向传播
def relu_reverse(signal): # relu 的导数
    return -np.minimum(0, signal)
dx = relu_reverse(dx)
print('第二层relu求导后',dx)
# print(x.shape) #(5, 3, 1)
# for dd, xx in zip(dx, h_out):
#     print(dd) #[ 73.6]
#     print(xx.T) #[[6 6]]
#     break
ddw = [np.dot(dd, xx.T) for dd, xx in zip(dx, h_out)]  # 根据链式法则，将反向传递回来的导数值乘以x，得到对参数的梯度
print('ddw',ddw)
dw = np.sum(ddw, axis=0) / h_out.shape[0]  #求误差对权重的平均值
print('dw',dw) #[658.64 658.64]
db = np.sum(dx, axis=0) / h_out.shape[0]  #
print('db',db)
# for dd in dx:
#     print(weights2.T) #(2,1)
#     # [[1]
#     #  [1]]
#     print(dd.shape) #[73.6]  (1,)
#     print( np.dot(weights2.T, dd).shape ) #(2,)
#     break
dx = np.array([np.dot(weights2.T, dd) for dd in dx])  # 继续反向传播梯度
print('继续传递',dx)

weights2 = weights2-lr * dw
bias2 = bias2 - lr * db
print(weights2)
print(bias2)




dx = relu_reverse(dx)
print('第一层relu求导后',dx)
# for dd, xx in zip(dx, x):
#     print(dd)
#     # [[-0.]
#     #  [-0.]]
#     print(xx.T) #[[1 2 3]]
#     print(np.dot(dd, xx.T))
#     break
ddw = [np.dot(dd, xx.T) for dd, xx in zip(dx, x)]
print('ddw',ddw)
dw = np.sum(ddw, axis=0) / x.shape[0]  #求误差对权重的平均值
print('dw',dw)
db = np.sum(dx, axis=0) / x.shape[0]  #
print('db',db)









