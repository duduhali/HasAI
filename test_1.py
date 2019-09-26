import tensorflow as tf
data1 = tf.constant( [[6,6]])
data2 = tf.constant( [[2],
                     [2]])
data3 = tf.constant( [[3,3]])
matMul = tf.matmul(data1, data2)
matMul2 = tf.multiply(data1, data2) #乘  不是矩阵运算
matAdd = tf.add(data1,data3)
with tf.Session() as sess:
    print( sess.run(matMul) )
    print(sess.run(matMul2))
    print(sess.run(matAdd))
    print(sess.run([matMul,matAdd]))
print('end!')