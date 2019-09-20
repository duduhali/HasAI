import numpy as np

def accuracy(x, label):
    acc = np.sum([np.argmax(xx) == ll for xx, ll in zip(x, label)])
    acc = 1.0 * acc / x.shape[0]
    return acc


