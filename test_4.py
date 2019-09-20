import numpy as np

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

print(softmax(np.array([1,2,3,5])))
