import numpy as np
from bpso_NN import linearize, binarise, sigmoid
from random import random
c2 = c1 = 1.49617

a = np.array([0, 1, 1, 0, 1, 1, 0])
b = np.array([1, 0, 1, 0, 0, 1, 0])
c = b - a
print(c)
v = 0.03 + c1 * random() * c
sv = sigmoid(v)
lv = linearize(v)
print(lv)
lv = binarise(lv)
sv = binarise(sv)
print(b - lv)
print(b - sv)
