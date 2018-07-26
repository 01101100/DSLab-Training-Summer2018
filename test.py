import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7])
x = np.reshape(x,(-1, 1))
print(x[np.array([0, 1, 4])])


