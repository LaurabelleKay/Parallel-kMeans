import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

X = np.random.rand(100, 1) * 3
Y = np.random.rand(100, 1) * 3
X1 = np.random.rand(100, 1) * 5 + 4
Y1 = np.random.rand(100, 1) * 5 + 4

X=np.asfarray(X)
Y = np.asfarray(Y)
X1 = np.asfarray(X1)
Y1 = np.asfarray(Y1)

X = np.concatenate((X, X1), axis=0)
Y = np.concatenate((Y, Y1), axis=0)

np.savetxt("X2.txt", X,  fmt='%1.1f')
np.savetxt("Y2.txt", Y,  fmt='%1.1f')

#for i in range(X):
 #   f.write(X[i] + "\n")

plt.scatter(X,Y, c= "black", s = 7)
plt.show()
