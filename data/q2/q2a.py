import matplotlib.pyplot as plt
import numpy as np
import csv,math

M = 1000000
theta = np.array([3,1,2])
x = np.ndarray(shape=(3,M),dtype=float)
x[0] = np.ones(M)
x[1] = np.random.normal( 3,2,M)
x[2] = np.random.normal(-1,2,M)
x = x.transpose()
noise_y = np.random.normal(0,math.sqrt(2),M)
y = [theta.dot(i) for i in x] + noise_y
#print(y[1]," = ",noise_y[1]," + ",theta,"*",x[1])
open('sampledPoints.csv','w+').write( "X,Y\n" + '\n'.join([ str(x[i])+","+str(y[i]) for i in range(M)]))
