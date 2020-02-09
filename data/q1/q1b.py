import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import csv, math
import numpy as np

LR = 0.1
EPSILON = 0.000001

def normalize(l):
	return [(i-np.mean(l))/math.sqrt(np.var(l)) for i in l]

x = np.array([ [1,i] for i in normalize([float(line[0]) for line in csv.reader(open('./linearX.csv'))]) ])
y = np.array(([float(line[0]) for line in csv.reader(open('./linearY.csv'))]))

idx = 0
theta = np.array([0,0])
thetaN = np.array([1,1])

while sum(abs(thetaN - theta) )/sum(abs(thetaN)) >= EPSILON:
	theta = [i for i in thetaN]
	error = y - [sum(np.multiply(theta,i)) for i in x]
	thetaN = theta - (LR * sum([np.multiply(-error[i],x[i]) for i in range(len(x))]) / len(x))
	idx += 1

print("parameters:",theta)
print("iterations required:",idx)
X = np.array(range(-2,5))
plt.scatter([i[1] for i in x], y, c = 'coral')
plt.xlabel('x')
plt.ylabel('y')
plt.title('linear regression')
plt.plot(X,(theta[0]+theta[1]*X),'b')
plt.show()
