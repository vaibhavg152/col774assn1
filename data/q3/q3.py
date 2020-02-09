import matplotlib.pyplot as plt
import numpy as np
import math,csv

x = [ [ 1, float(line[0]), float(line[1])] for line in csv.reader(open('logisticX.csv'))]
y = np.array([ float(line[0]) for line in csv.reader(open('logisticY.csv'))])

theta = np.array([0,0,0])

x1 = [x[i] for i in range(len(x)) if y[i]==1]
x0 = [x[i] for i in range(len(x)) if y[i]==0]
plt.scatter([i[1] for i in x0],[i[2] for i in x0],c='magenta',marker='_',linewidths=20)
plt.scatter([i[1] for i in x1],[i[2] for i in x1],c='brown',marker='+')

def h(theta,x):
	return 1/(1+math.exp(theta.dot(x)))
def L(x,y,theta):
	return sum([ y[i]*math.log(h(theta,x[i])) + (1-y[i])*math.log(1-h(theta,x[i])) for i in range(len(x)) ])

print(L(x,y,theta))

for idx in range(1):
	error = y - [sum(np.multiply(theta,i)) for i in x]
	gradient = sum([np.multiply(-error[i],x[i]) for i in range(len(x))]) / len(x)

	thetaN = theta - np.linalg.inv(H)*gradient
	theta = [i for i in thetaN]
