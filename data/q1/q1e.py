import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import csv, math
import numpy as np

EPSILON = 0.0005
WAIT_TIME = 0.2

def normalize(l):
	return [(i-np.mean(l))/math.sqrt(np.var(l)) for i in l]

x = np.array([ [1,i] for i in normalize([float(line[0]) for line in csv.reader(open('./linearX.csv'))]) ])
y = np.array(([float(line[0]) for line in csv.reader(open('./linearY.csv'))]))

for LR in [0.1,0.025,0.001,0.9,1.0]:
	J = []
	t0 = []
	t1 = []

	idx = 0
	theta = np.array([0,0])
	thetaN = np.array([1,1])
	fig = plt.figure()

	xx = np.arange(-1.5,1.5,0.01)
	yy = np.arange(-1.5,1.5,0.01)
	tgrid0, tgrid1 = np.meshgrid(xx,yy,sparse=True)
	z = sum([ (tgrid0*x[i][0]+tgrid1*x[i][1] - y[i])**2 for i in range(len(x)) ])/len(x)
	plt.contourf(xx, yy, z)
	plt.xlim(-1.5,1.5)
	plt.ylim(-1.5,1.5)

	while sum(abs(thetaN - theta) )/sum(abs(thetaN)) >= EPSILON:
		theta = [i for i in thetaN]
		error = y - [sum(np.multiply(theta,i)) for i in x]
		thetaN = theta - (LR * sum([np.multiply(-error[i],x[i]) for i in range(len(x))]) / len(x))
		J.append(sum([i**2 for i in error])/(2*len(x)))
		t0.append(theta[0])
		t1.append(theta[1])
		idx += 1

		plt.scatter(t0,t1,c='k',marker='_',linewidths=0.01)
		plt.title('Learning rate:'+str(LR)+' iterations:'+str(idx))
		plt.xlabel('theta0')
		plt.ylabel('theta1')
		plt.show(block=False)
		plt.pause(WAIT_TIME)

	plt.show(block=False)
