import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import csv, math
import numpy as np

LR = 0.01
EPSILON = 0.001
WAIT_TIME = 0.2

def normalize(l):
	return [(i-np.mean(l))/math.sqrt(np.var(l)) for i in l]

x = np.array([ [1,i] for i in normalize([float(line[0]) for line in csv.reader(open('./linearX.csv'))]) ])
y = np.array(normalize([float(line[0]) for line in csv.reader(open('./linearY.csv'))]))

J = []
t0 = []
t1 = []

idx = 0
theta = np.array([0,0])
thetaN = np.array([1,1])
fig = plt.figure()

xx = np.arange(-1,1.2,0.01)
yy = np.arange(-0.5,1.5,0.01)
tgrid0, tgrid1 = np.meshgrid(xx,yy,sparse=True)
z = sum([ (tgrid0*x[i][0]+tgrid1*x[i][1] - y[i])**2 for i in range(len(x)) ])/len(x)
plt.contourf(xx, yy, z)
plt.xlim(-1,1.2)
plt.ylim(-0.5,1.5)

while sum(abs(thetaN - theta) )/sum(abs(thetaN)) >= EPSILON:
	theta = [i for i in thetaN]
	error = y - [sum(np.multiply(theta,i)) for i in x]
	thetaN = theta - (LR * sum([np.multiply(-error[i],x[i]) for i in range(len(x))]) / len(x))
	J.append(sum([i**2 for i in error])/(2*len(x)))
	t0.append(theta[0])
	t1.append(theta[1])
	idx += 1

	plt.scatter(t0,t1,c='k',marker='_',linewidths=0.01)
	plt.show(block=False)
	plt.pause(WAIT_TIME)

"""	#plot figures
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(t0,t1,J,c='red')
	ax.set_title("Data Analysis ")
	ax.set_xlabel('theta0')
	ax.set_ylabel('theta1')
	ax.set_zlabel('J(theta)')
	ax.set_xlim3d(1,-0.1)
	ax.set_ylim3d(1,-0.1)
	ax.set_zlim3d(0,1)
"""

#h_theta = [sum(np.multiply(thetaN,[1,i])) for i in x]
#print("parameters:",theta)
#print("iterations required:",idx)
#print(len(t0),len(t1),len(J))
plt.show()
#X = np.array(range(-2,5))
#plt.scatter([i[1] for i in x], y, c = 'coral')
#plt.plot(X,(theta[0]+theta[1]*X),'b')
#plt.show()
