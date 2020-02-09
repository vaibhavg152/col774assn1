import matplotlib.pyplot as plt
import numpy as np
import csv,math,random
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D

M = 1000000
LR = 0.001
BATCH_SIZE = 1
EPSILON = 0.00001

theta = np.array([3,1,2])
x = np.ndarray(shape=(3,M),dtype=float)
x[0] = np.ones(M)
x[1] = np.random.normal( 3,2,M)
x[2] = np.random.normal(-1,2,M)

x = x.transpose()

noise_y = np.random.normal(0,math.sqrt(2),M)
y = [theta.dot(i) for i in x] + noise_y
#print(y[1]," = ",noise_y[1]," + ",theta,"*",x[1])

#part a done
curr_batch = 0
idx = 0
theta = np.array([0,0,0])
thetaN = np.array([1,1,1])
avg_theta = 0
avg_cost = [0 for i in range(1000)]
theta_prev = [ [0,0,0] for i in range(100)]

t0 = []
t1 = []
t2 = []
theta_all = []
fig = plt.figure()

while True:
	s = curr_batch*BATCH_SIZE
	e = min(s+BATCH_SIZE, M)
	error = y[s:e] - [sum(np.multiply(theta,i)) for i in x[s:e]]
	thetaN = theta - (LR * sum([np.multiply(-error[i],x[s+i]) for i in range((e-s))]) / (e-s))

	theta_prev = list(theta_prev[1:]) + [theta]
	avg_theta = [np.mean([i[j] for i in theta_prev]) for j in range(len(theta_prev[0]))]
	t0.append(theta[0])
	t1.append(theta[1])
	t2.append(theta[2])
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(t0,t1,t2) #,c='red')
	ax.set_title("Data Analysis ")
	ax.set_xlabel('theta0')
	ax.set_ylabel('theta1')
	ax.set_zlabel('theta2')
	ax.set_xlim3d(-1,4)
	ax.set_ylim3d(-1,2)
	ax.set_zlim3d(-1,3)

	plt.show(block=False)
	plt.pause(0.01)

#	avg_cost = list(avg_cost[1:]) + [sum([i**2 for i in error])/(2*BATCH_SIZE)]

	diff = sum(abs(avg_theta-thetaN))/100
	if diff < EPSILON:
		print('breaking...', idx)
		print('number of epochs:',idx*BATCH_SIZE/M)
		break
	theta = [i for i in thetaN]
	curr_batch += 1
	if (curr_batch)* BATCH_SIZE >= len(x):
		curr_batch = 0
	idx += 1

print(theta)
