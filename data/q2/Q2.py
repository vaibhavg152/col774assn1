import matplotlib.pyplot as plt
import csv
import math
import numpy as np

BATCH_SIZE = 10
LR = 0.025
EPSILON = 0.001

def normalize(l):
	return [(i-np.mean(l))/math.sqrt(np.var(l)) for i in l]

x = np.array([ [1,i] for i in normalize([float(line[0]) for line in csv.reader(open('./linearX.csv'))]) ])
y = np.array(normalize([float(line[0]) for line in csv.reader(open('./linearY.csv'))]))

curr_batch = 0
idx = 0
theta = np.array([0,0])
thetaN = np.array([0,0])
while True:
	s = curr_batch*BATCH_SIZE
	e = min(s+BATCH_SIZE, len(x))
	error = y[s:e] - [sum(np.multiply(theta,i)) for i in x[s:e]]
	thetaN = theta - (LR * sum([np.multiply(-error[i],x[s+i]) for i in range((e-s))]) / (e-s))
	diff = abs(sum([abs(thetaN[i]-theta[i]) for i in range(len(theta))]))
	if diff < EPSILON:
		print('breaking...', idx)
		break
	theta = [i for i in thetaN]
	curr_batch += 1
	if (curr_batch)* BATCH_SIZE >= len(x):
		curr_batch = 0
	idx += 1
#	loss  = sum([i**2 for i in error])/(2*BATCH_SIZE)
#	if idx %100 == 0:
#		print(theta[0])

h_theta = [sum(np.multiply(thetaN,[1,i])) for i in x]
print(theta)
plt.scatter([i[1] for i in x], [i[1] for i in h_theta])
plt.scatter([i[1] for i in x], y)
plt.show()
