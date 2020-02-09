import matplotlib.pyplot as plt
import numpy as np
import math,csv

EPSILON = 1e-9

def normalize(l):
	return np.array([(i-np.mean(l))/math.sqrt(np.var(l)) for i in l])

def h(t,x):
	return 1/(1+math.exp(-t.dot(x)))

def h_v(t,x):
	return np.array([h(t,i) for i in x])

#def L(x,y,t):
#	return sum([ y[i]*math.log(h(t,x[i])) + (1-y[i])*math.log(1-h(t,x[i])) for i in range(len(x)) ])

def delta(x,y,t):
	return (x.transpose().dot( h_v(t,x) - y ))/len(y)

def hessian(x,t):
	hp = np.array([h(t,i)*(1-h(t,i))*i for i in x])		#derivative of h_theta(x)
	return x.transpose().dot( hp )/len(x)

Y = np.array([ float(line[0]) for line in csv.reader(open('logisticY.csv'))])
X = np.array( [np.ones(len(Y))] + [ normalize([float(line[i]) for line in csv.reader(open('logisticX.csv'))]) for i in range(2) ] ).transpose()

theta = np.zeros(3)
thetaP = -np.ones(3)
grad = np.ones(3)
idx = 0

while sum(abs(grad))>EPSILON:
	invhess = np.linalg.inv(hessian(X,theta))
	grad = delta(X,Y,theta)
	theta -= invhess.dot(grad)
	idx+=1

print("iterations reequired:", idx)
print("parameters:",theta)
x1 = [X[i] for i in range(len(X)) if Y[i]==1]
x0 = [X[i] for i in range(len(X)) if Y[i]==0]
plt.scatter([i[1] for i in x0],[i[2] for i in x0],c='magenta',marker='_',linewidths=20)
plt.scatter([i[1] for i in x1],[i[2] for i in x1],c='brown',marker='+')
x = np.linspace(-2,2,20)
y = -(theta[0]+ theta[1]*x)/theta[2]
plt.plot(x,y)
plt.legend(['h_theta(x)=0.5','y=0','y=1'],loc=1)
plt.ylim(-3,4)
plt.xlim(-3,4)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
