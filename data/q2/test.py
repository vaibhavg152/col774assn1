import matplotlib.pyplot as plt
import csv, math
import numpy as np

x = [ line[:2] for line in csv.reader(open('q2test.csv')) ]
y = [ line[2]  for line in csv.reader(open('q2test.csv')) ]

x = [np.array([1,float(i[0]),float(i[1])]) for i in x[1:]]
y = np.array([float(i) for i in y[1:]])

errors = []
batch_sizes = [1,100,10000,1000000]
thetas = []
thetas.append([3,  1,  2])
thetas.append([2.935,  1.006,  2.003])
thetas.append([2.957,  1.013,  1.997])
thetas.append([2.963,  1.008,  1.999])
thetas.append([2.997,  1.000,  2.000])

for t in thetas:
	yp = np.array([ i.dot(t) for i in x ])
	errors.append(sum([ i**2 for i in y - yp])/(2*len(x)))
print(errors)

plt.plot([math.log(i,10) for i in batch_sizes],errors[1:])
plt.scatter([math.log(i,10) for i in batch_sizes],errors[1:])
plt.xlabel('log(batch_size)')
plt.ylabel('errors')
plt.show()
