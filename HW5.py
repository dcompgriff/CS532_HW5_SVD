import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


def part2():
	A = np.array([[3, -2],
	             [-1, 5]])
	U, S, V = linalg.svd(A, full_matrices=False)
	print('U: ')
	print(U)
	print('S: ')
	print(S)

	#Plot circle, image of circle, and left singular vectors.
	data = np.zeros((np.linspace(0, 2*3.141).size,2))
	pos = 0
	for i in np.linspace(0, 2*3.141):
		data[pos][0] = math.sin(i)
		data[pos][1] = math.cos(i)
		pos += 1
	plt.plot(list(map(lambda item: item[0], data)), list(map(lambda item: item[1], data)))
	plt.show()
	transformedData = A.dot(data.T)
	transformedData = transformedData.T
	plt.plot(list(map(lambda item: item[0], transformedData)), list(map(lambda item: item[1], transformedData)))
	ax1 = U[:, 0] * S[0]
	ax2 = U[:, 1] * S[1]
	plt.scatter(ax1[0], ax1[1])
	plt.scatter(ax2[0], ax2[1])
	plt.show()


def part3():
	global sdata
	print('Running HW5.py part 3...')
	sdata = np.loadtxt('./sdata.csv', delimiter=',')

	meanSdata = sdata.mean(axis=0)
	normSdata = sdata - (np.ones((sdata.shape))*meanSdata)

	#Find the svd of sdata.
	U, S, V = linalg.svd(normSdata, full_matrices=True)

	#3D Scatter Plot.
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	v1Line = []
	for c in np.linspace(-15, 12):
		v1Line.append(c*V[0])

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.plot(list(map(lambda item: item[0], v1Line)), list(map(lambda item: item[1], v1Line)), zs=list(map(lambda item: item[2], v1Line)), c='r')
	ax.scatter(list(normSdata[:, 0]), list(normSdata[:, 1]), zs=list(normSdata[:, 2]), c='b')

	#Project every point onto the line using the equation  proj_v(b) = (v*v^t)b
	temp = V[0].reshape(V[0].size, 1).dot(V[0].reshape(V[0].size, 1).T)
	xyz = temp.dot(normSdata.T).T
	#ax.scatter(list(map(lambda item: item[0], xyz)), list(map(lambda item: item[1], xyz)), zs=list(map(lambda item: item[2], xyz)), c='g')
	plt.show()

	#Find every wi using the equation wi = v1.T * x
	weights = V[0].reshape((1, V[0].size)).dot(normSdata.T).T
	plt.hist(weights, bins=20)
	plt.ylabel('Count')
	plt.xlabel('Weight')
	plt.title('Point Weight Histogram')
	plt.show()

def part4():
	global sdata
	print('Running HW5.py part 3...')

	#Q4 Part A
	bucky = np.loadtxt('./bucky.csv', delimiter=',')
	plt.imshow(bucky, cmap='gray')
	#plt.show()

	#Q4 Part B
	U, S, V = linalg.svd(bucky, full_matrices=True)
	plt.bar(list(range(0, len(S))), S)
	plt.xlabel('Singular Value Index')
	plt.ylabel('Singular Value Magnitude')
	#plt.show()

	#Q4 Part C
	U, S, V = linalg.svd(bucky, full_matrices=False)
	print('Initial Image size: ' + str(bucky.shape) + ', ' + str(bucky.shape[0]*bucky.shape[1]))
	for r in [10, 20, 50, 100]:
		Sr = np.diag(S[:r])
		img = U[:, :r].dot(Sr).dot(V[:r, :])
		plt.imshow(img, cmap='gray')
		#plt.show()
		#Q4 Part D
		print('Total size for r = ' + str(r) + ' = ' + str(U[:, :r].shape) + ' + ' + str(Sr.shape) + ' + ' + str(V[:r, :].shape))
		print(' = ' + str(U[:, :r].shape[0]*U[:, :r].shape[1]) + ' + ' + str(Sr.shape[0]*Sr.shape[1]) + ' + ' + str(V[:r, :].shape[0]*V[:r, :].shape[1]))
		print(' = ' + str(U[:, :r].shape[0]*U[:, :r].shape[1] + Sr.shape[0]*Sr.shape[1] + V[:r, :].shape[0]*V[:r, :].shape[1]))

def main():
	part2()
	#part3()
	#part4()

if __name__ == '__main__':
	main()

