import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D






def part2():
	pass


def part3():
	global sdata
	print('Running HW5.py part 3...')
	sdata = np.loadtxt('./sdata.csv', delimiter=',')

	meanSdata = sdata.mean(axis=0)
	normSdata = sdata - (np.ones((sdata.shape))*meanSdata)

	#Find the svd of sdata.
	U, S, V = linalg.svd(normSdata, full_matrices=True)


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
	pass









def main():
	part2()
	part3()
	part4()




if __name__ == '__main__':
	main()

