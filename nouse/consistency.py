# 2019-11-13, Ning
import numpy as np
import numpy.linalg as LA

class consistency:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def affinity(self, delta):
		n = self.x.shape[0]
		dis = np.zeros([n,n])
		for i in range(n):
			for j in range(i+1,n):
				dis[i,j] = np.exp(-np.square(LA.norm(self.x[i]-self.x[j]))/(2*delta**2))
				dis[j,i] = dis[i,j]
		return dis
if __name__ == '__main__':
	x = np.array([[1,2,3,4],[3,4,2,5],[7,5,8,9],[6,7,8,5],[9,6,7,8]])
	y = np.array([1,1,2,2,2])
	cos = consistency(x,y)
	affinity = cos.affinity(5)
	print(affinity)

