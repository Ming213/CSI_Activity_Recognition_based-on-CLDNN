import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

def process_mat_data(filename, key):
	data = sio.loadmat(filename)
	data = data[key]
	data_A = np.zeros([480, 30, 200, 3])
	
	for i in range(3):
		for j in range(16):
			for k in range(30):
				data_A[30*j+k,:,:,i] = data[i][30*j+k]

	return data_A
def load_data():
	filename_1 = 'mat/ling_chen.mat'
	filename_2 = 'mat/tian_data.mat'
	#filename = 'ling_chen.mat'
	data_A = process_mat_data(filename_1, 'ling_chen')
	data_B = process_mat_data(filename_2, 'lin')
	train_x = []
	test_x = []
	for i in range(16):
		for j in range(30):
			if j < 20:
				train_x.append(data_A[i*30 + j])
				train_x.append(data_B[i*30 + j])
			else:
				test_x.append(data_A[i*30 + j])
				test_x.append(data_B[i*30 + j])
	train_x = np.array(train_x)
	test_x = np.array(test_x)
	#train_x = train_x.reshape((400, 30, 200, 3))
	#test_x = test_x.reshape((80, 30, 200, 3))
	#train_x = train_x[:,:,:,0]
	#test_x = test_x[:,:,:,0]
	#train_x = train_x.reshape((400, 30, 200, 1))
	#test_x = test_x.reshape((80, 30, 200, 1))
	train_y = np.zeros([640, 1])
	test_y = np.zeros([320, 1])
	for i in range(16):
		for j in range(40):
			train_y[40*i + j] = i

	for i in range(16):
		for j in range(20):
			test_y[20*i + j] = i

	return train_x, train_y, test_x, test_y

