import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from process_mat_4 import *
import random
import xlrd
import xlwt
import scipy.io as sio

lr = 0.0005
num_epochs=500

#batch_size = 50
minibatch_size = 80
n_input = 16
n_step = 16
n_hidden_units = 128
n_class = 16
train_x, train_y, test_x, test_y = load_data()
loss_list=[]
train_accuracy_list=[]
test_accuracy_list=[]
flag_list=[]

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    with tf.name_scope('input_data') as scope:
        X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name='X')
        Y_orig = tf.placeholder(tf.int32, [None], name='Y_orig')
        Y = tf.one_hot(Y_orig, 16, axis = 1) #axis=0表示按行展开，每一行是一类，每一列是一个样本
        batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        return X, Y_orig, Y, batch_size

def print_activations(t):
	print(t.op.name, '', t.get_shape().as_list())

param_detail = []
time_length = 0

def layer_cnn(t_input):
	#t_input = tf.reshape(t_input, [-1, 28, 28, 1])
	with tf.name_scope('conv-1') as scope:
		kernel = tf.Variable(initial_value=tf.truncated_normal(shape=[3, 10, 3, 8], dtype=tf.float32, stddev=0.01), name='kernel')
		bias = tf.Variable(initial_value=tf.truncated_normal(shape=[8], dtype=tf.float32, stddev=0.01), name='bias')
		conv1 = tf.nn.conv2d(t_input, kernel, strides=[1,1,1,1], padding="SAME")
		out_1 = tf.nn.bias_add(conv1, bias)#tf.add的特殊形式，最后一维必须相同
		net = tf.nn.relu(out_1, name=scope)
		param_detail.append(net)
	with tf.name_scope('pool-1'):
		net = tf.nn.max_pool(net, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
		param_detail.append(net)
	with tf.name_scope('conv-2') as scope:
		kernel = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 8, 16], dtype=tf.float32, stddev=0.01), name='kernel')
		bias = tf.Variable(initial_value=tf.truncated_normal(shape=[16], dtype=tf.float32, stddev=0.01), name='bias')
		conv2 = tf.nn.conv2d(net, kernel, strides=[1,2,4,1], padding='SAME')
		out_2 = tf.nn.bias_add(conv2, bias)
		net = tf.nn.relu(out_2, name=scope)
		param_detail.append(net)
	return net

def linear_layer(t_input):
	t_input_shape = t_input.get_shape().as_list()
	f_width = t_input_shape[1]
	f_height = t_input_shape[2]
	#time_length = f_height
	f_maps = t_input_shape[3]
	flatten_size = f_width * f_height * f_maps
	net = tf.reshape(t_input, shape=[-1, flatten_size])
	param_detail.append(net)
	with tf.name_scope('linear_layer'):
		dense_w = tf.Variable(initial_value=tf.truncated_normal(shape=[flatten_size, 256], stddev=0.1), dtype=tf.float32)
		dense_b = tf.Variable(initial_value=tf.truncated_normal(shape=[256]), dtype=tf.float32)
		net = tf.nn.bias_add(tf.matmul(net, dense_w), dense_b)
		param_detail.append(net)
		return net

weight = {
	'in': tf.Variable(tf.random_normal([n_input, n_hidden_units])),
	'out': tf.Variable(tf.random_normal([n_hidden_units, n_class]))
}
bias = {
	'in': tf.constant(0.1, shape=[n_hidden_units,]),
	'out': tf.constant(0.1, shape=[n_class])
}

def rnn(X, Weights, biases, batch_size):
	X = tf.reshape(X, [-1, n_input])
	X_in = tf.matmul(X, weight['in']) + biases['in']
	X_in = tf.reshape(X_in, [-1, n_step, n_hidden_units])
	with tf.name_scope('lstm') as scope:
	    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias = 1.0, state_is_tuple=True)
	    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
	    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
	    results = tf.matmul(states[1], weight['out']) + biases['out']

	return results

def model(X_train, Y_train, X_test, Y_test):
	#saver = tf.train.Saver()
	(m, n_H0, n_W0, n_C0) = X_train.shape
	(m, n_y) = Y_train.shape
	X, Y_orig, Y, batch_size = create_placeholders(n_H0, n_W0, n_C0, n_y)
	cost = []
	cnn_out = layer_cnn(X)
	linear_out = linear_layer(cnn_out)
	pred = rnn(linear_out, weight, bias, batch_size)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y),)
	train_op = tf.train.AdamOptimizer(lr).minimize(cost)
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))#返回每一行最大位置的索引
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		writer = tf.summary.FileWriter('./graphs', sess.graph)
		sess.run(init)
		X_train = X_train.reshape((m, n_H0 * n_W0 * n_C0))
		train_data = np.hstack((X_train, Y_train))
		np.random.shuffle(train_data)
		X_train = train_data[:,:-1]
		X_train = X_train.reshape((m, n_H0, n_W0, n_C0))
		Y_train = train_data[:, -1]
		for epoch in range(num_epochs):
			epoch_loss = 0
			i = 0
			while (i+minibatch_size)<len(X_train):
				start = i
				end = i + minibatch_size
				batch_x = X_train[start:end]
				batch_y = Y_train[start:end]
				_,c = sess.run([train_op, cost], feed_dict = {X: batch_x, Y_orig: batch_y, batch_size: minibatch_size})
				epoch_loss += c
				i += minibatch_size
			test_accuracy_list.append(accuracy.eval({X: X_test, Y_orig: np.squeeze(Y_test), batch_size: 320}))
			train_accuracy_list.append(accuracy.eval({X: X_train, Y_orig: np.squeeze(Y_train), batch_size: 640}))
			print('%i:%f' %(epoch, epoch_loss))
			loss_list.append(epoch_loss)
			

		correct_prediction = tf.equal(tf.argmax(pred), tf.argmax(Y))
		#saver.save(sess, "model/cldnn")
		tf_predict = tf.argmax(pred,axis=1)
		y = tf.argmax(Y)
		print(tf_predict)
		predict = tf_predict.eval({X: test_x, Y_orig: np.squeeze(test_y), batch_size: 320})
		predict = np.array(predict)
		predict = predict.reshape(16, 20)
		Accu = np.zeros([16, 20])
		for i in range(16):
			for j in range(20):
				if predict[i][j] == i:
					Accu[i][j] = 1
		out = np.mean(Accu, axis = 1)
		print(out)
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		
		print("Train Accuracy:", accuracy.eval({X: X_train, Y_orig: np.squeeze(Y_train), batch_size: 640}))
		print("Test Accuracy:", accuracy.eval({X: X_test, Y_orig: np.squeeze(Y_test), batch_size: 320}))
	
		writer.close()
		i=1
		while i<=num_epochs:
		 	flag_list.append(i)
		 	i=i+1
		# # plt.plot(flag_list,test_accuracy_list,label='test',linewidth=1,color='r',marker='o', markerfacecolor='r',markersize=1)		
		# plt.plot(flag_list,loss_list,label='LOSS',linewidth=3,color='r',marker='o', markerfacecolor='blue',markersize=1)
		# # plt.plot(flag_list,train_accuracy_list,label='train',linewidth=1,color='b',marker='o', markerfacecolor='blue',markersize=1)
		# plt.xlabel('iterations')
		# plt.ylabel('loss')
		# plt.legend() 
		# plt.show()
		save_fn = 'loss.mat'
		save_array = np.array(loss_list)
		sio.savemat(save_fn, {'array': save_array}) #和上面的一样，存在了array变量的第一行

		save_fn = 'test_accuracy.mat'
		save_array = np.array(test_accuracy_list)
		sio.savemat(save_fn, {'array': save_array}) #和上面的一样，存在了array变量的第一行

		save_fn = 'train_accuracy.mat'
		save_array = np.array(train_accuracy_list)
		sio.savemat(save_fn, {'array': save_array}) #和上面的一样，存在了array变量的第一行


 



model(train_x, train_y, test_x, test_y)
num_epochs=num_epochs+3

