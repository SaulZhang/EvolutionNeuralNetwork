from data_fetcher import get_gaussian_quantiles, generate_xor
from standard_neat import start_neuroevolution
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import tensorflow as tf 
import numpy as np 

MODE = "train"

if MODE == "train":

	breast_cancer = load_breast_cancer()
	X_train, X_test, y_train, y_test = train_test_split(breast_cancer['data'], breast_cancer['target'], random_state=0)
	# sys.exit(0)
	#对数据的每一维度进行归一化(x-min)/(max-min)
	for i in range(X_train.shape[1]):
		X_train[:,i] = (X_train[:,i] - X_train[:,i].min())/(X_train[:,i].max()-X_train[:,i].min())

	for i in range(X_test.shape[1]):
		X_test[:,i] = (X_test[:,i] - X_test[:,i].min())/(X_test[:,i].max()-X_test[:,i].min())

	y_train_flatten = []

	for element in y_train:
		each_vector = np.zeros(2)
		each_vector[element] = 1
		y_train_flatten.append(each_vector)
	y_train_flatten = np.array(y_train_flatten)

	y_test_flatten = []
	for element in y_test:
		each_vector = list(np.zeros(2))
		each_vector[element] = 1
		y_test_flatten.append(each_vector)
	
	y_test_flatten = np.array(y_test_flatten)

	start_neuroevolution(X_train, y_train_flatten, X_test, y_test_flatten)
	
elif MODE == "test":
#wait to change..
	with tf.Session() as sess:
		new_saver = tf.train.import_meta_graph('model/best-model-1.0-6.meta')
		new_saver.restore(sess, tf.train.latest_checkpoint('./model'))

		graph = tf.get_default_graph()
		x0 = graph.get_tensor_by_name("input/x0:0")
		x1 = graph.get_tensor_by_name("input/x1:0")
		x2 = graph.get_tensor_by_name("input/x2:0")
		x3 = graph.get_tensor_by_name("input/x3:0")

		# x_test, y_test = generate_xor(n_samples=10000)
		
		feed_dict={x0: np.expand_dims(x_test[:, 0], 1), x1: np.expand_dims(x_test[:, 1], 1)}
		op_to_restore = graph.get_tensor_by_name("softmax_output/output_softmax:0")

		res = sess.run(op_to_restore,feed_dict)
		label = np.argmax(y_test,1)
		predict = np.argmax(res,1)
		count = 0
		for i in range(len(predict)):
			if(label[i] == predict[i]):
				count += 1
		print("The final accauracy is:",count/len(predict))
