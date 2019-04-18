import numpy as np
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle

import os

def main():
	names = os.listdir("../input/mllcnn/train/train")
	X = []
	Y = []
	for name in names:
	    images = os.listdir("../input/mllcnn/train/train/"+name)
	    for img in images:
	        im = cv2.imread("../input/mllcnn/train/train/"+name+"/"+img)
	        X.append(im)
	        Y.append(names.index(name))
	X = np.array(X)
	Y = np.array(Y)

	images = os.listdir("../input/testdata-vr/sample_test/sample_test")
	images.remove("instances.txt")
	X_test = []
	for img in images:
		im = cv2.imread("../input/testdata-vr/sample_test/sample_test/"+img)
		X_test.append(im)
	X_test = np.array(X_test)
	X.astype('float32')
	X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.25, random_state=42)
	X_train =  X_train.astype(np.float32, copy = False)
	X_val =  X_val.astype(np.float32, copy = False)
	X_test = X_test.astype(np.float32, copy = False)


	for i in range(X_train.shape[0]):
	    X_train[i] = (X_train[i]*1.0)/(X_train[i].max())
	for i in range(X_val.shape[0]):
	    X_val[i] = (X_val[i]*1.0)/(X_val[i].max())
	for i in range(X_test.shape[0]):
	    X_test[i] = (X_test[i]*1.0)/(X_test[i].max())        

	tf.reset_default_graph()
	X = tf.placeholder(tf.float32,[None,400,600,3])
	y = tf.placeholder(tf.int64,[None])

	iter_num = 1000
	batch_size = 16
	learning_rate = 0.003
	input_layer = tf.reshape(X, [-1, 400, 600,3])

	Wconv1 = tf.get_variable("Wconv1",shape=[5,5,3,32],initializer=tf.contrib.layers.xavier_initializer())
	bconv1 = tf.get_variable("bconv1",shape=[32],initializer = tf.contrib.layers.xavier_initializer())
	conv1 = tf.nn.conv2d(input_layer,Wconv1,strides=[1,2,2,1],padding="SAME") + bconv1
	a11 = tf.layers.batch_normalization(inputs = conv1)
	a12 = tf.nn.relu(a11)

	pool1 = tf.layers.max_pooling2d(inputs=a12, pool_size=[2, 2], strides=2)


	Wconv3 = tf.get_variable("Wconv3",shape=[5,5,32,64],initializer=tf.contrib.layers.xavier_initializer())
	bconv3 = tf.get_variable("bconv3",shape=[64],initializer = tf.contrib.layers.xavier_initializer())
	conv3 = tf.nn.conv2d(pool1,Wconv3,strides=[1,2,2,1],padding="SAME") + bconv3
	a31 = tf.layers.batch_normalization(inputs = conv3)
	a32 = tf.nn.relu(a31)

	pool2 = tf.layers.max_pooling2d(inputs=a32, pool_size=[2, 2], strides=2)

	pool2_flat = tf.reshape(pool2, [-1, pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])

	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

	logits = tf.layers.dense(inputs=dense,units=16)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=logits)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss=loss)

	predictions = tf.argmax(input = logits,axis = 1)

	correct = tf.equal(predictions, y)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))	
    
	saver = tf.train.Saver()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())    

	loss_trace = []

	for epoch in range(iter_num):
		batch_index = np.random.choice(len(X_train), size=batch_size)
		X_train_batch = X_train[batch_index]
		y_train_batch = y_train[batch_index]
		sess.run(train_op,feed_dict = {X:X_train_batch,y:y_train_batch})
		got_loss = sess.run(loss,feed_dict = {X:X_train_batch,y:y_train_batch})
		loss_trace.append(got_loss)

		if (epoch+1)%1 == 0:
			print("epoch = %d loss = %r"%(epoch,got_loss))
	
	plt.plot(loss_trace)

	prediction = []
	for i in range(X_val.shape[0]):
	    temp = X_val[i].reshape((1,400,600,3))
	    predict = sess.run(predictions,feed_dict={X:temp})
	    predict = predict.reshape((1))
	    prediction.append(predict)
	prediction = np.array(prediction)  
	
	got_logits = sess.run(logits,feed_dict={X:X_test})
	predict = sess.run(predictions,feed_dict = {X:X_test})
	print(predict)
	print(got_logits.shape)

	ranks = []
	for i in range(15):
		scores = []
		for j in range(16):
			scores.append((got_logits[i][j],j))
		scores.sort(key = lambda x : x[0])
		rank = []
		for j in range(16):
			rank.append(names[int(scores[15 - j][1])])
		ranks.append(rank)    	
	
	file = open('test_res','ab')
	pickle.dump(ranks,file)
	file = open('images','ab')
	pickle.dump(images,file)	

	for i in range(15):
	    print(images[i])
	    x,y = images[i].split('.')
	    file = open(x+".txt",'w')
	    for j in range(16):
	        print(ranks[i][j])
	        folder = os.listdir("../input/mllcnn/train/train/" + ranks[i][j])
	        for img in folder:
	        	file.write(ranks[i][j]+"_"+img+"\n")
	    print("-----------------------------")	
	
	count = 0
	for i in range(y_val.shape[0]):
	    if prediction[i][0] == y_val[i] :
	        count+=1


	validation_size = y_val.shape[0]
	print("Accuracy on validation set = %f"%(count/validation_size))

	np.reshape(prediction,(len(y_val),1))
	predicted_classes = []
	for i in range(len(y_val)):
	    predicted_classes.append(prediction[i][0])

	my_submission1 = pd.DataFrame({'predictions': predicted_classes})
	my_submission2 = pd.DataFrame({'actual': y_val})
	my_submission1.to_csv('submission1.csv', index=True)
	my_submission2.to_csv('submission2.csv', index=True)



main()    