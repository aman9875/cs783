import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle


extractor = cv2.xfeatures2d.SIFT_create(100)

def features(image, extractor):
	keypoints, descriptors = extractor.detectAndCompute(image, None)
	return keypoints, descriptors

n_clusters = 1000

with open("histogram.obj",'rb') as f:
	train_histogram = pickle.load(f)

with open("dict.obj",'rb') as f:
	Dict = pickle.load(f)

with open("cluster_centers.obj",'rb') as f:
	cluster_centers = pickle.load(f)

with open("tf_idf.obj", 'rb') as f:
	tf_idf = pickle.load(f)

with open("stop_words.obj", 'rb') as f:
	stop_words = pickle.load(f)

train_histogram = np.asarray(train_histogram)
tf_idf = np.asarray(tf_idf)
n_images = tf_idf.shape[0]
print(tf_idf.shape)

print(cluster_centers.shape)

folder = '/home/aman9875/Documents/test1'  #path for the test images
for filename in os.listdir(folder):
	img = cv2.imread(os.path.join(folder,filename))
	if img is not None:
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		kp, des = features(gray,extractor)
		img = cv2.drawKeypoints(gray,kp,img)
		cv2.imshow(filename,img)
		histogram = np.zeros(n_clusters)
		for j in range(0,len(des)):
			label = np.argmin(np.sum((cluster_centers-des[j])**2,axis=1), axis = 0)
			if label in stop_words:
				continue
			histogram[label] += 1

		c_histogram = histogram
		rank = [] 
		val = np.sum(c_histogram**2)
		for i in range(n_images):
			val1 = np.sum((train_histogram[i])**2)
			val2 = np.dot(train_histogram[i],c_histogram)
			sim = (val2)/(val1*val)
			rank.append((sim,i))
		rank.sort(key = lambda x : x[0],reverse = True)

		for i in range(0,10):
			k = rank[i][1]
			val = rank[i][0]
			print(Dict[k])
			print(val)

	print("\n")

	k = cv2.waitKey(0)
	if k == 27:         # wait for ESC key to exit
		cv2.destroyAllWindows()
