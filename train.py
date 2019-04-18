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

t = 0
descriptor_list = []
folders = '/home/aman9875/Documents/train'  #path of the training images
for folder in os.listdir(folders):
	for filename in os.listdir(os.path.join(folders,folder)):
		img = cv2.imread(os.path.join(folders,folder,filename))
		if img is not None:
			img = img[50:275,260:350]
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			kp, des = features(gray,extractor)
			descriptor_list.append(des)

n_images = len(descriptor_list)
descriptor_list = np.asarray(descriptor_list)
#print(descriptor_list.shape)
des_size = [0 for i in range(n_images)]
features = []
for i in range(0,n_images):
	l = descriptor_list[i].shape[0]
	des_size[i] = l
	for j in range(0,l):
		features.append(descriptor_list[i][j])

features = np.asarray(features)
features = np.reshape(features,(-1,128))
#print(features.shape) #(m * 128) all features of all images

n_clusters = 1000
kmeans =  KMeans(n_clusters = n_clusters,init='k-means++',n_init = 5).fit(features)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
t = 0
count = np.zeros(n_clusters)
for i in range(n_images):
	for j in range(des_size[i]):
		count[labels[t]] += 1
		t += 1

freq = []
for i in range(n_clusters):
	freq.append((count[i],i))

freq.sort(reverse = True)
stop_words = []
num_stop_words = 20

for i in range(num_stop_words):
	l = freq[i][1]
	stop_words.append(l)

histogram = [[0 for i in range(0,n_clusters)]for j in range(n_images)] #seperate histogram for each image
word_count = [0 for i in range(0,n_clusters)] #count of each word in the entire database
N = n_images #total documents in the database

t = 0
for i in range(n_images):
	for j in range(0,des_size[i]):
		if labels[t] in stop_words:
			t += 1
			continue
		histogram[i][labels[t]] += 1
		word_count[labels[t]] += 1
		t += 1

tf_idf = [[0 for i in range(0,n_clusters)]for j in range(0,n_images)]

for i in range(0,n_images):
	for j in range(0,n_clusters):
		if j in stop_words:
			continue
		tf_idf[i][j] = (histogram[i][j]/des_size[i])*(np.log(N/word_count[j]))


f = open(b"tf_idf.obj","wb")
pickle.dump(tf_idf,f)

f = open(b"cluster_centers.obj","wb")
pickle.dump(cluster_centers,f)

f = open(b"histogram.obj","wb")
pickle.dump(histogram,f)

f = open(b"stop_words.obj","wb")
pickle.dump(stop_words,f)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
	cv2.destroyAllWindows()