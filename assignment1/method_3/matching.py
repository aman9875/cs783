import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf
import pickle
import tensorflow_hub as hub


n_test = 15
n_images = 3456
path1 = '/home/aman9875/Documents/training_images/'
path2 = '/home/aman9875/Documents/test1/'
with open("results.obj",'rb') as f:
    result_train = pickle.load(f)

with open("results1.obj",'rb') as f:
	result_test = pickle.load(f)


def match_images(image_1_path,image_2_path):
  distance_threshold = 0.8

  # Read features.
  locations_1, descriptors_1 = result_train[image_1_path]
  num_features_1 = locations_1.shape[0]
  locations_2, descriptors_2 = result_test[image_2_path]
  num_features_2 = locations_2.shape[0]
  # Find nearest-neighbor matches using a KD tree.
  d1_tree = cKDTree(descriptors_1)
  _, indices = d1_tree.query(
      descriptors_2, distance_upper_bound=distance_threshold)

  # Select feature locations for putative matches.
  locations_2_to_use = np.array([
      locations_2[i,]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])
  locations_1_to_use = np.array([
      locations_1[indices[i],]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])

  f = 0
  if(len(locations_1_to_use) != 0 and len(locations_2_to_use) != 0):
	  # Perform geometric verification using RANSAC.
	  f = 1
	  _, inliers = ransac(
	      (locations_1_to_use, locations_2_to_use),
	      AffineTransform,
	      min_samples=3,
	      residual_threshold=20,
	      max_trials=10)

  if f==1 and inliers is not None:
  	print('Found %d inliers' % sum(inliers))
  	return sum(inliers)
  	_, ax = plt.subplots(figsize=(9, 18))
  	x , image_1_path = image_1_path.rsplit('/',1)
  	x , image_2_path = image_2_path.rsplit('/',1)
  	image_1_path = path1 + image_1_path
  	image_2_path = path2 + image_2_path
  	img_1 = mpimg.imread(image_1_path)
  	img_2 = mpimg.imread(image_2_path)
  	inlier_idxs = np.nonzero(inliers)[0]
  	plot_matches(
      ax,
      img_1,
      img_2,
      locations_1_to_use,
      locations_2_to_use,
      np.column_stack((inlier_idxs, inlier_idxs)),
      matches_color='b')
  	ax.axis('off')
  	ax.set_title('DELF correspondences')
  	plt.show()
  else:
  	print('Found 0 inliers')
  	return 0

c = 0
for key in result_test:
	print(c)
	count = []
	for key1 in result_train:
		count.append((match_images(key1,key),key1))
	count.sort(key = lambda x : x[0],reverse = True)
	_,t = key.rsplit('/',1)
	f = open(t,"w")
	for i in range(n_images):
		st = str(count[i][1])
		_,filename = st.rsplit('/',1)
		f.write("%s\n" %filename)