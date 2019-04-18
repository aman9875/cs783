import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import pickle



dict1 = {}

folders = '/home/aman9875/Documents/assignment2/data'
for folder in os.listdir(folders):
	print(folder)
	X = []
	y = []
	t = 0
	l1 = len(os.listdir(os.path.join(folders,folder)))	
	label = [0 for i in range(11)]
	for sub_folder in os.listdir(os.path.join(folders,folder)):
		print(sub_folder)
		l = len(os.listdir(os.path.join(folders,folder,sub_folder)))
		print(l)
		for filename in os.listdir(os.path.join(folders,folder,sub_folder)):
			img = Image.open(os.path.join(folders,folder,sub_folder,filename))
			img = img.resize((224,224))
			img = np.asarray(img)
			img = np.reshape(img,(224,224,3))	
			X.append(img)
			y.append(t)
			label[t] = int(sub_folder)
		t += 1
	X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.25,random_state = 42)
	dict_ = {}
	dict_['X_train'] = X_train
	dict_['y_train'] = y_train
	dict_['X_val'] = X_val
	dict_['y_val'] = y_val
	dict1[folder] = label
	with  open("%s.pkl" %folder , "wb") as f:
		pickle.dump(dict_,f)


with open("dict_fine.pkl" , "wb") as f:
	pickle.dump(dict1,f) 