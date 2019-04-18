import numpy as np
import cv2
from keras import models
from keras import layers
from keras import optimizers
from PIL import Image
from keras import Model
import pickle
import os
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from classification_models.resnet import ResNet18, preprocess_input

# load model
model = ResNet18(input_shape=(224,224,3), weights='imagenet', classes=1000)
layer_name = 'relu1'
intermidiate_layer_model = Model(inputs = model.input , outputs = model.get_layer(layer_name).output)

#print(model.summary())

folders = '/home/aman9875/Documents/assignment2/fine_data'
train_features = []
train_labels = []
validation_features = []
validation_labels = []

t = 0
for folder in os.listdir(folders):
	print(folder)
	for sub_folder in os.listdir(os.path.join(folders,folder)):
		print(sub_folder)
		l = len(os.listdir(os.path.join(folders,folder,sub_folder)))
		print(l)
		i = 0
		for filename in os.listdir(os.path.join(folders,folder,sub_folder)):
			img = Image.open(os.path.join(folders,folder,sub_folder,filename))
			img = img.resize((224,224))
			img = np.asarray(img)
			img = np.reshape(img,(1,224,224,3))	
			intermidiate_output = intermidiate_layer_model.predict(img)
			print(intermidiate_output.shape)
			if i < (3*l)//4:
				#print('train')
				train_features.append(intermidiate_output)
				train_labels.append(t)
			else:
				#print('validation')
				validation_features.append(intermidiate_output)
				validation_labels.append(t)
			i = i + 1
	t = t + 1

with open('train_features.obj' , 'wb') as f:
	pickle.dump(train_features,f)

with open('train_labels.obj' , 'wb') as f:
	pickle.dump(train_labels,f)

with open('validation_features.obj','wb') as f:
	pickle.dump(validation_features,f)

with open('validation_labels.obj','wb') as f:
	pickle.dump(validation_labels,f)
