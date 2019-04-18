import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import layers
from keras import optimizers
from PIL import Image
from keras import Model
import pickle
import os
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from classification_models.resnet import ResNet18, preprocess_input

with open("train_features.obj",'rb') as f:
	features = pickle.load(f)

with open("train_labels.obj",'rb') as f:
	labels = pickle.load(f)

with open("validation_features.obj" , 'rb') as f:
	validation_features = pickle.load(f)

with open("validation_labels.obj" , 'rb') as f:
	validation_labels = pickle.load(f)

print(features[0].shape)
print(len(validation_features))

features = np.asarray(features)
validation_features = np.asarray(validation_features)
labels = np.asarray(labels)
validation_labels = np.asarray(validation_labels)
print(features.shape)

batch_size  = 16
model = Sequential()
model.add(Flatten(input_shape = features[0].shape))
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(5,activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])
model.fit(features,labels,epochs = 50 , batch_size = batch_size , 
	validation_data = (validation_features , validation_labels))

model.save_weights('weights.h5')
model.save('coarse_model.h5')