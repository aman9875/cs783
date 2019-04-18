import numpy as np
import keras
from keras.models import Model
import pickle
from tflearn.data_utils import shuffle
import os
from keras.preprocessing.image import ImageDataGenerator

learning_rate = 0.3
decay_learning_rate = 1e-8
num_classes = 11

def outer_product(x):
    return keras.backend.batch_dot(
                x[0]
                , x[1]
                , axes=[1,1]
            ) / x[0].get_shape().as_list()[1] 

def signed_sqrt(x):
    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)

def L2_norm(x, axis=-1):
    return keras.backend.l2_normalize(x, axis=axis)

file = open("../input/resnet18/resnet18.pkl",'rb')
model = pickle.load(file)
print(model.summary())

x_detector = model.layers[-1].output
shape_detector = model.layers[-1].output_shape
shape_extractor = shape_detector

print(shape_extractor)

x_extractor = x_detector
print(x_extractor)

x_detector = keras.layers.Reshape([shape_detector[1] * shape_detector[2] , shape_detector[-1]])(x_detector)

print("After reshape",x_detector.shape)

x_extractor = keras.layers.Reshape([shape_extractor[1] * shape_extractor[2] , shape_extractor[-1]])(x_extractor)

print("After reshape",x_extractor.shape)

x = keras.layers.Lambda(outer_product)([x_detector, x_extractor])

print("Outer product",x.shape)

x = keras.layers.Reshape([shape_detector[-1]*shape_extractor[-1]])(x)

print("Reshape",x.shape)

x = keras.layers.Lambda(signed_sqrt)(x)

print("Signed sqrt",x.shape)

x = keras.layers.Lambda(L2_norm)(x)

print("l2 norm",x.shape)

x = keras.layers.Dense(units=num_classes,kernel_regularizer=keras.regularizers.l2(decay_learning_rate)
  ,kernel_initializer="glorot_normal")(x)

print("Dense",x.shape)

tensor_prediction = keras.layers.Activation("softmax")(x)

print("predictions",tensor_prediction.shape)  

model_bcnn = Model(inputs = model.input, outputs = [tensor_prediction])

for layer in model.layers:
    layer.trainable = False

for i in range(-4,0):
    model.layers[i].trainable = True

opt_sgd = keras.optimizers.SGD(lr=learning_rate,decay=decay_learning_rate,momentum=0.9,nesterov=False)

model_bcnn.compile(loss="sparse_categorical_crossentropy",optimizer=opt_sgd,
    metrics=[keras.metrics.sparse_categorical_accuracy])

#Load data

file = open("../input/finedataset/flowers_.pkl",'rb')
data = pickle.load(file)


print('Input data read complete')

X_train, Y_train, X_val, Y_val = data['X_train'], data['y_train'], data['X_val'], data['y_val']
X_train = np.asarray(X_train)
X_val = np.asarray(X_val)
Y_train = np.asarray(Y_train)
Y_val = np.asarray(Y_val)
np.reshape(Y_train,(Y_train.shape[0]))
np.reshape(Y_val,(Y_val.shape[0]))
#X_train = np.concatenate((X_train,X_val),axis = 0)
#Y_train = np.concatenate((Y_train,Y_val),axis = 0)
print(Y_train.dtype)

print("Data shapes -- (train, val, test)", X_train.shape, X_val.shape)
X_train, Y_train = shuffle(X_train, Y_train)
X_val, Y_val = shuffle(X_val, Y_val)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(X_train)

num_training_samples = X_train.shape[0]
num_validation_samples = X_val.shape[0]
batch_size = 20
num_epochs = 15

model_bcnn.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size), 
steps_per_epoch = (len(X_train) // batch_size) *3,epochs=num_epochs,validation_data = (X_val,Y_val),verbose = 1)

model_bcnn.save_weights("flowers_weight.h5")
    
