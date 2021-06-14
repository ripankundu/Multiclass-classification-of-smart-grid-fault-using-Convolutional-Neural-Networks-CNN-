# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 04:08:52 2021

@author: USER
"""


import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import normalize, to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.image as mpimg
#from keras.models import Sequential,optimizers
from keras import optimizers, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import normalize, to_categorical
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint, TensorBoard


#importing other required libraries
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import Sequential
from keras.applications import VGG19, VGG16, ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten, Dense, BatchNormalization, Activation,Dropout
from keras.utils import to_categorical
import tensorflow as tf
import random


X_train_before = np.load(r"C:\Users\USER\.spyder-py3\Tahmid_share/X_train_before_reshape.npy")
X_train_after = np.load(r"C:\Users\USER\.spyder-py3\Tahmid_share/X_train_after_reshape.npy")

y_train= np.load(r"C:\Users\USER\.spyder-py3\Tahmid_share/Y_train_now_now.npy")
X_test_before= np.load(r"C:\Users\USER\.spyder-py3\Tahmid_share/X_test_before_reshape.npy")
X_test_after= np.load(r"C:\Users\USER\.spyder-py3\Tahmid_share/X_test_after_reshape.npy")
y_test = np.load(r"C:\Users\USER\.spyder-py3\Tahmid_share/Y_test_now_now.npy", allow_pickle=True)


#Encoding the categorial value to integer number for the testing dataset
label_encoder = LabelEncoder()

y_test= label_encoder.fit_transform(y_test)#convert string to numreic value


### Normalize inputs 
X_train = normalize(X_train_after, axis=1)
X_test = normalize(X_test_after, axis=1)

#y_train = to_categorical(y_train_real)
y_test = to_categorical(y_test)


#Initializing ResNet50
base_model_resnet = ResNet50(include_top = False, weights = None, input_shape = (104,100,1), classes = y_train.shape[1])


#Adding layers to the ResNet50
model_resnet=Sequential()
#Add the Dense layers along with activation and batch normalization
model_resnet.add(base_model_resnet)
model_resnet.add(Flatten())
#Add the Dense layers along with activation and batch normalization
model_resnet.add(Dense(1024,activation=('relu'),input_dim=512))
model_resnet.add(Dense(512,activation=('relu'))) 
model_resnet.add(Dropout(.4))
model_resnet.add(Dense(256,activation=('relu'))) 
model_resnet.add(Dropout(.3))
model_resnet.add(Dense(128,activation=('relu')))
model_resnet.add(Dropout(.2))
model_resnet.add(Dense(2,activation=('softmax')))

#Summary of ResNet50 Model
model_resnet.summary()


from keras.utils import plot_model
plot_model(model_resnet, to_file='model.png')




#Defining the hyperparameters
batch_size= 100
epochs=5
learn_rate=.001
sgd=SGD(lr=learn_rate,momentum=.9,nesterov=False)
#Compiling ResNet50

model_resnet.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])


#Training the ResNet50 model


history=model_resnet.fit(X_train, y_train, epochs=epochs,batch_size=batch_size,validation_data = (X_test, y_test))
# history=model_resnet.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,steps_per_epoch = X_train.shape[0]//batch_size,validation_data = (X_test, y_test), validation_steps = 250,  verbose=1)
# model_resnet.fit_generator(train_generator.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, steps_per_epoch = x_train.shape[0]//batch_size, validation_data = val_generator.flow(x_val, y_val, batch_size = batch_size), validation_steps = 250, callbacks = [lrr], verbose=1)


#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training aaccuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



