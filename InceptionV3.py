# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 05:35:01 2021

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
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten
import cv2 
import numpy as np 
from keras.datasets import cifar10 
from keras import backend as K 
from keras.utils import np_utils

import math 
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler
import keras
import keras.backend as K
import tensorflow as tf

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



# importing keras librarys
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras import optimizers
import pandas as pd
import matplotlib.pyplot as plt 

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, History

from keras.optimizers import SGD
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, AveragePooling2D, Dropout



n_h, n_w, n_c =(104, 100, 1)

#weights_path = '../input/d/aeryss/keras-pretrained-    models/ResNet50_NoTop_ImageNet.h5'
#ResNet50 = keras.applications.ResNet50(weights=weights_path ,include_top=False, input_shape=(n_h, n_w, n_c))
Inception = InceptionV3(weights=None, include_top=False,
    input_tensor=Input(shape=(n_h, n_w, n_c)))

outputs = Inception.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.4)(outputs)
outputs = Dense(2, activation="softmax")(outputs)

model = Model(inputs=Inception.input, outputs=outputs)

for layer in Inception.layers:
    layer.trainable = False

model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
)


#Visualize the model
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model.png')


#Defining the hyperparameters
batch_size= 100
epochs=5
learn_rate=.001
sgd=SGD(lr=learn_rate,momentum=.9,nesterov=False)
#Compiling InceptionV3

model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])


#Training the InceptionV3 model


history=model.fit(X_train, y_train, epochs=epochs,batch_size=batch_size,validation_data = (X_test, y_test))



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
