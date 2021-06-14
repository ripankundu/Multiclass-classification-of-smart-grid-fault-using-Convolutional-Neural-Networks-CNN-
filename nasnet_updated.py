# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 00:06:24 2021

@author: USER
"""

import cv2
from glob import glob
import os
from builtins import range, input
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
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

n_h, n_w, n_c =(104, 100, 1)

#weights_path = '../input/d/aeryss/keras-pretrained-    models/ResNet50_NoTop_ImageNet.h5'
#ResNet50 = keras.applications.ResNet50(weights=weights_path ,include_top=False, input_shape=(n_h, n_w, n_c))
nasnet = NASNetLarge(weights=None, include_top=False,
    input_tensor=Input(shape=(n_h, n_w, n_c)))

outputs = nasnet.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.4)(outputs)
outputs = Dense(2, activation="softmax")(outputs)

model = Model(inputs=nasnet.input, outputs=outputs)

for layer in nasnet.layers:
    layer.trainable = False

model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
)


#Visualize the model
model.summary()


train_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)


#Redefining size of each image of the dataset
IMAGE_SIZE = [104,100]

#Defining hyperparameters for training
epochs = 3
batch_size = 64
history = model.fit(train_aug.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_test, y_test),
                    validation_steps=len(X_test) / batch_size,
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=epochs)
#Saving trained model and weights
model.save('nasnet_dem.h5')
model.save_weights('nasnet_weights_dem.hdf5')


#Now load the saved model
model = load_model('nasnet_dem.h5')



#Checking the model accuracy

plt.figure(figsize=(10,10))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('nasnet_dem_accuracy.png')
plt.show()




#Checking the model loss
plt.figure(figsize=(10,10))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('nasnet_dem_loss.png')
plt.show()




#Prediction the test value
y_pred = model.predict(X_test, batch_size=batch_size)


y_pred_bin = np.argmax(y_pred, axis=1)
y_test_bin = np.argmax(y_test, axis=1)





#Plotting the ROC curve
fpr, tpr, thresholds = roc_curve(y_test_bin, y_pred_bin)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for our model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)




#Calculate the confusion matrix

def plot_confusion_matrix(normalize):
  classes = ['SLG', '3P']
  tick_marks = [0.5,1.5]
  cn = confusion_matrix(y_test_bin, y_pred_bin,normalize=normalize)
  sns.heatmap(cn,cmap='plasma',annot=True)
  plt.xticks(tick_marks, classes)
  plt.yticks(tick_marks, classes)
  plt.title('Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

print('Confusion Matrix without Normalization')
plot_confusion_matrix(normalize=None)

print('Confusion Matrix with Normalized Values')
plot_confusion_matrix(normalize='true')



#Print thr classification report
from sklearn.metrics import classification_report
print(classification_report(y_test_bin, y_pred_bin))


