from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Conv2D, Conv1D, MaxPooling2D, TimeDistributed, Conv3D, Reshape, MaxPooling3D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import plot_model
from keras import regularizers

import tensorflow as tf


# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
# imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = ["Capture", "Pointing", "ZoomIn", "ZoomOut"]
data = []
labels = []
path = "FinalDataset"
 
# loop over the image paths
for imagePath in imagePaths:

	folders = [i for i in os.listdir(path + "/" + imagePath)]
	for folder in folders:
		photos = [i for i in os.listdir(path + "/" + imagePath + "/" + folder)]
		arr = []
		# arr = np.zeros((100,100,3))
		for photo in photos:
			# load the image
			image = cv2.imread(path + "/" + imagePath + "/" + folder + "/" + photo)
		
			# update the data and labels lists, respectively
			arr.append(image[:,:,0])

		data.append(arr)
		labels.append(imagePath)
        
# convert the data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)
print(data.shape)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(labels.shape)


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)
print(trainX.shape, trainY.shape, testX.shape, testY.shape)


trainX = trainX.reshape([len(trainX),1,15, 100, 100])
testX = testX.reshape([len(testX),1,15, 100, 100])

print(trainX.shape, trainY.shape, testX.shape, testY.shape)


model = Sequential()
model.add(Conv3D(32,kernel_size=(15,4, 4), padding="same", activation="relu", data_format="channels_first", input_shape = (1,15,100,100)))
model.add(Conv3D(32, kernel_size=(15,3,3), padding="same", activation="relu",kernel_regularizer = regularizers.l2(0.02)))
model.add(MaxPooling3D(pool_size=(2, 2,2)))

model.add(Conv3D(32,kernel_size=(15,4, 4), padding="same", activation="relu" ))
model.add(Conv3D(1, kernel_size=(15,3,3), padding="same", activation="relu",kernel_regularizer = regularizers.l2(0.02)))
model.add(MaxPooling3D(pool_size=(2, 2,2)))

model.add(Reshape((8,3,25)))
model.add(Conv2D(8, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size = (3,3)))


model.add(Flatten())
# model.add(Reshape((2,8)))
# model.add(MaxPooling2D(pool_size = (3,3)))
# model.add(LSTM(10))
# model.add(LSTM(20))

model.add(Dense(4, activation='softmax'))

model.summary()

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5, batch_size=10, shuffle=True)


# # Final evaluation of the model
# scores = model.evaluate(testX, testY, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))


# # save model and architecture to single file
# model.save("model3D2.h5")
# print("Saved model to disk")