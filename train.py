from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Conv2D, MaxPooling2D, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import plot_model

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



# initialize the set of labels from the spots activity dataset we are
# going to train our network on
LABELS = set(["Pointing", "Capture", "ZoomIn", "ZoomOut"])
 
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
		arr = np.zeros((100,100))
		# arr = np.zeros((100,100,3))
		for photo in photos:
			# load the image
			image = cv2.imread(path + "/" + imagePath + "/" + folder + "/" + photo)
		
			# update the data and labels lists, respectively
			# arr.append(image[:,:,0])
			# arr.append(image)
			arr = np.concatenate((arr, image[:,:,0]), axis=1)
			# arr = np.concatenate((arr, image), axis=1)
		# print(arr[:,100:].shape)
		data.append(arr[:,100:])
		labels.append(imagePath)


# convert the data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(data.shape)
print(labels.shape)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)


trainX = trainX.reshape([len(trainX),1,100, 1500])
print(trainX.shape, trainY.shape)
# # fix random seed for reproducibility
# numpy.random.seed(7)



model = Sequential()
model.add(Conv2D(1,kernel_size=(3, 3), padding="same", activation="relu", input_shape = (1,100,1500)))
model.add(Flatten())
model.add(Dense(2))
# for layer in model.layers:
# 	print(layer.output_shape)
# model.add(LSTM(250, activation='tanh', recurrent_activation='sigmoid', dropout=0.1))


model.build((1,100,1500))
model.summary()
# model.add(Dense())



# model = Sequential()
# # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(15,100,100)))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Flatten())
# model.add(TimeDistributed(Conv2D(32, (3, 3)),
#                           input_shape=(15, 100, 100,3)))
# # for layer in model.layers:
# #     print(layer.output_shape)
# # model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(15,100,100))))
# model.add(LSTM(250, activation='tanh'))
# model.add(Dense(4, activation='relu'))

# # model.build((15,100,100))
# model.summary()



# plot_model(model, to_file='model.png')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(trainX, trainY, validation_data=(testX, testY), epochs=1, batch_size=15)

# # Final evaluation of the model
# scores = model.evaluate(testX, testY, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))




