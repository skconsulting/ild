# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:05:33 2017

@author: sylvain
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K

'''
Creates a keras model with 3D CNNs and returns the model.
'''
def classifier(input_shape, kernel_size, pool_size):
	model = Sequential()

	model.add(Convolution3D(16, kernel_size[0], kernel_size[1], kernel_size[2],
	                        border_mode='valid',
	                        input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling3D(pool_size=pool_size))
	model.add(Convolution2D(32, kernel_size[0], kernel_size[1], kernel_size[2]))
	model.add(Activation('relu'))
	model.add(MaxPooling3D(pool_size=pool_size))
	model.add(Convolution3D(64, kernel_size[0], kernel_size[1], kernel_size[2]))
	model.add(Activation('relu'))
	model.add(MaxPooling3D(pool_size=pool_size))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2))
	model.add(Activation('softmax'))

	return model

def train_classifier(input_shape):
	model = classifier(input_shape, (3, 3, 3), (2, 2, 2))
	model.compile(loss='categorical_crossentropy',
		  optimizer='adadelta',
		  metrics=['accuracy'])
	'''
	Read the preprocessed datafiles chunk by chunnk and train the model on that batch (trainX, trainY) using:'''
	model.train_on_batch(trainX, trainY, sample_weight=None)
	'''The model can be trained on many epochs using for loops'''
	
	'''
	AFter training the dataset we test our model of the test dataset, read the test file chunk by chunk and 
	test on each chunk (trainX, trainY) using:'''
	print (model.test_on_batch(trainX, trainY, sample_weight=None))
	'''The average of all of this can be taken to obtain the final test score.'''
	
	'''After testing save the model using'''
	model.save('my_model.h5')
