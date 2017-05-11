# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017

@author: sylvain
"""

from param_pix import *
import os
import keras
import theano

import numpy as np
from keras.models import Model

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D 
from keras.optimizers import Adam,Adagrad
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import sklearn.metrics as metrics
from keras.models import load_model
#K.set_image_dim_ordering('th') 
#K.set_image_data_format('channels_first')
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
import cPickle as pickle
print keras.__version__
print theano.__version__

pickel_dirsource='pickle_ILD4'
pickel_train=pickel_dirsource

cwd=os.getcwd()

(cwdtop,tail)=os.path.split(cwd)
pickle_dir=os.path.join(cwdtop,pickel_dirsource)
pickle_dir_train=os.path.join(cwdtop,pickel_train)

def load_train_data(numpidir):
    X_train = pickle.load( open( os.path.join(numpidir,"X_train.pkl"), "rb" ))
    y_train = pickle.load( open( os.path.join(numpidir,"y_train.pkl"), "rb" ))
    class_weights=pickle.load(open( os.path.join(numpidir,"class_weights.pkl"), "rb" ))
    num_class= y_train.shape[3]
    img_rows=y_train.shape[1]
    img_cols=y_train.shape[2]
    num_images=y_train.shape[0]
    clas_weigh_l=[]
    for i in range (0,num_class):
#            print i,class_weights[i]
            clas_weigh_l.append(class_weights[i])
    print 'weights for classes:'
    for i in range (0,num_class):
                    print i, clas_weigh_l[i]
   
    return X_train, y_train,num_class,img_rows,img_cols,num_images,clas_weigh_l


coefcon={}
coefcon[1]=32 #32 for 320
for i in range (2,6):
    coefcon[i]=coefcon[i-1]*2
print coefcon


def get_unet(num_class,img_rows,img_cols):
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(coefcon[1], (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(coefcon[1], (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(coefcon[2], (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(coefcon[2], (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(coefcon[3], (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(coefcon[3], (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(coefcon[4], (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(coefcon[4], (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(coefcon[5], (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(coefcon[5], (3, 3), activation='relu', padding='same')(conv5)
#    print conv5.shape

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4],axis=3)
#    up6 = concatenate([conv5, conv4],axis=3)
 
    conv6 = Conv2D(coefcon[4], (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(coefcon[4], (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)

    conv7 = Conv2D(coefcon[3], (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(coefcon[3], (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(coefcon[2], (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(coefcon[2], (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(coefcon[1], (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(coefcon[1], (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(int(num_class), (1, 1), activation='softmax')(conv9)
#    conv11 =Reshape((3, 3))(conv10)
#    conv12 =Activation('softmax')(conv11)

    model = Model(inputs=[inputs], outputs=[conv10])

#    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
#    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#    model.compile(optimizer=Adagrad, loss='binary_crossentropy',
#               metrics=['accuracy'])
    return model



def evaluate(actual,pred):
    fscore = metrics.f1_score(actual, pred, average='macro')
    acc = metrics.accuracy_score(actual, pred)
    cm = metrics.confusion_matrix(actual,pred)
    return fscore, acc, cm

def load_model_set(pickle_dir_train):
    listmodel=[name for name in os.listdir(pickle_dir_train) if name.find('weights')==0]

    ordlist=[]
    for name in listmodel:
        nfc=os.path.join(pickle_dir_train,name)
        nbs = os.path.getmtime(nfc)
#        print name,nbs,type(nbs)
        
#        posp=name.find('-')+1
#        post=name.find('.h')
#        numepoch=float(name[posp:post])
        tt=(name,nbs)
#        ordlist.append (tt)
        ordlist.append(tt)

    ordlistc=sorted(ordlist,key=lambda col:col[1],reverse=True)

    namelast=ordlistc[0][0]

    namelastc=os.path.join(pickle_dir_train,namelast)
    print 'last weights :',namelast
    model=load_model(namelastc)
    return model



def train():
    
    print('-'*30)
    print('Loading train data...')
    print('-'*30)
#    listplickle=os.listdir(pickle_dir)
    listplickle=os.walk(pickle_dir).next()[1]
#    print listplickle
    numpidirb=pickle_dir
    fp=True
    for numpi in listplickle:
    
#        print numpi, numpidirb
        numpidir =os.path.join(pickle_dir,numpi)
        X_train, y_train,num_class,img_rows,img_cols,num_images,class_weights = load_train_data(numpidir)
        
#        print num_class
        
        if fp:
            print 'shape X_train :',X_train.shape
            print 'shape y_train :',y_train.shape
            print('-'*30)
            print 'number of images:', num_images
            print 'number of classes:', num_class
            print 'image number of rows :',img_rows
            print 'image number of columns :',img_cols
            print('-'*30)
            fp=False
        
        print('Creating and compiling model...')
        print('-'*30)
        
        listmodel=[name for name in os.listdir(numpidirb) if name.find('weights')==0]
        if len(listmodel)==0:
            listmodel=[name for name in os.listdir(numpidir) if name.find('weights')==0]
            if len(listmodel)==0:
                print 'first pass'
                model = get_unet(num_class,img_rows,img_cols)
            else:
                model=load_model_set(numpidir)
        else:
            model=load_model_set(numpidirb)
        numpidirb=os.path.join(pickle_dir,numpi)
        
        model_checkpoint = ModelCheckpoint(os.path.join(numpidir,'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', save_best_only=True)
    
        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        model.fit(X_train, y_train, batch_size=1, epochs=20, verbose=1, shuffle=True,
                  validation_split=0.2,  callbacks=[model_checkpoint],class_weight=class_weights)
        
train()