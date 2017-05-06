# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017

@author: sylvain
"""

#from __future__ import print_function

import os
import keras
import theano
import cv2
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D 
from keras.optimizers import Adam,Adagrad
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.core import Dense, Dropout,   Reshape, Activation
import sklearn.metrics as metrics
from data_roi import load_train_data, load_test_data
from keras.utils import np_utils
from numpy import argmax,amax
#K.set_image_dim_ordering('th') 
#K.set_image_data_format('channels_first')
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


print keras.__version__
print theano.__version__


red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255,255,0)
cyan=(0,255,255)
purple=(255,0,255)
white=(255,255,255)
darkgreen=(11,123,96)
pink =(255,128,255)
lightgreen=(125,237,125)
orange=(255,153,102)
lowgreen=(0,51,51)
parme=(234,136,222)
chatain=(139,108,66)


"""
classif ={
        'consolidation':0,
        'HC':1,
        'ground_glass':2,
        'healthy':3,
        'micronodules':4,
        'reticulation':5,
        'air_trapping':6,
        'cysts':7,
        'bronchiectasis':8,
#        'emphysema':10,
        'GGpret':9
        }
"""
classif ={
        'healthy':0,
        'HC':2,
        'ground_glass':1,
        }

classifc ={
    'consolidation':cyan,
    'HC':blue,
    'ground_glass':red,
    'healthy':darkgreen,
    'micronodules':green,
    'reticulation':yellow,
    'air_trapping':pink,
    'cysts':lightgreen,
    'bronchiectasis':orange,
    'emphysema':chatain,
    'GGpret': parme,



     'nolung': lowgreen,
     'bronchial_wall_thickening':white,
     'early_fibrosis':white,

     'increased_attenuation':white,
     'macronodules':white,
     'pcp':white,
     'peripheral_micronodules':white,
     'tuberculosis':white
 }

img_rows = 320
img_cols = 320
img_rows = 496
img_cols = 496
#img_rows = 96
#img_cols = 96

smooth = 1.

def fidclass(numero,classn):
    """return class from number"""
    found=False
#    print numero
    for cle, valeur in classn.items():

        if valeur == numero:
            found=True
            return cle
    if not found:
        return 'unknown'

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

coefcon={}
coefcon[1]=32 #32 for 320
for i in range (2,6):
    coefcon[i]=coefcon[i-1]*2
print coefcon


def get_unet():
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

    conv10 = Conv2D(3, (1, 1), activation='softmax')(conv9)
#    conv11 =Reshape((3, 3))(conv10)
#    conv12 =Activation('softmax')(conv11)

    model = Model(inputs=[inputs], outputs=[conv10])

#    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
#    model.compile(optimizer=Adagrad, loss='binary_crossentropy',
#               metrics=['accuracy'])


    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
#        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)
        imgs_p[i]=cv2.resize(imgs[i],(img_cols, img_rows),interpolation=cv2.INTER_LINEAR)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def preprocessnew(X_train,y_train):
    X_train = np.asarray(np.expand_dims(X_train,3))
    print ('ytrain :',y_train.shape)
    print ('Xtrain :',X_train.shape)
    uniquelbls = np.unique(y_train)
    nb_classes = int( uniquelbls.shape[0])
    print ('number of classes :', int(nb_classes)) 
    ytrainr=np.zeros((y_train.shape[0],img_rows, img_cols,int(nb_classes)),np.uint8)
    for i in range (y_train.shape[0]-1):
        for j in range (0,img_rows):
                ytrainr[i][j] = np_utils.to_categorical(y_train[i][j], nb_classes)
    print ('ytrainr :',ytrainr.shape)
    return (X_train,ytrainr)

def preprocessnewtest(X_test):
    X_test = np.asarray(np.expand_dims(X_test,1))
#    print ('Xtrain :',X_train.shape)
#    print ('ytest :',y_train.shape)
    X_test=X_test.swapaxes(1,3)
    print ('Xtest :',X_test.shape)
#    ooo
    return (X_test)
    

def normi(tabi):
     """ normalise patches"""

     max_val=float(np.max(tabi))
     min_val=float(np.min(tabi))

     mm=max_val-min_val
     mm=max(mm,1.0)
#     print 'tabi1',min_val, max_val,imageDepth/float(max_val)
     tabi2=(tabi-min_val)*(255/mm)
     tabi2=tabi2.astype('uint8')
     return tabi2

def evaluate(actual,pred):
    fscore = metrics.f1_score(actual, pred, average='macro')
    acc = metrics.accuracy_score(actual, pred)
    cm = metrics.confusion_matrix(actual,pred)

    return fscore, acc, cm

def maxproba(proba):
    """looks for max probability in result"""
    lenp = len(proba)
    m=0
    for i in range(0,lenp):
        if proba[i]>m:
            m=proba[i]
            im=i
    return im,m


def predict():
    
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()
   
#    imgs_train = preprocess(imgs_train)


    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test= preprocessnewtest(imgs_test)
#    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std
    model = get_unet()
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')
    
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test=np.load('imgs_mask_test.npy' )
#    imgs_mask_test = model.predict(imgs_test, verbose=1,batch_size=1)
   
    print imgs_mask_test.shape
    imgs = np.ndarray((imgs_mask_test.shape[0], img_rows, img_cols,3), dtype=np.uint8)
    for i in range (imgs_mask_test.shape[0]):
        for j in range (0,img_rows):
            for k in range(img_cols):
                proba=imgs_mask_test[i][j][k]
                numpat=argmax(proba)
                pat=fidclass(numpat,classif)
                imgs[i][j][k]=classifc[pat]
#                if j%10 ==0 and k%10 ==0:
#                    print proba
#                print numpat
#                print pat
#                print classifc[pat]
#                ooo
                
#        print imgr.shape
#        print imgr[10][50]
      
#    print imgs.min(),imgs.max(),imgs.shape
#    proba = model.predict_proba(imgs_test, verbose=1,batch_size=1)
    
#    y_val_subset = imgs_mask_test[:]
#    y_test_subset = imgs_mask_train[:]
#
#    # argmax functions shows the index of the 1st occurence of the highest value in an array
##    y_actual = np.argmax(y_val_subset)
##    y_predict = np.argmax(y_test_subset)
#    
#    fscore, acc, cm = evaluate(y_test_subset, y_val_subset)
#    print 'f-score is : ', fscore
#    print 'accuray is : ', acc
#    print 'confusion matrix'
#    print cm
#    
    
    
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
        
        
        
    for image, image_id in zip(imgs, imgs_id_test):
        print image.min(),image.max(),image.shape
    
#        image = image[:, :, 0]
#        print image[0][0]
#        np.putmask(image,image>0.5,100)
#        image=image.astype(np.uint8)
#        image = (image[:, :, 0] * 255.).astype(np.uint8)
#        print image.min(),image.max(),image.shape
#        image=normi(image)
#        print image.min(),image.max()
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.bmp'), image)

def train():
    
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()
    print imgs_train.min(),imgs_train.max(),imgs_train.shape
    print imgs_mask_train.min(),imgs_mask_train.max(),imgs_mask_train.shape

    imgs_train,imgs_mask_train = preprocessnew(imgs_train,imgs_mask_train)
    print imgs_train.min(),imgs_train.max(),imgs_train.shape
#    imgs_mask_train = preprocessnew(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std
    
#    imgs_mask_train = imgs_mask_train.astype('uint8')

#    print imgs_mask_train.min(),imgs_mask_train.max(),imgs_mask_train.shape
#    ooo
#    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    
    model = get_unet()
    
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=20, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
    


if __name__ == '__main__':
#    train()
    predict()