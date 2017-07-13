# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017

@author: sylvain
"""
setdata='CHU'
#setdata='S2'



import collections
import cPickle as pickle
import cv2
import datetime
import dicom
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax,amax
import os
from PIL import Image
import random
import shutil
from scipy.io import loadmat
from scipy.misc import bytescale
import sklearn.metrics as metrics
from skimage.io import imsave
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import sys
import time


import keras
import theano
from keras import applications
from keras.constraints import maxnorm
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers import Input, concatenate, UpSampling2D ,ZeroPadding2D
from keras.layers import Reshape,Permute,Conv2DTranspose,BatchNormalization
from keras.optimizers import Adam,Adagrad,SGD
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.models import load_model,Model,Sequential
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger,EarlyStopping
from keras.utils import np_utils,layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
K.set_image_dim_ordering('tf')

#from __future__ import print_function
from utils import fcn32_blank, fcn_32s_to_8s,prediction
#from nicolovmodel import add_softmax,nicolov
from zf_unet_224_model import *
import cnn_model
import ild_helpers
print keras.__version__
print theano.__version__
print ' keras.backend.image_data_format :',keras.backend.image_data_format()


#modelName='fcn8s'
modelName='unet'
#modelName='vgg16'
#modelName='zf'
image_rows = 512
image_cols = 512
#image_rows = 96
#image_cols = 96


num_bit=1
learning_rate=1e-4
#image_rows = 16
#image_cols = 16
#image_rows = 400
#image_cols = 400
front_enabled=False #defined if front predict is enabled or not
cwd=os.getcwd()
#
(cwdtop,tail)=os.path.split(cwd)

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

bmpname='scan_bmp'
transbmp='trans_bmp'
#directory with lung mask dicom
lungmask='lung'
lungmask1='lung_mask'
#directory to put  lung mask bmp
lungmaskbmp='scan_bmp'
sourcedcm = 'source'


typei='jpg' #can be jpg
typeibmp='bmp'
sroi='sroi'
#avgPixelSpacing=0.734

black=(0,0,0)
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



if setdata=='CHU':

#CHU
    classif ={
        'back_ground':0,
        'healthy':1,    
        'ground_glass':2,
        'HC':3,
        'reticulation':4,
        'bronchiectasis':5,
        'cysts':6,
         'consolidation':7,
        'micronodules':8,
        'air_trapping':9,
        'GGpret':10
        } 
elif setdata=='ILD1':
#ILD1
    classif ={
        'back_ground':0,
        'healthy':1,    
        'ground_glass':2,
        'consolidation':3,

        'reticulation':4,
        'bronchiectasis':5,
        'cysts':6,
        'HC':7,
        'micronodules':8,
        'air_trapping':9,
        'GGpret':10
        } 
elif setdata=='ILD6':
#ILD6
    classif ={
        'back_ground':0,
        'healthy':1,    
        'ground_glass':2,
        'reticulation':3,
        'HC':4,
        'micronodules':5
        } 
elif setdata=='ILD0':
#ILD0
    classif ={
        'back_ground':0,
        'healthy':1,    
        'ground_glass':2,
        } 
elif setdata=='S1':
#ILD0
    classif ={
        'back_ground':0,
        'healthy':1,    
        } 
elif setdata=='S2':
#S2
    print 'this is S2'
    classif ={
        'back_ground':0,
        'healthy':1,   
        'ground_glass':2,
        }
elif setdata=='lungpatch':
#S2
    print 'this is lungpatch'
    classif ={
        'back_ground':0,
        'healthy':1,   
        'ground_glass':2,
        }     
elif setdata=='lc':
#S2
    print 'this is lc'
    classif ={
        'back_ground':0,
        'healthy':1,    
        'ground_glass':2,
        'reticulation':3,
        'HC':4,
        'micronodules':5
        }     
else:
    print 'error: not defined set'
print 'patterns for set :',setdata
for i,j in classif.items():
    print i, j
print '---------------'
usedclassif=[
        'back_ground',
        'healthy',
        'ground_glass',
        'reticulation',
        'HC',
        'micronodules',
        'consolidation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        ]

notusedclassif=[]
for i in classif:
    if i not in usedclassif:
        notusedclassif.append(i)

classifnotvisu=['back_ground','healthy',]
classifnotvisu=['back_ground']
#        'healthy',
#        'reticulation',
##        'HC',
#        'micronodules',
#        'consolidation',
#        'air_trapping',
#        'cysts',
##        'bronchiectasis'
#        ]

classifc ={
    'back_ground':black,
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

def rsliceNum(s,c,e):
    endnumslice=s.find(e)
    if endnumslice <0:
        return -1
    else:
        posend=endnumslice
        while s.find(c,posend)==-1:
            posend-=1
        debnumslice=posend+1
        return int((s[debnumslice:endnumslice]))
    
def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
#         print 'path exist'
         # remove if exists
         shutil.rmtree(path)
         time.sleep(1)
# Now the directory is empty of files

def normi(img):
#     tabi2=bytescale(img, low=0, high=255)
     tabi1=img-img.min()
     maxt=float(tabi1.max())
     if maxt==0:
         maxt=1
     tabi2=tabi1*(255/maxt)
     tabi2=tabi2.astype(np.uint8)
     return tabi2
 
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
    
def normalize(image):
    image1= (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image1[image1>1] = 1.
    image1[image1<0] = 0.
    return image1

def zero_center(image):
    image1 = image - PIXEL_MEAN
    return image1

def norm(image):
    image1=normalize(image)
    image2=zero_center(image1).astype(np.float32)
    return image2

def normHU(image): #normalize HU images
    image1= (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image1[image1>1] = 1.
    image1[image1<0] = 0.
    image2=image1 - PIXEL_MEAN
    return image2

def preprocess_batch(batch):
    batch=batch.astype(np.float32)
    batch /= 256
    batch -= 0.5
    return batch

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def evaluate(actual,pred,num_class):
    fscore = metrics.f1_score(actual, pred, average='weighted')
    acc = metrics.accuracy_score(actual, pred)
    labl=[]
    for i in range(num_class):
        labl.append(i)
    cm = metrics.confusion_matrix(actual,pred,labels=labl)
    return fscore, acc, cm

def double_conv_layerunet(x, size, dropout, batch_norm):
    kernel_size=(3,3)
    conv = Conv2D(size,kernel_size, activation='relu',kernel_constraint=maxnorm(4.),padding='same')(x)
#    conv = Conv2D(size,kernel_size,kernel_constraint=maxnorm(4.,axis=-1),padding='same')(x)

    if batch_norm == True:
        conv = BatchNormalization( )(conv)
#    conv = Activation('relu')(conv)
#    conv = LeakyReLU(alpha=0.15)(conv)
#    if dropout > 0:
#        conv = Dropout(dropout)(conv)

    conv = Conv2D(size, kernel_size,activation='relu',kernel_constraint=maxnorm(4.), padding='same')(conv)
#    conv = Conv2D(size,kernel_size,kernel_constraint=maxnorm(4.,axis=-1),padding='same')(conv)

    if batch_norm == True:
        conv = BatchNormalization()(conv)
#    conv = Activation('relu')(conv)
#    conv = LeakyReLU(alpha=0.15)(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    return conv

def get_unet(num_class,num_bit,img_rows,img_cols):
#    global weights
    print 'this is model UNET new'
    
#    weights=w
    coefcon={}
    coefcon[1]=16 #32 for 320
#coefcon[1]=16 #32 for 320
    for i in range (2,6):
        coefcon[i]=coefcon[i-1]*2
    print coefcon
    dor={}
    dor[1]=0.04 #0.04 f
#    dor[1]=0 #0.04 f

    for i in range (2,6):
        dor[i]=min(dor[i-1]*2,0.5)
    print 'do coeff :',dor
    batch_norm=False
    print 'barchnorm :', batch_norm
   
    inputs = Input((img_rows, img_cols,num_bit))
    conv1=double_conv_layerunet(inputs, coefcon[1], dor[1], False)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2=double_conv_layerunet(pool1, coefcon[2], dor[2], batch_norm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3=double_conv_layerunet(pool2, coefcon[3], dor[3], batch_norm)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4=double_conv_layerunet(pool3, coefcon[4],dor[4], batch_norm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5=double_conv_layerunet(pool4, coefcon[5], dor[5], batch_norm)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4],axis=3)
 
    conv6=double_conv_layerunet(up6, coefcon[4], dor[4], batch_norm)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    
    conv7=double_conv_layerunet(up7, coefcon[3], dor[3], batch_norm)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    
    conv8=double_conv_layerunet(up8, coefcon[2], dor[2], batch_norm)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    
    conv9=double_conv_layerunet(up9, coefcon[1], dor[1], False)    
     
    conv10 = Conv2D(int(num_class), (1,1), activation='softmax',padding='same')(conv9) #softmax?
    
    model = Model(inputs=[inputs], outputs=[conv10])
    print model.layers[-1].output_shape
#    model = Model(inputs=[inputs], outputs=[conv10])
#    print model.layers[-1].output_shape #== (None, 16, 16, 21)
    
    return model

def VGG_16(num_class,num_bit,img_rows,img_cols):
    
    print 'VGG16 with num_class :',num_class

    padding='same'
    kernel_size=(3,3)
    model = Sequential()
    model.add(Conv2D(64, kernel_size, activation='relu', padding=padding , input_shape=(img_rows,img_cols,num_bit)))
    model.add(Conv2D(64, kernel_size, activation='relu', padding=padding ))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Conv2D(128, kernel_size, activation='relu', padding=padding ))
    model.add(Conv2D(128, kernel_size, activation='relu', padding=padding ))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Conv2D(256, kernel_size, activation='relu', padding=padding ))
    model.add(Conv2D(256, kernel_size, activation='relu', padding=padding ))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Conv2D(512, kernel_size, activation='relu', padding=padding ))
    model.add(Conv2D(512, kernel_size, activation='relu', padding=padding ))
    model.add(Conv2D(512, kernel_size, activation='relu', padding=padding ))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Conv2D(512, kernel_size, activation='relu', padding=padding ))
    model.add(Conv2D(512, kernel_size, activation='relu', padding=padding ))
    model.add(Conv2D(512, kernel_size, activation='relu', padding=padding ))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(int(num_class), activation='softmax')) 
    
    '''
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                             WEIGHTS_PATH,  cache_subdir='models')
                            
    model.load_weights(weights_path)
    '''
    return model
   
def copy_mat_to_keras(kmodel, verbose=True):   
    print 'copy mat to keras'
    kerasnames = [lr.name for lr in kmodel.layers]
    prmt = (0,1,2,3) # WARNING : important setting as 2 of the 4 axis have same size dimension    
    for i in range(0, p.shape[1]):
        
        if USETVG:
            matname = p[0,i].name[0][0:-1]
            matname_type = p[0,i].name[0][-1] # "f" for filter weights or "b" for bias
        else:
            matname = p[0,i].name[0].replace('_filter','').replace('_bias','')
            matname_type = p[0,i].name[0].split('_')[-1] # "filter" or "bias"
        
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            if verbose:
                print 'found : ', (str(matname), str(matname_type), kindex)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            if  matname_type in ['f','filter']:
                l_weights = p[0,i].value
                f_l_weights = l_weights.transpose(prmt)
                f_l_weights = np.flip(f_l_weights, 2)
                f_l_weights = np.flip(f_l_weights, 3)
                if True: # WARNING : this depends on "image_data_format": put True when "channels_last" in keras.json file
                    f_l_weights = np.flip(f_l_weights, 0)
                    f_l_weights = np.flip(f_l_weights, 1)
                try:
                    assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
                    current_b = kmodel.layers[kindex].get_weights()[1]
                    kmodel.layers[kindex].set_weights([f_l_weights, current_b])
                except:
                    print 'not correct dim'
                        
            elif matname_type in ['b','bias']:
                l_bias = p[0,i].value
                assert (l_bias.shape[1] == 1)
                try:
                     assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
                     current_f = kmodel.layers[kindex].get_weights()[0]
                     kmodel.layers[kindex].set_weights([current_f, l_bias[:,0]])
                except:
                        print 'not correct dim'
        else:
            print 'not found : ', str(matname)

class weighted_categorical_crossentropy(object):
    """
    A weighted version of keras.objectives.categorical_crossentropy  
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    Usage:
        loss = weighted_categorical_crossentropy(weights).loss
        model.compile(loss=loss,optimizer='adam')
    """  
    def __init__(self,weights):
        self.weights = K.variable(weights)
    def myloss(self,y_true, y_pred):        
#         scale preds so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred,axis=-1, keepdims=True)
        # clip
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        # calc
        loss = y_true*K.log(y_pred)*self.weights
        loss =-K.sum(loss,-1)
        return loss
 
def get_model(num_class,num_bit,img_rows,img_cols,mat_t_k,weights):
    if modelName == 'unet':
#    unet
        model = get_unet(num_class,num_bit,img_rows,img_cols)
        mloss = weighted_categorical_crossentropy(weights).myloss
        model.compile(optimizer=Adam(lr=learning_rate), loss=mloss, metrics=['categorical_accuracy'])
#        model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    
    elif  modelName == 'fcn8s':
    #fcn8s
        USETVG = True
        if USETVG:
            data = loadmat('pascal-fcn8s-tvg-dag.mat', matlab_compatible=False, struct_as_record=False)
            l = data['layers']
            p = data['params']
            description = data['meta'][0,0].classes[0,0].description
        else:
            data = loadmat('pascal-fcn8s-dag.mat', matlab_compatible=False, struct_as_record=False)
#            l = data['layers']
#            p = data['params']
            description = data['meta'][0,0].classes[0,0].description
            print(data.keys())
        class2index = {}
        for i, clname in enumerate(description[0,:]):
            class2index[str(clname[0])] = i
        fcn32model = fcn32_blank(num_class,num_bit,img_rows)
        model = fcn_32s_to_8s(num_class,fcn32model)
        if mat_t_k:
            print 'import public weights'
            copy_mat_to_keras(model,False)
        mloss = weighted_categorical_crossentropy(weights).myloss
        model.compile(optimizer=Adam(lr=learning_rate), loss=mloss, metrics=['categorical_accuracy'])
#        model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    elif  modelName == 'vgg16':
        #VGG16
        model =VGG_16(num_class,num_bit,img_rows,img_cols)
        mloss = weighted_categorical_crossentropy(weights).myloss
        model.compile(optimizer=Adam(lr=1e-5), loss=mloss, metrics=['categorical_accuracy'])
#        model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
#        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
#        model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])
    elif  modelName == 'zf':
    #ZF
        model=ZF_UNET_224(num_class,img_rows,img_cols,0.05, True)
        mloss = weighted_categorical_crossentropy(weights).myloss
        model.compile(optimizer=Adam(lr=1e-5), loss=mloss, metrics=['categorical_accuracy'])
# 
# nicolov model
#    model = add_softmax(nicolov(num_class,img_rows,img_cols),img_rows,img_cols)
#    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
#    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
   weights=[]
   image_size=96
   num_bit=1
   num_class=3
   model=get_model(num_class,num_bit,image_size,image_size,False,weights)
   imarr = np.ones((image_size,image_size,num_bit))
   imarr = np.expand_dims(imarr, axis=0)
   print 'imarr.shape',imarr.shape
#   print 'model.predict(imarr).shape ',model.predict(imarr,verbose=1).shape
   model.summary()
#   model_json=model.to_json()
#   with open("model.json", "w") as json_file:
#    json_file.write(model_json)
#   file=open('model8s.json','w')
#   
#   file.write(mjson)
#   file.close()
   
#   
#   orig_stdout = sys.stdout
#   f = open('out1.txt', 'w')
#   sys.stdout = f
#   print(model.summary())
#   sys.stdout = orig_stdout
#   f.close()