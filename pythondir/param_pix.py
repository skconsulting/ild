# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017

@author: sylvain
"""
import os
import shutil
import sklearn.metrics as metrics
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D 
from keras.optimizers import Adam,Adagrad
#from __future__ import print_function
from keras import backend as K
from keras.models import Model
from keras import applications
K.set_image_dim_ordering('th')
import numpy as np

image_rows = 512
image_cols = 512
#image_rows = 400
#image_cols = 400

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

bmpname='scan_bmp'
#directory with lung mask dicom
lungmask='lung_mask'
#directory to put  lung mask bmp
lungmaskbmp='scan_bmp'


typei='jpg' #can be jpg
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


classif ={
        'back_ground':0,
        'healthy':1,
        'ground_glass':2,
        'reticulation':3,
        'HC':4,
        'micronodules':5,
        'consolidation':6,
        'air_trapping':7,
        'cysts':8,
        'bronchiectasis':9,
        'GGpret':10
        } 

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

classifnotvisu=['back_ground']
#classifnotvisu=['back_ground',
#        'healthy',
#        'reticulation',
#        'HC',
#        'micronodules',
#        'consolidation',
#        'air_trapping',
#        'cysts',
#        'bronchiectasis',]

classifc ={
    'back_ground':chatain,
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
# Now the directory is empty of files


def normi(img):
     tabi1=img-img.min()
     maxt=float(tabi1.max())
     if maxt==0:
         maxt=1
     tabi2=tabi1*(255/maxt)
     tabi2=tabi2.astype('uint8')
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
    image2=zero_center(image1)
    return image2


def evaluate(actual,pred,num_class):
    fscore = metrics.f1_score(actual, pred, average='weighted')
    acc = metrics.accuracy_score(actual, pred)
    labl=[]
    for i in range(num_class):
        labl.append(i)
    cm = metrics.confusion_matrix(actual,pred,labels=labl)
    return fscore, acc, cm
coefcon={}
coefcon[1]=32 #32 for 320
#coefcon[1]=16 #32 for 320
for i in range (2,6):
    coefcon[i]=coefcon[i-1]*2
print coefcon
def get_unet(num_class,img_rows,img_cols):
#    global weights
#    weights=w
    kernel_size=(3,3)
    inputs = Input((1,img_rows, img_cols))
    conv1 = Conv2D(coefcon[1], kernel_size, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(coefcon[1], kernel_size, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(coefcon[2], kernel_size, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(coefcon[2], kernel_size, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(coefcon[3], kernel_size, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(coefcon[3], kernel_size, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(coefcon[4], kernel_size, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(coefcon[4], kernel_size, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(coefcon[5], kernel_size, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(coefcon[5], kernel_size, activation='relu', padding='same')(conv5)
#    print conv5.shape

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4],axis=1)
#    up6 = concatenate([conv5, conv4],axis=3)
 
    conv6 = Conv2D(coefcon[4], kernel_size, activation='relu', padding='same')(up6)
    conv6 = Conv2D(coefcon[4], kernel_size, activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)

    conv7 = Conv2D(coefcon[3], kernel_size, activation='relu', padding='same')(up7)
    conv7 = Conv2D(coefcon[3], kernel_size, activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(coefcon[2],kernel_size, activation='relu', padding='same')(up8)
    conv8 = Conv2D(coefcon[2], kernel_size, activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(coefcon[1], kernel_size, activation='relu', padding='same')(up9)
    conv9 = Conv2D(coefcon[1], kernel_size, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(int(num_class), (1,1), activation='softmax',padding='same')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model
   
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
#        return  w_categorical_crossentropy(y_true, y_pred, weights)
        y_true=np.moveaxis(y_true,1,3)
        y_pred=np.moveaxis(y_pred,1,3)
        
#         scale preds so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred,axis=-1, keepdims=True)
        # clip
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        # calc
        loss = y_true*K.log(y_pred)*self.weights
        loss =-K.sum(loss,-1)
        return loss
    
def get_model(num_class,img_rows,img_cols,weights):
    model = get_unet(num_class,img_rows,img_cols)
#    model=img_rows, img_cols, img_channel = 224, 224, 3

#    model = applications.VGG16(weights=None, include_top=True, 
#                               classes = num_classes,input_shape=(1,image_rows,image_cols))
    
    mloss = weighted_categorical_crossentropy(weights).myloss
#        model.compile(optimizer=Adam(lr=1e-5), loss=mloss, metrics=['categorical_accuracy'])
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model