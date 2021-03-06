# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017

@author: sylvain
"""
setdata='set1'

#modelName='fcn8s'
modelName='unet'
#modelName='sk4'
#modelName='alexnet'
#modelName='vgg16
#setdata='S2'

#import collections
import cPickle as pickle

import numpy as np

import os
import shutil
from scipy.io import loadmat
#from scipy.misc import bytescale
import sklearn.metrics as metrics

import sys
import time
from itertools import product
import functools
import random
import cv2

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
#K.set_image_dim_ordering('th') #for patch
K.set_image_dim_ordering('tf') #for pixel (for correct weighted loss function)

#from __future__ import print_function
from utils import fcn32_blank, fcn_32s_to_8s,prediction
#from nicolovmodel import add_softmax,nicolov
from zf_unet_224_model import ZF_UNET_224
from AlexNet_Original  import create_model
#import cnn_model
#import ild_helpers
print keras.__version__
print theano.__version__
print ' keras.backend.image_data_format :',keras.backend.image_data_format()


#modelName='zf'
image_rows = 512
image_cols = 512
#image_rows = 96
#image_cols = 96

DIM_ORDERING=keras.backend.image_data_format()
#print DIM_ORDERING

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
#
#limitHU=1700.0
#centerHU=-662.0

#limitHU=1424.0
#centerHU=-312.0
#
##
#minb=centerHU-(limitHU/2)
#maxb=centerHU+(limitHU/2)

minb=-1024.0
maxb=400.

#minb=-1024.0
#maxb=400.
#MIN_BOUND = -1000.0
#MAX_BOUND = 400.0

MIN_BOUND =minb
MAX_BOUND = maxb
PIXEL_MEAN = 0.346
#PIXEL_MEAN = 0.87#0.8681 for next
#PIXEL_MEAN = 0.275 #for ILD patch only

print 'MIN_BOUND:',MIN_BOUND,'MAX_BOUND:',MAX_BOUND,'PIXEL_MEAN',PIXEL_MEAN

scan_bmp='scan_bmp'
transbmp='trans_bmp'
#directory with lung mask dicom
lungmask='lung'
lungmask1='lung_mask'
#directory to put  lung mask bmp
lung_namebmp='bmp'
source = 'source'


typei='jpg' #can be jpg
typei1='bmp' #can be jpg
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

#cv2.destroyAllWindows()
layertokeep= [
        'bronchiectasis',
        ]

reservedword=['REPORT_SCORE']

if setdata=='set0':
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
 
    derivedpat=[
        'GGpret',
        ]
    
elif setdata=='set1':   
     classif ={
        'back_ground':0,
        'consolidation':1,
        'HC':2,
        'ground_glass':3,
        'healthy':4,
        'micronodules':5,
        'reticulation':6,
        'bronchiectasis':7,
        'emphysema':8,
        'GGpret':9
        } 

     derivedpat=[
        'GGpret'
        ]
    
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
#print 'patterns for set :',setdata
#for i,j in classif.items():
#    print i, j
print '---------------'
hugeClass=['healthy','back_ground']
hugeClass=[]
#notusedclassif=[]
#for i in classif:
#    if i not in usedclassif:
#        notusedclassif.append(i)

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
def colorimage(image,color):
#    im=image.copy()
    im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    np.putmask(im,im>0,color)
    return im

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
#    print 'bound', MIN_BOUND,MAX_BOUND
#    print 'image source',image.min(),image.max()
#    print 'image res',image1.min(),image1.max()
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

def geneshiftv(img,s):
    if s!=0:
        shap=img.shape[0]
        shi=int(shap*s/100.)
        tr=np.roll(img,shi,axis=0)
        if shi>0:
            remp=img[0]
            for i in range (shi):
                tr[i]=remp
        else:
            remp=img[-1]
            for i in range (-shi):
                tr[img.shape[0]-i-1]=remp
    else:
        tr=img
    return tr

def geneshifth(img,s):
    if s!=0:
        shap=img.shape[1]
        shi=int(shap*s/100.)
        tr=np.roll(img,shi,axis=1)       
        if shi>0:
            remp=img[:,0]
            for i in range (shi):
                tr[:,i]=remp
        else:
            remp=img[:,-1]
            for i in range (-shi):
                tr[:,img.shape[1]-i-1]=remp    
    else:
                tr=img
    return tr

def generesize(img,r,nearest):
    if r !=0:
        shapx=img.shape[1]
        shapy=img.shape[0]  
        types=type(img[0][0])
        imgc=img.copy().astype('float64')
        if nearest:
            imgr=cv2.resize(imgc,None,fx=(100+r)/100.,fy=(100+r)/100.,interpolation=cv2.INTER_NEAREST)  
        else:
            imgr=cv2.resize(imgc,None,fx=(100+r)/100.,fy=(100+r)/100.,interpolation=cv2.INTER_LINEAR)  
        imgr=imgr.astype(types)
        newshapex=imgr.shape[1]
        newshapey=imgr.shape[0]
        valcor=(imgc[0][0] +imgc[0][shapx-1]+imgc[shapy-1][0]+imgc[shapy-1][shapx-1])/4.
    
        if newshapex>shapx:
            dx=int(newshapex-shapx)/2
            dy=int(newshapey-shapy)/2
            imgrf=imgr[dy:dy+shapy,dx:dx+shapx]
        else:
            dx=int(shapx-newshapex)/2
            dy=int(shapy-newshapey)/2
            imgrf=np.zeros((shapy,shapx),types)
            np.putmask(imgrf,imgrf==0,valcor)
            imgrf[dy:dy+newshapey,dx:dx+newshapex]=imgr
    else:
            imgrf=img
    return imgrf

def generot(image,tt):
    if tt==0:
        imout=image
    elif tt==1:
    # 1 90 deg
        imout = np.rot90(image,1)
    elif tt==2:
    #2 180 deg
        imout = np.rot90( image,2)
    elif tt==3:
    #3 270 deg
        imout = np.rot90(image,3)
    elif tt==4:
    #4 flip fimage left-right
            imout=np.fliplr(image)
    elif tt==5:
    #5 flip fimage left-right +rot 90
        imout = np.rot90(np.fliplr(image))
    elif tt==6:
    #6 flip fimage left-right +rot 180
        imout = np.rot90(np.fliplr(image),2)
    elif tt==7:
    #7 flip fimage left-right +rot 270
        imout = np.rot90(np.fliplr(image),3)
    return imout


def genescaleint(img,s):
    if s!=0:
        minint=-1.
        maxint=1.
        types=type(img[0][0])
        acts=s*(maxint-minint)/100.0
        imgc=img.copy().astype('float64')    
        imgr=imgc+acts
        imgr=np.clip(imgr,minint,maxint)
        imgr=imgr.astype(types)
    else:
        imgr=img
    return imgr


def genesmultint(img,s):
    if s!=0:
        minint=-1.
        maxint=1.      
        types=type(img[0][0])
        acts=(100+s)/100.0
        imgc=img.copy().astype('float64')    
        imgr=imgc*acts
        imgr=np.clip(imgr,minint,maxint)
        imgr=imgr.astype(types)
    else:
        imgr=img
    return imgr


def geneaug(img,scaleint,multint,rotimg,resiz,shiftv,shifth,nearest):
    
    imgr=geneshifth(img,shifth)
    imgr=geneshiftv(imgr,shiftv)
    imgr=generot(imgr,rotimg)
    imgr=generesize(imgr,resiz,nearest)
    imgr=genesmultint(imgr,multint)
    imgr=genescaleint(imgr,scaleint)

    return imgr
    

def generandom(maxscaleint,maxmultint,maxrot,maxresize,maxshiftv,maxshifth,keepaenh):
    
    scaleint = random.randint(-100, 100)
    scaleint =keepaenh*maxscaleint*scaleint/100.
    
    multint = random.randint(-100, 100)
    multint =keepaenh*maxmultint*multint/100.
    
    rotimg = random.randint(0, maxrot)
    
    resiz = random.randint(-100,100)
    resiz =keepaenh*maxresize*resiz/100.
    
    shiftv = random.randint(-100, 100)
    shiftv =keepaenh*maxshiftv*shiftv/100.

    
    shifth = random.randint(-100, 100)
    shifth =keepaenh*maxshifth*shifth/100.

    return scaleint,multint,rotimg,resiz,shiftv,shifth



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
    print 'fscore'
    fscore = metrics.f1_score(actual, pred, average='weighted')
    print 'accuracy'
    acc = metrics.accuracy_score(actual, pred)
    labl=[]
    for i in range(num_class):
        labl.append(i)
    print 'cm'
    cm = metrics.confusion_matrix(actual,pred,labels=labl)
    return fscore, acc, cm

def double_conv_layerunet(x, size, dropout, batch_norm,dim_org,ke_i,kernel_siz):
    
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3

    conv = Conv2D(size,kernel_siz, activation='relu',kernel_initializer=ke_i,
                  data_format=dim_org,kernel_constraint=maxnorm(4.),padding='same')(x)
          
    if batch_norm == True:
        conv = BatchNormalization(axis=axis)(conv)
    
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    conv = Conv2D(size, kernel_siz,activation='relu',kernel_initializer=ke_i,
                  data_format=dim_org,kernel_constraint=maxnorm(4.), padding='same')(conv)

    if batch_norm == True:
        conv = BatchNormalization(axis=axis)(conv)
#    conv = LeakyReLU(alpha=0.15)(conv)
    
    return conv


def get_unet(num_class,num_bit,img_rows,img_cols,INP_SHAPE,dim_org,CONCAT_AXIS):

    print 'this is model UNET new2'
    ke_i='he_normal'
#    ke_i='glorot_uniform'
    kernel_size=(3,3)
    stride=(2,2)
    coefcon={}
    coefcon[1]=16 #32 for 320

#coefcon[1]=16 #32 for 320 #16 for gpu
    for i in range (2,6):
        coefcon[i]=coefcon[i-1]*2
    print coefcon
    dor={}
    dor[1]=0.04 #0.04 f
#    dor[1]=0 #0.04 f #best

    for i in range (2,6):
        dor[i]=min(dor[i-1]*2,0.5)
    print 'do coeff :',dor
    batch_norm=False
    print 'batchnorm :', batch_norm
   
#    inputs = Input((img_rows, img_cols,num_bit))
    inputs = Input(INP_SHAPE)
#    print INP_SHAPE

    conv1=double_conv_layerunet(inputs, coefcon[1], dor[1], False,dim_org,ke_i,kernel_size)
    pool1 = MaxPooling2D(pool_size=(2, 2),data_format=dim_org)(conv1)

    conv2=double_conv_layerunet(pool1, coefcon[2], dor[2], batch_norm,dim_org,ke_i,kernel_size)
    pool2 = MaxPooling2D(pool_size=(2, 2),data_format=dim_org)(conv2)

    conv3=double_conv_layerunet(pool2, coefcon[3], dor[3], batch_norm,dim_org,ke_i,kernel_size)
    pool3 = MaxPooling2D(pool_size=(2, 2),data_format=dim_org)(conv3)

    conv4=double_conv_layerunet(pool3, coefcon[4],dor[4], batch_norm,dim_org,ke_i,kernel_size)
    pool4 = MaxPooling2D(pool_size=(2, 2),data_format=dim_org)(conv4)

    conv5=double_conv_layerunet(pool4, coefcon[5], dor[5], batch_norm,dim_org,ke_i,kernel_size)
    
    #    up6 = concatenate([UpSampling2D(size=(2, 2),data_format=dim_org)(conv5), conv4],axis=CONCAT_AXIS)
    up6 = concatenate([Conv2DTranspose(coefcon[4], kernel_size, strides=(2, 2), padding='same',data_format=dim_org)(conv5), conv4], axis=CONCAT_AXIS) 
    conv6=double_conv_layerunet(up6, coefcon[4], dor[4], batch_norm,dim_org,ke_i,kernel_size)

#    up7 = concatenate([UpSampling2D(size=(2, 2),data_format=dim_org)(conv6), conv3], axis=CONCAT_AXIS)
    up7 = concatenate([Conv2DTranspose(coefcon[3], kernel_size, strides=stride, padding='same',data_format=dim_org)(conv6), conv3], axis=CONCAT_AXIS)    
    conv7=double_conv_layerunet(up7, coefcon[3], dor[3], batch_norm,dim_org,ke_i,kernel_size)

#    up8 = concatenate([UpSampling2D(size=(2, 2),data_format=dim_org)(conv7), conv2], axis=CONCAT_AXIS)
    up8 = concatenate([Conv2DTranspose(coefcon[2], kernel_size, strides=stride, padding='same',data_format=dim_org)(conv7), conv2], axis=CONCAT_AXIS)  
    conv8=double_conv_layerunet(up8, coefcon[2], dor[2], batch_norm,dim_org,ke_i,kernel_size)

#    up9 = concatenate([UpSampling2D(size=(2, 2),data_format=dim_org)(conv8), conv1], axis=CONCAT_AXIS)
    up9 = concatenate([Conv2DTranspose(coefcon[1], kernel_size, strides=stride, padding='same',data_format=dim_org)(conv8), conv1], axis=CONCAT_AXIS)  
    conv9=double_conv_layerunet(up9, coefcon[1], dor[1], False,dim_org,ke_i,kernel_size)    
     
    conv10 = Conv2D(int(num_class), (1,1), activation='softmax',data_format=dim_org,padding='same',kernel_initializer=ke_i)(conv9) #softmax?
    
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


def sk3(num_class,num_bit,img_rows,img_cols,INP_SHAPE,dim_org):
    print INP_SHAPE

    kernel_size=(3,3)
    
    print 'sk3 with num_class :',num_class
    model = Sequential() 
    model.add(Conv2D(16, kernel_size, input_shape=INP_SHAPE, padding='same', 
                     activation='relu',data_format=dim_org, kernel_constraint=maxnorm(3)))
    model.add(Conv2D(16, kernel_size, padding='same', 
                     activation='relu',data_format=dim_org, kernel_constraint=maxnorm(3)))
#    model.add(AveragePooling2D(pool_size=(2, 2),data_format=dim_org))
    model.add(Dropout(0.1)) 
    model.add(Conv2D(32, kernel_size,  padding='same', 
                     activation='relu',data_format=dim_org, kernel_constraint=maxnorm(3)))
    model.add(AveragePooling2D(pool_size=(2, 2),data_format=dim_org))
    model.add(Dropout(0.2)) 
    model.add(Conv2D(64, kernel_size,  padding='same', 
                     activation='relu',data_format=dim_org, kernel_constraint=maxnorm(3)))
    model.add(AveragePooling2D(pool_size=(2, 2),data_format=dim_org))
#    model.add(Dropout(0.2)) 
#    model.add(Conv2D(128, kernel_size,  padding='same', 
#                     activation='relu',data_format=dim_org, kernel_constraint=maxnorm(3)))
#    model.add(MaxPooling2D(pool_size=(2, 2),data_format=dim_org))    
    model.add(Dropout(0.3)) 
    model.add(Conv2D(128, kernel_size,  padding='same', 
                     activation='relu',data_format=dim_org, kernel_constraint=maxnorm(3)))
    model.add(UpSampling2D(size=(2, 2),data_format=dim_org))    
    model.add(Dropout(0.3))
    model.add(Conv2DTranspose(64, kernel_size,  padding='same', 
                     activation='relu',data_format=dim_org, kernel_constraint=maxnorm(3)))
    model.add(UpSampling2D(size=(2, 2),data_format=dim_org))
    model.add(Dropout(0.2)) 
    model.add(Conv2DTranspose(32, kernel_size,  padding='same', 
                     activation='relu',data_format=dim_org, kernel_constraint=maxnorm(3)))
#    model.add(UpSampling2D(size=(2, 2),data_format=dim_org))
    model.add(Dropout(0.1)) 
    model.add(Conv2DTranspose(num_class, kernel_size, activation='softmax', 
                     data_format=dim_org,padding='same')) 

    return model


def sk4(num_class,num_bit,img_rows,img_cols,INP_SHAPE,dim_org):
    print INP_SHAPE

    kernel_size=(3,3)
    kernel_sizes=(3,3)
    pool_siz=(2,2)
#    padding='valid'
    padding='same'
    maxnormv=3
    startnum=32
    coef={}
    coef[0]=startnum
    for i in range(1,4):
        coef[i]=coef[i-1]*2
        print i,coef[i]
    
    print 'sk4 with num_class :',num_class
    model = Sequential() 
    
    model.add(Conv2D(coef[0], kernel_sizes, input_shape=INP_SHAPE, padding=padding, 
                      data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))    
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(coef[0], kernel_sizes, input_shape=INP_SHAPE, padding=padding, 
                      data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))    
    model.add(LeakyReLU(alpha=0.01))
    model.add(AveragePooling2D(pool_size=pool_siz,data_format=dim_org))
    model.add(Dropout(0.25)) 
    
    model.add(Conv2D(coef[1], kernel_size, padding=padding, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(coef[1], kernel_size, padding=padding, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(AveragePooling2D(pool_size=pool_siz,data_format=dim_org))
    model.add(Dropout(0.25)) 
    
    model.add(Conv2D(coef[2], kernel_size, padding=padding, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(coef[2], kernel_size, padding=padding, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(AveragePooling2D(pool_size=pool_siz,data_format=dim_org))
    model.add(Dropout(0.25)) 
    
    model.add(Conv2D(coef[3], kernel_size, padding=padding, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(coef[3], kernel_size, padding=padding, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=0.01))
#    model.add(AveragePooling2D(pool_size=pool_siz,data_format=dim_org))
    model.add(Dropout(0.25)) 
    
#    model.add(Conv2D(coef[3], kernel_size, padding=padding, 
#                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
#    model.add(LeakyReLU(alpha=0.01))
#    model.add(Conv2D(coef[3], kernel_size, padding=padding, 
#                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
#    model.add(LeakyReLU(alpha=0.01))
#    model.add(UpSampling2D(size=pool_siz,data_format=dim_org))
#    model.add(Dropout(0.25)) 
    
    model.add(Conv2D(coef[2], kernel_size, padding=padding, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(coef[2], kernel_size, padding=padding, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25)) 
    model.add(UpSampling2D(size=pool_siz,data_format=dim_org))
    
    model.add(Conv2D(coef[1], kernel_size, padding=padding, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(coef[1], kernel_size, padding=padding, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25)) 
    model.add(UpSampling2D(size=pool_siz,data_format=dim_org))
    
    model.add(Conv2D(coef[0], kernel_size, padding=padding, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(coef[0], kernel_size, padding=padding, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25)) 
    model.add(UpSampling2D(size=pool_siz,data_format=dim_org))

    model.add(Conv2D(num_class, kernel_sizes, activation='softmax', 
                     data_format=dim_org,padding=padding)) 

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

class weighted_categorical_crossentropyold(object):
    """
    author: wassname
    A weighted version of keras.objectives.categorical_crossentropy  
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    Usage:
        loss = weighted_categorical_crossentropy(weights).loss
        model.compile(loss=loss,optimizer='adam')
    """  
    def __init__(self,weights):
        self.weights = K.variable(weights,dtype='float32')
    def myloss(self,y_true, y_pred):        
#         scale preds so that the class probas of each sample sum to 1
#        print 'ypred shape',y_pred.shape
#        y_pred = K.clip(y_pred, K.epsilon(), 10)
#        y_pred=K.permute_dimensions(y_pred,[0,2,3,1])
#        y_true=K.permute_dimensions(y_true,[0,2,3,1])
        y_pred /= K.sum(y_pred,axis=-1, keepdims=True)
#         clip
        y_pred = K.clip(y_pred, K.epsilon(), 1)
#         calc

        loss = y_true*K.log(y_pred)*self.weights
#        loss = y_true*K.log(y_pred)


        loss =-K.sum(loss,-1)
        return loss
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

def categorical_accuracy(y_true, y_pred):
 return K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())

 

def w_categorical_crossentropyold(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=0)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

def w_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
#        nb_cl = len(weights)
#        final_mask = K.zeros_like(y_pred[:, 0])
#        y_pred_max = K.max(y_pred, axis=1, keepdims=True)
#        y_pred_max_mat = K.equal(y_pred, y_pred_max)
#        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
#            final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
#        return K.categorical_crossentropy(y_pred, y_true) * final_mask
        return K.categorical_crossentropy(y_pred, y_true)

    return loss


def single_class_accuracy(interesting_class_id):
    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
#        accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')

        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return fn

def squeezed_accuracy(y_true, y_pred):

        class_id_true = K.argmax(K.squeeze(y_true,axis=0), axis=-1)
        class_id_preds = K.argmax(K.squeeze(y_pred,axis=0), axis=-1)
        # Replace class_id_preds with class_id_true for recall here
#        accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
#        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32')
        class_acc = K.mean(K.equal(class_id_true, class_id_preds))
        return class_acc

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def get_model(num_class,num_bit,img_rows,img_cols,mat_t_k,weights,weightedl,namelastc,learning_rate):
    DIM_ORDERING=keras.backend.image_data_format()
    if DIM_ORDERING == 'channels_first':
        INP_SHAPE = (num_bit, image_rows, image_cols)  
#        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 1
#        dim_org='channels_first'
    elif DIM_ORDERING == 'channels_last':
        INP_SHAPE = (image_rows, image_cols, num_bit)  
#        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 3
#        dim_org='channels_last'
        
    if modelName == 'unet':
#    unet
        model = get_unet(num_class,num_bit,img_rows,img_cols,INP_SHAPE,DIM_ORDERING,CONCAT_AXIS)
        if namelastc != 'NAN':
            model.load_weights(namelastc) 
        
        if weightedl:
            print 'weighted loss'
            mloss = weighted_categorical_crossentropy(weights).myloss
            model.compile(optimizer=Adam(lr=learning_rate), loss=mloss, metrics=['categorical_accuracy'])
        else:
            model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
            print 'NO weighted loss'
        
#        model.compile(optimizer=SGD(lr=learning_rate,decay=1e-6, momentum=0.9, nesterov=True), loss=mloss, metrics=['categorical_accuracy'])
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
        
    elif  modelName == 'alexnet':
    #alexnet
        DROPOUT = 0.5
        WEIGHT_DECAY = 0.0005   # L2 regularization factor
        USE_BN = True           # whether to use batch normalization
# Theano - 'th' (channels, width, height)
    # Tensorflow - 'tf' (width, height, channels)
        DIM_ORDERING = 'tf'
        model=create_model(num_class,num_bit,img_rows,img_cols,DIM_ORDERING,WEIGHT_DECAY,USE_BN,DROPOUT)
        mloss = weighted_categorical_crossentropy(weights).myloss
        model.compile(optimizer=Adam(lr=1e-5), loss=mloss, metrics=['categorical_accuracy'])
        
    
    elif  modelName == 'sk3':
         
        model=sk3(num_class,num_bit,img_rows,img_cols,INP_SHAPE,DIM_ORDERING)
#        weights = np.ones(512)
        if weightedl:
            print 'weighted loss'
            mloss = weighted_categorical_crossentropy(weights).myloss
            model.compile(optimizer=Adam(lr=learning_rate), loss=mloss, metrics=['categorical_accuracy'])
        else:
            model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
            print 'NO weighted loss'
    
    elif  modelName == 'sk4':        
        model=sk4(num_class,num_bit,img_rows,img_cols,INP_SHAPE,DIM_ORDERING)
        if namelastc != 'NAN':
            model.load_weights(namelastc)  
#        weights = np.ones(512)
        if weightedl:
            print 'weighted loss'
#            for layer in model.layers[:28]:
#                layer.trainable = False
#            for layer in model.layers[0:3]:
#                layer.trainable = True
            mloss = weighted_categorical_crossentropy(weights).myloss
#            INTERESTING_CLASS_ID=3
#            model.compile(optimizer=Adam(lr=learning_rate), loss=mloss, metrics=[single_class_accuracy(INTERESTING_CLASS_ID)])
#            model.compile(optimizer=Adam(lr=learning_rate), loss=mloss, metrics=[mean_pred])

            model.compile(optimizer=Adam(lr=learning_rate), loss=mloss, metrics=['categorical_accuracy'])

        else:
#            for layer in model.layers[:28]:
#                layer.trainable = False
#            for layer in model.layers[11:25]:
#                layer.trainable = False
            model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#            INTERESTING_CLASS_ID=0
#            model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=[single_class_accuracy(INTERESTING_CLASS_ID)])
#            model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy',mean_pred])
            print 'NO weighted loss'
        
        
# 
# nicolov model
#    model = add_softmax(nicolov(num_class,img_rows,img_cols),img_rows,img_cols)
#    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
#    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
   
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
#    

    print 'last model layer',model.layers[-1].output_shape #== (None, 16, 16, 21)
    return model

if __name__ == "__main__":
   weights=[]
   image_size=512
   num_bit=1
   num_class=11
   weightedl=False
   model=get_model(num_class,num_bit,image_size,image_size,False,weights,weightedl,'NAN',learning_rate)
   model.summary()
   print 'last model shape', model.layers[-1].output_shape #== (None, 16, 16, 21)
   DIM_ORDERING=keras.backend.image_data_format()
   print DIM_ORDERING
   
   if DIM_ORDERING == 'channels_first':
        imarr = np.ones((num_bit,image_size,image_size))
   else:
        imarr = np.ones((image_size,image_size,num_bit))
   imarr = np.expand_dims(imarr, axis=0)
   print 'imarr.shape',imarr.shape
   print 'model.predict(imarr).shape ',model.predict(imarr,verbose=1).shape
   
   json_string = model.to_json()
   pickle.dump(json_string,open( modelName+'CNN.h5', "wb"),protocol=-1)
   
   
   print modelName
   print model.metrics_names
   
#   model_json=model.to_json()
#   with open("model.json", "w") as json_file:
#    json_file.write(model_json)
#   file=open('model8s.json','w')
#   
#   file.write(mjson)
#   file.close()
   
#   
   orig_stdout = sys.stdout
   f = open(modelName+'_model.txt', 'w')
   sys.stdout = f
   print(model.summary())
   sys.stdout = orig_stdout
   f.close()