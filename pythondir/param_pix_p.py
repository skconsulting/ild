# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017

@author: sylvain
"""
import argparse
from appJar import gui
import cPickle as pickle
import cv2
import dicom
import numpy as np
from numpy import argmax,amax
import os
import random
import scipy
from scipy.misc import bytescale
import shutil
from skimage import measure
import sklearn.metrics as metrics
import sys
import time
from time import time as mytime
import webbrowser


import keras
import theano
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.models import load_model
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

import cnn_model as CNN4
import ild_helpers as H


print keras.__version__
print theano.__version__
print ' keras.backend.image_data_format :',keras.backend.image_data_format()

setdata='set2'

writeFile=False

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

dimpavx=16
dimpavy=16
pxy=float(dimpavx*dimpavy)

avgPixelSpacing=0.734   # average pixel spacing in mm

surfelem=avgPixelSpacing*avgPixelSpacing
volelem=surfelem*avgPixelSpacing

#print volelem

modelname='ILD_CNN_model.h5'
pathjs='../static'

datacrossn='datacross'
datafrontn='datafront'
path_data='data'
path_pickle='CNNparameters'
predictout='predicted_results'
predictout3d='predicted_results_3d'
predictout3dn='predicted_results_3dn'
predictout3d1='predicted_results_3dn1'
#predictout3dr='predicted_results_3dr'
predictoutmerge='predicted_results_merge'
dicompadirm='predict_dicom'
dicomcross='cross'
dicomfront='front'
dicomcross_merge='merge'

source_name='source'
jpegpath='jpegpath'
jpegpath3d='jpegpath3d'
jpegpadirm='jpegpadirm'

lung_name='lung'
lung_namebmp='bmp'
lung_name_gen='lung'
#directory with lung mask dicom
lungmask='lung'
lungmask1='lung_mask'
#directory to put  lung mask bmp
lungmaskbmp='scan_bmp'
lungimage='lungimage'


scan_bmp='scan_bmp'
transbmp='trans_bmp'
source='source'

bgdir='bgdir3d'

typei='jpg'
typei1='bmp'
typeid='jpg' #can be png for 16b
typej='jpg'
typeiroi1='jpg'
typeiroi2='bmp'

#excluvisu=['healthy']
excluvisu=['']


bmpname='scan_bmp'

sourcedcm = 'source'

sroi='sroi'
sroi3d='sroi3d'
volumeweb = 'volume.txt'
htmldir='html'
threeFileTxt='uip.txt'
threeFile='uip.html'
threeFilejs='world.js'

threeFileTxtMerge='uipMerge.txt'
threeFileMerge='uipMerge.html'
threeFilejsMerge='worldMerge.js'

threeFileTxt3d='uip3d.txt'
threeFile3d='uip3d.html'
threeFilejs3d='world3d.js'

threeFileTop0='uiptop0.html'
threeFileTop1='uiptop1.html'
threeFileTop2='uiptop2.html'
threeFileBot='uipbot.html'


black=(0,0,0)
grey=(100,100,100)
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

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
dirpickle=os.path.join(cwdtop,path_pickle)

classifnotvisu=['healthy',]
if setdata=='set2':

#set2
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
    usedclassif = [
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        'GGpret'
        ]
else:
    print 'error: not defined set'

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
volcol={
    'consolidation':'boxMaterialCyan',
    'HC':'boxMaterialBlue',
    'ground_glass':'boxMaterialRed',
    'healthy':'boxMaterialGrey',
    'micronodules':'boxMaterialGreen',
    'reticulation':'boxMaterialYellow',
    'air_trapping':'boxMaterialPink',
    'cysts':'boxMaterialLightgreen',
     'bronchiectasis':'boxMaterialOrange',
     'emphysema':'boxMaterialChatain',
     'GGpret': 'boxMaterialParme',

     'bronchial_wall_thickening':'boxMaterialWhite',
     'early_fibrosis':'boxMaterialWhite',
     'increased_attenuation':'boxMaterialWhite',
     'macronodules':'boxMaterialWhite',
     'pcp':'boxMaterialWhite',
     'peripheral_micronodules':'boxMaterialWhite',
     'tuberculosis':'boxMaterialWhite'
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


def normi(tabi):
     """ normalise patches"""
     tabi2=bytescale(tabi, low=0, high=255)
#     max_val=float(np.max(tabi))
#     min_val=float(np.min(tabi))
#     mm=max_val-min_val
#     if mm ==0:
#         mm=1
##     print 'tabi1',min_val, max_val,imageDepth/float(max_val)
#     tabi2=(tabi-min_val)*(255/mm)
#     tabi2=tabi2.astype('uint8')
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

def maxproba(proba):
    """looks for max probability in result"""  
    im=argmax(proba)
    m=amax(proba)
    return im,m