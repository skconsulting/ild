# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017

@author: sylvain
19 august 2017
version 1.3

"""

from numpy import argmax,amax
import os
import numpy as np
#from scipy.misc import bytescale
import shutil

import time
import sklearn.metrics as metrics

import keras
import theano
import sys
from keras import backend as K
K.set_image_dim_ordering('th')


print keras.__version__
print theano.__version__
print ' keras.backend.image_data_format :',keras.backend.image_data_format()

oldFormat=False #for compatibility with old format

writeFile=False
minmax=False #to be put to True for min and max on 5 slice
"""
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25
"""

minb=-1024.0
maxb=400.
PIXEL_MEAN = 0.275


MIN_BOUND = minb
MAX_BOUND = maxb


#limitHU=1424.0
#centerHU=-312.0
#PIXEL_MEAN = 0.2725
#
##limitHU=1700.0
##centerHU=-662.0
##PIXEL_MEAN = 0.52
#
#minb=centerHU-(limitHU/2)
#maxb=centerHU+(limitHU/2)
#MIN_BOUND =minb
#MAX_BOUND = maxb
#
#PIXEL_MEAN = 0.2725

print 'MIN_BOUND:',MIN_BOUND,'MAX_BOUND:',MAX_BOUND,'PIXEL_MEAN',PIXEL_MEAN
#sys.exit()
dimpavx=16
dimpavy=16
pxy=float(dimpavx*dimpavy) #surface in pixel

avgPixelSpacing=0.734   # average pixel spacing in mm

surfelemp=avgPixelSpacing*avgPixelSpacing # for 1 pixel in mm2
surfelem= surfelemp*pxy/100 #surface of 1 patch in cm2

volelemp=avgPixelSpacing*avgPixelSpacing*avgPixelSpacing # for 1 pixel
volelem= volelemp*pxy/1000 #in ml, to multiply by slicepitch in mm

modelname='ILD_CNN_model.h5'
pathjs='../static'

path_data='data'
path_pickle='CNNparameters'
path_pickleArch='modelArch'
modelArch='CNN.h5'

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
reportfile='report'
reportdir='report'

source_name='source'
jpegpath='jpegpath'
jpegpath3d='jpegpath3d'
jpegpadirm='jpegpadirm'

reportalldir='REPORT_SCORE'

lung_name='lung'
lung_namebmp='bmp'
#lung_name_gen='lung'
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
typei2='png' 

#volumeroifile='volumeroi'
#volumeroifilep='volumeroip'

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

#RGB
black=(0,0,0)
grey=(100,100,100)
red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
lblue=(30,30,255)
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
chatainlow=(109,98,46)
chatain=(139,108,66)
highgrey=(240,240,240)

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
dirpickle=os.path.join(cwdtop,path_pickle)
dirpickleArch=os.path.join(dirpickle,path_pickleArch)


classifnotvisu=['healthy',]
layertokeep= [
        'bronchiectasis',
        ]
setdataref='set1'

classifdict={}
usedclassifdict={}
derivedpatdict={}

##CHU
classifdict['CHU'] ={       
        'consolidation':0,
        'HC':1,
        'ground_glass':2,
        'healthy':3,
        'reticulation':4,
        'bronchiectasis':5,
        'lung':6
        
        }
usedclassifdict['CHU'] = [
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'reticulation',
        'bronchiectasis'
        ]
derivedpatdict['CHU']=[
  
        ]

##set0
classifdict['set0'] ={
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
        'GGpret':9,
        'lung':10
        }
usedclassifdict['set0'] = [
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
derivedpatdict['set0']=[
        'GGpret',
        ]

classifdict['set0p'] ={
        'back_ground':0,
        'consolidation':1,
        'HC':2,
        'ground_glass':3,
        'healthy':4,
        'micronodules':5,
        'reticulation':6,
        'air_trapping':7,
        'cysts':8,
        'bronchiectasis':9,
#        'emphysema':10,
        'GGpret':10,
        'lung':11
        }
usedclassifdict['set0p'] = [
        'back_ground',
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
derivedpatdict['set0p']=[
        'GGpret',
        ]
##set1

classifdict['set1'] ={
        
        'consolidation':0,
        'HC':1,
        'ground_glass':2,
        'healthy':3,
        'micronodules':4,
        'reticulation':5,
        'bronchiectasis':6,
        'emphysema':7,
        'GGpret':8,
        'lung':9
        }

usedclassifdict['set1'] = [
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'micronodules',
        'reticulation',
        'bronchiectasis',
        'emphysema',
        'GGpret'
        ]
derivedpatdict['set1']=[
        'GGpret',
        ]

classifdict['set1p'] ={
       'back_ground':0,
        'consolidation':1,
        'HC':2,
        'ground_glass':3,
        'healthy':4,
        'reticulation':5,
        'air_trapping':6,        
        'bronchiectasis':7,
        'GGpret':8,
        'lung':9
        }

usedclassifdict['set1p'] = [
        'back_ground',
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'reticulation',
        'air_trapping',
        'bronchiectasis',
        'GGpret'
        ]
derivedpatdict['set1p']=[
        'GGpret',
        ]

classifdict['set2'] ={
        'consolidation':0,
        'HC':1,
        'ground_glass':2,
        'healthy':3,
        'reticulation':4,
        'air_trapping':5,
                'cysts':6,
        'bronchiectasis':7,
        'GGpret':8,
        'lung':9
        }

usedclassifdict['set2'] = [
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        'GGpret'
        ]
derivedpatdict['set2']=[
        'GGpret',
        ]
#set2p###########################☻
classifdict['set2p'] ={
        'back_ground':0,
        'consolidation':1,
        'HC':2,
        'ground_glass':3,
        'healthy':4,
        'reticulation':5,
        'air_trapping':6,
        'cysts':7,
        'bronchiectasis':8,
        'GGpret':9,
        'lung':10
        }

usedclassifdict['set2p'] = [
        'back_ground',
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        'GGpret'
        ]
derivedpatdict['set2p']=[
        'GGpret',
        ]

classifc ={
    'back_ground':chatainlow,
    'consolidation':cyan,
    'HC':lblue,
    'ground_glass':red,
    'healthy':darkgreen,
    'micronodules':green,
    'reticulation':yellow,
    'air_trapping':pink,
    'cysts':lightgreen,
    'bronchiectasis':orange,
    'emphysema':chatain,
    'GGpret': parme,
     'lung': highgrey,
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
    'back_ground':'boxMaterialGrey',
    'micronodules':'boxMaterialGreen',
    'reticulation':'boxMaterialYellow',
    'air_trapping':'boxMaterialPink',
    'cysts':'boxMaterialLightgreen',
     'bronchiectasis':'boxMaterialOrange',
     'emphysema':'boxMaterialChatain',
     'GGpret': 'boxMaterialParme',
     'lung': 'boxMaterialGrey',

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
#     tabi2=bytescale(tabi, low=0, high=255)
     max_val=float(np.max(tabi))
     min_val=float(np.min(tabi))
     mm=max_val-min_val
     if mm ==0:
         mm=1
#     print 'tabi1',min_val, max_val,imageDepth/float(max_val)
     tabi2=(tabi-min_val)*(255/mm)
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

def maxproba(proba):
    """looks for max probability in result"""  
    im=argmax(proba)
    m=amax(proba)
    return im,m

def evaluate(actual,pred,num_class,label):
#    fscore = metrics.f1_score(actual, pred,labels=label, average='weighted')
#    acc = metrics.accuracy_score(actual, pred)
    pres = metrics.precision_score(actual, pred,labels=label,average='weighted')
    recall = metrics.recall_score(actual, pred,labels=label,average='weighted')
#    labl=[]
#    for i in range(num_class):
#        labl.append(i)
#    cm = metrics.confusion_matrix(actual,pred,labels=labl)
#    return fscore, acc, cm,pres,recall
    return pres,recall


def evaluatefull(actual,pred,num_class):
#    fscore = metrics.f1_score(actual, pred,labels=label, average='weighted')
#    acc = metrics.accuracy_score(actual, pred)
    labl=[]
    for i in range(1,num_class):
        labl.append(i)
    pres = metrics.precision_score(actual, pred,labels=labl,average='weighted')
    recall = metrics.recall_score(actual, pred,labels=labl,average='weighted')
    
#    for i in range(num_class):
#        labl.append(i)
#    cm = metrics.confusion_matrix(actual,pred,labels=labl)
#    return fscore, acc, cm,pres,recall
    return pres,recall

def evaluatef(actual,pred,num_class):
#    fscore = metrics.f1_score(actual, pred,labels=label, average='weighted')
#    acc = metrics.accuracy_score(actual, pred)
#    pres = metrics.precision_score(actual, pred,labels=label,average='weighted')
#    recall = metrics.recall_score(actual, pred,labels=label,average='weighted')
    labl=[]
    for i in range(1,num_class):
             labl.append(i)

    cm = metrics.confusion_matrix(actual,pred,labels=labl)
    return cm
