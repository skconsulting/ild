# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017

@author: sylvain Kritter 
Version 1.5 

06 September 2017
"""

import numpy as np
import os

import shutil

import time


setdata='set0'

#avgPixelSpacing=0.734   # average pixel spacing in mm
#
#surfelemp=avgPixelSpacing*avgPixelSpacing # for 1 pixel in mm2
#surfelem= surfelemp/100 #surface of 1 pixel in cm2

imageDepth=255
dimtabnorm=512
#dimtaby=512
dimtabmenu=190

source_name='source'
roi_name='sroi'
path_patient='path_patient'
scan_bmp='scan_bmp'
source='source'

typei='jpg'
typei1='bmp'
typei2='png' 

lung_mask='lung'
lung_mask1='lung_mask'
lung_mask_bmp1='scan_bmp'
lung_mask_bmp='bmp'
path_data='data'
volumeroifile='volumeroir'
reportalldir='REPORT_SCORE'

black=(0,0,0)
grey=(100,100,100)
highgrey=(200,200,200)
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
parme1=(232,136,224)
parme2=(230,136,226)
parme3=(228,136,228)
parme4=(226,136,230)
chatain=(139,108,66)
chatainlow=(109,98,46)
lowyellow=(239,228,176)

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)

if setdata=='set0':
#set0
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
        'emphysema':9,
#        'GGpret':10,
#        'HCpret':12,
#        'HCpbro':13,
#        'GGpbro':14,
#        'bropret':15,
        'unclass':10,
        'lung':11,


        'erase':13               
        }
    
    usedclassif =[
        'consolidation',
        'HC',
        'ground_glass',
        'emphysema',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
#        'HCpret',
#        'HCpbro',
#        'GGpbro',
#        'GGpret',
#        'bropret',
        'lung',
        'unclass',
        'erase'
        ]

    classifcontour=['lung']
elif setdata=='set0p':
#set0
    classif ={
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
        'emphysema':10,
#        'GGpret':11,
#        'HCpret':12,
#        'HCpbro':13,
#        'GGpbro':14,
#        'bropret':15,
        'lung':11,

        'erase':13               
        }
    
    usedclassif =[
#       'back_ground',
        'consolidation',
        'HC',
        'ground_glass',
        'emphysema',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
#        'HCpret',
#        'HCpbro',
#        'GGpbro',
#        'GGpret',
#        'bropret',
        'lung',
        'erase'
        ]

    classifcontour=['lung']
else:
    print 'error: not defined set'
    


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
    'HCpret': parme1,
    'HCpbro': parme2,
    'GGpbro': parme3,
    'bropret': parme4,
     'lung': highgrey,
     'bronchial_wall_thickening':white,
     'early_fibrosis':white,

     'increased_attenuation':white,
     'macronodules':white,
     'pcp':white,
     'peripheral_micronodules':white,
     'tuberculosis':white,
     'erase':white,
     'unclass':lowyellow
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
    