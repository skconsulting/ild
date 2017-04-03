# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:51:27 2017

@author: sylvain
"""
import os
import cv2
import dicom
#import dircache
import sys
import shutil
from scipy import misc
import numpy as np
from sklearn.cross_validation import train_test_split
import cPickle as pickle
#from sklearn.cross_validation import train_test_split
import random
import math
from math import *
def normi(tabi,n):
     """ normalise patches"""
    
     max_val=float(np.max(tabi))
     min_val=float(np.min(tabi))
    
     mm=max_val-min_val
     mm=max(mm,1.0)
#     print 'tabi1',min_val, max_val,imageDepth/float(max_val)
     tabi2=(tabi-min_val)*(n/mm)
     tabi2=tabi2.astype('uint8')

     return tabi2
HUG= 'pr14/aa/source'
HUGw='pr14/aa/write'
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
dirHUG=os.path.join(cwdtop,HUG)
dirWHUG=os.path.join(cwdtop,HUGw)
fmbmp=dirHUG
listdcm=[name for name in  os.listdir(fmbmp) if name.lower().find('.dcm')>0]
#print listdcm
FilesDCM =(os.path.join(fmbmp,listdcm[0]))  
    #           
RefDs = dicom.read_file(FilesDCM)
print RefDs
dsr= RefDs.pixel_array
print dsr.shape,type(dsr[0][0]),dsr.min(),dsr.max()
print type(dsr)
dsr=normi(dsr,255)
print dsr.shape,type(dsr[0][0]),dsr.min(),dsr.max()


ndc=np.zeros((512,512,3),np.uint8)
ndc[200][200]=200
print RefDs.PhotometricInterpretation
RefDs.PhotometricInterpretation='RGB'
RefDs.HighBit=7
RefDs.BitsStored=8
RefDs.BitsAllocated=8
RefDs.SamplesPerPixel=3
RefDs.PlanarConfiguration=0
RefDs.SeriesDescription='bb' 
#RefDs.SOPInstanceUID                 ='1.3'  
a=random.randint(0,1000)
RefDs.StudyInstanceUID                 =str(a)  
print RefDs

#RefDs.BitsStored=24
#print RefDs
#print RefDs.PhotometricInterpretation
dsr = cv2.cvtColor(dsr, cv2.COLOR_GRAY2RGB)
#print dsr.shape
#a=dsr[0][0]
#print a
#print type(a)
#dsr[200:220,300:320]=1000
#ndc[200:220,300:320]=(256,256,256)
dsr[200:220,300:380]=(255,00,00)
dsr[100:140,200:240]=(255,255,00)
#ndc[300:320,400:420]=(0,0,256)
RefDs.PixelData=dsr
SliceThickness=RefDs.SliceThickness
filesDCMW =(os.path.join(dirWHUG,listdcm[0]))  
RefDs.save_as(filesDCMW)
