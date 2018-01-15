# coding: utf-8
#Sylvain Kritter 4 aout 2017
"""Top file to generate patches from DICOM database HU method and pixle out from CHU Grenoble
include new patterns when patterns are super imposed, cross view
it is for cross view only
includes back_ground
version 1.0
18 august 2017
S. Kritter

"""

#from param_pix_t import *

from param_pix_t import usedclassifall
from param_pix_t import dimpavx,dimpavy,typei,typei1,avgPixelSpacing,thrpatch,perrorfile,plabelfile,pxy
from param_pix_t import remove_folder,normi,genelabelloc,totalpat,totalnbpat,fidclass,rsliceNum
from param_pix_t import white,medianblur,average3
from param_pix_t import patchpicklename,scan_bmp,lungmask,lungmask1,sroi,patchesdirname,derivedpat
from param_pix_t import imagedirname,picklepath,source,lungmaskbmp,layertokeep,reservedword,augmentation
import os
#import sys
#import png
import numpy as np
import datetime
#import scipy as sp
#import scipy.misc
import dicom
#import PIL

import cv2
#import matplotlib.pyplot as plt
import cPickle as pickle
#general parameters and file, directory names
#######################################################
#global directory for scan file
topdir='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool'
#namedirHUG = 'CHU2'
#namedirHUG = 'CHU2'
#namedirHUG = 'REFVALnew'
namedirHUG = 'HUG'

#subHUG='UIPJC'
subHUG='ILD_TXT'

#namedirHUG = 'CHU2new'



#subdir for roi in text
#subHUG='UIP'
#subHUG='UIP'
#subHUG='UIPTR14'

######################  end ############################################

#full path names
#cwd=os.getcwd()
#(cwdtop,tail)=os.path.split(cwd)
cwdtop=topdir
#path for HUG dicom patient parent directory
path_HUG=os.path.join(cwdtop,namedirHUG)
##directory name with patient databases
namedirtopc =os.path.join(path_HUG,subHUG)
print('directory ',namedirtopc)
if not os.path.exists(namedirtopc):
    print('directory ',namedirtopc, ' does not exist!')


#end general part
#########################################################
#log files

listHug= [ name for name in os.listdir(namedirtopc) if os.path.isdir(os.path.join(namedirtopc, name)) and \
          name not in reservedword]
print 'list of patients :',listHug




eferror=os.path.join(namedirtopc,perrorfile)

errorfile = open(eferror, 'a')
errorfile.write('---------------\n')
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
errorfile.write('started ' +namedirHUG+' '+subHUG+' at :'+todayn)
errorfile.write('--------------------------------\n')
errorfile.write('--------------------------------\n')
errorfile.write('source directory '+namedirtopc+'\n')
errorfile.write('th : '+ str(thrpatch)+'\n')
errorfile.write( 'list of patients :'+str(listHug)+'\n')
errorfile.write('using pattern set: \n')
for pat in usedclassifall:
    errorfile.write(pat+'\n')
errorfile.write('--------------------------------\n')
errorfile.close()
#filetowrite=os.path.join(namedirtopc,'lislabel.txt')

#end customisation part for datataprep
#######################################################


roitab={}

def genepara(namedirtopcf):
#    dirFileP = os.path.join(namedirtopcf, source)
    dirFileP = namedirtopcf

    listsln=[]
        #list dcm files
    fileList =[name for name in  os.listdir(dirFileP) if ".dcm" in name.lower()]
    slnt=0
    

    FilesDCM =(os.path.join(dirFileP,fileList[0]))
    FilesDCM1 =(os.path.join(dirFileP,fileList[1]))
    RefDs = dicom.read_file(FilesDCM,force=True)
    RefDs1 = dicom.read_file(FilesDCM1,force=True)
    patientPosition=RefDs.PatientPosition
#    SliceThickness=RefDs.SliceThickness
    try:
            slicepitch = np.abs(RefDs.ImagePositionPatient[2] - RefDs1.ImagePositionPatient[2])
    except:
            slicepitch = np.abs(RefDs.SliceLocation - RefDs1.SliceLocation)

    
    SliceThickness=RefDs.SliceThickness
    PatientAge=RefDs.PatientAge
    PatientSex=RefDs.PatientSex
    
    try:
        ConvolutionKernel=RefDs.ConvolutionKernel
    except:
            ConvolutionKernel=' NAN'
    try:
            SliceSpacingB=RefDs.SpacingBetweenSlices
    except AttributeError:
             print "Oops! No Slice spacing..."
             SliceSpacingB=0
    print 'number of slices', slnt
    print 'slice Thickness :',SliceThickness
    print 'Slice spacing',SliceSpacingB
    print 'slice pitch in z :',slicepitch
    print 'patient position :',patientPosition
    errorfile = open(eferror, 'a')
    errorfile.write('---------------\n')

    errorfile.write('number of slices :'+str(slnt)+'\n')
    errorfile.write('slice Thickness :'+str(SliceThickness)+'\n')
    errorfile.write('slice spacing :'+str(SliceSpacingB)+'\n')
    errorfile.write('slice pitch in z :'+str(slicepitch)+'\n')
    errorfile.write('patient position  :'+str(patientPosition)+'\n')

    slnt=slnt+1
    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
    dsr= RefDs.pixel_array
    dimtabxo=dsr.shape[0]
    dimtabyo=dsr.shape[1]
    dsr= dsr-dsr.min()
    dsr=dsr.astype('uint16')
    errorfile.write('patient shape  :'+str(dsr.shape[0])+'\n')
    errorfile.write('--------------------------------\n')
    errorfile.close()  
#    dsrresize = scipy.ndimage.interpolation.zoom(dsr, fxs, mode='nearest')
    dsrresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
    dimtabx=int(dsrresize.shape[0])
    dimtaby=int(dsrresize.shape[1])
    return dimtabxo,dimtabyo,dimtabx,dimtaby,slnt,fxs,listsln,PatientAge,PatientSex,ConvolutionKernel,patientPosition,SliceThickness,slicepitch


#############################################################

listdirc= [ name for name in os.listdir(namedirtopc) if os.path.isdir(os.path.join(namedirtopc, name)) and \
            name not in reservedword]

print 'class used :',usedclassifall

patdata={}
for f in listdirc:

    print('work on:',f)
    errorfile = open(eferror, 'a')
    listroi=[]
    tn = datetime.datetime.now()
    todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
    errorfile.write('started ' +namedirHUG+' '+f+' at :'+todayn)
    errorfile.close()

   
    namedirtopcf=os.path.join(namedirtopc,f)

   
    dimtabxo,dimtabyo,dimtabx,dimtaby,slnt,fxs,listsln,PatientAge,PatientSex,ConvolutionKernel,patientPosition,SliceThickness,slicepitch = genepara(namedirtopcf)

    patdata[f]={}
    patdata[f]['PatientAge']=PatientAge
    patdata[f]['PatientSex']=PatientSex
    patdata[f]['ConvolutionKernel']=ConvolutionKernel
    patdata[f]['patientPosition']=patientPosition
    patdata[f]['SliceThickness']=str(round(SliceThickness,3))
    patdata[f]['SliceThickness']= "{0:.2f}".format(SliceThickness)
    patdata[f]['slicepitch']="{0:.2f}".format(slicepitch)

    

   

errorfile = open(eferror, 'a')
for f in patdata:
    print patdata[f]['PatientAge'],patdata[f]['PatientSex'], patdata[f]['ConvolutionKernel'], patdata[f]['patientPosition'], patdata[f]['SliceThickness'],patdata[f]['slicepitch']
    errorfile.write(("%10s" % patdata[f]['PatientAge'])+
                    ("%10s" %patdata[f]['PatientSex'])+ 
                    ("%10s" %patdata[f]['ConvolutionKernel'])+
                    ("%10s" %patdata[f]['patientPosition'])+ 
                    ("%10s" %patdata[f]['SliceThickness'])+
                    ("%10s" %patdata[f]['slicepitch'])+'\n')
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
errorfile.write('completed ' +namedirHUG+' '+subHUG+' at :'+todayn)
errorfile.write('---------------\n')
errorfile.close()
#print listslice
print('completed')