# coding: utf-8
#Sylvain Kritter 4 aout 2017
"""Top file to generate patches from DICOM database HU method and pixle out from CHU Grenoble
include new patterns when patterns are super imposed, cross view
it is for cross view only
version 1.0
18 august 2017
S. Kritter

"""

#from param_pix_t import *

from param_pix_t import derivedpatall,classifc,usedclassifall,classifall
from param_pix_t import dimpavx,dimpavy,typei,typei1,avgPixelSpacing,thrpatch,perrorfile,plabelfile,pxy
from param_pix_t import remove_folder,normi,genelabelloc,totalpat,totalnbpat,fidclass,rsliceNum
from param_pix_t import white,medianblur,average3,median3,numbit,minmax
from param_pix_t import patchpicklename,scan_bmp,lungmask,lungmask1,sroi,patchesdirname,derivedpat
from param_pix_t import imagedirname,picklepath,source,lungmaskbmp,layertokeep,reservedword,augmentation
import os
import sys
#import png
import numpy as np
import datetime
#import scipy as sp
#import scipy.misc
import dicom
#import PIL
import copy
import cv2
#import matplotlib.pyplot as plt
import cPickle as pickle
#general parameters and file, directory names
#######################################################
#global directory for scan file
topdir='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool'
namedirHUG = 'CHU2new'
#namedirHUG = 'CHU2'
#namedirHUG = 'CHU'
#namedirHUG = 'REFVALnew'


subHUG='UIP'
#subHUG='ILD_TXT'
#subHUG='UIPJC'
#subHUG='UIPJCAN'


#global directory for output patches file
toppatch= 'TOPPATCH'
#extension for output dir
#extendir='val'
extendir='all'
extendir='essai1'

if medianblur:
    exta1='m'
elif average3:
    exta1='a'
elif median3:
    exta1='med'
else:
    exta1=''
if augmentation:
    exta2='3'
else:
    exta2=''
if numbit: 
    if minmax: 
        exta3='_3bm53'
    else:
        exta3='_3b'
else:
    exta3='_1b'
#extendir1=namedirHUG+exta1+exta2+exta3
extendir1=namedirHUG+'_'+subHUG+'_'+exta1+exta2+exta3


alreadyDone =[ 'S107260', 'S139370', 'S139430', 'S139431', 'S145210', 
              'S14740', 'S15440', 'S1830', 'S274820', 
              'S275050', 'S28200', 'S335940', 'S359750', 
              'S4550', 'S72260', 'S72261']
alreadyDone =['']
#labelEnh=('consolidation','reticulation,air_trapping','bronchiectasis','cysts')
labelEnh=()
locabg='anywhere_CHUG'
forceDcm=False #true to force dcm for ROI, otherwise first put bmp
########################################################################
######################  end ############################################
########################################################################
if len (extendir1)>0:
    extendir1='_'+extendir1
patchesdirnametop = 'th'+str(round(thrpatch,2))+'_'+toppatch+'_'+extendir+extendir1
print 'name of directory for patches :', patchesdirnametop


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

patchtoppath=os.path.join(cwdtop,patchesdirnametop)
#create patch and jpeg directory
patchpath=os.path.join(patchtoppath,patchesdirname)
#path for patch pickle
picklepathdir =os.path.join(patchtoppath,picklepath)
#print 'picklepathdir',picklepathdir

#define the name for jpeg files
jpegpath=os.path.join(patchtoppath,imagedirname)
#print jpegpath

#patchpath = final/patches
#remove_folder(patchtoppath)
if not os.path.isdir(patchtoppath):
    os.mkdir(patchtoppath)

#remove_folder(patchpath)
if not os.path.isdir(patchpath):
    os.mkdir(patchpath)

if not os.path.isdir(picklepathdir):
    os.mkdir(picklepathdir)

#remove_folder(jpegpath)
if not os.path.isdir(jpegpath):
    os.mkdir(jpegpath)

listHug= [ name for name in os.listdir(namedirtopc) if os.path.isdir(os.path.join(namedirtopc, name)) and \
            name not in alreadyDone and name not in reservedword]
print 'list of patients :',listHug



eferror=os.path.join(patchtoppath,perrorfile)
errorfile = open(eferror, 'a')
errorfile.write('---------------\n')
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
errorfile.write('started ' +namedirHUG+' '+subHUG+' at :'+todayn)
errorfile.write('--------------------------------\n')
if medianblur:
    errorfile.write('median blur\n')
else:
    errorfile.write('NO median blur\n')
if average3:
   errorfile.write('average3\n')
else:
    errorfile.write('NO average3\n')
if median3:
   errorfile.write('median3\n')
else:
    errorfile.write('NO median3\n')
if numbit:
    if minmax:
     errorfile.write('3 dimension patch with min max\n')
    else:
     errorfile.write('3 dimension patch without min max\n')
else:
    errorfile.write('1 dimension patch\n')
    
if augmentation:
    errorfile.write('roi -1/+1 slice\n')
else:
    errorfile.write('roi 1 slice only\n')
    
    
errorfile.write('--------------------------------\n')
errorfile.write('source directory '+namedirtopc+'\n')
errorfile.write('th : '+ str(thrpatch)+'\n')
errorfile.write('name of directory for patches :'+ patchesdirnametop+'\n')
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
    dirFileP = os.path.join(namedirtopcf, source)
    listsln=[]
        #list dcm files
    fileList =[name for name in  os.listdir(dirFileP) if ".dcm" in name.lower()]
    slnt=0
    for filename in fileList:
        FilesDCM =(os.path.join(dirFileP,filename))
        RefDs = dicom.read_file(FilesDCM,force=True)
        scanNumber=int(RefDs.InstanceNumber)
        if scanNumber>slnt:
            slnt=scanNumber
        listsln.append(scanNumber)

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
    return dimtabxo,dimtabyo,dimtabx,dimtaby,slnt,fxs,listsln

def tagviews(tab,text,x,y):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,text,(x, y), font,0.3,white,1)
    return viseg

def resizescan(tabs,fxs):
    tabs= tabs.astype('float32')
    dsrmin= tabs.min()
    dsrmax= tabs.max()        
    tabs=cv2.resize(tabs,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
#                dsr=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_CUBIC)
    tabs=np.clip(tabs,dsrmin,dsrmax)
    tabs=tabs.astype('int16')
    return tabs


def genebmp(dirName, sou,tabscanName,fxs,listsln,listroi):
    """generate patches from dicom files and sroi"""
#    print ('generate  bmp files from dicom files in :',dirName, 'directory :',sou)
    dirFileP = os.path.join(dirName, sou)
    (top,tail)=os.path.split(dirName)
    listslnCopy=copy.copy(listsln)
    tabscanm1=np.zeros((slnt,dimtabx,dimtaby),np.int16)
    tabscanp1=np.zeros((slnt,dimtabx,dimtaby),np.int16)
    # for source scan
    if sou ==source:
        tabscano=np.zeros((slnt,dimtabxo,dimtabyo),np.int16)
        tabscan=np.zeros((slnt,dimtabx,dimtaby),np.int16)
        
        dirFilePbmp=os.path.join(dirFileP,scan_bmp)
        remove_folder(dirFilePbmp)
        os.mkdir(dirFilePbmp)
        fileList =[name for name in  os.listdir(dirFileP) if ".dcm" in name.lower()]
        for filename in fileList:
                FilesDCM =(os.path.join(dirFileP,filename))
                RefDs = dicom.read_file(FilesDCM,force=True)
                dsr= RefDs.pixel_array
                dsr=dsr.astype('int16')
#                fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
#                print 'fxs',fxs
                scanNumber=int(RefDs.InstanceNumber)
                endnumslice=filename.find('.dcm')
                imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)
                dsr[dsr == -2000] = 0
                intercept = RefDs.RescaleIntercept
                slope = RefDs.RescaleSlope
                if slope != 1:
                    dsr = slope * dsr.astype(np.float64)
                    dsr = dsr.astype(np.int16)
                dsr += np.int16(intercept)
                tabscano[scanNumber]=dsr 
                tabscanName[scanNumber]=imgcoredeb

        for scanNumber in range (1,slnt):
            
                tabscanos= tabscano[scanNumber].astype('float32')
                dtx=tabscanos.shape[0]
                if medianblur:
                    tabscanos=cv2.medianBlur(tabscanos,3)             
                if average3:
                    ts9 = tabscano[max(scanNumber-1,1)].astype('float32')
                    ts11 = tabscano[min(scanNumber+1,slnt-1)].astype('float32') 
                    tabscanos=(tabscanos+ts9+ts11)/3.
                    
                if median3:
                    ts9 = tabscano[max(scanNumber-1,1)].astype('float32')
                    ts11 = tabscano[min(scanNumber+1,slnt-1)].astype('float32')
                    tmedian=np.median(np.dstack((ts9,tabscanos,ts11)),axis=2)
                    tabsref = np.zeros((dtx,dtx),'float32')    
                    dift=tabscanos-ts9
                    dift1=ts11-tabscanos
                    tmaxabs=np.maximum(abs(dift),abs(dift1))
                    np.putmask(tabsref,tmaxabs<201,tabscanos)
                    np.putmask(tabsref,tmaxabs>200,tmedian)
                    tabscanos=tabsref
               
                    
                tabscan[scanNumber]=resizescan(tabscanos,fxs)
                dsrforimage=normi(tabscan[scanNumber])
                                            
#                imgcored=tabscanName[scanNumber]+'.'+typei1
#                bmpfiled=os.path.join(dirFilePbmp,imgcored)

                textw='n: '+tail+' scan: '+str(scanNumber)

#                cv2.imwrite (bmpfiled, dsrforimage)
                dsrforimage=cv2.cvtColor(dsrforimage,cv2.COLOR_GRAY2BGR)
                dsrforimage=tagviews(dsrforimage,textw,2,20)
#                cv2.imwrite (bmpfileroi, dsrforimage)
                tabsroi[scanNumber]=dsrforimage
        if numbit:
            for scanNumber in range (1,slnt):
                    tabscanm1[scanNumber]=tabscan[max(scanNumber-1,1)]
                    tabscanp1[scanNumber]=tabscan[min(scanNumber+1,slnt-1)]

    # for lung
    elif sou == lungmask or sou == lungmask1 :

        dirFilePbmp=os.path.join(dirFileP,lungmaskbmp)
        tabscan=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
        if not os.path.exists(dirFilePbmp):
            os.mkdir(dirFilePbmp)
            
        fileList = [name for name in os.listdir(dirFilePbmp) if '.'+typei1 in name.lower()]

        if len(fileList)>0:
            for fil in fileList:
                namefile=os.path.join(dirFilePbmp,fil)
#                print fil,namefile
                img=cv2.imread(namefile,0)
#                print fxs
#                print img.shape
#                cv2.imshow('img',img)
                
                dsrresizer=cv2.resize(img,(dimtabx,dimtabx),interpolation=cv2.INTER_LINEAR)
                scanNumber=rsliceNum(namefile,'_','.'+typei1)

                np.putmask(dsrresizer,dsrresizer>0,100)
                tabscan[scanNumber]=dsrresizer
                listslnCopy.remove(scanNumber)
        if len(listslnCopy)>0:     
            print 'lung  bmp not  complete' 
            fileList =[name for name in  os.listdir(dirFileP) if ".dcm" in name.lower()]
            if len(fileList)>0:
                for filename in fileList:
                    FilesDCM =(os.path.join(dirFileP,filename))
                    RefDs = dicom.read_file(FilesDCM,force=True)
                    dsr= RefDs.pixel_array
                    dsr=dsr.astype('int16')
    #                fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
    #                print 'fxs',fxs
                    scanNumber=int(RefDs.InstanceNumber)
                    if tabscan[scanNumber].max()==0:
                        endnumslice=filename.find('.dcm')
                        imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)
                        dsr=normi(dsr)
                        if dsr.max()>0:
                            dsrresize=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)                        
                            imgcored=tabscanName[scanNumber]+'.'+typei1
                            bmpfiled=os.path.join(dirFilePbmp,imgcored)
                            imgc=colorimage(dsrresize,classifc[sou])
                            cv2.imwrite (bmpfiled, imgc)
                            dsrresizer=np.copy(dsrresize)
    #                        np.putmask(dsrresizer,dsrresizer==1,0)
                            np.putmask(dsrresizer,dsrresizer>0,100)
                            tabscan[scanNumber]=dsrresizer
            else:
                 print 'no lung dcm'
        else:
              print 'lung  bmp complete' 
    #for roi
    else:
        tabscan=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
        if not os.path.exists(dirFileP):
            os.mkdir(dirFileP)
        if not forceDcm:
            #not force roi from dcm
            fileList = [name for name in os.listdir(dirFileP) if '.'+typei1 in name.lower()]
            if len(fileList)>0:
                print 'roi in bmp'
                augm=True
                for fil in fileList:
                    namefile=os.path.join(dirFileP,fil)
                    img=cv2.imread(namefile,0)
                    if img.max()>0:
                        
                        scanNumberr=rsliceNum(namefile,'_','.'+typei1)
                        if augmentation:
                            if augm:
                                print 'augmentation'
                                augm=False
                            for i in range (-1,2):                            
                                scanNumber=scanNumberr+i
    #                            print scanNumber
                                if scanNumber not in listroi:
                                        listroi.append(scanNumber)
                                dsrresizer=cv2.resize(img,(dimtabx,dimtabx),interpolation=cv2.INTER_LINEAR)
                                np.putmask(dsrresizer,dsrresizer>0,100)
                                tabscan[scanNumber]=dsrresizer
                        else:
                            if scanNumberr not in listroi:
                                        listroi.append(scanNumberr)
                            dsrresizer=cv2.resize(img,(dimtabx,dimtabx),interpolation=cv2.INTER_LINEAR)
                            np.putmask(dsrresizer,dsrresizer>0,100)
                            tabscan[scanNumberr]=dsrresizer
            else:
                print 'no roi in bmp'                    
            fileList =[name for name in  os.listdir(dirFileP) if ".dcm" in name.lower()]
            if len(fileList)>0:
                print ' roi in dcm'
                for filename in fileList:
                    FilesDCM =(os.path.join(dirFileP,filename))
                    RefDs = dicom.read_file(FilesDCM,force=True)
                    dsr= RefDs.pixel_array
                    dsr=dsr.astype('int16')
#                    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
    #                print 'fxs',fxs
                    scanNumber=int(RefDs.InstanceNumber)
                    if tabscan[scanNumber].max()==0:
                        endnumslice=filename.find('.dcm')
                        imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)
                        dsr=normi(dsr)
                        if dsr.max()>0:
                            if scanNumber not in listroi:
                                listroi.append(scanNumber)
                            dsrresize=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)                  
                            imgcored=tabscanName[scanNumber]+'.'+typei1
                            bmpfiled=os.path.join(dirFileP,imgcored)
                            imgc=colorimage(dsrresize,classifc[sou])
                            cv2.imwrite (bmpfiled, imgc)
                            dsrresizer=np.copy(dsrresize)
                            np.putmask(dsrresizer,dsrresizer==1,0)
                            np.putmask(dsrresizer,dsrresizer>0,100)
                            tabscan[scanNumber]=dsrresizer   
            else: 
                print 'no roi in dcm'                
        else:
        #force roi from dcm
            print 'force roi from dcm'
            fileList =[name for name in  os.listdir(dirFileP) if ".dcm" in name.lower()]
            if len(fileList)>0:
                for filename in fileList:
                    FilesDCM =(os.path.join(dirFileP,filename))
                    RefDs = dicom.read_file(FilesDCM,force=True)
                    dsr= RefDs.pixel_array
                    dsr=dsr.astype('int16')
#                    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
    #                print 'fxs',fxs
                    scanNumber=int(RefDs.InstanceNumber)
                    if tabscan[scanNumber].max()==0:
                        endnumslice=filename.find('.dcm')
                        imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)
                        dsr=normi(dsr)
                        if dsr.max()>0:
                            if scanNumber not in listroi:
                                listroi.append(scanNumber)
                            dsrresize=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)                  
                            imgcored=tabscanName[scanNumber]+'.'+typei1
                            bmpfiled=os.path.join(dirFileP,imgcored)
                            imgc=colorimage(dsrresize,classifc[sou])
                            cv2.imwrite (bmpfiled, imgc)
                            dsrresizer=np.copy(dsrresize)
                            np.putmask(dsrresizer,dsrresizer==1,0)
                            np.putmask(dsrresizer,dsrresizer>0,100)
                            tabscan[scanNumber]=dsrresizer   

#    print sou, tabscan.min(),tabscan.max()
    return tabscan,tabsroi,tabscanName,listroi,tabscanm1,tabscanp1

def reptfulle(tabc,dx,dy):
    imgi = np.zeros((dx,dy,3), np.uint8)
    cv2.polylines(imgi,[tabc],True,(1,1,1))
    cv2.fillPoly(imgi,[tabc],(1,1,1))
    tabzi = np.array(imgi)
    tabz = tabzi[:, :,1]
    return tabz, imgi


def tagview(tab,label,x,y):
    """write text in image according to label and color"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    col=classifc[label]
    labnow=classifall[label]
#    print (labnow, text)
    if label == 'back_ground':
        deltay=30
    else:
#        deltay=25*((labnow-1)%5)
        deltay=40+10*(labnow-1)

    viseg=cv2.putText(tab,label,(x, y+deltay), font,0.3,col,1)
    return viseg



def contour2(im,l):
    col=classifc[l]
    vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
#    im=im.astype('uint8')
    ret,thresh = cv2.threshold(im,0,255,0)
    im2,contours0, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
    cv2.drawContours(vis,contours,-1,col,-1,cv2.LINE_AA)
    return vis

def pavs (dirName,pat,slnt,dimtabx,dimtaby,tabscanName,listroi,tabscanm1,tabscan,tabscanp1):
    """ generate patches from ROI"""
    print 'pav :',dirName,'pattern :',pat
    ntotpat=0

    tabf=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    _tabsroi=np.zeros((slnt,dimtabx,dimtaby,3),np.uint8)
#    _tabbg = np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    _tabscan = np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    patpickle=[]
    (top,tail)=os.path.split(dirName)


    nampadir=os.path.join(patchpath,pat)
    nampadirl=os.path.join(nampadir,locabg)
    if not os.path.exists(nampadir):
         os.mkdir(nampadir)
    if not os.path.exists(nampadirl):
         os.mkdir(nampadirl)

    pathpicklepat=os.path.join(picklepathdir,pat)
#    print pathpicklepat
    pathpicklepatl=os.path.join(pathpicklepat,locabg)

#    patchpicklenamepatient=namedirHUG+'_'+tail+'_'+patchpicklename
    patchpicklenamepatient=namedirHUG+'_'+subHUG+'_'+tail+'_'+patchpicklename

    pathpicklepatfile=os.path.join(pathpicklepatl,patchpicklenamepatient)

    if not os.path.exists(pathpicklepat):
         os.mkdir(pathpicklepat)
    if not os.path.exists(pathpicklepatl):
         os.mkdir(pathpicklepatl)
    if os.path.exists(pathpicklepatfile):
        os.remove(pathpicklepatfile)

    for scannumb in listroi:
#       print 'wrok on scan: ',scannumb

       tabp = np.zeros((dimtabx, dimtaby), dtype='i')
       tabf=np.copy(tabroipat[pat][scannumb])

       tabfc=np.copy(tabf)
       np.putmask(tabfc,tabfc>0,100)
       nbp=0

       if tabf.max()>0:
           vis=contour2(tabf,pat)
           if vis.sum()>0:

                _tabsroi = np.copy(tabsroi[scannumb])
                imn=cv2.addWeighted(vis,0.5,_tabsroi,1,0)
                imn=tagview(imn,pat,0,100)
                tabsroi[scannumb]=imn
                imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)
                sroifile=tabscanName[scannumb]+'.'+typei1
                filenamesroi=os.path.join(sroidir,sroifile)
                cv2.imwrite(filenamesroi,imn)

                atabf = np.nonzero(tabf)

                xmin=atabf[1].min()
                xmax=atabf[1].max()
                ymin=atabf[0].min()
                ymax=atabf[0].max()

                np.putmask(tabf,tabf>0,1)
                _tabscan=tabscan[scannumb]
                if numbit:
#                    _tabscanm2=tabscanm1[max(scannumb-1,1)]
                    _tabscanm1=tabscanm1[scannumb]
                    _tabscanp1=tabscanp1[scannumb]
                    _tabscanm2=tabscanm1[max(1,scannumb-1)]
                    _tabscanp2=tabscanp1[min(scannumb+1,slnt-1)]
#                    _tabscanp2=tabscanp1[min(scannumb+1,slnt-1)]
                  
                for  i in range(xmin,xmax+1):
                    j=ymin
                    while j<=ymax:
                        tabpatch=tabf[j:j+dimpavy,i:i+dimpavx]
                        area= tabpatch.sum()
                        targ=float(area)/pxy
                        
                        if targ >thrpatch:
                            imgray = _tabscan[j:j+dimpavy,i:i+dimpavx]
                            if numbit:
                                imgraym1 = _tabscanm1[j:j+dimpavy,i:i+dimpavx]
                                imgrayp1 = _tabscanp1[j:j+dimpavy,i:i+dimpavx]
                                imgraym2 = _tabscanm2[j:j+dimpavy,i:i+dimpavx]
                                imgrayp2 = _tabscanp2[j:j+dimpavy,i:i+dimpavx]
#                                imgraym2 = _tabscanm2[j:j+dimpavy,i:i+dimpavx]
#                                imgrayp2 = _tabscanp2[j:j+dimpavy,i:i+dimpavx]
                            max_val= imgray.max()
                            min_val=imgray.min()

                            if  max_val - min_val>2:
                                nbp+=1
                                if numbit:
                                    if minmax:
                                        imgrayminimum=np.minimum(imgray,imgraym2)
                                        imgrayminimum=np.minimum(imgraym1,imgrayminimum)                                      
                                        imgrayminimum=np.minimum(imgrayp1,imgrayminimum)
                                        imgrayminimum=np.minimum(imgrayp2,imgrayminimum)
                                        
                                        imgraymaximum=np.maximum(imgray,imgraym2)
                                        imgraymaximum=np.maximum(imgraym1,imgraymaximum)
                                        imgraymaximum=np.maximum(imgrayp1,imgraymaximum)
                                        imgraymaximum=np.maximum(imgrayp2,imgraymaximum)

                                        
                                        imgraystack=np.dstack((imgrayminimum,imgraym1,imgray,imgrayp1,imgraymaximum))
#                                        print imgraystack.shape,imgraystack[0,0],imgraym2[0,0],imgraym1[0,0],imgray[0,0],imgrayp1[0,0],imgrayp2[0,0]
#                                        ooo

                                        
                                    else:                                   
#                                        imgraystack=np.dstack((imgraym2,imgraym1,imgray,imgrayp1,imgrayp2))
                                        imgraystack=np.dstack((imgraym1,imgray,imgrayp1))

#                                        print imgraystack.shape,imgraystack[0,0],imgraym2[0,0],imgraym1[0,0],imgray[0,0],imgrayp1[0,0],imgrayp2[0,0]
#                                        ooo
                                    patpickle.append(imgraystack)
                                    
                                else:
                                    patpickle.append(imgray)
                                cv2.rectangle(tabp,(i,j),(i+dimpavx,j+dimpavy),200,0)
                                tabf[j:j+dimpavy,i:i+dimpavx]=0
                                j+=dimpavy-1
                        j+=1
                    i+=1

       if nbp>0:
#             print tabfc.shape,tabp.shape
             tabfc =tabfc+tabp
             ntotpat=ntotpat+nbp
             if scannumb not in listsliceok:
                    listsliceok.append(scannumb)
             stw=namedirHUG+'_'+tail+'_slice_'+str(scannumb)+'_'+pat+'_'+locabg+'_'+str(nbp)
             stww=stw+'.txt'
             flw=os.path.join(jpegpath,stww)
             mfl=open(flw,"w")
             mfl.write('#number of patches: '+str(nbp)+'\n')
             mfl.close()
             stww=stw+'.'+typei
             flw=os.path.join(jpegpath,stww)
             cv2.imwrite(flw, tabfc)
#             print 'pathpicklepatfile',pathpicklepatfile
#             print patpickle
             pickle.dump(patpickle, open(pathpicklepatfile, "wb"),protocol=-1)
#             print '1' ,scannumb,listij

   
    return ntotpat

def colorimage(image,color):
#    im=image.copy()
    im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    np.putmask(im,im>0,color)

    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    return im


def calnewpat(dirName,pat,slnt,dimtabx,dimtaby,tabscanName):
    print 'new pattern : ',pat
#    (top,tail)=os.path.split(dirName)
    tab=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    if pat=='HCpret':
        pat1='HC'
        pat2='reticulation'

    elif pat=='HCpbro':
        pat1='HC'
        pat2='bronchiectasis'

    elif pat=='GGpbro':
        pat1='ground_glass'
        pat2='bronchiectasis'

    elif pat == 'GGpret':
        pat1='ground_glass'
        pat2='reticulation'

    elif pat=='bropret':
        pat1='bronchiectasis'
        pat2='reticulation'

    tab1=np.copy(tabroipat[pat1])
    tab2=np.copy(tabroipat[pat2])
    tab3=np.copy(tabroipat[pat])
    np.putmask(tab1,tab1>0,255)
    np.putmask(tab2,tab2>0,255)
#    np.putmask(tab3,tab3>0,255)

    nm=False

    for i in range (0,slnt):
#        if i == 145 and pat=='bropret':
#        if tab1[i].max()>0:
#            cv2.imshow('tab1',tab1[i])
#            cv2.imshow('tab2',tab2[i])
#            cv2.imshow('tab3',tab3[i])
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()

        tab3[i]=np.bitwise_and(tab1[i],tab2[i])
        
        if tab3[i].max()>0:
            
            tab[i]=np.bitwise_not(tab3[i])
            if pat1 not in layertokeep:          
                tab1[i]=np.bitwise_and(tab1[i],tab[i])
                tabroipat[pat1][i]= tab1[i]
            if pat2 not in layertokeep:
                tab2[i]=np.bitwise_and(tab2[i],tab[i])
                tabroipat[pat2][i]= tab2[i]
            nm=True
#            cv2.imshow('tab11',tab1[i])
#            cv2.imshow('tab21',tab2[i])
#            cv2.imshow('tab31',tab3[i])
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()

    if nm:
            npd=os.path.join(namedirtopcf,pat)
            remove_folder(npd)
            os.mkdir(npd)
          
            for i in range (0,slnt):
                 if tab3[i].max()>0:                 

                    naf3=tabscanName[i]+'.'+typei1
                    npdn3=os.path.join(npd,naf3)
                    imgc=colorimage(tab3[i],classifc[pat])
                    cv2.imwrite(npdn3,imgc)
                    
                    if pat2 not in layertokeep:
#                        print 'keptpat2',pat2
                        naf2=tabscanName[i]+'.'+typei1
                        npd2=os.path.join(namedirtopcf,pat2+'_m')
                        if not os.path.exists(npd2):
                            os.mkdir(npd2)
                        npdn2=os.path.join(npd2,naf2)
                        imgc=colorimage(tab2[i],classifc[pat2])
                        cv2.imwrite(npdn2,imgc)
#                    print 'pat1', pat1
                    if pat1 not in layertokeep:
#                        print 'keptpat1',pat1
                        naf1=tabscanName[i]+'.'+typei1
                        npd1=os.path.join(namedirtopcf,pat1+'_m')
                        if not os.path.exists(npd1):
#                            remove_folder(npd1)
                            os.mkdir(npd1)
                        npdn1=os.path.join(npd1,naf1)
                        imgc=colorimage(tab1[i],classifc[pat1])                   
                        cv2.imwrite(npdn1,imgc)
    return tab3


def genebackground(namedir,listroi):
    for sln in listroi:
        tabpbac=np.copy(tabslung[sln])
#        
        patok=False
        for pat in usedclassifall:
            if pat !=fidclass(0,classifall):
#                print sln,pat
                tabpat=tabroipat[pat][sln]

                if tabpat.max()>0:
                    patok=True
#                    tabp=cv2.cvtColor(tabpat,cv2.COLOR_BGR2GRAY)
                    np.putmask(tabpat,tabpat>0,255)
                    mask=np.bitwise_not(tabpat)
                    tabpbac=np.bitwise_and(tabpbac,mask)
#                    print tabroipat[fidclass(0,classif)][sln].shape
                    tabroipat[fidclass(0,classifall)][sln]=tabpbac

        if patok:
            labeldir=os.path.join(namedir,fidclass(0,classifall))
            if not os.path.exists(labeldir):
               os.mkdir(labeldir)
            namepat=tabscanName[sln]+'.'+typei1
            imgcoreScan=os.path.join(labeldir,namepat)
    #                imgcoreScan=os.path.join(locadir,namepat)
            tabtowrite=colorimage(tabroipat[fidclass(0,classifall)][sln],classifc[fidclass(0,classifall)])
#            tabtowrite=colorimage(tabroipat[fidclass(0,classifall)][sln],(100,100,100))

#            tabtowrite=cv2.cvtColor(tabtowrite,cv2.COLOR_BGR2RGB)
            cv2.imwrite(imgcoreScan,tabtowrite)    

#############################################################

listdirc= [ name for name in os.listdir(namedirtopc) if os.path.isdir(os.path.join(namedirtopc, name)) and \
            name not in alreadyDone and name not in reservedword]

print 'class used :',usedclassifall

for f in listdirc:
    print('work on:',f)
    errorfile = open(eferror, 'a')
    listroi=[]
    tn = datetime.datetime.now()
    todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
    errorfile.write('started ' +namedirHUG+' '+f+' at :'+todayn)
    errorfile.close()

    nbpf=0
    listsliceok=[]
    tabroipat={}
    namedirtopcf=os.path.join(namedirtopc,f)

    if os.path.isdir(namedirtopcf):
        sroidir=os.path.join(namedirtopcf,sroi)
        remove_folder(sroidir)
        os.mkdir(sroidir)

    dimtabxo,dimtabyo,dimtabx,dimtaby,slnt,fxs,listsln = genepara(namedirtopcf)

    tabsroi=np.zeros((slnt,dimtabx,dimtaby,3),np.uint8)
    tabscan =np.zeros((slnt,dimtabx,dimtaby),np.uint16)
    tabslung =np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    tabscanName={}

    tabscan,tabsroi,tabscanName,_,tabscanm1,tabscanp1=genebmp(namedirtopcf, source,tabscanName,fxs,listsln,listroi)

    if  os.path.exists(os.path.join(namedirtopcf, lungmask)):
        lugmaskt=lungmask
    elif  os.path.exists(os.path.join(namedirtopcf, lungmask1)):
            lugmaskt=lungmask1
    else:
        lugmaskt=lungmask
        os.mkdir(os.path.join(namedirtopcf, lungmask))
            
    tabslung,_,_,_,_,_=genebmp(namedirtopcf, lugmaskt,tabscanName,fxs,listsln,listroi)

    for i in usedclassifall:       
            tabroipat[i]=np.zeros((slnt,dimtabx,dimtaby),np.uint8)

    contenudir = [name for name in os.listdir(namedirtopcf) if name in usedclassifall and name not in derivedpatall]
    for i in contenudir:
            tabroipat[i],_,_,listroi,_,_=genebmp(namedirtopcf, i,tabscanName,fxs,listsln,listroi)

#    for i in derivedpatall:
    for i in  derivedpat:
#        i='HCpret'
        tabroipat[i]=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
        tabroipat[i]=calnewpat(namedirtopcf,i,slnt,dimtabx,dimtaby,tabscanName)

    contenudir = [name for name in os.listdir(namedirtopcf) if name in usedclassifall]
    for i in contenudir:
            nbp=pavs(namedirtopcf,i,slnt,dimtabx,dimtaby,tabscanName,listroi,tabscanm1,tabscan,tabscanp1)  
            nbpf=nbpf+nbp
    namenbpat=namedirHUG+'_nbpat_'+f+'.txt'
    ofilepw = open(os.path.join(jpegpath,namenbpat), 'w')
    ofilepw.write('number of patches: '+str(nbpf))
    ofilepw.close()
    errorfile = open(eferror, 'a')
    tn = datetime.datetime.now()
    todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
    errorfile.write('completed ' +f+' at :'+todayn)
    errorfile.close()
#

#################################################################
totalpat(jpegpath)
totalnbpat (patchtoppath,picklepathdir)
genelabelloc(patchtoppath,plabelfile,jpegpath)
##########################################################

errorfile = open(eferror, 'a')
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
errorfile.write('completed ' +namedirHUG+' '+subHUG+' at :'+todayn)
errorfile.write('---------------\n')
errorfile.close()
#print listslice
print('completed')