# coding: utf-8
#Sylvain Kritter 10 January 2017
""" generate predict and visu with lung in bmp file already there """
#general parameters and file, directory names"

import os
import cv2
import datetime
import time
import dicom
import scipy
import sys
import shutil
import matplotlib.pyplot as plt    
import numpy as np
#import Tkinter as Tk
import keras
import PIL
from PIL import Image as ImagePIL
from PIL import  ImageFont, ImageDraw 
import cPickle as pickle
import ild_helpers as H
import cnn_model as CNN4
from keras.models import model_from_json
from keras.models import load_model
from Tkinter import *
print keras.__version__


#global environment

picklefileglobal='MHKpredictglobaluip.pkl'
instdirMHK='MHKpredict'
workdiruser='Documents/boulot/startup/radiology/PREDICT'

tempofile=os.path.join(os.environ['TMP'],picklefileglobal)
workingdir= os.path.join(os.environ['USERPROFILE'],workdiruser)
instdir=os.path.join(os.environ['LOCALAPPDATA'],instdirMHK)

#
print 'instdir', instdir
#print 'workdir', workingfull

#########################################################
#define if the patch set is limited to the only pattrens, or all (set0)
#patchSet='set2'
#picklefile='pickle_sk32'
listset=['set0','set1','set2']
picklefile={}
dimpavxs={}
dimpavys={}

#pattern set definition
#ex16:32x32 no mean, no global cotrast, local contrast int keras 1.0
#keras 1.1:
#ex27: 16 with not centered data on mean no global
#ex28: 16 with centered data on mean 
#ex23 16 not centered
#ex30 16 centered, keras 1.1 No global contrast, internal contrast
#ex31: 16 Global Contrast cv2,no local contrast,mean
#ex32: 16 Global Contrast int,no local contrast
#ex33: 32 NO Global Contrast, local int contrast 
#ex34: 32 Global Contrast cv2, no local contrast 
#ex35: 32  Global Contrast cv2,  no local contrast , cmean defined by treaning
#ex371 32 global cont cv2, local contr int, no cmean, full depth
#ex37 32 global cont cv2, local contr int, no cmean, full depth Geneva only back_ground
#ex38 32 global cont cv2, local contr int, no cmean, full depth geneva + Gre
#ex40 16 global cont cv2, local contr int, no cmean, full depth, with back-ground
#ex41 16 global cont cv2, local contr int, no cmean, full depth, no back-ground
#ex42 16 global cont cv2, local contr int, no cmean, full depth, no back-ground, only GNB HC
#ex43 16 global cont cv2, No local contr int, no cmean, full depth, no back-ground, only GNB HC
#ex44 16 global cont cv2, local contr int, no cmean, full depth,  back-ground, only GNB HC
#ex45 16 global cont cv2, no local contr int, no cmean, full depth, back-ground, only GNB HC, 8 bits
#ex46 16 global cont int, No local contr int, no cmean, full depth,  back-ground, only GNB HC, 16 bits
#ex47 16 global cont int, local contr int, no cmean, full depth,  back-ground, limited subset HC, 16 bits
#ex54 16 global cont int, No local contr, no cmean, full depth,  back-ground, limited subset HC, 13 bits
#ex57 16 global cont int, No local contr, no cmean, full depth,  back-ground,only gre, 13 bits



globalHist=True  #use histogram equalization on full image
globalHistInternal=True  #use histogram equalization on full image with internal (True) or opencv

contrast = False # to enhance contrast on patch put True       
normiInternal=True #normalization internal procedure or openCV

thrpatch = 0.9             # threshold for patch acceptance in area ratio
cMean=False # if data are centered on mean on the full set
trainingMean=0.371#for HUG 32 pixels
usetrainingMean=True # if cMean then  training mean or actual mean is used
useFullImageDepth=True #to use 255 or 128 (from ex35)
#imageDepth=255 #number of bits used on dicom images (2 **n)
#imageDepth=65535 #number of bits used on dicom images (2 **n) 16 bits
imageDepth=8191 #number of bits used on dicom images (2 **n) 13 bits
wbg = True                # with or without bg (true if with back_ground)
gp='pickle_ex54'
dx=16

threeFileTxt='uip.txt'
threeFile='uip.html'
threeFilejs='world.js'
threeFileTop='uiptop.html'
threeFileBot='uipbot.html'

#pset==0: 'consolidation', 'HC','ground_glass', 'micronodules', 'reticulation'
picklefile['set0']=gp
dimpavxs['set0'] =dx
dimpavys['set0'] = dx    

picklefile['set1']=gp
dimpavxs['set1'] =dx
dimpavys['set1'] = dx

picklefile['set2']=gp
dimpavxs['set2'] =dx
dimpavys['set2'] = dx

#subdirectory name to colect pkl filesfor lung  prediction
picklefilelung='pickle_sk8_lung'
#patch size in pixels for lung 32 * 32
lungdimpavx =15
lungdimpavy = 15

#parameters for tools


thrprobaUIP=0.5 #probability to take into account patches for UI
thrproba=0.5 # #probability to take into account patches for visu
avgPixelSpacing=0.734   # average pixel spacing
subErosion= 15 #erosion factor for subpleura in mm


#Specific for Grenoble images
rotScan180=False
#sepIsDash=True
Double =False
#to enhance contrast on lung patch for predictionput True
contrastScanLung=False
#normalization internal procedure or openCV
normiInternalLung=False
#probability for lung acceptance
thrlung=0.7

##################################
#subsample by default
subsdef=10
#name of directory to store background
vbg='A'
#workingdirectory='C:\Users\sylvain\Documents\boulot\startup\radiology\PREDICT'
#installdirectory='C:\Users\sylvain\Documents\boulot\startup\radiology\UIP\python'
#global directory for predict file
namedirtop = 'predict_essai'

#directory for storing image out after prediction
predictout='predicted_results'

#directory with lung mask dicom
lungmask='lung_mask'

#directory to put  lung mask bmp
lungmaskbmp='bmp'

#directory name with scan with roi
sroi='sroi'

#subdirectory name to put images
jpegpath = 'patch_jpeg'

#directory with bmp from dicom
scanbmp='scan_bmp'
scanbmpbmp='scan_bmpbmp'

Xprepkl='X_predict.pkl'
Xrefpkl='X_file_reference.pkl'

lungXprepkl='lung_X_predict.pkl'
lungXrefpkl='lung_X_file_reference.pkl'

#file to store different parameters
subsamplef='subsample.pkl'

#subdirectory name to colect pkl files for prediction
modelname= 'ILD_CNN_model.h5'

# list label not to visualize
#excluvisu=['back_ground','healthy'
#            ,'micronodules',
#            'consolidation',
#            'HC',
#            'ground_glass',
#            'healthy',
#            'micronodules',
#            'reticulation',
#            'air_trapping',            
#             'bronchial_wall_thickening',
#             'early_fibrosis',
#             'emphysema',
#             'increased_attenuation',
#             'macronodules',
#             'pcp',
#             'peripheral_micronodules',
#             'tuberculosis'     
#                 ]
excluvisu=['back_ground','healthy']

#dataset supposed to be healthy
datahealthy=['138']

#image  patch format
typeiroi='bmp' 
typei='bmp'
typeid='png' #can be jpg for 16 bits


#########################################################################
cwd=os.getcwd()
glovalf=tempofile
path_patient = os.path.join(workingdir,namedirtop)
varglobal=(thrpatch,thrproba,path_patient,subsdef)

def setva():
        global varglobal,thrproba,thrpatch,subsdef,path_patient
        if not os.path.exists(glovalf) :
            pickle.dump(varglobal, open( glovalf, "wb" ))
        else:
            dd = open(glovalf,'rb')
            my_depickler = pickle.Unpickler(dd)
            varglobal = my_depickler.load()
            dd.close() 
#            print varglobal
            path_patient=varglobal[2]
#            print path_patient
            thrproba=varglobal[1]
            print 'thrproba setva', thrproba
            thrpatch=varglobal[0]
            
            subsdef=varglobal[3]



def newva():
 pickle.dump(varglobal, open( glovalf, "wb" ))

(cwdtop,tail)=os.path.split(cwd)

#if not os.path.exists(path_patient):
#    print 'patient directory does not exists'
#    sys.exit()

setva()   
picklein_file={}

for setn in listset:
    picklein_file[setn] = os.path.join(instdir,picklefile[setn])
#    print picklein_file[setn]
    if not os.path.exists(picklein_file[setn]):
        
        print 'model and weight directory does not exists for: ',setn
        sys.exit()
    lpck= os.listdir(picklein_file[setn])
    ph5=False
#    print lpck
    for l in lpck:
        if l.find('.h5',0)>0 or l.find('.json',0)>0:
            ph5=True
    if not(ph5):
        print 'model and/or weight files does not exists for : ',setn
        sys.exit()     

picklein_lung_file = os.path.join(instdir,picklefilelung)
if not os.path.exists(picklein_lung_file):
    print 'model and weight directory for lung does not exists'
    sys.exit()
lpck= os.listdir(picklein_lung_file)
pson=False
pweigh=False
for l in lpck:
    if l.find('.json',0)>0:
        pson=True
    if l.find('ILD_CNN_model_weights',0)==0:

        pweigh=True
if not(pweigh and pson):
    print 'model and/or weight files for lung does not exists'
    sys.exit()     

pxys={}
for nset in listset:
    pxys[nset]=float(dimpavxs[nset]*dimpavys[nset])

#end general part
#font file imported in top directory
font20 = ImageFont.truetype( 'arial.ttf', 20)
font10 = ImageFont.truetype( 'arial.ttf', 10)
#########################################################
#print path_patient
#if os.path.exists(path_patient):
#    errorfile = open(path_patient+'/predictlog.txt', 'w') 

#color of labels
black=(0,0,0)
red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255,255,0)
cyan=(0,255,255)
purple=(255,0,255)
white=(255,255,255)
darkgreen=(11,123,96)
pink =(255,128,150)
lightgreen=(125,237,125)
orange=(255,153,102)
lowgreen=(0,51,51)

classiflung={
    'nolung':0,
    'lung':1,}
    
classifclung ={
'nolung':red,
'lung':white
 }
# color for TREE JS 
volcol={          
'back_ground':'boxMaterialGrey',
#'consolidation':'boxMaterialRed',
'consolidation':'boxMaterialCyan',
'HC':'boxMaterialBlue',
#'ground_glass':'boxMaterialYellow',
'ground_glass':'boxMaterialRed',
'healthy':'boxMaterialGrey',
'micronodules':'boxMaterialGreen',
#'reticulation':'boxMaterialPurple',
'reticulation':'boxMaterialYellow',
'air_trapping':'boxMaterialPink',
'cysts':'boxMaterialLightgreen',
 'bronchiectasis':'boxMaterialOrange',
 'bronchial_wall_thickening':'boxMaterialWhite',
 'early_fibrosis':'boxMaterialWhite',
 'emphysema':'boxMaterialWhite',
 'increased_attenuation':'boxMaterialWhite',
 'macronodules':'boxMaterialWhite',
 'pcp':'boxMaterialWhite',
 'peripheral_micronodules':'boxMaterialWhite',
 'tuberculosis':'boxMaterialWhite'}            
             
 
#only label we consider, number will start at 0 anyway
 
 
classifSet0bg={
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
             'bronchial_wall_thickening':10,
             'early_fibrosis':11,
             'emphysema':12,
             'increased_attenuation':13,
             'macronodules':14,
             'pcp':15,
             'peripheral_micronodules':16,
             'tuberculosis':17      
                 }
       
classifSet1bg={
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
             'bronchial_wall_thickening':10,
             'early_fibrosis':11,
             'emphysema':12,
             'increased_attenuation':13,
             'macronodules':14,
             'pcp':15,
             'peripheral_micronodules':16,
             'tuberculosis':17                      
            }
classifSet2bg={
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
             'bronchial_wall_thickening':10,
             'early_fibrosis':11,
             'emphysema':12,
             'increased_attenuation':13,
             'macronodules':14,
             'pcp':15,
             'peripheral_micronodules':16,
             'tuberculosis':17      
                 }
classifSet3bg={
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
             'bronchial_wall_thickening':10,
             'early_fibrosis':11,
             'emphysema':12,
             'increased_attenuation':13,
             'macronodules':14,
             'pcp':15,
             'peripheral_micronodules':16,
             'tuberculosis':17      
                 }  
       
classifSet0Nbg={
            'consolidation':0,
            'HC':1,
            'ground_glass':2,
            'healthy':3,
            'micronodules':4,
            'reticulation':5,
            'air_trapping':6,
            'cysts':7,
            'bronchiectasis':8,            
             'bronchial_wall_thickening':9,
             'early_fibrosis':10,
             'emphysema':11,
             'increased_attenuation':12,
             'macronodules':13,
             'pcp':14,
             'peripheral_micronodules':15,
             'tuberculosis':16      
                 }
       
classifSet1Nbg={
           'consolidation':0,
            'HC':1,
            'ground_glass':2,
            'healthy':3,
            'micronodules':4,
            'reticulation':5,
            'air_trapping':6,
            'cysts':7,
            'bronchiectasis':8,            
             'bronchial_wall_thickening':9,
             'early_fibrosis':10,
             'emphysema':11,
             'increased_attenuation':12,
             'macronodules':13,
             'pcp':14,
             'peripheral_micronodules':15,
             'tuberculosis':16      
                 }
classifSet2Nbg={
            'consolidation':0,
            'HC':1,
            'ground_glass':2,
            'healthy':3,
            'micronodules':4,
            'reticulation':5,
            'air_trapping':6,
            'cysts':7,
            'bronchiectasis':8,            
             'bronchial_wall_thickening':9,
             'early_fibrosis':10,
             'emphysema':11,
             'increased_attenuation':12,
             'macronodules':13,
             'pcp':14,
             'peripheral_micronodules':15,
             'tuberculosis':16      
            }
classifSet3Nbg={
            'consolidation':0,
            'HC':1,
            'ground_glass':2,
            'healthy':3,
            'micronodules':4,
            'reticulation':5,
            'air_trapping':6,
            'cysts':7,
            'bronchiectasis':8,            
             'bronchial_wall_thickening':9,
             'early_fibrosis':10,
             'emphysema':11,
             'increased_attenuation':12,
             'macronodules':13,
             'pcp':14,
             'peripheral_micronodules':15,
             'tuberculosis':16     
             }
classif={}                    
if wbg:
    print 'with back-ground'
    classif['set0']=classifSet0bg
    classif['set1']=classifSet1bg 
    classif['set2']=classifSet2bg
    classif['set3']=classifSet3bg
else:
    classif['set0']=classifSet0Nbg
    classif['set1']=classifSet1Nbg
    classif['set2']=classifSet2Nbg
    classif['set3']=classifSet3Nbg                                     
    

classifc ={
'back_ground':darkgreen,
#'consolidation':red,
'consolidation':cyan,
'HC':blue,
'ground_glass':red,
#'ground_glass':yellow,
'healthy':darkgreen,
#'micronodules':cyan,
'micronodules':green,
#'reticulation':purple,
'reticulation':yellow,
'air_trapping':pink,
'cysts':lightgreen,
 'bronchiectasis':orange,
 'nolung': lowgreen,
 'bronchial_wall_thickening':white,
 'early_fibrosis':white,
 'emphysema':white,
 'increased_attenuation':white,
 'macronodules':white,
 'pcp':white,
 'peripheral_micronodules':white,
 'tuberculosis':white
 }


def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path,ignore_errors=True)

def rsliceNum(s,c,e):
    ''' look for  afile according to slice number'''
    #s: file name, c: delimiter for snumber, e: end of file extension
    endnumslice=s.find(e)
    posend=endnumslice
    while s.find(c,posend)==-1:
        posend-=1
    debnumslice=posend+1
    return int((s[debnumslice:endnumslice])) 
    
def rotateImage(image, angle):
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image    
    
def genebmp(dirName,fn,subs):
    """generate patches from dicom files"""
    global dimtabx, dimtaby,slicepitch,fxsref,errorfile,fxslungref
    print ('load dicom files in :',dirName, 'scan name:',fn)
    #directory for patches
    bmp_dir = os.path.join(dirName, scanbmp)
    bmp_bmp_dir = os.path.join(dirName, scanbmpbmp)

    FilesDCM =(os.path.join(dirName,fn))  
#           
    RefDs = dicom.read_file(FilesDCM)
#    print RefDs
    SliceThickness=RefDs.SliceThickness
    try:
        SliceSpacingB=RefDs. SpacingBetweenSlices
    except AttributeError:
         print "Oops! No Slice spacing..."
         SliceSpacingB=0

    slicepitch=float(SliceThickness)+float(SliceSpacingB)
#    print SliceThickness
#    print SliceSpacingB
 
    print 'slice pitch in z :',slicepitch
  
    dsr= RefDs.pixel_array
#    min_val=np.min(dsr)
#    max_val=np.max(dsr)
#    print 'dsr orig',min_val, max_val
            #scale the dicom pixel range to be in 0-255
    dsr= dsr-dsr.min()
#    c=float(imageDepth)/dsr.max()
#    dsr=dsr*c
    if imageDepth <256:
        dsr=dsr.astype('uint8')
    else:
        dsr=dsr.astype('uint16')
#    min_val=np.min(dsr)
#    max_val=np.max(dsr)
#    print 'dsr',min_val, max_val

    #resize the dicom to have always the same pixel/mm
    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing  
#    print 'fxs',fxs
    if fxsref==0:
        fxsref=fxs
    else:
        if fxs!=fxsref:
            print 'ERROR!'
            errorfile.write('ERROR fxs: '+str(fxs)+' scan name '+fn+ ' dirname '+dirName)
            errorfile.close()
            sys.exit()
    
    slicenumber=int(RefDs.InstanceNumber)
    endnumslice=fn.find('.dcm')
    if imageDepth <256:
        imgcore=fn[0:endnumslice]+'_'+str(slicenumber)+'.'+typei
    else:
        imgcore=fn[0:endnumslice]+'_'+str(slicenumber)+'.'+typeid   
    imgcorebmp=fn[0:endnumslice]+'_'+str(slicenumber)+'.'+typei  
    bmpfile=os.path.join(bmp_dir,imgcore)
    bmp_bmp_file=os.path.join(bmp_bmp_dir,imgcorebmp)
    imgresize1=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
#    imgresize1= scipy.misc.imresize(dsr,fxs,interp='bicubic',mode=None)
    if globalHist:
                if globalHistInternal:
                    imgresize = normi(imgresize1) 
#                    min_val=np.min(imgresize)
#                    max_val=np.max(imgresize)
#                    print 'imgresize',min_val, max_val
#                    ooo
#                    min_val=np.min(imgresize)
#                    max_val=np.max(imgresize)
#                    print 'imgresize',min_val, max_val
#                    ooo
                else:
                    if imageDepth <256:
                        imgresize = cv2.equalizeHist(imgresize1) 
                    else:
                        imgresize= cv2.normalize(imgresize1, 0, imageDepth,cv2.NORM_MINMAX);
    else:
                imgresize=imgresize1    
    
    
#    if globalHist:
#            imgresize = cv2.equalizeHist(imgresize1) 
#    else:
#            imgresize=imgresize1
    if rotScan180:
        imgresize=rotateImage(imgresize, 180)
        
   
    if imageDepth <256:
        scipy.misc.imsave(bmpfile,imgresize)
        scipy.misc.imsave(bmp_bmp_file,imgresize)
    else:
        cv2.imwrite (bmpfile, imgresize,[int(cv2.IMWRITE_PNG_COMPRESSION),0]) 
        scipy.misc.imsave(bmp_bmp_file,imgresize)
#    min_val=np.min(imgresize)
#    max_val=np.max(imgresize)
#    print 'imgresize',min_val, max_val
#    ooo
    dimtabx=imgresize.shape[0]
    dimtaby=imgresize.shape[1]
#    print dimtabx
    lung_dir = os.path.join(dirName, lungmask)
    
    lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)
    
    lung_bmp_listr = os.listdir(lung_bmp_dir)
    lungFound=False
    for llung in  lung_bmp_listr:
         if '.'+typei in llung.lower():
             slicescan=rsliceNum(llung,'_','.'+typei)   
             if Double :
                 slicescan=slicescan*2

             if slicescan== slicenumber:
                lungFound=True
                break
             
    if not lungFound:            
        lunglist = [name for name in os.listdir(lung_dir) if ".dcm" in name.lower()]
#        print('lungdir',lung_dir)
        for l in lunglist:
#                print(l)

                lungfile=os.path.join(lung_dir,l)
                RefDslung = dicom.read_file(lungfile)
                slicescan=int(RefDslung.InstanceNumber)
                endnumslice=l.find('.dcm')
                imgcorescan=l[0:endnumslice]+'_'+str(slicescan)+'.'+typei  

                if Double :
                    slicescan=slicescan*2
                    
#                        print slicescan
                if slicescan== slicenumber:                                            
                    lungFound=True
                    
#                    RefDslung = dicom.read_file(lungfile)
                    dsrLung= RefDslung.pixel_array
                    dsrLung= dsrLung-dsrLung.min()
                    c=255.0/dsrLung.max()
                    dsrLung=dsrLung*c
                    dsrLung=dsrLung.astype('uint8')  
                    fxslung=float(RefDslung.PixelSpacing[0])/avgPixelSpacing
#                    print 'fxslung',fxslung
                   
                    if fxslungref==0:
                       fxslungref=fxslung                    
                    else :
                        if fxslungref!=fxslung:
                            print 'ERROR fxslung: ',str(fxslung),' scan number:'+str(slicescan),' scan name',fn, ' dirname ',dirName
                            errorfile.write('ERROR fxslung: '+str(fxslung)+' scan number:'+str(slicescan)+' scan name'+fn+ ' dirname '+dirName)
                            errorfile.close() 
                            sys.exit()
                    listlungdict=cv2.resize(dsrLung,None,fx=fxslung,fy=fxslung,interpolation=cv2.INTER_LINEAR)
#                    listlungdict= scipy.misc.imresize(dsrLung,fxslung,interp='bicubic',mode=None) 
#                    print imgresize.shape,listlungdict.shape
                    lungcoref=os.path.join(lung_bmp_dir,imgcorescan)
                    scipy.misc.imsave(lungcoref,listlungdict)
#                    print 'aaa'
#                    scipy.misc.imsave('b.bmp',listlungdict)
#                     cv2.imshow('scan',imgt) 
#                     cv2.imshow('lung',listlungdict) 
#                     cv2.waitKey(0)    
#                     cv2.destroyAllWindows()
    
                    break
            
    if  not lungFound:
        print 'lung not found'
        generatelungmask(dirName,imgcore,slicenumber)

def normiold(tabi):
     """ normalise patches 0 255"""
     tabi1=tabi-tabi.min()
     max_val=np.max(tabi1)
#     print 'tabi1',min_val, max_val,imageDepth/float(max_val)
     tabi2=tabi1*(imageDepth/float(max_val))
     if imageDepth<256:
         tabi2=tabi2.astype('uint8')
     else:
         tabi2=tabi2.astype('uint16')

     return tabi2

def normi(tabi):
     """ normalise patches 0 255"""
    
     max_val=float(np.max(tabi))
     min_val=float(np.min(tabi))
     
#     print 'tabi1',min_val, max_val,imageDepth/float(max_val)
     tabi2=(tabi-min_val)*(imageDepth/(max_val-min_val))
     if imageDepth<256:
         tabi2=tabi2.astype('uint8')
     else:
         tabi2=tabi2.astype('uint16')

     return tabi2

def lungpredict(dn,nd,sln):

        global patch_list_lung
        dataset_list_lung=[]
        nameset_list_lung=[]
        (top,tail)=os.path.split(dn)
        for fil in patch_list_lung:
            if fil[0]==tail:
               
                dataset_list_lung.append(fil[4])
                nameset_list_lung.append(fil[0:3])
        
        X = np.array(dataset_list_lung)
#        print X[1].shape
        X_predict1 = np.asarray(np.expand_dims(X,1))/float(128)   
        if cMean:
            m=np.mean(X_predict1)
#            print 'mean of Xtrain :',m
            X_predict=X_predict1-m
        else:
            X_predict=X_predict1
        
        jsonf= os.path.join(picklein_lung_file,'ILD_CNN_model.json')
#        print jsonf
        weigf= os.path.join(picklein_lung_file,'ILD_CNN_model_weights')
#        print weigf
#model and weights fr CNN
        args  = H.parse_args()                          
        train_params = {
     'do' : float(args.do) if args.do else 0.5,        
     'a'  : float(args.a) if args.a else 0.3,          # Conv Layers LeakyReLU alpha param [if alpha set to 0 LeakyReLU is equivalent with ReLU]
     'k'  : int(args.k) if args.k else 4,              # Feature maps k multiplier
     's'  : float(args.s) if args.s else 1,            # Input Image rescale factor
     'pf' : float(args.pf) if args.pf else 1,          # Percentage of the pooling layer: [0,1]
     'pt' : args.pt if args.pt else 'Avg',             # Pooling type: Avg, Max
     'fp' : args.fp if args.fp else 'proportional',    # Feature maps policy: proportional, static
     'cl' : int(args.cl) if args.cl else 5,            # Number of Convolutional Layers
     'opt': args.opt if args.opt else 'Adam',          # Optimizer: SGD, Adagrad, Adam
     'obj': args.obj if args.obj else 'ce',            # Minimization Objective: mse, ce
     'patience' : args.pat if args.pat else 5,         # Patience parameter for early stoping
     'tolerance': args.tol if args.tol else 1.005,     # Tolerance parameter for early stoping [default: 1.005, checks if > 0.5%]
     'res_alias': args.csv if args.csv else 'res'      # csv results filename alias
         }
#        model = H.load_model()

        model = model_from_json(open(jsonf).read())
        model.load_weights(weigf)

        model.compile(optimizer='Adam', loss=CNN4.get_Obj(train_params['obj']))        

        proba = model.predict_proba(X_predict, batch_size=100)
        picklefileout_f_dir = os.path.join( dn,picklefilelung)
        xfp=os.path.join(picklefileout_f_dir,lungXprepkl)
        pickle.dump(proba, open( xfp, "wb" ))
        xfpr=os.path.join(picklefileout_f_dir,lungXrefpkl)
        pickle.dump(patch_list_lung, open( xfpr, "wb" ))

def pavgenelung(dn,nd,sln):
    print('generate pavement  on: ',nd, 'slice:',sln)
    global patch_list_lung
    (dptop,dptail)=os.path.split(dn)
    bmpdir = os.path.join(dn,scanbmp)
    jpegpathf=os.path.join(dn,jpegpath)
#    listbmp= os.listdir(bmpdir)
#    for img in listbmp:
    bmpfile = os.path.join(bmpdir,nd)
#    print bmpfile
    tabf = cv2.imread(bmpfile,1)
#             im = cv2.imread(bmpfile,1)
    im = ImagePIL.open(bmpfile)
#        tabim=np.array(im)
#
    atabf = np.nonzero(im)
            #tab[y][x]  convention
    xmin=atabf[1].min()
    xmax=atabf[1].max()-lungdimpavx
    ymin=atabf[0].min()
    ymax=atabf[0].max()-lungdimpavy
#    print xmin,xmax,ymin,ymax           
    i=xmin
    while i <= xmax:
             j=ymin
    #        j=maxj
             while j<=ymax:
                    crorig = im.crop((i, j, i+lungdimpavx, j+lungdimpavy))
                    imagemax=crorig.getbbox()
                                            
                    min_val=np.min(crorig)
                    max_val=np.max(crorig)
#                    print imagemax,min_val,max_val
                    if imagemax!=None and max_val-min_val>10:                                  
                        imgray =np.array(crorig)
#                        imgray = cv2.cvtColor(imgra,cv2.COLOR_BGR2GRAY)                           
                        if contrastScanLung:
                             if normiInternalLung:
                                 tabi2=normi(imgray)
                             else:
                                 tabi2 = cv2.equalizeHist(imgray)                                        
                                 patch_list_lung.append((dptail,sln,i,j,tabi2))
#                                    n+=1
                        else:
                            patch_list_lung.append((dptail,sln,i,j,imgray))
                        x=0
                        while x < lungdimpavx:
                            y=0
                            while y < lungdimpavy:
                                tabf[y+j][x+i]=[255,0,0]
                                if x == 0 or x == lungdimpavx-1 :
                                    y+=1
                                else:
                                    y+=lungdimpavy-1
                            x+=1                                             
                        
                    j+=lungdimpavy

             i+=lungdimpavx

    scipy.misc.imsave(jpegpathf+'/'+'l_'+str(sln)+'.'+typei, tabf)
# 
        
def addpatch(col,lab, xt,yt,px,py):
    imgi = np.zeros((dimtabx,dimtaby,3), np.uint8)
    tablint=[(xt,yt),(xt,yt+py),(xt+px,yt+py),(xt+px,yt)]
    tabtxt=np.asarray(tablint)
    cv2.polylines(imgi,[tabtxt],True,col)
    cv2.fillPoly(imgi,[tabtxt],col)
    return imgi        
        
        
        
def  genelung(dn,nd,sln):
#    print('generate lung bmp  on: ',nd, 'slice:',sln)
    (preprob,listnamepatch)=loadpkllung(dn)
    dirpatientfdb=os.path.join(dn,scanbmp)

    lung_dir = os.path.join(dn, lungmask)              
    lung_dir_bmp=os.path.join(lung_dir, lungmaskbmp)
    
    predictout_dir = os.path.join(dn,predictout)

#            print img
    imgt = np.zeros((dimtabx,dimtaby,3), np.uint8)
#            listlabel={}
    imgc=os.path.join(dirpatientfdb,nd)    
    endnumslice=nd.find('.'+typei)
    imgcore=nd[0:endnumslice]
#            
#    #        print imgcore
    posend=endnumslice
    while nd.find('-',posend)==-1:
                posend-=1
    imgcorwosln=nd[0:posend]              
    imscan = ImagePIL.open(imgc)
    imscanc= imscan.convert('RGB')
    tablscan = np.array(imscanc)

    ill = 0

    for ll in listnamepatch:
                    slicename=ll[0]    
                    proba=preprob[ill]          
                    prec, mprobai = maxproba(proba)
                    classlabel=fidclass(prec,classiflung) 
                    classcolor=classifclung[classlabel]
                    xpat=ll[1]
                    ypat=ll[2]
        #            print slicenumber, slicename,dptail
                    if slicename == sln: 
                        if mprobai> thrlung and classlabel=='lung':
#                                print classcolor,classlabel
                                imgi=addpatch(classcolor,classlabel,xpat,ypat,lungdimpavx,lungdimpavy)
                                imgt=cv2.add(imgt,imgi)
        
                    ill+=1
  

    predscan=cv2.add(tablscan,imgt)

    kernele=np.ones((16,16),np.uint8)
    kernelc=np.ones((3,3),np.uint8)
    kerneld=np.ones((19,19),np.uint8)

        
    erosion = cv2.erode(imgt,kernele,iterations = 1)             
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernelc)                
    dilation = cv2.dilate(opening,kerneld,iterations = 1)
#    cv2.imshow('image',imgt) 
#    cv2.imshow('imageo',tablscan) 
#    cv2.imshow('erosion',erosion) 
#
#    cv2.imshow('opening',opening) 
#
#    cv2.imshow('dilatation',dilation) 
#    cv2.waitKey(0)    
#    cv2.destroyAllWindows()
    
    imgcorefull=imgcore+'.'+typei
    imgname=os.path.join(predictout_dir,'lung_'+imgcorefull)    
#    print 'imgname',imgname
    lungname= 'agene_lung'+imgcorwosln+'_'+str(sln)+'.'+typei
    
    imgnamelung=os.path.join(lung_dir_bmp,lungname)
#    print imgname
    cv2.imwrite(imgname,predscan)
#    cv2.imwrite(imgnamelung,imgt)
    cv2.imwrite(imgnamelung,dilation)


def generatelungmask(dn,nd,sln):
     print'generate lung mask',' top:',dn,' slicename:',nd,' slicenumber:',  sln
     """ generate lung mask"""
     global patch_list_lung,classdirec
#     print('generate patches on: ',nd, 'slice:',sln) 
     patch_list_lung=[]
     pavgenelung(dn,nd,sln)
     lungpredict(dn,nd,sln)
     genelung(dn,nd,sln)
    


def pavgene (namedirtopcf,nset):
        """ generate patches from scan"""
        patch_list=[]
        print('generate patches on: ',namedirtopcf)
        (dptop,dptail)=os.path.split(namedirtopcf)
        namemask1=os.path.join(namedirtopcf,lungmask)
        namemask=os.path.join(namemask1,lungmaskbmp)
#        print namemask
        bmpdir = os.path.join(namedirtopcf,scanbmp)
        bmp_bmp_dir = os.path.join(namedirtopcf,scanbmpbmp)
        jpegpathf=os.path.join(namedirtopcf,jpegpath)
        
        listbmp= os.listdir(bmpdir)
#        print(listbmp)
        
        listlungbmp= os.listdir(namemask)    
        dimpavx=dimpavxs[nset]
        dimpavy=dimpavys[nset]
        pxy=pxys[nset]
        
        for img in listbmp:
#             print img
#             tflung=False
#             tab2w = np.zeros((dimtabx,dimtaby), np.uint8)
             tabfw = np.zeros((dimtabx,dimtaby,3), np.uint8)
             if imageDepth<256:
                slicenumber= rsliceNum(img,'_','.'+typei)
             else:
                slicenumber= rsliceNum(img,'_','.'+typeid)
             endname=img.find('.'+typeid)
             imgbmp=img[0:endname]+'.'+typei
             

             if listlungbmp!=[]:
#                    print(listlungbmp)
                    for llung in listlungbmp:
                        
#                       print llung
                       
                       slicelung=rsliceNum(llung,'_','.'+typei)
                       if Double :
                               slicelung=slicelung*2
                      
                       if slicelung == slicenumber:
                       
                        
                            lungfile = os.path.join(namemask,llung)
        #                    print(lungfile)
                            tablung = cv2.imread(lungfile,0)
#                            tablungcopy=np.copy(tablung)
#                            np.putmask(tablungcopy,tablungcopy==1,0)
#                            np.putmask(tablungcopy,tablungcopy>0,255)
                            
                            np.putmask(tablung,tablung==1,0)
                            np.putmask(tablung,tablung>0,1)

             bmpfile = os.path.join(bmpdir,img)
#             tabf = cv2.imread(bmpfile,1)
             im = ImagePIL.open(bmpfile)
#             im=cv2.imread(bmpfile,-1)
             tabf=np.array(im)
             
             bmp_bmpfile = os.path.join(bmp_bmp_dir,imgbmp)
#             tabf = cv2.imread(bmpfile,1)
             imbmp = ImagePIL.open(bmp_bmpfile).convert('L')
             tabfbmp=np.array(imbmp)
             tabfrgb=cv2.cvtColor(tabfbmp,cv2.COLOR_GRAY2BGR)
#             tabfrgb=cv2.cvtColor(tablungcopy,cv2.COLOR_GRAY2BGR)
#             cv2.imshow('image',tablung) 
#             cv2.waitKey(0)
             nz= np.count_nonzero(tablung)
             if nz>0:
            
                atabf = np.nonzero(tablung)
                #tab[y][x]  convention
                xmin=atabf[1].min()
                xmax=atabf[1].max()
                ymin=atabf[0].min()
                ymax=atabf[0].max()
               
             else:
                xmin=0
                xmax=0
                ymin=0
                ymax=0
                
             i=xmin
#             print xmin,xmax,ymin,ymax
#             ooo
             while i <= xmax:
                 j=ymin
        #        j=maxj
                 while j<=ymax:
#                     print(i,j)
                     tabpatch=tablung[j:j+dimpavy,i:i+dimpavx]
#                     t=np.copy(tabpatch)
                     area= tabpatch.sum()  
#                     print tabpatch.shape
#                     print(i,j,area)
                     
#                     np.putmask(t,t>0,255)
#                    check if area above threshold

                     targ=float(area)/pxy
#                     if i>400 and j >250 and i <410 and j<280 and targ>0.8:
#                         print i,j,targ,area
                     if targ>thrpatch:
#                        print i,j,targ,area
#                        ooo
                        imgray = tabf[j:j+dimpavy,i:i+dimpavx]
                        if imageDepth<256:
                            imgray=imgray.astype('uint8')  
                        else:                                        
                            imgray=imgray.astype('uint16')
                        
#                        if i ==136 and j==254:
#                            print i,j,targ,area
#                            tab2w[j:j+dimpavy,i:i+dimpavx]=t
#                            scipy.misc.imsave('a.bmp',tab2w)
#                        imgray =np.array(crorig)
#                        imgray = cv2.cvtColor(imgra,cv2.COLOR_BGR2GRAY)

                        imagemax= cv2.countNonZero(imgray)
                        min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(imgray)
#                        print '1', min_val,max_val,min_loc,max_loc
#                        ooo
             
#                        if imagemax > 0 and max_val - min_val>10:
            
                        if imagemax > 0 and min_val != max_val:
#                            namepatch=patchpathf+'/p_'+str(slicenumber)+'_'+str(i)+'_'+str(j)+'.'+typei
                            if contrast:
                                    if normiInternal:    
                                        tabi2=normi(imgray)
 
                                    else:
                                        if imageDepth<256:
                                            tabi2 = cv2.equalizeHist(imgray)     
                                        else:
#                                           tabi2=normi(imgray)
                                           tabi2= cv2.normalize(imgray, 0, imageDepth,cv2.NORM_MINMAX);
                    
#                                    print tabi2
#                                    if PcMean:
#                                            tabi2=tabi2.astype('float64')
#                                            tabi2 -=np.mean(tabi2)
#                                            min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(tabi2)
#                                            print '2', min_val,max_val,min_loc,max_loc
#                                            tabi2 /= np.std(tabi2)
#                                            min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(tabi2)
#                                            print '3', min_val,max_val,min_loc,max_loc
                                    patch_list.append((dptail,slicenumber,i,j,tabi2))
#                                    min_val=np.min(tabi2)
#                                    max_val=np.max(tabi2)
#                                    print 'patch norm', min_val, max_val
                                  
#                                    n+=1
                            else:
#                                imgray -=np.mean(imgray)
#                                if PcMean:
#                                    imgray=imgray.astype('float64')
#                                    imgray -=np.mean(imgray)
#                                    imgray /= np.std(imgray)
                                  
                                patch_list.append((dptail,slicenumber,i,j,imgray))
#                                min_val=np.min(imgray)
#                                max_val=np.max(imgray)
#                                print 'patch no norm', min_val, max_val
#                            ooo
                            tablung[j:j+dimpavy,i:i+dimpavx]=0
                            x=0
                            while x < dimpavx:
                                y=0
                                while y < dimpavy:
                                    if y+j<dimtaby and x+i<dimtabx:
                                        tabfw[y+j][x+i]=[255,0,0]
#                                        tabfw[y+j][x+i]=255

                                    if x == 0 or x == dimpavx-1 :
                                        y+=1
                                    else:
                                        y+=dimpavy-1
                                x+=1                                             
                            j+=dimpavy-1
#                     j+=dimpavy
                     j+=1
#                 i+=dimpavx
                 i+=1
#        print namedirtopcf,n
             
             scipy.misc.imsave(jpegpathf+'/'+'s_'+str(slicenumber)+'.'+typei, cv2.add(tabfw,tabfrgb))
#             scipy.misc.imsave(jpegpathf+'/'+'s_'+str(slicenumber)+'.'+typei, cv2.add(tabfw,tablung))

        return patch_list
# 
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
        
def ILDCNNpredict(patient_dir_s,setn,patch_list):     
        
#        print ('predict patches on: ',patient_dir_s) 
        (top,tail)=os.path.split(patient_dir_s)
        print ('predict patches on: ',tail, 'set:',setn) 
        dataset_list=[]
        nameset_list=[]
 
        for fil in patch_list:
            if fil[0]==tail:
               
                dataset_list.append(fil[4])
                nameset_list.append(fil[0:3])
        
        X0=len(dataset_list) 
      
#        for i in range (0,X0-1):
#            if dataset_list[i].shape[0] !=16 or dataset_list[i].shape[1] !=16:
#                print dataset_list[i].shape
        if useFullImageDepth:
            X_predict1 =np.expand_dims(dataset_list,1)/float(imageDepth)
#            print 'full'
        else:            
#            X_predict1 =np.expand_dims(dataset_list,1)
#        else:            
            X_predict1 =np.expand_dims(dataset_list,1)/float(imageDepth)/2  

        if cMean:
            m=np.mean(X_predict1)
            if usetrainingMean:
                X_predict=X_predict1-trainingMean
                print 'per training mean:',trainingMean
            else:
                X_predict=X_predict1-m
                print 'actual mean of Xtrain :',m
            
            X_predict=X_predict1-trainingMean
        else:
            X_predict=X_predict1
        

        args  = H.parse_args()                          
        train_params = {
     'do' : float(args.do) if args.do else 0.5,        
     'a'  : float(args.a) if args.a else 0.3,          # Conv Layers LeakyReLU alpha param [if alpha set to 0 LeakyReLU is equivalent with ReLU]
     'k'  : int(args.k) if args.k else 4,              # Feature maps k multiplier
     's'  : float(args.s) if args.s else 1,            # Input Image rescale factor
     'pf' : float(args.pf) if args.pf else 1,          # Percentage of the pooling layer: [0,1]
     'pt' : args.pt if args.pt else 'Avg',             # Pooling type: Avg, Max
     'fp' : args.fp if args.fp else 'proportional',    # Feature maps policy: proportional, static
     'cl' : int(args.cl) if args.cl else 5,            # Number of Convolutional Layers
     'opt': args.opt if args.opt else 'Adam',          # Optimizer: SGD, Adagrad, Adam
     'obj': args.obj if args.obj else 'ce',            # Minimization Objective: mse, ce
     'patience' : args.pat if args.pat else 5,         # Patience parameter for early stoping
     'tolerance': args.tol if args.tol else 1.005,     # Tolerance parameter for early stoping [default: 1.005, checks if > 0.5%]
     'res_alias': args.csv if args.csv else 'res'      # csv results filename alias
         }

        modelfile= os.path.join(picklein_file[setn],modelname)
#        if keras.__version__ =='1.1.2':
        model= load_model(modelfile)
#        else:
#           jsonf= os.path.join(picklein_file[setn],'ILD_CNN_model.json')
#           #        print jsonf
#           weigf= os.path.join(picklein_file[setn],'ILD_CNN_model_weights')
#           model = model_from_json(open(jsonf).read())
#           model.load_weights(weigf)
#        print weigf

        model.compile(optimizer='Adam', loss=CNN4.get_Obj(train_params['obj']))        
             
        if X0>0:
            proba = model.predict_proba(X_predict, batch_size=100)
        else:
            proba=()
#        picklefileout_f_dir = os.path.join( patient_dir_s,picklefile[setn])  
        picklefileout_f_dir = os.path.join( patient_dir_s,setn)     

        xfp=os.path.join(picklefileout_f_dir,Xprepkl)
        pickle.dump(proba, open( xfp, "wb" ))
        xfpr=os.path.join(picklefileout_f_dir,Xrefpkl)
        pickle.dump(patch_list, open( xfpr, "wb" ))
        print 'number of patches', len(patch_list), 'in :',setn
#        for i in range (0,len(patch_list)):
#            prec, mprobai = maxproba(proba[i])
#            classlabel=fidclass(prec,classif['set0']) 
#            if classlabel=='HC':
#                print classlabel,mprobai,patch_list[i][2:4]
#        ooo
#        print proba[0]




    
def tagviews(b,fig,t0,x0,y0,t1,x1,y1,t2,x2,y2,t3,x3,y3,t4,x4,y4,t5,x5,y5):
    """write simple text in image """
    imgn=ImagePIL.open(fig)
    draw = ImageDraw.Draw(imgn)
    if b:
        draw.rectangle ([x1, y1,x1+100, y1+15],outline='black',fill='black')
        draw.rectangle ([140, 0,dimtabx,75],outline='black',fill='black')
    draw.text((x0, y0),t0,white,font=font10)
    draw.text((x1, y1),t1,white,font=font10)
    draw.text((x2, y2),t2,white,font=font10)
    if not b:
        draw.text((x3, y3),t3,white,font=font10)
    draw.text((x4, y4),t4,white,font=font10)
    draw.text((x5, y5),t5,white,font=font10)
    imgn.save(fig)

def maxproba(proba):
    """looks for max probability in result"""
    lenp = len(proba)
    m=0
    for i in range(0,lenp):
        if proba[i]>m:
            m=proba[i]
            im=i
    return im,m


def loadpkl(do,nset):
    """crate image directory and load pkl files"""
#    global classdirec
    

    picklefileout_f_dir = os.path.join( do,nset)

    xfp=os.path.join(picklefileout_f_dir,Xprepkl)
    dd = open(xfp,'rb')
    my_depickler = pickle.Unpickler(dd)
    probaf = my_depickler.load()
    dd.close()  
    
    xfpr=os.path.join(picklefileout_f_dir,Xrefpkl)
    dd = open(xfpr,'rb')
    my_depickler = pickle.Unpickler(dd)
    patch_listr = my_depickler.load()
    dd.close() 
    
    preprob=[]
    prefile=[]
    
    (top,tail)=os.path.split(do)

    n=0
    for fil in patch_listr:        
        if fil[0]==tail:
#            print n, proba[n]
            preprob.append(probaf[n])
            prefile.append(fil[1:4])
        n=n+1
    return (preprob,prefile)
    

def loadpkllung(do):
    """crate image directory and load pkl files"""
#    global classdirec
    
    picklefileout_f_dir = os.path.join( do,picklefilelung)
    xfp=os.path.join(picklefileout_f_dir,lungXprepkl)
    dd = open(xfp,'rb')
    my_depickler = pickle.Unpickler(dd)
    probaf = my_depickler.load()
    dd.close()  
    
    xfpr=os.path.join(picklefileout_f_dir,lungXrefpkl)
    dd = open(xfpr,'rb')
    my_depickler = pickle.Unpickler(dd)
    patch_listr = my_depickler.load()
    dd.close() 
    
    preprob=[]
    prefile=[]
    
    (top,tail)=os.path.split(do)

    n=0
    for fil in patch_listr:        
        if fil[0]==tail:
#            print n, proba[n]
            preprob.append(probaf[n])
            prefile.append(fil[1:4])
        n=n+1
    return (preprob,prefile)



def drawContour(imi,ll):
    
    vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
    for l in ll:
#        print l
        col=classifc[l]

        masky=cv2.inRange(imi,col,col)
        outy=cv2.bitwise_and(imi,imi,mask=masky)
        imgray = cv2.cvtColor(outy,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,0,255,0)
        im2,contours0, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_SIMPLE)        
        contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
#        cv2.drawContours(vis,contours,-1,col,1,cv2.LINE_AA)
        cv2.drawContours(vis,contours,-1,col,1)

    return vis

def tagviewn(fig,label,pro,nbr,x,y,nset):
    """write text in image according to label and color"""

    col=classifc[label]
#    print col, label
    if wbg :
        labnow=classif[nset][label]-1
    else:
        labnow=classif[nset][label]
#    print (labnow, text)
    if label == 'back_ground':
        x=0
        y=0        
        deltax=0
        deltay=60
    else:                
        deltax=130*((labnow)//5)
        deltay=11*((labnow)%5)
    gro=-x*0.0027+1.2
#    if label=='ground_glass':
#            print label,x,deltax, y,deltay,labnow,gro
    cv2.putText(fig,str(nbr)+' '+label+' '+pro,(x+deltax, y+deltay+10),cv2.FONT_HERSHEY_PLAIN,gro,col,1)

def genethreef(dirpatientdb,nset,lsn,cla):
      
        (dptop,dptail)=os.path.split(dirpatientdb)
        if cla==1:
            topdir=dptail
        else:
            (dptop1,dptail1)=os.path.split(dptop)
            topdir=dptail1
        (preprob,listnamepatch)=loadpkl(dirpatientdb,nset)
        cwd=os.getcwd()
        souuip=os.path.join(cwd,threeFileTop)
#        print souuip
        threeFilel=topdir+'_'+threeFile
        desouip=os.path.join(dirpatientdb,threeFilel)
        shutil.copyfile(souuip,desouip)
        threeFileTxtl=topdir+'_'+threeFileTxt
        volumefileT = open(os.path.join(dirpatientdb,threeFileTxtl), 'w')
        jsfile=topdir+'_'+threeFilejs
        jsfilel=os.path.join(dirpatientdb,jsfile)
        volumefile = open(os.path.join(dirpatientdb,threeFilel), 'a')
        volumefilejs = open(jsfilel, 'w')
#        jsfilel.replace("\\","/")
#        print jsfilel
        volumefile.write( '<script src="'+jsfilel+'"></script>\n)')	
        volumefilejs.write( 'function buildobj()	{\n')	
        pz=slicepitch/avgPixelSpacing
#        zxd=slicepitch*lsn/2/avgPixelSpacing
        zxd=round(pz*lsn/2,3)
        volumefileT.write('camera.position.set(0 '+', -'+str(dimtabx)+', 0 );\n')  
        
        volumefileT.write( 'var boxGeometry = new THREE.BoxGeometry( '+str(dimpavxs[nset])+' , '+\
        str(dimpavxs[nset])+' , '+str(round(pz,3))+') ;\n\n')
        volumefileT.write('var voxels = [\n')
        volumefilejs.write( 'var boxGeometry = new THREE.BoxGeometry( '+str(dimpavxs[nset])+' , '+\
        str(dimpavxs[nset])+' , '+str(round(pz,3))+') ;\n')
#        zxd=slicepitch*lsn/2/avgPixelSpacing
        print 'pz',pz
     
#        volumefile.write('camera.position.set('+str(dimtabx/2)+',-'+str(dimtabx/2)+','+str(zxd)+' );\n') 
#        volumefile.write('controls.target.set('+str(dimtabx/2)+','+str(dimtabx/2)+','+str(zxd)+') ;\n') 
        volumefilejs.write('camera.position.set(0 '+', -'+str(dimtabx)+', 0 );\n') 
        volumefilejs.write('controls.target.set(0 , 0 , 0 );\n') 
        
#    print  >> volumefile ,dictSurf
#    volumefile.close()
#    print dictSurf
        ill = 0
        for ll in listnamepatch:
            
            slicename=ll[0] 
#            print slicename
            xpat=dimtabx-ll[1]-(dimtabx/2)
            ypat=ll[2]-(dimtabx/2)
#            print xpat,ypat
            proba=preprob[ill]          
            prec, mprobai = maxproba(proba)
            mproba=round(mprobai,2)
#            print classif[nset],nset,prec
            classlabel=fidclass(prec,classif[nset]) 
#            if classlabel=='reticulation':
#                print classlabel,mprobai,xpat,ypat
#            print classlabel
#            if mprobai >thrproba:
            bm=volcol[classlabel]            
            bmn = 'boxMesh'+str(ill)
            zpa=round((lsn-slicename)*pz-zxd,3)
#                print (slicename,zpa)
            if ill ==0:
                volumefileT.write('{"x": '+str(xpat)+', "y": '+str(ypat)+', "z": '+str(zpa)\
                 +', "class": "'+classlabel+'", "proba": '+str(mproba)+' },\n')
            elif ill==1:
                 volumefileT.write('{"x": '+str(xpat)+', "y": '+str(ypat)+', "z": '+str(zpa)\
                 +', "class": "'+classlabel+'", "proba": '+str(mproba)+' }')
            else:                     
                 volumefileT.write(',\n{"x": '+str(xpat)+', "y": '+str(ypat)+', "z": '+str(zpa)\
                 +', "class": "'+classlabel+'", "proba": '+str(mproba)+' }')

            volumefilejs.write( 'var newMaterial= '+bm+'.clone();\n'); 
            if mproba> thrprobaUIP:
                mproba=(mproba-thrprobaUIP)/thrprobaUIP
                volumefilejs.write( 'newMaterial.opacity= '+str(mproba)+';\n'); 
            else:
                volumefilejs.write( 'newMaterial.opacity= 0;\n'); 
            volumefilejs.write( bmn+' = new THREE.Mesh(boxGeometry,newMaterial)\n'); 
            volumefilejs.write( bmn+'.position.set('+str(xpat)+', '+str(ypat)+','+str(zpa)+');\n')
            volumefilejs.write('scene.add('+bmn+');\n') 
#            else:
#                bm=volcol['healthy']            
#                bmn = 'boxMesh'+str(ill)
#                zpa=(lsn-slicename)*pz
#                volumefile.write( bmn+' = new THREE.Mesh(boxGeometry, '+bm+');\n'); 
#                volumefile.write( bmn+'.position.set('+str(xpat)+', '+str(ypat)+','+str(zpa)+');\n')
#                volumefile.write('scene.add('+bmn+');\n') 
            ill+=1
        volumefileT.write('\n]\n')
        vtb=open(os.path.join(cwd,threeFileBot),'r')
        app=vtb.read()
        volumefile.write(app)
        volumefilejs.write('\n}\n')
        volumefile.close()
        volumefileT.close()
        vtb.close()
        volumefilejs.close()




def  visua(dirpatientdb,cla,wra,nset,dx):
    global errorfile
    dimpavx=dimpavxs[nset]
    dimpavy=dimpavys[nset]
    (dptop,dptail)=os.path.split(dirpatientdb)
    if cla==1:
        topdir=dptail
    else:
        (dptop1,dptail1)=os.path.split(dptop)
        topdir=dptail1
#        print 'topdir visua',topdir
    for i in range (0,len(classif[nset])):
#        print 'visua dptail', topdir
        listelabelfinal[topdir,fidclass(i,classif[nset])]=0
    #directory name with predict out dabasase, will be created in current directory
    predictout_dir = os.path.join(dirpatientdb, predictout)
    predictout_dir_bv = os.path.join(predictout_dir,vbg)
    predictout_dir_th = os.path.join(predictout_dir,str(thrproba))
    (preprob,listnamepatch)=loadpkl(dirpatientdb,nset)
#    print preprob[0], listnamepatch
    dirpatientfdb=os.path.join(dirpatientdb,scanbmpbmp)
    dirpatientfsdb=os.path.join(dirpatientdb,sroi)
    listbmpscan=os.listdir(dirpatientfdb)
#    print dirpatientfdb
    listlabelf={}
    sroiE=False
    if os.path.exists(dirpatientfsdb):
        sroiE=True
        listbmpsroi=os.listdir(dirpatientfsdb)
  
    for img in listbmpscan:
#        print img
#        if imageDepth<256:
        slicenumber= rsliceNum(img,'_','.'+typei)
#        else:
#                slicenumber= rsliceNum(img,'_','.'+typeid)
#        slicenumber=rsliceNum(img,'_','.'+typei)
        imgt = np.zeros((dimtabx,dimtaby,3), np.uint8)
        listlabelaverage={}
        listlabel={}
        listlabelrec={}
#        print dirpatientfsdb,dirpatientfdb
        if sroiE:
            for imgsroi in listbmpsroi:
#                print imgsroi
                slicenumbersroi=rsliceNum(imgsroi,'_','.'+typeiroi)
                if slicenumbersroi==slicenumber:
                    imgc=os.path.join(dirpatientfsdb,imgsroi)
                    break
            
#            print 'sroi'
            
        else:
#            print 'scan'
            imgc=os.path.join(dirpatientfdb,img)
 
#        print 'img:',img, 'imgc:',imgc        
        
        tablscan=cv2.imread(imgc,1)
#        print imgc
#        print tablscan
#        print imgc

        ill = 0
#    print  >> volumefile ,dictSurf
#    volumefile.close()
#    print dictSurf
        foundp=False
        for ll in listnamepatch:
            
            
            slicename=ll[0] 
            xpat=ll[1]
            ypat=ll[2]
            proba=preprob[ill]          
            prec, mprobai = maxproba(proba)
            mproba=round(mprobai,2)
#            print classif[nset],nset,prec
            classlabel=fidclass(prec,classif[nset]) 
#            if classlabel=='reticulation':
#                print classlabel,mprobai,xpat,ypat
#            print classlabel
            classcolor=classifc[classlabel]            
#            print slicenumber, slicename,dptail
            if slicenumber == slicename and\
            (dptail in datahealthy or (classlabel not in excluvisu)):
#                    print slicenumber, slicename,dptail
                    foundp=True
                    if classlabel in listlabel:
                        numl=listlabel[classlabel]
                        listlabel[classlabel]=numl+1
                    else:
                        listlabel[classlabel]=1
                    if classlabel in listlabelf:
                        nlt=listlabelf[classlabel]
                        listlabelf[classlabel]=nlt+1
                    else:
                        listlabelf[classlabel]=1
#            if xpat ==84:
#            print classlabel,mprobai,xpat            
            if mprobai >thrproba and slicenumber == slicename and\
             (dptail in datahealthy or (classlabel not in excluvisu)):
#                    if classlabel=='reticulation':
                    
                     

                    if classlabel in listlabelrec:
                        numl=listlabelrec[classlabel]
                        listlabelrec[classlabel]=numl+1
                        cur=listlabelaverage[classlabel]
                        averageproba= round((cur*numl+mproba)/(numl+1),2)
                        listlabelaverage[classlabel]=averageproba
                    else:
                        listlabelrec[classlabel]=1
                        listlabelaverage[classlabel]=mproba

                    imgi=addpatch(classcolor,classlabel,xpat,ypat,dimpavx,dimpavy)
                    imgt=cv2.add(imgt,imgi)
                        
                        

            ill+=1

        if wra:        
#            imgcorefull=imgcore+'.bmp'
#            imgcorefull=img
            imgnameth=os.path.join(predictout_dir_th,img)
            imgnamebv=os.path.join(predictout_dir_bv,img)
    #        print 'imgname',imgname    
            cv2.imwrite(imgnamebv,tablscan)
            
            tablscan = cv2.cvtColor(tablscan, cv2.COLOR_BGR2RGB)

            vis=drawContour(imgt,listlabel)

#put to zero the contour in image in order to get full visibility of contours
            img2gray = cv2.cvtColor(vis,cv2.COLOR_BGR2GRAY)
#            print tablscan.shape
#            tablscang = cv2.cvtColor(tablscan,cv2.COLOR_BGR2GRAY)
#            tablscang=tablscan
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            tablscan=cv2.cvtColor(tablscan,cv2.COLOR_BGR2GRAY)
#            mask_inv=cv2.cvtColor(mask_inv,cv2.COLOR_BGR2GRAY)
#            print mask_inv.shape,tablscan.shape
            img1_bg = cv2.bitwise_and(tablscan,tablscan,mask = mask_inv)  
            img1_bg=cv2.cvtColor(img1_bg,cv2.COLOR_GRAY2BGR)
#superimpose scan and contours      
#            print img1_bg.shape, vis.shape
            imn=cv2.add(img1_bg,vis)


            if foundp:
#            tagviews(imgname,'average probability',0,0)           
                for ll in listlabelrec:
                    delx=int(dx*0.6-120)
                    tagviewn(imn,ll,str(listlabelaverage[ll]),listlabelrec[ll],delx,0,nset)
            else:   
#            tagviews(imgname,'no recognised label',0,0)
                errorfile.write('no recognised label in: '+str(topdir)+' '+str (img)+'\n' )

            imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)
#            cv2.imwrite(imgnamebv,tablscan)
            cv2.imwrite(imgnameth,imn)            
            
       
            if foundp:
                t0='average probability'
            else:
                t0='no recognised label'
            t1='n: '+topdir+' scan: '+str(slicenumber)        
            t2='CONFIDENTIAL - prototype - not for medical use'
            t3='For threshold: '+str(thrproba)+' :'
            t4=time.asctime()
            t5= picklefile['set0']
            tagviews(True,imgnamebv,t0,0,0,t1,0,20,t2,20,dimtaby-27,t3,0,38,t4,0,dimtaby-10,t5,0,dimtaby-20)
            tagviews(False,imgnameth,t0,0,0,t1,0,20,t2,20,dimtaby-27,t3,0,38,t4,0,dimtaby-10,t5,0,dimtaby-20)

        
            errorfile.write('\n'+'number of labels in :'+str(topdir)+' '+str(dptail)+str (img)+'\n' )
#    print listlabelf
    for classlabel in listlabelf:  
          listelabelfinal[topdir,classlabel]=listlabelf[classlabel]
          print 'set:',nset,'patient: ',topdir,', label:',classlabel,': ',listlabelf[classlabel]
          string=str(classlabel)+': '+str(listlabelf[classlabel])+'\n' 
#          print string
          errorfile.write(string )

#    
#def renomscan(fa):
#        num=0
#        contenudir = os.listdir(fa)
##        print(contenudir)
#        for ff in contenudir:
##            print ff
#            if ff.find('.dcm')>0 and ff.find('-')<0:     
#                num+=1    
#                corfpos=ff.find('.dcm')
#                cor=ff[0:corfpos]
#                ncff=os.path.join(fa,ff)
##                print ncff
#                if num<10:
#                    nums='000'+str(num)
#                elif num<100:
#                    nums='00'+str(num)
#                elif num<1000:
#                    nums='0'+str(num)
#                else:
#                    nums=str(num)
#                newff=cor+'-'+nums+'.dcm'
#    #            print(newff)
#                shutil.copyfile(ncff,os.path.join(fa,newff) )
#                os.remove(ncff)
def dd(i):
    if (i)<10:
        o='0'+str(i)
    else:
        o=str(i)
    return o


def nothings(x):
    global imgtext
    imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
    pass

def nothing(x):
    pass

def contrasti(im,r):   
     tabi = np.array(im)
     r1=0.5+r/100.0
     tabi1=tabi*r1     
     tabi2=np.clip(tabi1,0,255)
     tabi3=tabi2.astype(np.uint8)
     return tabi3

def lumi(im,r):
    tabi = np.array(im)
    r1=r
    tabi1=tabi+r1
    tabi2=np.clip(tabi1,0,255)
    return tabi2

# mouse callback function
def draw_circle(event,x,y,flags,img):
    global ix,iy,quitl,patchi
    patchi=False

    if event == cv2.EVENT_RBUTTONDBLCLK:
        print x, y
    if event == cv2.EVENT_LBUTTONDBLCLK:
 
#        print('identification')
        ix,iy=x,y
       
        patchi=True
#        print 'identification', ix,iy, patchi
        if x>250 and x<270 and y>dimtaby-30 and y<dimtaby-10:
            print 'quit'
            ix,iy=x,y
            quitl=True

def addpatchn(col,lab, xt,yt,imgn,nset):
#    print col,lab
    dimpavx=dimpavxs[nset]
    dimpavy=dimpavys[nset]
    cv2.rectangle(imgn,(xt,yt),(xt+dimpavx,yt+dimpavy),col,1)
    return imgn
 
def retrievepatch(x,y,top,sln,pr,li,nset):
    tabtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
    dimpavx=dimpavxs[nset]
    dimpavy=dimpavys[nset]
    ill=-1
    pfound=False
    for f in li:
        ill+=1 
        slicenumber=f[0]

        if slicenumber == sln:

            xs=f[1]
            ys=f[2]
#            print xs,ys
            if x>xs and x < xs+dimpavx and y>ys and y<ys+dimpavy:
                     print x, y
                     proba=pr[ill]
                     pfound=True

                     n=0
                     cv2.putText(tabtext,'X',(xs-5+dimpavx/2,ys+5+dimpavy/2),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
                     for j in range (0,len(proba)):
                         
#                     for j in range (0,2):
                         if proba[j]>0.01:
                             n=n+1
                             strw=fidclass(j,classif[nset])+ ' {0:.1f}%'.format(100*proba[j])                             
                             cv2.putText(tabtext,strw,(dimtabx-142,(dimtaby-60)+10*n),cv2.FONT_HERSHEY_PLAIN,0.8,(0,255,0),1)
                             
                             print fidclass(j,classif[nset]), ' {0:.2f}%'.format(100*proba[j])
                     print'found'
                     break 
#    cv2.imshow('image',tabtext)                
    if not pfound:
            print'not found'
    return tabtext

def drawpatch(t,lp,preprob,k,top,nset,dx):
    imgn = np.zeros((dimtabx,dimtaby,3), np.uint8)
    ill = 0
#    endnumslice=k.find('.bmp')
    slicenumber=rsliceNum(k,'_','.'+typei)
#    print imgcore
#    posend=endnumslice
#    while k.find('-',posend)==-1:
#            posend-=1
#    debnumslice=posend+1
#    slicenumber=int((k[debnumslice:endnumslice])) 
    th=t/100.0
    listlabel={}
    listlabelaverage={}
#    print slicenumber,th
    for ll in lp:

#            print ll
            slicename=ll[0]          
            xpat=ll[1]
            ypat=ll[2]        
        #we find max proba from prediction
            proba=preprob[ill]
           
            prec, mprobai = maxproba(proba)

            classlabel=fidclass(prec,classif[nset])
            classcolor=classifc[classlabel]
       
            
            if mprobai >th and slicenumber == slicename and\
            (top in datahealthy or (classlabel not in excluvisu)):
#                    print classlabel
                    if classlabel in listlabel:
#                        print 'found'
                        numl=listlabel[classlabel]
                        listlabel[classlabel]=numl+1
                        cur=listlabelaverage[classlabel]
#                               print (numl,cur)
                        averageproba= round((cur*numl+mprobai)/(numl+1),2)
                        listlabelaverage[classlabel]=averageproba
                    else:
                        listlabel[classlabel]=1
                        listlabelaverage[classlabel]=mprobai

                    imgn= addpatchn(classcolor,classlabel,xpat,ypat,imgn,nset)


            ill+=1
#            print listlabel        
    for ll1 in listlabel:
#                print ll1,listlabelaverage[ll1]
                delx=int(dx*0.6-120)
                tagviewn(imgn,ll1,str(round(listlabelaverage[ll1],2)),listlabel[ll1],delx,0,nset)
    ts='Treshold:'+str(t)

    cv2.putText(imgn,ts,(0,42),cv2.FONT_HERSHEY_PLAIN,0.8,white,1,cv2.LINE_AA)
    return imgn
    
def opennew(dirk, fl,L):
    pdirk = os.path.join(dirk,L[fl])
    img = cv2.imread(pdirk,1)
    return img,pdirk

def reti(L,c):
    for i in range (0, len(L)):
     if L[i]==c:
         return i
         break
     

def openfichier(k,dirk,top,L,nset):
    nseed=reti(L,k) 
#    print 'openfichier', k, dirk,top,nseed
  
    global ix,iy,quitl,patchi,classdirec
    global imgtext, dimtabx,dimtaby
   
    patchi=False
    ix=0
    iy=0
    ncf1 = os.path.join(path_patient,top)
    dop =os.path.join(ncf1,picklefile[nset])
    if classdirec==2:
        ll=os.listdir(ncf1)
        for l in ll:
            ncf =os.path.join(ncf1,l)
            dop =os.path.join(ncf,picklefile[nset])
    else:
        ncf=ncf1
            
    subsample=varglobal[3]
    pdirk = os.path.join(dirk,k)
    img = cv2.imread(pdirk,1)

    dimtabx= img.shape[0]
    dimtaby= dimtabx
    imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
#    print 'openfichier:',k , ncf, pdirk,top
    
    (preprob,listnamepatch)=loadpkl(ncf,nset)      
    
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)

    cv2.createTrackbar( 'Brightness','image',0,100,nothing)
    cv2.createTrackbar( 'Contrast','image',50,100,nothing)
    cv2.createTrackbar( 'Threshold','image',50,100,nothing)
    cv2.createTrackbar( 'Flip','image',nseed,len(L)-1,nothings)
        
    while(1):
        cv2.setMouseCallback('image',draw_circle,img)
        c = cv2.getTrackbarPos('Contrast','image')
        l = cv2.getTrackbarPos('Brightness','image')
        tl = cv2.getTrackbarPos('Threshold','image')
        fl = cv2.getTrackbarPos('Flip','image')

        img,pdirk= opennew(dirk, fl,L)
#        print pdirk
        
        
        (topnew,tailnew)=os.path.split(pdirk)
#        endnumslice=tailnew.find('.bmp',0)
#        posend=endnumslice
        slicenumber=rsliceNum(tailnew,'_','.'+typei)
#        while tailnew.find('_',posend)==-1:
#            posend-=1
#            debnumslice=posend+1
#        slicenumber=int((tailnew[debnumslice:endnumslice])) 
#        
        imglumi=lumi(img,l)
        imcontrast=contrasti(imglumi,c)        
        imcontrast=cv2.cvtColor(imcontrast,cv2.COLOR_BGR2RGB)
#        print imcontrast.shape, imcontrast.dtype
        imgn=drawpatch(tl,listnamepatch,preprob,L[fl],top,nset,dimtabx)
#        imgn=cv2.cvtColor(imgn,cv2.COLOR_BGR2RGB)
        imgngray = cv2.cvtColor(imgn,cv2.COLOR_BGR2GRAY)
#        print imgngray.shape, imgngray.dtype
        mask_inv = cv2.bitwise_not(imgngray)              
        outy=cv2.bitwise_and(imcontrast,imcontrast,mask=mask_inv)
        imgt=cv2.add(imgn,outy)
 
       
        cv2.rectangle(imgt,(250,dimtaby-10),(270,dimtaby-30),red,-1)
        cv2.putText(imgt,'quit',(260,dimtaby-10),cv2.FONT_HERSHEY_PLAIN,1,yellow,1,cv2.LINE_AA)
        imgtoshow=cv2.add(imgt,imgtext)        
        imgtoshow=cv2.cvtColor(imgtoshow,cv2.COLOR_BGR2RGB)

        
        cv2.imshow('image',imgtoshow)

        if patchi :
            print 'retrieve patch asked'
            imgtext=retrievepatch(ix,iy,top,slicenumber,preprob,listnamepatch,nset)
            patchi=False

        if quitl or cv2.waitKey(20) & 0xFF == 27 :
#            print 'on quitte', quitl
            break
    quitl=False
#    print 'on quitte 2'
    cv2.destroyAllWindows()


def listfichier(dossier):
    Lf=[]
    L= os.listdir(dossier)
#    print L, dossier
    for k in L:
        if '.'+typei in k.lower(): 
            Lf.append(k)
    return Lf

def listbtn2(L,dirk,top,nset):
    for widget in cadreim.winfo_children():
        widget.destroy()
    canvas = Canvas(cadreim, borderwidth=2, width=200,height=600,background="blue")
    frame = Frame(canvas, background="blue")
    vsb = Scrollbar(cadreim, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)

    vsb.pack(side="right", fill="y")
    canvas.pack(side="right", fill="both", expand=True)
    canvas.create_window((1,1), window=frame, anchor="nw")
#    canvas.create_window((1,1), window=frame)

    frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
       
    for k in L: 
           Button(frame,text=k,command=lambda k = k:\
           openfichier(k,dirk,top,L,nset)).pack(side=TOP,expand=1)

    
def opendir(k,nset):
    global classdirec
#    
#    for widget in cadrepn.winfo_children():
#        widget.destroy()
    for widget in cadrestat.winfo_children():
        widget.destroy()
    Label(cadrestat, bg='lightgreen',text='patient:'+k).pack(side=TOP,fill=X,expand=1)
    tow=''
    fdir=os.path.join(path_patient,k)
    if classdirec==1:   
#        fdir=os.path.join(path_patient,k)
        bmp_dir = os.path.join(fdir, scanbmp)
    else:
        ldir=os.listdir(fdir)
        for ll in ldir:
             fdir = os.path.join(fdir, ll)
             bmp_dir = os.path.join(fdir, scanbmp)
    
    separator = Frame(cadrestat,height=2, bd=10, relief=SUNKEN)
    separator.pack(fill=X)
#    print 'bmp dir', bmp_dir      
    listscanfile =os.listdir(bmp_dir)
   
    ldcm=[]
    for ll in listscanfile:
      if  ll.lower().find('.'+typei,0)>0:
         ldcm.append(ll)
    numberFile=len(ldcm)        
    tow='Number of sub sampled scan images: '+ str(numberFile)+\
    '\n\n'+'Predicted patterns: '+'\n' 
    for cle, valeur in listelabelfinal.items():
#             print 'cle valeur', cle,valeur
             for c in classif[nset]:
#                 print (k,c)
                 if (k,c) == cle and listelabelfinal[(k,c)]>0:

                     tow=tow+c+' : '+str(listelabelfinal[(k,c)])+'\n'

    Label(cadrestat, text=tow,bg='lightgreen').pack(side=TOP, fill='both',expand=1)
#    print tow
    dirkinter=os.path.join(fdir,predictout)
    dirk=os.path.join(dirkinter,vbg)
    L=listfichier(dirk)
    listbtn2(L,dirk,k,nset)
    
       
def listdossier(dossier): 
    L= os.walk(dossier).next()[1]  
    return L
    
def listbtn(L,nset):   
    cwt = Label(cadrerun,text="Select a patient")
    cwt.pack()
    for widget in cadrerun.winfo_children():       
                widget.destroy()    
    for k in L:
            Button(cadrerun,text=k,command=lambda k = k: opendir(k,nset)).pack(side=LEFT,fill="both",\
            expand=1)

def runf(nset):

    listbtn(listdossier( path_patient ),nset)

    
def onFrameConfigure(canvas):
    '''Reset the scroll region to encompass the inner frame'''
    canvas.configure(scrollregion=canvas.bbox("all"))
    
    
def quit():
    global fenetre
    fenetre.quit()
    fenetre.destroy()   

def posP(sln):
    global lungSegment
    if sln in lungSegment['upperset']:
        psp='upperset'
    elif sln in lungSegment['middleset']:
        psp='middleset'
    else:
        psp='lowerset'
    return psp

def  calcMed (ntp, nset):
    '''calculate the median position in between left and right lung'''
#    print 'number of subpleural for : ',pat
#    print 'subpleural', ntp, pat
    global lungSegment
    tabMed={}
#    print subErosion,avgPixelSpacing,subErosionPixel
    lung_dir = os.path.join(ntp, lungmask)  
    lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)
    lunglist = os.listdir(lung_bmp_dir)
    for l in lunglist:
        
        slicename= rsliceNum(l,'_','.'+typei)
        if Double:
            slicename=2*slicename

        if slicename in lungSegment['allset']:
#             print 'slicename',slicename
             lf=os.path.join(lung_bmp_dir,l)
#             print lf
             imscan = ImagePIL.open(lf).convert('L')          
             imgngray=np.array(imscan)              
             ke=5
             kernele=np.ones((ke,ke),np.uint8)
             kerneld=np.ones((ke,ke),np.uint8)
        
             erosion = cv2.erode(imgngray,kernele,iterations = 1)                      
             dilation = cv2.dilate(erosion,kerneld,iterations = 1)            
             
             im2,contours0, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,\
                      cv2.CHAIN_APPROX_SIMPLE)   
#             cv2.imshow('lung2',imgngray)
             xmed=np.zeros((2), np.uint16)
             xmaxi=np.zeros((2), np.uint16)
             xmini=np.zeros((2), np.uint16)
#             print len(contours0)
             if len(contours0)>1:
                 areaArray =[]
                 for i,c in enumerate(contours0):
                      area = cv2.contourArea(c)
                      areaArray.append(area)
    
    #first sort the array by area
                 sorteddata = sorted(zip(areaArray, contours0), key=lambda x: x[0], reverse=True)
    
    #find the nth largest contour [n-1][1], in this case 2
                 xmed= np.zeros(3, np.uint16)
                 xmini=np.zeros(3, np.uint16)
                 xmaxi=np.zeros(3, np.uint16)
                 
                 firstlargestcontour = sorteddata[0][1]
                 visa = np.zeros((dimtabx,dimtaby,3), np.uint8)
                 cv2.drawContours(visa,firstlargestcontour,-1,white,1)
                 npvisa=np.array(visa)
                 atabf = np.nonzero(npvisa)
                 xmin=atabf[1].min()
                 xmax=atabf[1].max()
                 xmed[0]=(xmax+xmin)/2
                 xmini[0]=xmin
                 xmaxi[0]=xmax
#                 cv2.imshow('cont',visa)
#                 cv2.waitKey(0)    
#                 cv2.destroyAllWindows()
                         
                 
                 secondlargestcontour = sorteddata[1][1]
                 visa = np.zeros((dimtabx,dimtaby,3), np.uint8)
                 cv2.drawContours(visa,secondlargestcontour,-1,white,1)
                 npvisa=np.array(visa)
                 atabf = np.nonzero(npvisa)
                 xmin=atabf[1].min()
                 xmax=atabf[1].max()
                 xmed[1]=(xmax+xmin)/2
                 xmini[1]=xmin
                 xmaxi[1]=xmax
    #         
#                 cv2.imshow('cont',visa)
#                 cv2.waitKey(0)    
#                 cv2.destroyAllWindows()
                         
                 xmedf=0     
    #             print n
                 ifinmin=0
                 for i in range (0,2):
    #                 print '3', i, xmed[i],xmedf
                     if xmed[i]>xmedf:
                    
                         xmedf=xmed[i]
                         ifinmax=i
                     else:
                         ifinmin=i
                         
                 xmedian=    (xmini[ifinmax]+xmaxi[ifinmin])/2
                 tabMed[slicename]=xmedian
             else:
                 xmedian=dimtabx/2
             
             if xmedian<0.75*dimtabx/2 or xmedian>1.25*dimtabx/2:
                 xmedian=dimtabx/2
             tabMed[slicename]=xmedian
#             print xmedian
#             tabm=np.zeros((dimtabx,dimtabx,3),np.uint8)
#             tabm[:,xmedian]=(0,125,0)
#
#             imgngrayc = cv2.cvtColor(imgngray,cv2.COLOR_GRAY2BGR)
#             cv2.imshow('image',cv2.add(imgngrayc,tabm) )
##             cv2.imshow('lung1',imgngray)
#             cv2.waitKey(0)    
#             cv2.destroyAllWindows()
    
    return tabMed


def  calcSupNp (ntp, pat,nset,tabmed):
    '''calculate the number of reticulation and HC in subpleural'''
#    print 'number of subpleural for : ',pat
#    print 'subpleural', ntp, pat
    global lungSegment
    
    dimpavx=dimpavxs[nset]
    dimpavy=dimpavys[nset]
    pxy=pxys[nset]
    dictP={}
#    dictPS={}
    dictP['upperset']=(0,0)
    dictP['middleset']=(0,0)
    dictP['lowerset']=(0,0)
    dictP['all']=(0,0)

    preprob,prefile=loadpkl(ntp,nset)
    predictout_dir = os.path.join(ntp,predictout)
    predictout_dir_bv = os.path.join(predictout_dir,vbg)
    listbg=os.listdir(predictout_dir_bv)

#subErosion= 10 in mm
#avgPixelSpacing=0.734 in mm/ pixel  
    subErosionPixel=int(round(2*subErosion/avgPixelSpacing))
#    print subErosion,avgPixelSpacing,subErosionPixel
    lung_dir = os.path.join(ntp, lungmask)  
    lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)
    lunglist = os.listdir(lung_bmp_dir)
    for l in lunglist:
#        print 'l',l
        slicename= rsliceNum(l,'_','.'+typei)
        if Double:
            slicename=2*slicename

        ill=0
        if slicename in lungSegment['allset']:
#             print 'slicename',slicename
             vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
             ill+=1
             for lbg in listbg:

                 slicenamebg=rsliceNum(lbg,'_','.'+typei)
              
#                 slicenamebg=rsliceNum(lbg,'_','.bmp')
#                 endnumslicebg=lbg.find('.bmp')
#                 posendbg=endnumslicebg
#                 while lbg.find('_',posendbg)==-1:
#                        posendbg-=1
#                 debnumslicebg=posendbg+1
#                 slicenamebg=int(lbg[debnumslicebg:endnumslicebg]) 
                 if  slicenamebg== slicename:
                     lfbg=os.path.join(predictout_dir_bv,lbg)             
             lf=os.path.join(lung_bmp_dir,l)
             imscan = ImagePIL.open(lf).convert('L')
             imbg= ImagePIL.open(lfbg)            
             imgngray=np.array(imscan) 
             imbgarray=np.array(imbg)
  
             kernele=np.ones((subErosionPixel,subErosionPixel),np.uint8)
             erosion = cv2.erode(imgngray,kernele,iterations = 1)     
             
             ret, mask = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY)
             mask_inv = cv2.bitwise_not(mask)  
             
             subpleurmask = cv2.bitwise_and(imgngray,imgngray,mask = mask_inv) 

             im2,contours0, hierarchy = cv2.findContours(subpleurmask,cv2.RETR_TREE,\
                      cv2.CHAIN_APPROX_SIMPLE)      
                                
                 
             contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
             cv2.drawContours(vis,contours,-1,white,1)
             cv2.imwrite(lfbg,cv2.add(vis,imbgarray))

             np.putmask(subpleurmask,subpleurmask>0,100)

             ill=0
             for ll in prefile:
                    
                    slicenamepatch=ll[0] 
                    xpat=ll[1]
                    ypat=ll[2]
                    proba=preprob[ill]          
                    prec, mprobai = maxproba(proba)
#                    mproba=round(mprobai,2)
                    classlabel=fidclass(prec,classif[nset]) 
                    if classlabel==pat and mprobai>thrprobaUIP:
                        if slicename == slicenamepatch:
#                                print ll
#                            print slicename,xpat,ypat,classlabel
                            midx=tabmed[slicename]
                    
                            if xpat >= midx:
                              pospr=1
                              pospl=0
                            else:
                              pospr=0
                              pospl=1
                            tabpatch = np.zeros((dimtabx,dimtaby), np.uint8)
#                            vis3 = np.zeros((dimtabx,dimtaby,3), np.uint8)
#                            subpr = np.zeros((dimtabx,dimtaby), np.uint8)
                            tabpatch[ypat:ypat+dimpavy,xpat:xpat+dimpavx]=100
#                            print slicename,subpleurmask.shape,tabpatch.shape
                            tabsubpl = cv2.bitwise_and(subpleurmask,subpleurmask,mask = tabpatch)  
                            np.putmask(tabsubpl,tabsubpl>0,1) 
                            area= tabsubpl.sum()                              
#                            np.putmask(tabsubpl,tabsubpl>0,150)
#                    check if area above threshold
                            targ=float(area)/pxy
                            psp=posP(slicename)

                            if targ>thrpatch:                            
#                                print area, targ,thrpatch
                                dictP['all']=(dictP['all'][0]+pospl,dictP['all'][1]+pospr)
                                
                                dictP[psp]=(dictP[psp][0]+pospl,dictP[psp][1]+pospr)
                
#                                dictP[psp]=dictP[psp]+1
                    ill+=1
    return dictP
  

def diffMicro(do, pat,nset,tabMed):
    '''calculate number of diffuse micronodules, left and right '''

    dictP={}
    dictPS={}
    dictP['upperset']=(0,0)
    dictP['middleset']=(0,0)
    dictP['lowerset']=(0,0)
    dictP['all']=(0,0)
    dictPS['all']=(0,0)
    dictPS['upperset']=(0,0)
    dictPS['middleset']=(0,0)
    dictPS['lowerset']=(0,0)

    preprob,prefile=loadpkl(do,nset)
        
    ill=0
    for ll in prefile:
            slicename=ll[0] 
           
            xpat=ll[1]
            proba=preprob[ill]          
            prec, mprobai = maxproba(proba)
            classlabel=fidclass(prec,classif[nset]) 
            psp=posP(slicename)
            midx=tabMed[slicename]
#            print midx
            if xpat >= midx:
                    pospr=1
                    pospl=0
            else:
                    pospr=0
                    pospl=1
#            if classlabel=='HC':
#                print mprobai,xpat
            
            if classlabel==pat and mprobai>thrprobaUIP:       
#                if classlabel=='micronodules' and pospr==1 and psp=='lowerset':
#                    print 'pat', xpat,mprobai,midx
                dictP[psp]=(dictP[psp][0]+pospl,dictP[psp][1]+pospr)
                dictP['all']=(dictP['all'][0]+pospl,dictP['all'][1]+pospr)
                   
            dictPS['all']=(dictPS['all'][0]+pospl,dictPS['all'][1]+pospr)
            dictPS[psp]=(dictPS[psp][0]+pospl,dictPS[psp][1]+pospr)
                            
            ill+=1
    return dictP,dictPS 


def cvs(p,f,d,ds,s,dc,ws):
    dictint={}
    
    llungloc=(('lowerset','lower'),('middleset','middle'),('upperset','upper'))
    llunglocsl=(('lowerset','left_sub_lower'),('middleset','left_sub_middle'),('upperset','left_sub_upper'))
    llunglocsr=(('lowerset','right_sub_lower'),('middleset','right_sub_middle'),('upperset','right_sub_upper'))
    llunglocl=(('lowerset','left_lower'),('middleset','left_middle'),('upperset','left_upper'))
    llunglocr=(('lowerset','right_lower'),('middleset','right_middle'),('upperset','right_upper'))
    f.write(p+': ')
    for i in llungloc:
        st=s[i[0]][0]+s[i[0]][1]
        if st>0:
            l=100*float(d[i[0]][0]+d[i[0]][1])/st
            l=str(round(l,3))
        else:
            l='0'
        dictint[i[1]]=l
        f.write(l+' ')
            

    if ws:
        for i in llunglocsl:
            st=s[i[0]][0]
            if st>0:
                l=100*float(ds[i[0]][0])/st
                l=str(round(l,3))
            else:
                l='0'
            dictint[i[1]]=l
            f.write(l+' ')
            
        for i in llunglocsr:
            st=s[i[0]][1]
            if st>0:
                l=100*float(ds[i[0]][1])/st
                l=str(round(l,3))
            else:
                l='0'
            dictint[i[1]]=l
            f.write(l+' ')
        

    else:  
        slol='None'
        dictint['left_sub_lower']=slol
        f.write('None ')

        smil='None'
        dictint['left_sub_middle']=smil
        f.write('None ')

        supl='None'
        dictint['left_sub_upper']=supl
        f.write('None ')
        
        slor='None'
        dictint['right_sub_lower']=slor
        f.write('None ')
        
        smir='None'
        dictint['right_sub_middle']=smir
        f.write('None ')
        
        supr='None'
        dictint['right_sub_upper']=supr
        f.write('None ')

    for i in llunglocl:
        st=s[i[0]][0]
        if st>0:
            l=100*float(d[i[0]][0])/st
            l=str(round(l,3))
        else:
            l='0'
        dictint[i[1]]=l
        f.write(l+' ')
        
    for i in llunglocr:
        st=s[i[0]][1]
        if st>0:
            l=100*float(d[i[0]][1])/st
            l=str(round(l,3))
        else:
            l='0'
        dictint[i[1]]=l
        f.write(l+' ')

       
    dc[p]=dictint
    f.write('\n')
    return dc
    
def writedic(p,v,d):
    v.write(p+ ' '+d[p]['lower']+' '+d[p]['middle']+' '+d[p]['upper']+' ')
    v.write(d[p]['left_sub_lower']+' '+d[p]['left_sub_middle']+' '+d[p]['left_sub_upper']+' ')
    v.write(d[p]['right_sub_lower']+' '+d[p]['right_sub_middle']+' '+d[p]['right_sub_upper']+' ')
    v.write(d[p]['left_lower']+' '+d[p]['left_middle']+' '+d[p]['left_upper']+' ')
    v.write(d[p]['right_lower']+' '+d[p]['right_middle']+' '+d[p]['right_upper']+'\n')    
    
def  uiptree(ntp):
    '''calculate the number of reticulation and HC in total and subpleural 
    and diffuse micronodules'''
    
    (top, tail)= os.path.split(ntp)
#    print tail
#    classpatch(ntp)
    tabMed=calcMed(ntp, 'set0')
    dictSurf={}
    dictRET,dictPSset2LR=diffMicro(ntp, 'reticulation','set0',tabMed)
    dictHC,dictPSset2LR=diffMicro(ntp, 'HC','set0',tabMed)
    print '-------------------------------------------'
    print 'surface total for set2 by segment Left Right:'
    print dictPSset2LR
    print '-------------------------------------------'
    
    print 'reticulation total for set0:'
    print dictRET
    print '-------------------------------------------'
    
    print 'HC total for set0:'
    print dictHC
    print '-------------------------------------------'
    
    dictSubRET=calcSupNp(ntp, 'reticulation','set0',tabMed)
    dictSubHC=calcSupNp(ntp, 'HC','set0',tabMed)
    print 'reticulation subpleural for set0:'
    print dictSubRET
    print '-------------------------------------------'
    
    print 'HC subleural for set0:'
    print dictSubHC
    print '-------------------------------------------'    
   
    dictGG,dictPSset0LR= diffMicro(ntp, 'ground_glass','set0',tabMed) #waiting set1    
    print 'surface total for set0 left and right:'
    print dictPSset0LR
    print '-------------------------------------------'
    
    print 'extensive GG for set0:'
    print dictGG   
    print '-------------------------------------------'
    
    dictDiffMicro,dictPSset2LR= diffMicro(ntp, 'micronodules','set0',tabMed)
    print 'diffuse micronodules for set0:'
    print dictDiffMicro
    print '-------------------------------------------'
    
    dictPlobAir,dictPSset0LR= diffMicro(ntp, 'air_trapping','set0',tabMed) #waiting for set3
#    print 'dictPSset0LR', dictPSset0LR
    print 'air_trapping for set0:' #waiting for set3
    print dictPlobAir
    print '-------------------------------------------'
    
    dictPlobConso,dictPSset0LR= diffMicro(ntp, 'consolidation','set0',tabMed)
    print '-------------------------------------------'
    print 'Peribronchial or lobar consolidation for set0:'#waiting for set1
    print dictPlobConso    
    print '-------------------------------------------'
    
#    volumefile = open(ntp+'_volume.txt', 'w')
    dimpavx=dimpavxs['set0']
    volumefile = open(os.path.join(top,str(dimpavx)+'_'+str(tail)+'_volume.txt'), 'w')
    volumefile.write('patient: '+tail+' PC UIP\n')
    volumefile.write('pattern   lower  middle  upper')
    volumefile.write('  left_sub_lower  left_sub_middle  left_sub_upper ')
    volumefile.write('  right_sub_lower  right_sub_middle  right_sub_upper ')
    volumefile.write('  left_lower  left_middle  left_upper ')
    volumefile.write(' right_lower  right_middle  right_upper\n')
    dictSurf=cvs('reticulation',volumefile,dictRET,dictSubRET,dictPSset2LR,dictSurf,True)
    dictSurf=cvs('HC',volumefile,dictHC,dictSubHC,dictPSset2LR,dictSurf,True)
    dictSurf=cvs('ground_glass',volumefile,dictGG,dictGG,dictPSset0LR,dictSurf,False)
    dictSurf=cvs('micronodules',volumefile,dictDiffMicro,dictDiffMicro,dictPSset2LR,dictSurf,False)
    dictSurf=cvs('air_trapping',volumefile,dictPlobAir,dictPlobAir,dictPSset0LR,dictSurf,False)
    dictSurf=cvs('consolidation',volumefile,dictPlobConso,dictPlobConso,dictPSset0LR,dictSurf,False)
    volumefile.write(' ---------------------------\n')
#    print dictSurf
#    writedic('reticulation',volumefile,dictSurf)
#    writedic('HC',volumefile,dictSurf)
#    writedic('ground_glass',volumefile,dictSurf)
#    writedic('micronodules',volumefile,dictSurf)
#    writedic('air_trapping',volumefile,dictSurf)
#    writedic('consolidation',volumefile,dictSurf)    
#    volumefile.write('micronodules: '+lo+' ,'+mi+' ,'+up+' ,'+' ,'+' ,'+' ,'+lol+' ,'+mil+' ,'+upl+' ,'+lor+' ,'+mir+' ,'+upr+'\n')
#    volumefile.close()
#    
##    print os.path.join(top,str(dimpavx)+'_'+str(tail)+'_volume.txt')
#    volumefile = open(os.path.join(top,str(dimpavx)+'_'+str(tail)+'_volume.txt'), 'a')
    print  >> volumefile ,dictSurf
    volumefile.close()
#    

def runpredict(pp,subs,thrp, thpro,retou):
    
    global classdirec,path_patient, patch_list
    global  proba,subsdef,varglobal,thrproba,errorfile
    global dimtabx, dimtaby
    global lungSegment,fxsref,fxslungref
    for widget in cadretop.winfo_children():       
                widget.destroy()    
    for widget in cadrelistpatient.winfo_children():
               widget.destroy()
    for widget in cadreparam.winfo_children():
               widget.destroy()
    for widget in cadrerun.winfo_children():
               widget.destroy()
    for widget in cadrestat.winfo_children():
               widget.destroy()
    for widget in cadreim.winfo_children():
               widget.destroy()
                
    cw = Label(cadrestatus, text="Running",fg='red',bg='blue')
    cw.pack(side=TOP,fill=X)
    thrpatch=thrp
    thrproba=thpro
    subsdef =subs
    path_patient=pp
    
    varglobal=(thrpatch,thrproba,path_patient,subsdef)  
    newva()
    runl()
#    print path_patient
    if os.path.exists(path_patient):
       print path_patient
       errorfile = open(path_patient+'/predictlog.txt', 'w') 
       patient_list= os.walk(path_patient).next()[1]
       for f in patient_list:
            lungSegment={}
            print('================================================') 
            print('work on:',f, 'with subsamples :', subs)  
            fxsref=0  
            fxslungref=0
            namedirtopcf1 = os.path.join(path_patient,f)           
            listscanfile1= os.listdir(namedirtopcf1)
            for ll in listscanfile1:
                namedirtopcf=os.path.join(namedirtopcf1,ll)
                if os.path.isdir(namedirtopcf):
#                    print 'it is a dir'
                    listscanfile= os.listdir(namedirtopcf)
                    classdirec=2
        #    for ll in patient_list2:
                elif ll.find('.dcm',0)>0:
        #            print 'it is not a dir'
                    listscanfile=listscanfile1
                    namedirtopcf=namedirtopcf1
#                    print 'write classider'
                    classdirec=1
                    break
                
            ldcm=[]
            lscanumber=[]
            listscanf= [name for name in listscanfile if name.lower().find('.dcm',0)>0]
#            print listscanf
            for ll in listscanf:
                llf = os.path.join(namedirtopcf,ll)
                RefDs = dicom.read_file(llf)
                ldcm.append(ll)  
                scann=int(RefDs.InstanceNumber)
                lscanumber.append(scann)
              
            
#            print ldcm[0]
            if Double:
                ldcmd=[]
                lsd=[]
                for i in range(0,len(ldcm)):
                    if i%2>0:              
#                    print i
                        lsd.append(lscanumber[i])
                        ldcmd.append(ldcm[i])
                ldcm=ldcmd
                lscanumber=lsd
            numberFile=len(ldcm) 
            if retou==1:
                
                #directory for scan in bmp
                bmp_dir = os.path.join(namedirtopcf, scanbmp)
                remove_folder(bmp_dir)    
                os.mkdir(bmp_dir) 
                bmp_bmp_dir = os.path.join(namedirtopcf, scanbmpbmp)
                remove_folder(bmp_bmp_dir)    
                os.mkdir(bmp_bmp_dir) 
                #directory for lung mask
                lung_dir = os.path.join(namedirtopcf, lungmask)
                lung_bmp_dir = os.path.join(lung_dir, lungmaskbmp)
                if os.path.exists(lung_dir)== False:
                   os.mkdir(lung_dir)
                if os.path.exists(lung_bmp_dir)== False:
                   os.mkdir(lung_bmp_dir)
                   
                #directory for pickle from cnn and status
                for nset in listset:
                    
                    pickledir = os.path.join( namedirtopcf,nset)
#                    print pickledir
                    remove_folder(pickledir)
                    os.mkdir(pickledir) 
            #directory for picklefile lung
                pickledir_lung = os.path.join( namedirtopcf,picklefilelung)
                remove_folder(pickledir_lung)
                os.mkdir(pickledir_lung)
                #directory for bpredicted images
                predictout_f_dir = os.path.join( namedirtopcf,predictout)
                if not os.path.exists(predictout_f_dir):
                    os.mkdir(predictout_f_dir)
                
                predictout_f_dir_bg = os.path.join( predictout_f_dir,vbg)
                remove_folder(predictout_f_dir_bg)
                os.mkdir(predictout_f_dir_bg)  
                
                predictout_f_dir_th = os.path.join( predictout_f_dir,str(thrproba))
                remove_folder(predictout_f_dir_th)
                os.mkdir(predictout_f_dir_th) 
                
                #directory for the pavaement in jpeg                
                jpegpathf = os.path.join( namedirtopcf,jpegpath)
                remove_folder(jpegpathf)    
                os.mkdir(jpegpathf)
#                print len(lscanumber)
                lastScanNumber=lscanumber[len(lscanumber)-1]
               
                Nset=numberFile/3
                print 'total number of scans: ',numberFile, 'in each set: ', Nset, 'last:',lastScanNumber

                upperset=[]
                middleset=[]
                lowerset=[]
                allset=[]
                for scanumber in range(0,numberFile):
        #            print scanumber
                    if scanumber%subs==0:
                        allset.append(lscanumber[scanumber])
#                        print 'loop',scanumber
                        
                        if scanumber < Nset:
                            upperset.append(lscanumber[scanumber])
                        elif scanumber < 2*Nset:
                            middleset.append(lscanumber[scanumber])
                        else:
                            lowerset.append(lscanumber[scanumber])
                        lungSegment['upperset']=upperset
                        lungSegment['middleset']=middleset
                        lungSegment['lowerset']=lowerset
                        lungSegment['allset']=allset
                       
                        scanfile=ldcm[scanumber]  
#                        print 'scanfile: ',scanfile
                        
                        genebmp(namedirtopcf,scanfile,subs)
               
                print lungSegment             
                patch_list=pavgene(namedirtopcf,'set0')
                ILDCNNpredict(namedirtopcf,'set0',patch_list)
                genethreef(namedirtopcf,'set0',lastScanNumber,classdirec)
#                print 'end genethreef'
               
                visua(namedirtopcf,classdirec,True,'set0',dimtabx)
#                
#                patch_list=pavgene(namedirtopcf,'set0')
#                ILDCNNpredict(namedirtopcf,'set0',patch_list)
#                
#                visua(namedirtopcf,classdirec,True,'set2',dimtabx)  
                uiptree(namedirtopcf)
#                classpatch(namedirtopcf)
                pickledir = os.path.join( namedirtopcf,listset[0]) 
                spkl=os.path.join(pickledir,subsamplef)
                data_scan=(subs,dimtabx)
                pickle.dump(data_scan, open( spkl, "wb" ))
                
            else:  
                for nset in listset:
                    pickledir = os.path.join( namedirtopcf,picklefile[nset])             
                pickledir = os.path.join( namedirtopcf,listset[0])                  
                spkl=os.path.join(pickledir,subsamplef)              
                dd = open(spkl,'rb')
                my_depickler = pickle.Unpickler(dd)
                data_scan = my_depickler.load()
                dd.close()  
                dimtabx=data_scan[1]
                dimtaby=dimtabx
#                print dimtabx
                visua(namedirtopcf,classdirec,False,nset,dimtabx)
#                uiptree(namedirtopcf,'set2')
#                classpatch(namedirtopcf)
            print('completed on: ',f)    
            print('================================================')  
            print('================================================') 
       
       (top, tail)= os.path.split(path_patient)
       for widget in cadrestatus.winfo_children():       
                widget.destroy()
       wcadrewait = Label(cadrestatus, text="completed for "+tail,fg='darkgreen',bg='lightgreen',width=85)
       wcadrewait.pack()

       runf('set0')
    else:
    #            print 'path patient does not exist'
        wer = Label(cadrestatus, text="path for patients does not exist",\
               fg='red',bg='yellow',width=85)
        wer.pack(side=TOP,fill='both')
        bouton1_run = Button(cadrestatus, text="continue", fg='red',\
              bg='yellow',command= lambda: runl1())
        bouton1_run.pack()


def runl1 ():
    for widget in cadrelistpatient.winfo_children():
               widget.destroy()
    for widget in cadreparam.winfo_children():
               widget.destroy()
    for widget in cadrestatus.winfo_children():
                widget.destroy()
    for widget in cadretop.winfo_children():
                widget.destroy()
    for widget in cadrerun.winfo_children():
                widget.destroy()
    for widget in cadrestat.winfo_children():
                widget.destroy()
    for widget in cadreim.winfo_children():
                widget.destroy()
#    for widget in cadrepn.winfo_children():
#                widget.destroy()
    runl()

def chp(newp):
    global varglobal
    varglobal=(thrpatch,thrproba,newp,subsdef)
    for widget in cadrelistpatient.winfo_children():
               widget.destroy()
    for widget in cadreparam.winfo_children():
               widget.destroy()
    for widget in cadrestatus.winfo_children():
                widget.destroy()
    for widget in cadretop.winfo_children():
                widget.destroy()
    for widget in cadrerun.winfo_children():
                widget.destroy()
    for widget in cadrestat.winfo_children():
                widget.destroy()
    for widget in cadreim.winfo_children():
                widget.destroy()
#    for widget in cadrepn.winfo_children():
#                widget.destroy()
   
#    print varglobal
    runl()




def runl ():
    global path_patient,varglobal
    runalready=False
#    print path_patient  varglobal=(thrpatch,thrproba,path_patient,subsdef)
    bouton_quit = Button(cadretop, text="Quit", command= quit,bg='red',fg='yellow')
    bouton_quit.pack(side="top")
    separator = Frame(cadretop,height=2, bd=10, relief=SUNKEN)
    separator.pack(fill=X)
    w = Label(cadretop, text="path for patients:")
    w.pack(side=LEFT,fill='both')
    
    clepp = StringVar()
    e = Entry(cadretop, textvariable=clepp,width=80)
    e.delete(0, END)
    e.insert(0, varglobal[2])
#    e.insert(0, workingdir)
    e.pack(side=LEFT,fill='both',expand=1)
    boutonp = Button(cadretop, text='change patient dir',command= lambda : chp(clepp.get()),bg='green',fg='blue')
    boutonp.pack(side=LEFT)
##   
#    print varglobal
    if os.path.exists(varglobal[2]):
        pl=os.listdir(varglobal[2])
        ll = Label(cadrelistpatient, text='list of patient(s):')
        ll.pack()
        tow=''
        for l in pl:
            ld=os.path.join(varglobal[2],l)
            if os.path.isdir(ld):
                tow =tow+l+' - '
                pdir=os.path.join(ld,picklefile['set0'])
                if os.path.exists(pdir):
                    runalready=True
                else:
                    psp=os.listdir(ld)
                    for ll in psp:
                        if ll.find('.dcm')<0:
                            pdir1=os.path.join(ld,ll)
                            pdir=os.path.join(pdir1,picklefile['set0'])
                            if os.path.exists(pdir):
                                runalready=True
                            
            ll = Label(cadrelistpatient, text=tow,fg='blue')
            
        ll.pack(side =TOP)
       
             

    else:     
        print 'do not exist'
        ll = Label(cadrelistpatient, text='path_patient does not exist:',fg='red',bg='yellow')
        ll.pack()

    separator = Frame(cadrelistpatient,height=2, bd=10, relief=SUNKEN)
    separator.pack(fill=X)

    wcadre5 = Label(cadreparam, text="subsample:")
    wcadre5.pack(side=LEFT)
    clev5 = IntVar()
    e5 = Entry(cadreparam, textvariable=clev5,width=5)
    e5.delete(0, END)
    e5.insert(0, str(varglobal[3]))
    e5.pack(fill='x',side=LEFT)
    wcadresep = Label(cadreparam, text=" | ",bg='purple')
    wcadresep.pack(side=LEFT)  

    wcadre6 = Label(cadreparam, text="patch ovelapp [0-1]:")
    wcadre6.pack(side=LEFT)    
    clev6 = DoubleVar()
    e6 = Entry(cadreparam, textvariable=clev6,width=5)
    e6.delete(0, END)
    e6.insert(0, str(varglobal[0]))    
    e6.pack(fill='x',side=LEFT)
    wcadresep = Label(cadreparam, text=" | ",bg='purple')
    wcadresep.pack(side=LEFT) 

    wcadre7 = Label(cadreparam, text="predict proba acceptance[0-1]:")
    wcadre7.pack(side=LEFT)
    clev7 = DoubleVar()
    e7 = Entry(cadreparam, textvariable=clev7,width=5)
    e7.delete(0, END)
    e7.insert(0, str(varglobal[1]))
    e7.pack(fill='x',side=LEFT)
    wcadresep = Label(cadreparam, text=" | ",bg='purple')   
    wcadresep.pack(side=LEFT)
    
#    retour0=IntVar(cadreparam)
#    bouton0 = Radiobutton(cadreparam, text='run predict',variable=retour0,value=1,bd=2)
#    bouton0.pack(side=RIGHT)
#    if runalready:
#         bouton1 = Radiobutton(cadreparam, text='visu only',variable=retour0,value=0,bd=2)
#         bouton1.pack(side=RIGHT)
#    print runalready
    if runalready:
       bouton_run1 = Button(cadreparam, text="Run visu", bg='green',fg='blue',\
             command= lambda: runpredict(clepp.get(),clev5.get(),clev6.get(),clev7.get(),0))
       bouton_run1.pack(side=RIGHT)
    bouton_run2 = Button(cadreparam, text="Run predict", bg='green',fg='blue',\
             command= lambda: runpredict(clepp.get(),clev5.get(),clev6.get(),clev7.get(),1))
    bouton_run2.pack(side=RIGHT)
#    separator = Frame(cadretop,height=2, bd=10, relief=SUNKEN)
#    separator.pack(fill=X, padx=5, pady=2)


##########################################################
    
t = datetime.datetime.now()
today = str('date: '+dd(t.month)+'-'+dd(t.day)+'-'+str(t.year)+\
'_'+dd(t.hour)+':'+dd(t.minute)+':'+dd(t.second))

print today


quitl=False
patchi=False
listelabelfinal={}
oldc=0
#imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)

fenetre = Tk()
fenetre.title("predict")
fenetre.geometry("700x800+100+50")



cadretop = LabelFrame(fenetre, width=700, height=20, text='top',borderwidth=5,bg="purple",fg='yellow')
cadretop.grid(row=0,sticky=NW)
cadrelistpatient = LabelFrame(fenetre, width=700, height=20, text='list',borderwidth=5,bg="purple",fg='yellow')
cadrelistpatient.grid(row=1,sticky=NW)
cadreparam = LabelFrame(fenetre, width=700, height=20, text='param',borderwidth=5,bg="purple",fg='yellow')
cadreparam.grid(row=2,sticky=NW)
cadrestatus = LabelFrame(fenetre,width=700, height=20,text="status run",bg='purple',fg='yellow')
cadrestatus.grid(row=3,sticky=NW)
cadrerun = LabelFrame(fenetre,text="select a patient",width=700, height=20,fg='yellow',bg='purple')
cadrerun.grid(row=4,sticky=NW)
#cadrepn = LabelFrame(fenetre,text="patient name list:",width=700, height=20,bg='purple',fg='yellow')
#cadrepn.grid(row=5,sticky=NW)
cadrestat=LabelFrame(fenetre,text="statistic", width=350,height=20,fg='yellow',bg='purple')
cadrestat.grid(row=6,  sticky=NW )
cadreim=LabelFrame(fenetre,text="images", width=350,height=20,fg='yellow',bg='purple')
cadreim.grid(row=6,  sticky=E)
    
#setva()
runl()


#dataset_list=[]
#nameset_list=[]
#proba=[]
fenetre.mainloop()

#visuinter()
errorfile.close() 
