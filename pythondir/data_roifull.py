# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:03:33 2017

@author: sylvain
generate data for segmentation roi
"""
#from __future__ import print_function

import os
import dicom
import numpy as np
import cv2
#from keras.utils import np_utils
from skimage.io import imsave, imread
from skimage.transform import resize
from PIL import Image, ImageFont, ImageDraw
import cPickle as pickle
import shutil
import time
cwd=os.getcwd()


(cwdtop,tail)=os.path.split(cwd)
nameHug='HUG'
subHUG='ILD1'
path_HUG=os.path.join(cwdtop,nameHug)
#path_HUG=os.path.join(nameHug,namsubHug)
namedirtopc =os.path.join(path_HUG,subHUG)

toppatch= 'TOPPATCH'
#extension for output dir
extendir='essai'

image_rows = 496
image_cols = 496

classif ={
        'back_ground':0,
        'healthy':1,
        'HC':2,
        'ground_glass':3,
        'consolidation':4,
        'micronodules':5,
        'reticulation':6,
        'air_trapping':7,
        'cysts':8,
        'bronchiectasis':9,
#        'emphysema':10,
        'GGpret':10
        } 

usedclassif = [
        'back_ground',
        'consolidation',
        'HC',
        'fibrosis',
        'ground_glass',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        'emphysema',

        'bronchial_wall_thickening',
        'early_fibrosis',
        'increased_attenuation',
        'macronodules',
        'pcp',
        'peripheral_micronodules',
        'tuberculosis'
        ]

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
col1= (142,180,227)
col2=(155,215,204)
col3=(214,226,144)
col4=(234,136,222)
col5=(218,163,152)

classifc ={
'back_ground':darkgreen,
'consolidation':cyan,
'HC':blue,
'ground_glass':red,
'healthy':darkgreen,
'micronodules':green,
'reticulation':yellow,
'air_trapping':pink,
'cysts':lightgreen,
 'bronchiectasis':orange,
 'HCpret': col1,
 'HCpbro': col2,
 'GGpbro': col3,
 'GGpret': col4,
 'bropret': col5,
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




patchesdirnametop = toppatch+'_'+extendir
patchtoppath=os.path.join(path_HUG,patchesdirnametop)
patchpicklename='picklepatches.pkl'
picklepath = 'picklepatches'
picklepathdir =os.path.join(patchtoppath,picklepath)

if not os.path.isdir(patchtoppath):
    os.mkdir(patchtoppath)

if not os.path.isdir(picklepathdir):
    os.mkdir(picklepathdir)
#define the name of directory for normalised patches
#pickle for patches

patchfile='patchfile'
bmpname='scan_bmp'
#directory with lung mask dicom
lungmask='lung_mask'
#directory to put  lung mask bmp
lungmaskbmp='scan_bmp'
typei='jpg' #can be jpg
sroi='sroi'

font5 = ImageFont.truetype( 'arial.ttf', 5)
font10 = ImageFont.truetype( 'arial.ttf', 10)
font20 = ImageFont.truetype( 'arial.ttf', 20)
patchtoppath=os.path.join(path_HUG,patchesdirnametop)
avgPixelSpacing=0.734


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

def genepara(fileList):
    fileList =[name for name in  os.listdir(namedirtopcf) if ".dcm" in name.lower()]
    slnt=0
    for filename in fileList:
        FilesDCM =(os.path.join(namedirtopcf,filename))
        RefDs = dicom.read_file(FilesDCM)
        scanNumber=int(RefDs.InstanceNumber)
        if scanNumber>slnt:
            slnt=scanNumber
    print 'number of slices', slnt
    slnt=slnt+1
    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
    dsr= RefDs.pixel_array
    dsr= dsr-dsr.min()
    dsr=dsr.astype('uint16')
    dsrresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
    dimtabx=int(dsrresize.shape[0])
    dimtaby=int(dsrresize.shape[1])
    return dimtabx,dimtaby,slnt

def normi(img):
     tabi1=img-img.min()
     maxt=float(tabi1.max())
     if maxt==0:
         maxt=1
     tabi2=tabi1*(255/maxt)
     tabi2=tabi2.astype('uint8')
     return tabi2

def tagviews(tab,text,x,y):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,text,(x, y), font,0.3,white,1)
    return viseg
 
def genebmp(dirName,fileList,slnt,dimtabx,dimtaby):
    """generate patches from dicom files and sroi"""
    print ('generate  bmp files from dicom files in :',f)
    (top,tail)=os.path.split(dirName)
#    global constPixelSpacing, dimtabx,dimtaby
    #directory for patches
    bmp_dir = os.path.join(dirName, bmpname)
    remove_folder(bmp_dir)
    os.mkdir(bmp_dir)
    lung_dir = os.path.join(dirName, lungmask)
    lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)
    remove_folder(lung_bmp_dir)
    os.mkdir(lung_bmp_dir)
    
    lunglist = [name for name in os.listdir(lung_dir) if ".dcm" in name.lower()]

    tabscan=np.zeros((slnt,dimtabx,dimtaby),np.int16)
    tabslung=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    tabsroi=np.zeros((slnt,dimtabx,dimtaby,3),np.uint8)
#    os.listdir(lung_dir)
    for filename in fileList:
#            print(filename)
#        if ".dcm" in filename.lower():  # check whether the file's DICOM
            FilesDCM =(os.path.join(dirName,filename))
            RefDs = dicom.read_file(FilesDCM)
            dsr= RefDs.pixel_array
            dsr=dsr.astype('int16')
            fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
#                print 'fxs',fxs
            scanNumber=int(RefDs.InstanceNumber)
            endnumslice=filename.find('.dcm')
            imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)+'.'
#            bmpfile=os.path.join(dirFilePbmp,imgcore)
            dsr[dsr == -2000] = 0
            intercept = RefDs.RescaleIntercept
#            print intercept
            slope = RefDs.RescaleSlope
            if slope != 1:
                dsr = slope * dsr.astype(np.float64)
                dsr = dsr.astype(np.int16)

            dsr += np.int16(intercept)
            dsr = dsr.astype('int16')
#                        print dsr.min(),dsr.max(),dsr.shape
            dsr=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
            dsrforimage=normi(dsr)

            tabscan[scanNumber]=dsr

            imgcored=imgcoredeb+typei
            bmpfiled=os.path.join(bmp_dir,imgcored)
            imgcoresroi='sroi_'+str(scanNumber)+'.'+typei
            bmpfileroi=os.path.join(sroidir,imgcoresroi)
#            print imgcoresroi,bmpfileroi
            textw='n: '+tail+' scan: '+str(scanNumber)

            cv2.imwrite (bmpfiled, dsrforimage,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
            dsrforimage=cv2.cvtColor(dsrforimage,cv2.COLOR_GRAY2BGR)
            dsrforimage=tagviews(dsrforimage,textw,0,20)
            cv2.imwrite (bmpfileroi, dsrforimage)
            tabsroi[scanNumber]=dsrforimage


    for lungfile in lunglist:
#             print(lungfile)
#             if ".dcm" in lungfile.lower():  # check whether the file's DICOM
                FilesDCM =(os.path.join(lung_dir,lungfile))
                RefDs = dicom.read_file(FilesDCM)
                dsr= RefDs.pixel_array
                dsr=dsr.astype('int16')
                fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
#                print 'fxs',fxs
                scanNumber=int(RefDs.InstanceNumber)
                endnumslice=filename.find('.dcm')
                imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)+'.'
                imgcore=imgcoredeb+typei
                bmpfile=os.path.join(lung_bmp_dir,imgcore)
                dsr=normi(dsr)
                dsrresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
                cv2.imwrite (bmpfile, dsrresize)
#                np.putmask(dsrresize,dsrresize==1,0)
                np.putmask(dsrresize,dsrresize>0,1)
                tabslung[scanNumber]=dsrresize

    return tabscan,tabsroi,tabslung

def contour2(im,l):
    col=classifc[l]
    vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
#    im=im.astype('uint8')
    ret,thresh = cv2.threshold(im,0,255,0)
    im2,contours0, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
    cv2.drawContours(vis,contours,-1,col,1,cv2.LINE_AA)
    return vis

def tagview(tab,label,x,y):
    """write text in image according to label and color"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    col=classifc[label]
    labnow=classif[label]
#    print (labnow, text)
    if label == 'back_ground':
        deltay=30
    else:
#        deltay=25*((labnow-1)%5)
        deltay=40+10*(labnow-1)

    viseg=cv2.putText(tab,label,(x, y+deltay), font,0.3,col,1)
    return viseg

def peparescan(numslice,tabs,tabl):
    tablc=tabl.copy().astype(np.int16)
    taba=cv2.bitwise_and(tabs,tabs,mask=tabl)
    np.putmask(tablc,tablc==0,-1000)
    np.putmask(tablc,tablc==1,0)
    tabab=cv2.bitwise_or(taba,tablc)             
    datascan[numslice]=tabab


def preparroi(namedirtopcf):
    (top,tail)=os.path.split(namedirtopcf)
    pathpicklepat=os.path.join(picklepathdir,tail)
    if not os.path.exists (pathpicklepat):
                os.mkdir(pathpicklepat)
    
    for num in numsliceok:
        scan_list=[]
        mask_list=[]
        scan_list.append(datascan[num] )
        patchpicklenamepatient=str(num)+'_'+patchpicklename        
        tabl=tabslung[num].copy()
        pathpicklepatfile=os.path.join(pathpicklepat,patchpicklenamepatient)
        
        maskr=tabroi[num].copy()        
        np.putmask(maskr,maskr>0,1)
        roi=cv2.bitwise_xor(tabl,maskr)
        roif=cv2.bitwise_or(roi,tabroi[num])  
        
        mask_list.append(roif)
        patpickle=(scan_list,mask_list)
        pickle.dump(patpickle, open(pathpicklepatfile, "wb"),protocol=-1)

def create_test_data(namedirtopcf,pat,tabscan,tabsroi,tabslung):
    
    (top,tail)=os.path.split(namedirtopcf)
    print 'create test data for :', tail, 'pattern :',pat
    pathpat=os.path.join(namedirtopcf,pat)
    list_pos=os.listdir(pathpat)
 

    for d in list_pos:
        print 'localisation : ',d
        pathpat2=os.path.join(pathpat,d)
        list_image=os.listdir(pathpat2)
   
        for l in list_image:
            pos=l.find('.')      
            ext=l[pos:len(l)]
            numslice=rsliceNum(l,'_',ext)
            if numslice not in numsliceok:
                numsliceok.append(numslice)
                peparescan(numslice,tabscan[numslice],tabslung[numslice])
                tabroi[numslice]=np.zeros((tabscan.shape[1],tabscan.shape[2]), np.uint8)

#            tabl=tabslung[numslice].copy()
            img_maskc = cv2.imread(os.path.join(pathpat2, l), 0)
            
            maskr=img_maskc.copy()
            roir=tabroi[numslice].copy()
            
            np.putmask(maskr,maskr>0,1)
            np.putmask(roir,roir>0,1)

            tabroix=cv2.bitwise_xor(maskr,roir)
            
            np.putmask(img_maskc,img_maskc>0,classif[pat])
            
            tabroinum=tabroi[numslice]
            tabroif=cv2.bitwise_and(tabroinum,tabroinum,mask=tabroix)
            
            tabroi[numslice]=cv2.add(tabroif,img_maskc)
#                                       
            if img_maskc.max()>0:
               vis=contour2(img_maskc,pat)
               if vis.sum()>0:
                _tabsroi = np.copy(tabsroi[numslice])
                imn=cv2.add(vis,_tabsroi)
                imn=tagview(imn,pat,0,100)
                tabsroi[numslice]=imn
                imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)
                sroifile='sroi_'+str(numslice)+'.'+typei
                filenamesroi=os.path.join(sroidir,sroifile)
                cv2.imwrite(filenamesroi,imn)
        
    return tabsroi

listdirc= (os.listdir(namedirtopc))
for f in listdirc:
    print('work on:',f)
    numsliceok=[]
    namedirtopcf=os.path.join(namedirtopc,f)
    sroidir=os.path.join(namedirtopcf,sroi)
    if os.path.exists(sroidir):
        shutil.rmtree(sroidir)
        time.sleep(1)
    os.mkdir(sroidir)
    contenudir = [name for name in os.listdir(namedirtopcf) if name.find('.dcm')>0]

    dimtabx,dimtaby,slnt = genepara(contenudir)
    tabscan,tabsroi,tabslung=genebmp(namedirtopcf,contenudir,slnt,dimtabx,dimtaby)
    contenupat = [name for name in os.listdir(namedirtopcf) if name in classif]
    datascan={}
    datamask={}
    tabroi={}
    for pat in contenupat:
        print 'work on :',pat
        tabsroi=create_test_data(namedirtopcf,pat,tabscan,tabsroi,tabslung)
    preparroi(namedirtopcf)
    
#    (top,tail)=os.path.split(namedirtopcf)
#    pathpicklepat=os.path.join(picklepathdir,tail)
#    listpkl=os.listdir(pathpicklepat)
#    for l in listpkl:
#        readpkl=pickle.load(open(os.path.join(pathpicklepat,l) ,"rb"))
#        print len(readpkl)
#        print len(readpkl[0])
#        for i in range (len(readpkl[0])):
#            s=normi(readpkl[0][i])
#            m=normi(readpkl[1][i])
#    #            l=normi(readpkl[2][i])
##            t=cv2.add(s,m)
#            cv2.imshow('scan',s)
#            cv2.imshow('mask',m)
##            cv2.imshow(str(i)+' '+tail,t)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
#    