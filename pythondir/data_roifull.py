# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:03:33 2017

@author: sylvain
generate data for segmentation roi
"""
#from __future__ import print_function
from param_pix import *
import os
import dicom
import numpy as np
import cv2
 
import cPickle as pickle
import shutil
import time
cwd=os.getcwd()


(cwdtop,tail)=os.path.split(cwd)
nameHug='HUG'
subHUG='ILD4'
path_HUG=os.path.join(cwdtop,nameHug)
#path_HUG=os.path.join(nameHug,namsubHug)
namedirtopc =os.path.join(path_HUG,subHUG)

toppatch= 'TOPPATCH'
#extension for output dir
#extendir='pix1'
#extendir='essai3'
extendir=subHUG


patchesdirnametop = toppatch+'_'+extendir
patchtoppath=os.path.join(path_HUG,patchesdirnametop)
patchpicklename='picklepatches.pkl'
picklepath = 'picklepatches'
picklepathdir =os.path.join(patchtoppath,picklepath)

if not os.path.isdir(patchtoppath):
    os.mkdir(patchtoppath)

if not os.path.isdir(picklepathdir):
    os.mkdir(picklepathdir)

patchtoppath=os.path.join(path_HUG,patchesdirnametop)

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
        np.putmask(tabl,tabl>0,1)
        
        pathpicklepatfile=os.path.join(pathpicklepat,patchpicklenamepatient)
        
        maskr=tabroi[num].copy()   
#        print 'maskr',maskr.min(),maskr.max()
#        maskrc=maskr.copy()

        np.putmask(maskr,maskr>0,255)
        
        maskr=np.bitwise_not(maskr)          
        
        roi=cv2.bitwise_and(tabl,tabl,mask=maskr)
        
        roif=cv2.add(roi,tabroi[num]) 

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
#            np.putmask(tabl,tabl>0,1)

            newroi = cv2.imread(os.path.join(pathpat2, l), 0)            
            newroic=newroi.copy()
            np.putmask(newroic,newroic>0,1)            

            oldroi=tabroi[numslice].copy().astype(np.uint8)
            tabroinum=oldroi.copy()
            np.putmask(oldroi,oldroi>0,255)
           
            oldroi=np.bitwise_not(oldroi)          
#            np.putmask(oldroi,oldroi<255,0)
           
            
            tabroix=cv2.bitwise_and(newroic,newroic,mask=oldroi)
            
            np.putmask(newroi,newroi>0,classif[pat])
            
            tabroif=cv2.bitwise_and(newroi,newroi,mask=tabroix)
            tabroif=cv2.add(tabroif,tabroinum)
            tabroi[numslice]=tabroif
#            o=normi(oldroi)
#            n=normi(newroi)
#            x=normi(tabroix)
#            f=normi(tabroif)
#            cv2.imshow('oldroi',o)
#            cv2.imshow('newroi',n)
#            cv2.imshow('tabroix',x)
#            cv2.imshow('tabroif',f)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
                
#                                       
            if newroi.max()>0:
               vis=contour2(newroi,pat)
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
    
