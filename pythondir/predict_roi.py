# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017

@author: sylvain
"""

#from __future__ import print_function
from param_pix import *
import os
import keras
import theano
import cv2
from keras.optimizers import Adam,Adagrad

from skimage.io import imsave
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')
 
from numpy import argmax,amax
import shutil
import time
import dicom
import cPickle as pickle
 
print keras.__version__
print theano.__version__
predict_source='predict_new'
pickel_train='pickle_ILD6/0'

cwd=os.getcwd()

(cwdtop,tail)=os.path.split(cwd)
namedirtopc=os.path.join(cwdtop,predict_source)

pickle_dir_train=os.path.join(cwdtop,pickel_train)

predict_result='predict_result'



def genepara(fileList):
    fileList =[name for name in  os.listdir(namedirtopcf) if ".dcm" in name.lower()]
    slnt=0
    for filename in fileList:
        FilesDCM =(os.path.join(namedirtopcf,filename))
        RefDs = dicom.read_file(FilesDCM)
        scanNumber=int(RefDs.InstanceNumber)
        if scanNumber not in numsliceok:
                numsliceok.append(scanNumber)
        if scanNumber>slnt:
            slnt=scanNumber
    print 'number of slices', slnt
    slnt=slnt+1

    return slnt


def maxproba(proba):
    """looks for max probability in result"""
    lenp = len(proba)
    m=0
    for i in range(0,lenp):
        if proba[i]>m:
            m=proba[i]
            im=i
    return im,m

def load_model_set(pickle_dir_train):
    print('Loading saved weights...')
    print('-'*30)
    listmodel=[name for name in os.listdir(pickle_dir_train) if name.find('weights')==0]

    ordlist=[]
    for name in listmodel:
        nfc=os.path.join(pickle_dir_train,name)
        nbs = os.path.getmtime(nfc)
#        print name,nbs,type(nbs)
        
#        posp=name.find('-')+1
#        post=name.find('.h')
#        numepoch=float(name[posp:post])
        tt=(name,nbs)
#        ordlist.append (tt)
        ordlist.append(tt)

    ordlistc=sorted(ordlist,key=lambda col:col[1],reverse=True)

    namelast=ordlistc[0][0]

    namelastc=os.path.join(pickle_dir_train,namelast)
    print 'last weights :',namelast

    return namelastc


def predictr(X_predict,dimtabx,dimtaby,num_list):

    print('-'*30)
    print('Predicting masks on test data...')

    imgs_mask_test = model.predict(X_predict, verbose=1,batch_size=1)
#   
#    print imgs_mask_test.shape
    return imgs_mask_test

def tagviewn(tab,label,x,y):
    """write text in image according to label and color"""
    col=classifc[label]
    font = cv2.FONT_HERSHEY_SIMPLEX
#    print col, label
    labnow=classif[label]

    deltax=130*((labnow)//10)
    deltay=11*((labnow)%10)
#    gro=-x*0.0027+1.2
    gro=0.3
#    print x+deltax,y+deltay,label
    viseg=cv2.putText(tab,label,(x+deltax, y+deltay+10), font,gro,col,1)
    return viseg


def visu(namedirtopcf,imgs_mask_test,num_list,dimtabx,dimtaby,tabscan,sroidir):
    roip=False
    if os.path.exists(sroidir):
        listsroi=os.listdir(sroidir)
        roip=True
        tabroi={}
        for roil in listsroi:
            posu=roil.find('_')+1
            posp=roil.find('.')
            nums=int(roil[posu:posp])
            tabroi[nums]=os.path.join(sroidir,roil)
    
    imgs = np.zeros((imgs_mask_test.shape[0], dimtabx, dimtaby,3), dtype=np.uint8)
    patlistf=[]
    for i in range (imgs_mask_test.shape[0]):
        imi=imgs_mask_test[i]
        imi=np.moveaxis(imi,0,2)
        patlist=[]
        for j in range (0,dimtabx):
            for k in range(dimtaby):
                proba=imi[j][k]
                numpat=argmax(proba)
                 
                pat=fidclass(numpat,classif)
                patlistf.append(pat)
                if pat not in classifnotvisu:
                    if pat not in patlist:
                        patlist.append(pat)
                    imgs[i][j][k]=classifc[pat]
                
        for p in patlist:
            delx=int(dimtaby*0.6-120)
            imgs[i]=tagviewn(imgs[i],p,delx,0)

    uniquelbls = np.unique(patlistf)
    print 'list of patterns',uniquelbls
    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    pred_dir=os.path.join(namedirtopcf,predict_result)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
        
    for image, image_id in zip(imgs, num_list):
        if roip:
            imgn=cv2.imread(tabroi[image_id])
            imgnc=cv2.resize(imgn,(dimtabx, dimtaby),interpolation=cv2.INTER_LINEAR)
            imgnc= cv2.cvtColor(imgnc,cv2.COLOR_RGB2BGR)
        else:
            imgn=normi(tabscan[image_id])
            imgnc= cv2.cvtColor(imgn,cv2.COLOR_GRAY2BGR)
        image2=cv2.add(image,imgnc)
#        image2=image

        imsave(os.path.join(pred_dir, str(image_id) + '.'+typei), image2)


def tagviews(tab,text,x,y):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,text,(x, y), font,0.3,white,1)
    return viseg

   
def genebmp(dirName,fileList,slnt,dimtabx,dimtaby):
    """generate patches from dicom files and sroi"""
    print ('generate  bmp files from dicom files in :',f)
    sliceok=[]
    (top,tail)=os.path.split(dirName)
#    global constPixelSpacing, dimtabx,dimtaby
    #directory for patches
    bmp_dir = os.path.join(dirName, bmpname)
    if os.path.exists(bmp_dir):
        shutil.rmtree(bmp_dir)
    os.mkdir(bmp_dir)
    lung_dir = os.path.join(dirName, lungmask)
    lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)
    if os.path.exists(lung_bmp_dir):
        shutil.rmtree(lung_bmp_dir)
    os.mkdir(lung_bmp_dir)
    
    lunglist = [name for name in os.listdir(lung_dir) if ".dcm" in name.lower()]

    tabscan=np.zeros((slnt,dimtabx,dimtaby),np.int16)
    tabslung=np.zeros((slnt,dimtabx,dimtaby),np.uint8)

#    os.listdir(lung_dir)
    for filename in fileList:
#            print(filename)
#        if ".dcm" in filename.lower():  # check whether the file's DICOM
            FilesDCM =(os.path.join(dirName,filename))
            RefDs = dicom.read_file(FilesDCM)
            dsr= RefDs.pixel_array
            dsr=dsr.astype('int16')
#            fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
#                print 'fxs',fxs
            scanNumber=int(RefDs.InstanceNumber)
            if scanNumber not in sliceok:
                sliceok.append(scanNumber)
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

            dsr=cv2.resize(dsr,(image_rows, image_cols),interpolation=cv2.INTER_LINEAR)

            dsrforimage=normi(dsr)

            tabscan[scanNumber]=dsr

            imgcored=imgcoredeb+typei
            bmpfiled=os.path.join(bmp_dir,imgcored)
#            print imgcoresroi,bmpfileroi
            textw='n: '+tail+' scan: '+str(scanNumber)

            cv2.imwrite (bmpfiled, dsrforimage,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
            dsrforimage=cv2.cvtColor(dsrforimage,cv2.COLOR_GRAY2BGR)
            dsrforimage=tagviews(dsrforimage,textw,0,20)


    for lungfile in lunglist:
#                print(lungfile)
#             if ".dcm" in lungfile.lower():  # check whether the file's DICOM
                FilesDCM =(os.path.join(lung_dir,lungfile))
                RefDs = dicom.read_file(FilesDCM)
                dsr= RefDs.pixel_array
                dsr=dsr.astype('int16')
#                fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
#                print 'fxs',fxs
                scanNumber=int(RefDs.InstanceNumber)
                if scanNumber in sliceok:
                    endnumslice=filename.find('.dcm')
                    imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)+'.'
                    imgcore=imgcoredeb+typei
                    bmpfile=os.path.join(lung_bmp_dir,imgcore)
                    dsr=normi(dsr)
    #                dsrresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
                    dsrresize=cv2.resize(dsr,(image_rows, image_cols),interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite (bmpfile, dsrresize)
    #                np.putmask(dsrresize,dsrresize==1,0)
                    np.putmask(dsrresize,dsrresize>0,1)
#                    print tabslung.shape,scanNumber
                    tabslung[scanNumber]=dsrresize

    return tabscan,tabslung


def preparscan(namedirtopcf,tabscan,tabslung):
    (top,tail)=os.path.split(namedirtopcf)
    scan_list=[]
    num_list=[]
    for num in numsliceok:
        tabl=tabslung[num].copy()
        scan=tabscan[num].copy()
#        print scan.min(),scan.max()
        
        tablc=tabl.astype(np.int16)
        taba=cv2.bitwise_and(scan,scan,mask=tabl)
        np.putmask(tablc,tablc==0,-1000)
        np.putmask(tablc,tablc==1,0)
        tabab=cv2.bitwise_or(taba,tablc) 
        tababn=norm(tabab)
        scan_list.append(tababn)
#        print tabab.min(),tabab.max()
        num_list.append(num)
    X_train = np.asarray(np.expand_dims(scan_list,1))      
   
    return X_train,num_list

listdirc= (os.listdir(namedirtopc))


def load_train_data(numpidir):

    class_weights=pickle.load(open( os.path.join(numpidir,"class_weights.pkl"), "rb" ))
    clas_weigh_l=[]
    num_class=len(class_weights)
    print 'number of classes :',num_class
    for i in range (0,num_class):
#            print i,class_weights[i]
            clas_weigh_l.append(class_weights[i])
    print 'weights for classes:'
    for i in range (0,num_class):
                    print i, clas_weigh_l[i]
    class_weights_r=np.array(clas_weigh_l)
#    numpatl={}
#    for pat in usedclassif:
#        numpatl[pat]=0
##        print fidclass(classif[pat],classif)
##        print pat, numpatl[pat]
#    y_test1=np.moveaxis(y_test,1,3)
#    for i in range (y_test1.shape[0]):
#        for j in range (0,y_test1.shape[1]):
#            for k in range(y_test1.shape[2]):
#                proba=y_test1[i][j][k]
#                numpat=argmax(proba)     
#                pat=fidclass(numpat,classif)
#                numpatl[pat]+=1
#     
#    for pat in usedclassif:
#        print pat,numpatl[pat]
#    tot=numpatl['back_ground']*1.0
#    for pat in usedclassif:
#        print pat,tot/numpatl[pat]
#        

   
  
    return class_weights_r,num_class

def loadmodel():

    weights,num_class=load_train_data(pickle_dir_train)
 
#    model = get_unet(num_class,image_rows,image_cols)
    model = get_model(num_class,image_rows,image_cols,weights)
#    mloss = weighted_categorical_crossentropy(weights).myloss
#    model.compile(optimizer=Adam(lr=1e-5), loss=mloss, metrics=['categorical_accuracy'])
#    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    namelastc=load_model_set(pickle_dir_train)
    model.load_weights(namelastc)
    return model
#model =load_model_set(pickle_dir_train)

model=loadmodel()

for f in listdirc:
    print('work on:',f)
    numsliceok=[]
    namedirtopcf=os.path.join(namedirtopc,f)
    sroidir=os.path.join(namedirtopcf,sroi)
    contenudir = [name for name in os.listdir(namedirtopcf) if name.find('.dcm')>0]

    slnt = genepara(contenudir)
    
#    print dimtabx,dimtaby,slnt
    tabscan,tabslung=genebmp(namedirtopcf,contenudir,slnt,image_cols,image_rows)
    X_predict,num_list=preparscan(namedirtopcf,tabscan,tabslung)
    print 'Xpredict :', X_predict.shape

#    
    imgs_mask_test=predictr(X_predict,image_cols,image_rows,num_list)
    print 'imgs_mask_test :', imgs_mask_test.shape
#    print imgs_mask_test[0][200][200]
#    pickle.dump(imgs_mask_test, open('imgs_mask_test', "wb"),protocol=-1)
#    imgs_mask_test=pickle.load(open('imgs_mask_test', "rb"))
    visu(namedirtopcf,imgs_mask_test,num_list,image_cols,image_rows,tabscan,sroidir)
    
    