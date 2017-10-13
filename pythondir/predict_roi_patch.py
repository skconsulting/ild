# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017

@author: sylvain

predict images in predict_source, with weights in pickle_train
can be used for patch recosntructed images , only one image per data set
"""

#from __future__ import print_function

from param_pix import classif,classifc
from param_pix import black
from param_pix import normi,get_model
from param_pix import bmpname,image_rows,image_cols,num_bit
from param_pix import sroi

import cPickle as pickle

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imsave
import os


predicttop='PREDICT'
trainsettop='TRAIN_SET'
#predict_source='predict_dum'
#predict_source='predict_new'
predict_source='predict_patc1'

classifnotvisu=[]

#pickel_train='pickle_lu_f6'
pickel_train='best_wesk3'
#pickel_train='best_weunet'

num_class=len(classif)

#pickel_train='pickle_ILD_TXT'

thrproba=0.1
#ldummy=False

attn=0.4 #attenuation color
#if ldummy:
#    print 'mode dummy'
cwd=os.getcwd()

(cwdtop,tail)=os.path.split(cwd)
namedirtopc=os.path.join(cwdtop,predicttop)
namedirtopc=os.path.join(namedirtopc,predict_source)

pickle_dir_train=os.path.join(cwdtop,trainsettop)
pickle_dir_train=os.path.join(pickle_dir_train,pickel_train)

predict_result='predict_result'


def load_model_set(pickle_dir_train):
    print('Loading saved weights...')
    print('-'*30)
    listmodel=[name for name in os.listdir(pickle_dir_train) if name.find('weights')==0]
#    print listmodel

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


def predictr(X_predict):

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


def visu(namedirtopcf,imgs_mask_test,num_list,dimtabx,dimtaby,tabscan,sroidir,Xpr):
    print 'visu starts'
    souceimage=normi(Xpr[0])
    print souceimage.shape
    souceimage= cv2.cvtColor(souceimage,cv2.COLOR_GRAY2RGB)


    np.set_printoptions(threshold=np.nan)
    print tabscan.shape,tabscan[0][0][0]
    
    imclassroi = np.argmax(tabscan, axis=3)
#    print imclassroi.shape
#    print imclassroi[0][0][0]

    tabroi={}
    for i in num_list:
        tabroi[i]=imclassroi[i].astype(np.uint8)
    
    imgs = np.zeros((imgs_mask_test.shape[0], dimtabx, dimtaby,3), dtype=np.uint8)
#    imgpat = np.zeros((dimtabx, dimtaby,3), dtype=np.uint8)
    
    for i in range (imgs_mask_test.shape[0]):
        patlist=[]
        imi=imgs_mask_test[i]
        imclass=np.argmax(imi, axis=2).astype(np.uint8)
        imamax=np.amax(imi, axis=2)
        np.putmask(imamax,imamax>=thrproba,255)
        np.putmask(imamax,imamax<thrproba,0)
        imamax=imamax.astype(np.uint8)
#        print type(imamax[0][0])
        imclass=np.bitwise_and(imamax,imclass)

        imclassc = np.expand_dims(imclass,2)
        imclassc=np.repeat(imclassc,3,axis=2) 

        for key,value in classif.items():
            
            if key not in classifnotvisu:
                imcc=imclassc.copy()
                bl=(value,value,value)
                blc=[]
                zz=classifc[key]
#                print zz
                for z in range(3):
                    blc.append(int(zz[z]))
#                print imcc[200][200][z]*0.5)
#                print blc
        
#                print key,bl
#                print imcc[200][200]
                np.putmask(imcc,imcc!=bl,black)
#                print imcc[200][200]
    #        print 'im4',imclassc[200][200],imclassc.min(),imclassc.max(),type(imclassc[0][0][0]), np.unique(imclassc)
                np.putmask(imcc,imcc==bl,blc)
#                print imcc[200][200]
                if imcc.max()>0:
                    patlist.append(key)               
                imgs[i]=cv2.add(imgs[i],imcc)
                for p in patlist:
                    delx=int(dimtaby*0.6-120)
                    imgs[i]=tagviewn(imgs[i],p,delx,0)

#            imgs[i]=tagviewn(imgs[i],p,delx,0)

        uniquelbls = np.unique(patlist)
        print 'list of patterns',uniquelbls
        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)
        pred_dir = 'preds'
        pred_dir=os.path.join(namedirtopcf,predict_result)
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
    print num_list
    for image, image_id in zip(imgs, num_list):
        img=tabroi[image_id]
        print img.shape
        print type(img[0][0])
        imgrgb= cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        imgn=normi(img)
#        imgnc=cv2.resize(imgn,(dimtabx, dimtaby),interpolation=cv2.INTER_LINEAR)
#        imgnc= cv2.cvtColor(imgn,cv2.COLOR_GRAY2RGB)
        imgncolor = np.zeros((dimtabx, dimtaby,3), dtype=np.uint8)
        
        for key,value in classif.items():
                print key,value
                imcc=imgrgb.copy()
                print imcc.shape
                print imcc[0][0]
                bl=(value,value,value)
                blc=[]
                zz=classifc[key]
#                print zz
                for z in range(3):
                    blc.append(int(zz[z]))
                print bl, blc
#                print imcc[200][200][z]*0.5)
#                print blc
        
#                print key,bl
#                print imcc[200][200]
                np.putmask(imcc,imcc!=bl,black)
#                imcc1=imcc.copy()
#                print imcc[200][200]
    #        print 'im4',imclassc[200][200],imclassc.min(),imclassc.max(),type(imclassc[0][0][0]), np.unique(imclassc)
                np.putmask(imcc,imcc==bl,blc)
#                imcc2=imcc.copy()
#                print imcc[200][200]              
                imgncolor=cv2.add(imgncolor,imcc)
#        imgnc=np.repeat(imgn,3,axis=2) 
#        print imgnc.shape,imgn.shap
#                cv2.imshow('imgnc',imgnc)
#                cv2.imshow('imcc1',imcc1)
#                cv2.imshow('imcc2',imcc2)
#                cv2.imshow('imgncolor',imgncolor)
#                
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()
                
            
#        image2=cv2.add(image,imgncolor)
#        image2=image
        plt.figure(figsize = (10, 7))
        plt.subplot(1,3,1)
        plt.title('source')
        plt.imshow(imgn)
        
        plt.subplot(1,3,2)
        plt.title('predict')
        plt.imshow( image )
        
        plt.subplot(1,3,3)
        plt.title('source+predict')
        plt.imshow( imgncolor )
#        plt.imshow( imclassc, alpha=0.5 )
        imsave(os.path.join(pred_dir, str(image_id) + '.bmp'), image)
        imsave(os.path.join(sroidir, str(image_id) + '.bmp'), imgncolor)
        imsave(os.path.join(sroidir, str(image_id) + 'b.bmp'), imgn)
        imsave(os.path.join(scanbmp, str(image_id) + '.bmp'), souceimage)



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
  
    return class_weights_r,num_class

def loadmodel(num_class):
    print 'load model'
#    weights,num_class=load_train_data(pickle_dir_train)
    weights=[]
#    model = get_unet(num_class,image_rows,image_cols)
#    model = get_model(num_class,image_rows,image_cols,weights)
    model = get_model(num_class,num_bit,image_rows,image_cols,False,weights)
#    mloss = weighted_categorical_crossentropy(weights).myloss
#    model.compile(optimizer=Adam(lr=1e-5), loss=mloss, metrics=['categorical_accuracy'])
#    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    namelastc=load_model_set(pickle_dir_train)
    model.load_weights(namelastc)
    return model


def fullprint(*args, **kwargs):
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold='nan')
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)


############################################
listdirc= (os.listdir(namedirtopc))
model=loadmodel(num_class)
print('work in directory:',namedirtopc)

for f in listdirc:
    print '-----------'
    print('work on:',f)
    print '-----------'
    numsliceok=[]
    namedirtopcf=os.path.join(namedirtopc,f)
    scanbmp=os.path.join(namedirtopcf,bmpname)
    sroidir=os.path.join(namedirtopcf,sroi)
    if not os.path.exists(sroidir):
        os.mkdir(sroidir)
    if not os.path.exists(scanbmp):
        os.mkdir(scanbmp)
    
    contenudir = [name for name in os.listdir(namedirtopcf) if name.find('.pkl')>0]
    print contenudir
    
    if len(contenudir)>0:
            y_test=pickle.load(open( os.path.join(namedirtopcf,"y_test.pkl"), "rb" ))
            X_test=pickle.load(open( os.path.join(namedirtopcf,"X_test.pkl"), "rb" ))
            slnt =y_test.shape[0]
#            print y_test[0][0][0]
#            tabscan,tabslung=genebmp(namedirtopcf,contenudir,slnt,True)
    else:
        print 'error'
        exit()
    X_predict=  np.expand_dims(X_test[0],0)
    num_list=[0]
    print 'Xpredict :', X_predict.shape
    print 'Xpredict min max :', X_predict.min(),X_predict.max()
#    
    imgs_mask_test=predictr(X_predict)
#    imgs_mask_test=np.flip(imgs_mask_test,3)
    print 'imgs_mask_test shape :', imgs_mask_test.shape
    print 'imgs_mask_test[0][100][100]',imgs_mask_test[0][100][100]
    print 'X_predict[0][100][100]',X_predict[0][100][100]
    print 'imgs_mask_test[0][0][200]',imgs_mask_test[0][0][200]
    
#    print 'X_predict shape :', X_predict.shape
#    print 'X_predict[0][50][200]',X_predict[0][50][0]
#    print 'X_predict[0][0][200]',X_predict[0][0][200]
    
    
    imamax=np.amax(imgs_mask_test[0], axis=2)
#    print 'imamax min max',imamax.min(), imamax.max(),imamax[100][200]
    plt.hist(imamax.flatten(), bins=80, color='c')
    plt.xlabel("proba")
    plt.ylabel("Frequency")
    plt.show()
#    np.putmask(imamax,imamax>0.9,1)
#    np.putmask(imamax,imamax<=0.9,0)
#    print 'imamax[50][0]',imamax[50][0]
#    print 'imamax[0][200]',imamax[0][200]
    
#    np.putmask(imamax,imamax<=0.7,0)
#    np.putmask(imamax,imamax>0.7,1)

#    imclass = np.argmax(imgs_mask_test[0], axis=2)
#    print 'imclass[100][200]',imclass[100][200]
#    print 'imclass[0][0]',imclass[0][0]
#    cv2.imwrite('a.bmp',normi(imclass))
#    cv2.imwrite('b.bmp',imclass*100)
#    cv2.imwrite('c.bmp',normi(imamax))
#    cv2.imwrite('d.bmp',imamax)
#    cv2.imwrite('e.bmp',X_predict[0])
#    print 'imclass shape :',imclass.shape    
#    plt.figure(figsize = (10, 5))
#    plt.subplot(1,2,1)
#    plt.title('imclass')
#    plt.imshow( imclass )
#    plt.subplot(1,2,2)
#    plt.title('image')
#    plt.imshow( np.asarray(crpim) )
#    masked_imclass = np.ma.masked_where(imclass == 0, imclass)
#    plt.imshow( X_predict[0][:,:,0] )
#    print X_predict[0][:,:,0].shape
   
    visu(namedirtopcf,imgs_mask_test,num_list,image_cols,image_rows,y_test,sroidir,X_predict)
    
    