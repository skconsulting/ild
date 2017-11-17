# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017

@author: sylvain
predict images in predict_source, with weights in pickle_train
"""

#from __future__ import print_function

from param_pix import classif,classifc
from param_pix import white,black
from param_pix import normi,remove_folder,norm,rsliceNum,preprocess_batch,get_model
from param_pix import scan_bmp,lungmask,lungmask1,lung_namebmp,image_rows,image_cols,typei,num_bit
from param_pix import sroi,source,MAX_BOUND,typei1

import cPickle as pickle
import dicom
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
#from skimage.io import imsave
predict_source_TOP='PREDICT'
predict_source='predict_new' #path for images to  predict 
predict_source='predict_chu' #path for images to  predict 


pickle_dir_model_top='TRAIN_SET'
pickle_dir_modelp='pickle_train_set_2'#path for class weight
pickel_train='pickle' #path for weights

classifnotvisu=['back_ground','healthy']

thrproba=0.
ldummy=False

attn=0.4 #attenuation color
if ldummy:
    print 'mode dummy'
cwd=os.getcwd()

(cwdtop,tail)=os.path.split(cwd)
namedirtopc=os.path.join(cwdtop,predict_source_TOP)
namedirtopc=os.path.join(namedirtopc,predict_source)

pickle_dir_model=os.path.join(cwdtop,pickle_dir_model_top)
pickle_dir_model=os.path.join(pickle_dir_model,pickle_dir_modelp)

pickle_dir_train=os.path.join(pickle_dir_model,pickel_train)


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

    deltax=100*((labnow)//5)
    deltay=11*((labnow)%5)
#    gro=-x*0.0027+1.2
    gro=0.3
#    print x+deltax,y+deltay,label
    newv=np.zeros((tab.shape[0],tab.shape[1],3),dtype=np.uint8)
    viseg=cv2.putText(newv,label,(x+deltax, y+deltay+10), font,gro,col,1)
    visegb = cv2.cvtColor(viseg ,cv2.COLOR_RGB2GRAY)
    visegb = cv2.cvtColor(visegb ,cv2.COLOR_GRAY2RGB)
    tabc=np.copy(tab)
    np.putmask(tabc,visegb>0,0)
    viseg=cv2.putText(tab,label,(x+deltax, y+deltay+10), font,gro,col,1)
    
    return viseg

def colorimage(image,color):
#    im=image.copy()
    im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    np.putmask(im,im>0,color)
    return im

def visu(namedirtopcf,imgs_mask_test,num_list,dimtabx,dimtaby,tabscan,sroidir,tabscanLung):
    print 'visu starts'    
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
#    imgpat = np.zeros((dimtabx, dimtaby,3), dtype=np.uint8)
#    pred_dir = 'preds
    pred_dir=os.path.join(namedirtopcf,predict_result)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for slicenumber in num_list:   
        for i in range (imgs_mask_test.shape[0]):
            patlist=[]
        num=num_list.index(slicenumber)
        imclass0=np.argmax(imgs_mask_test[num], axis=2).astype(np.uint8)
        imamax=np.amax(imgs_mask_test[num], axis=2)
        
        imamaxc=imamax.copy()
        tablung1=np.copy(tabscanLung[slicenumber])   
        np.putmask(tablung1,tablung1>0,255)
   
        np.putmask(imamax,imamax>=thrproba,255)
        np.putmask(imamax,imamax<thrproba,0)
    
        imamax=imamax.astype(np.uint8)
        imclass=np.bitwise_and(imamax,imclass0)
        np.putmask(imclass,imamaxc<thrproba,classif['healthy'])
        imclass=np.bitwise_and(tablung1, imclass) 
               
        for key,value in classif.items():
                
                if key not in classifnotvisu:
                    imcc=imclass.copy()
                  
                    np.putmask(imcc,imcc!=classif[key],0)
                    if imcc.max()>0:
                        imgnpc=colorimage(imcc,classifc[key])
                        patlist.append(key)               
                        imgs[num]=cv2.add(imgs[num],imgnpc)
                    
        for p in patlist:
                        imgs[num]=tagviewn(imgs[num],p,340,440)
    
    
        uniquelbls = np.unique(patlist)
        print 'list of patterns',uniquelbls
        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)
		
        if roip:
            imgn=cv2.imread(tabroi[slicenumber])
            imgnc=cv2.resize(imgn,(dimtabx, dimtaby),interpolation=cv2.INTER_LINEAR)
#            
        else:
            print 'no roi'
            imgnc=normi(tabscan[slicenumber])
        imgnc= cv2.cvtColor(imgnc,cv2.COLOR_RGB2BGR)
#        imgs[num] = cv2.cvtColor(imgs[num] ,cv2.COLOR_RGB2BGR)
        image2=cv2.add(imgnc,imgs[num])
		
        plt.figure(figsize = (10, 7))
		
        plt.subplot(1,3,1)
        plt.title('source')
        plt.imshow(imgnc)
        
        plt.subplot(1,3,2)
        plt.title('predict')
        plt.imshow( imgs[num] )
        
        plt.subplot(1,3,3)
        plt.title('source+predict')
        plt.imshow( imgnc )
        plt.imshow( imgs[num], alpha=0.5 )
        plt.show()
#        image2= cv2.cvtColor(image2,cv2.COLOR_RGB2BGR)
        image2= cv2.cvtColor(image2,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(pred_dir, str(slicenumber) + '.bmp'), image2)
    
 


def tagviews(tab,text,x,y):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,text,(x, y), font,0.3,white,1)
    return viseg

def genebmp(dirName,fileList,slnt,hug):
    """generate patches from dicom files and sroi"""
    print ('generate  bmp files from dicom files in :',f)
    sliceok=[]
    (top,tail)=os.path.split(dirName)
#    global constPixelSpacing, dimtabx,dimtaby
    #directory for patches
    bmp_dir = os.path.join(dirName, scan_bmp)
    if os.path.exists(bmp_dir):
        remove_folder(bmp_dir)

    os.mkdir(bmp_dir)
    if hug:
        lung_dir = os.path.join(dirName, lungmask)
        if not os.path.exists(lung_dir):
            lung_dir = os.path.join(dirName, lungmask1)
        
    else:
        (top,tail)=os.path.split(dirName)
        lung_dir = os.path.join(top, lungmask)
        if not os.path.exists(lung_dir):
            lung_dir = os.path.join(top, lungmask1)
    lung_bmp_dir = os.path.join(lung_dir,lung_namebmp)

    lunglist = [name for name in os.listdir(lung_dir) if ".dcm" in name.lower()]
    lunglistbmp = [name for name in os.listdir(lung_bmp_dir) if ".bmp" in name.lower()]


    tabscan=np.zeros((slnt,image_rows,image_cols),np.int16)
    tabslung=np.zeros((slnt,image_rows,image_cols),np.uint8)
#    os.listdir(lung_dir)
    for filename in fileList:
#            print(filename)
            FilesDCM =(os.path.join(dirName,filename))
            RefDs = dicom.read_file(FilesDCM,force=True)
            dsr= RefDs.pixel_array
            dsr=dsr.astype('int16')

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
            if dsr.shape[0]!= image_cols:
                 dsr = dsr.astype('float32')
                 dsr=cv2.resize(dsr,(image_rows,image_cols),interpolation=cv2.INTER_LINEAR)

            dsr = dsr.astype('int16')

#            dsr=cv2.resize(dsr,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)

            dsrforimage=normi(dsr)
#            dsrforimage = bytescale(dsr, low=0, high=255)
            
           
            tabscan[scanNumber]=dsr
             
            imgcored=imgcoredeb+typei
            bmpfiled=os.path.join(bmp_dir,imgcored)
            textw='n: '+tail+' scan: '+str(scanNumber)
#            print imgcoresroi,bmpfileroi
            dsrforimage=tagviews(dsrforimage,textw,0,20)
#            dsrforimage=cv2.cvtColor(dsrforimage,cv2.COLOR_GRAY2BGR)
            cv2.imwrite (bmpfiled, dsrforimage,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
                       

    if len(lunglist)>0:
        for lungfile in lunglist:
    #             print(lungfile)
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
                        dsrresize=cv2.resize(dsr,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)
        
                        cv2.imwrite (bmpfile, dsrresize)
        #                np.putmask(dsrresize,dsrresize==1,0)
                        np.putmask(dsrresize,dsrresize>0,1)
                        tabslung[scanNumber]=dsrresize
    else:

            for img in lunglistbmp:
                slicenumber= rsliceNum(img,'_','.'+typei1)
                imr=cv2.imread(os.path.join(lung_bmp_dir,img),0) 
    
                if imr.max()>0: 
                    imr=cv2.resize(imr,(image_cols,image_rows),interpolation=cv2.INTER_LINEAR)
                    np.putmask(imr,imr>0,100)          
                    tabslung[slicenumber]=imr
            
            
    return tabscan,tabslung


def preparscan(tabscan,tabslung):
#    (top,tail)=os.path.split(namedirtopcf)

    
    scan_list=[]
    num_list=[]
    for num in numsliceok:
        print num
        tabl=tabslung[num].copy()
        scan=tabscan[num].copy()    
#        print scan.min(),scan.max()
#        print tabl.min(),tabl.max()
        np.putmask(tabl,tabl>0,255)

        taba=cv2.bitwise_and(scan,scan,mask=tabl)
#        print taba.min(),taba.max()
#        np.putmask(tabl,tabl>0,1)
        tablc=tabl.astype(np.int16)
 
        np.putmask(tablc,tablc==0,MAX_BOUND)
        np.putmask(tablc,tablc==255,0)
#        print type(taba[0][0]),taba.min(),taba.max()
#        print type(tablc[0][0]),tablc.min(),tablc.max()
        tabab=taba+tablc
        tababn=norm(tabab)

        scan_list.append(tababn)
#        print tabab.min(),tabab.max()
        num_list.append(num)
    X_train = np.asarray(np.expand_dims(scan_list,3)) 
    if num_bit ==3:
     X_train=np.repeat(X_train,3,axis=3)     
   
    return X_train,num_list


def preparscandummy(namedirtopcf):
    print 'pre dummy',namedirtopcf
    
    listimage =[name for name in os.listdir(namedirtopcf) if name.find('.bmp')>0]
#    (top,tail) =os.split(namedirtopcf)
    print listimage
    tabscan=np.zeros((100,image_rows,image_cols),np.uint8)
    scan_list=[]
    num_list=[]
    for i in listimage:
        imgf=os.path.join(namedirtopcf,i)
        if num_bit==3:
            img=cv2.imread(imgf)
        else:
            img=cv2.imread(imgf,0)
            
        scan_list.append(img)
        print img.shape, img.min(), img.max()
        
        num=rsliceNum(imgf,'_','.bmp')
#        print tabab.min(),tabab.max()
#        print num
        num_list.append(num)
        tabscan[num]=img
#        print img.shape, img.min(), img.max()
#        print  tabscan[num].shape,  tabscan[num].min(),  tabscan[num].max()
#        
#        print num
#        cv2.imshow('tabscan[num]',normi(tabscan[num]))
#        cv2.imshow('img',normi(img))
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    X_train = np.asarray(scan_list) 
    print X_train.shape
    X_train=preprocess_batch(X_train)
    if num_bit==1:
        X_train = np.expand_dims(X_train,3) 
#    print X_train.shape
    return X_train,num_list,tabscan




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

    weights,num_class=load_train_data(pickle_dir_model)

 
#    model = get_unet(num_class,image_rows,image_cols)
#    model = get_model(num_class,image_rows,image_cols,weights)
    model = get_model(num_class,num_bit,image_rows,image_cols,False,weights,False)
#    mloss = weighted_categorical_crossentropy(weights).myloss
#    model.compile(optimizer=Adam(lr=1e-5), loss=mloss, metrics=['categorical_accuracy'])
#    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    namelastc=load_model_set(pickle_dir_train)
#    namelastc=load_model_set('C:/ProgramData/MedikEye/Score2/CNNparameters/sk0_c0')

    model.load_weights(namelastc)
    return model
#model =load_model_set(pickle_dir_train)

def genepara(fileList,namedir):
    fileList =[name for name in  os.listdir(namedir) if ".dcm" in name.lower()]
    slnt=0
    for filename in fileList:
        FilesDCM =(os.path.join(namedir,filename))
        RefDs = dicom.read_file(FilesDCM,force=True)
        scanNumber=int(RefDs.InstanceNumber)
        if scanNumber not in numsliceok:

                numsliceok.append(scanNumber)
        if scanNumber>slnt:
            slnt=scanNumber
    print 'number of slices', slnt
    slnt=slnt+1
    return slnt



model=loadmodel()

def fullprint(*args, **kwargs):
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold='nan')
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)

print('work in directory:',namedirtopc)
for f in listdirc:
    print '-----------'
    print('work on:',f)
    print '-----------'
    numsliceok=[]
    namedirtopcf=os.path.join(namedirtopc,f)
    sroidir=os.path.join(namedirtopcf,sroi)
    contenudir = [name for name in os.listdir(namedirtopcf) if name.find('.dcm')>0]

    
    if not ldummy:
        if len(contenudir)>0:
            slnt = genepara(contenudir,namedirtopcf)
            tabscan,tabslung=genebmp(namedirtopcf,contenudir,slnt,True)
        else:
              namedirtopcfs=os.path.join(namedirtopcf,source)
              contenudir = [name for name in os.listdir(namedirtopcfs) if name.find('.dcm')>0]
              slnt = genepara(contenudir,namedirtopcfs)
              print slnt
              tabscan,tabslung=genebmp(namedirtopcfs,contenudir,slnt,False)
        X_predict,num_list=preparscan(tabscan,tabslung)       
    else:
         X_predict,num_list,tabscan=preparscandummy(namedirtopcf) 
    print 'Xpredict :', X_predict.shape
    print 'Xpredict min max :', X_predict.min(),X_predict.max()
#    cv2.imwrite('b.bmp',normi(X_predict[0]))
#    abmp=cv2.imread('C:/ProgramData/MedikEye/Score2/modulepython/a.bmp')
#    difab=normi(X_predict[0])-abmp
#    print difab.min(),difab.max()
#    cv2.imwrite('c.bmp',difab)
#    ooo
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
   
    visu(namedirtopcf,imgs_mask_test,num_list,image_cols,image_rows,tabscan,sroidir,tabslung)
    
    