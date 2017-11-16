# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017
Create Xtrain etc data for training 
specific for patch based images
2nd step
@author: sylvain

"""

#from __future__ import print_function
from param_pix import cwdtop,image_rows,image_cols

from param_pix import remove_folder,normi,norm,fidclass
from param_pix import classif

import cPickle as pickle
import cv2
import collections
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
#######################################################################################################

pklnum=100 #number of total images to generate
print 'number of turns :',pklnum
perval = 0.2 #percentage for validation

nametopdummy='DUMMY' # name of top directory for dummy images with patches
toppatchdummy= 'TOPPATCH' #for dummy scan with patches
namesubdummy='lu_training' #for dummy scan with patches
nsubsubdummy='lu_f'  #for dummy scan with patches
dummyinclude=False #to add patch roi after each set of pattern

nameHug='IMAGEDIR'
#nameHug='CHU'
#nameHug='DUMMY'

extendirdummy=namesubdummy

toppatch= 'TOPPATCH' #for scan classified ROI
#extendir='ILD_TXT'  #for scan classified ROI
#extendir='ILD0'  #for scan classified ROI
extendir='1'  #for scan classified ROI
#extendir='UIP2'  #for scan classified ROI

pickel_dirsource='TRAIN_SET/pickle_train_set' #path for data fort training
pickel_dirsourcenum='q' #extensioon for path for data for training


##############################################################
validationdir='V'
extendir2=''

pickel_dirsource_root='pickle'


#sepextend2='ROI'
if len (extendir2)>0:
    extendir2='_'+extendir2
#path for cnn training data recording
pickel_dirsource=pickel_dirsource+'_'+pickel_dirsourcenum+extendir2


#cwd=os.getcwd()
#
#(cwdtop,tail)=os.path.split(cwd)
pickle_dir=os.path.join(cwdtop,pickel_dirsource)
print 'path to write data for training',pickle_dir


remove_folder(pickle_dir)
os.mkdir(pickle_dir)


path_HUG=os.path.join(cwdtop,nameHug)
patchesdirnametop = toppatch+'_'+extendir
patchtoppath=os.path.join(path_HUG,patchesdirnametop)


patchpicklename='picklepatches.pkl'
#roipicklepath = 'roipicklepatches'
picklepatches='picklepatches'
classpatch='classpatch'
picklepathdir =os.path.join(patchtoppath,picklepatches) # path scan 
picklepathdir =os.path.join(picklepathdir,classpatch) # path scan 
print 'work on :',picklepathdir , 'for scan data input '


#pathdummy =os.path.join(cwdtop+'/TRAIN_SET',nametopdummy)
#pathdummy =os.path.join(pathdummy,toppatchdummy+'_'+namesubdummy)
#pathdummy =os.path.join(pathdummy,picklepatches)
#pathdummy =os.path.join(pathdummy,nsubsubdummy) #path for dummy scan with patches
#
#print 'path for dummy images with patches' ,pathdummy
#lisdummy=os.listdir(pathdummy)
#lenlisdummy=len(lisdummy)
#print 'number of images in dummy',lenlisdummy

def get_class_weights(y):
    counter = collections.Counter(y)
    majority = max(counter.values())
#    for cls, count in counter.items():
#        print cls, count
    return  {cls: float(majority/count) for cls, count in counter.items()}

#remove_folder(pickle_dir)


def readclasses(lisscan,picklepathdir,indexpat,indexaug):
        
    patpick=os.path.join(picklepathdir,lisscan[indexpat])
    readpkl=pickle.load(open(patpick, "rb"))
                                            
    scanr=readpkl[0]
    scan=geneaug(scanr,indexaug)
    maskr=readpkl[1]
    mask=geneaug(maskr,indexaug)

    return scan, mask  

def numbclasses(y):
    y_train = np.array(y)
    uniquelbls = np.unique(y_train)
    print 'list of roi:'
    for pat in uniquelbls:
        print  fidclass(pat,classif)
    print '-----------'
    nb_classes = int( uniquelbls.shape[0])
    print ('number of classes  in this data:', int(nb_classes)) 
    nb_classes = len( classif)
    print ('number of classes  in this set:', int(nb_classes)) 
    y_flatten=y_train.flatten()
    class_weights= get_class_weights(y_flatten)
#    class_weights[0]=class_weights[0]/200.0
#    class_weights[1]=class_weights[1]/100.0
    
    return int(nb_classes),class_weights

def readclasses2(num_classes,X_traini,y_traini,X_testi,y_testi):

       
    X_train = np.asarray(np.expand_dims(X_traini,3)) 
    X_test = np.asarray(np.expand_dims(X_testi,3))  

    y_train = np.array(y_traini)
    y_test = np.array(y_testi)

    lytrain=y_train.shape[0]
    lytest=y_test.shape[0]
#    print lytrain,lytest
   
    ytrainr=np.zeros((lytrain,image_rows,image_cols,int(num_classes)),np.uint8)
    ytestr=np.zeros((lytest,image_rows, image_cols,int(num_classes)),np.uint8)
    
    
    for i in range (lytrain):
        for j in range (0,image_rows):
            
            ytrainr[i][j] = np_utils.to_categorical(y_train[i][j], num_classes)


    for i in range (lytest):
        for j in range (0,image_rows):
            ytestr[i][j] = np_utils.to_categorical(y_test[i][j], num_classes)
    
    return X_train, X_test, ytrainr, ytestr  



#start main
def geneaug(image,tt):
    if tt==0:
        imout=image
    elif tt==1:
    # 1 90 deg
        imout = np.rot90(image,1)
    elif tt==2:
    #2 180 deg
        imout = np.rot90( image,2)
    elif tt==3:
    #3 270 deg
        imout = np.rot90(image,3)
    elif tt==4:
    #4 flip fimage left-right
            imout=np.fliplr(image)
    elif tt==5:
    #5 flip fimage left-right +rot 90
        imout = np.rot90(np.fliplr(image))
    elif tt==6:
    #6 flip fimage left-right +rot 180
        imout = np.rot90(np.fliplr(image),2)
    elif tt==7:
    #7 flip fimage left-right +rot 270
        imout = np.rot90(np.fliplr(image),3)
  
    return imout



###########################################################"

lisscan=[name for name in os.listdir(picklepathdir)]
#
numscan=len(lisscan)
print '-----------'
print'number of scan images in scan:', numscan
print '-----------'

print '-----------'
patch_list=[]
label_list=[]
    
for numgen in range(pklnum):
      
        indexpat=random.randint(0, numscan-1)
        indexaug = random.randint(0, 7)

#        print numgen ,indexpat,numscan
        scan,mask=readclasses(lisscan,picklepathdir,indexpat,indexaug)  
        patch_list.append(scan)
        label_list.append(mask)
    
print 'number of data',len(patch_list)
print '-----------'

X_train, X_test, y_train, y_test = train_test_split(patch_list,
                                        label_list,test_size=perval)
#                                        , random_state=42)
num_classes,class_weights=numbclasses(y_test)
print 'weights:'
setvalue=[]
for key,value in class_weights.items():
   print key, fidclass (key,classif), value
   setvalue.append(key)
print('-' * 30)
print 'after adding for non existent :'
for numw in range(num_classes):
    if numw not in setvalue:
        class_weights[numw]=1.0
for key,value in class_weights.items():
   print key, fidclass (key,classif), value
print('-' * 30)

X_train, X_test, y_train, y_test= readclasses2(num_classes,X_train,y_train,X_test,y_test)

print 'shape X_train :',X_train.shape
print 'shape X_test :',X_test.shape
print 'shape y_train :',y_train.shape
print 'shape y_test :',y_test.shape
print '-----------'
print X_train.min(),X_train.max(),np.mean(X_train)
diri=os.path.join(pickle_dir,validationdir)
remove_folder(diri)
os.mkdir(diri)
pickle.dump(X_test, open( os.path.join(diri,"X_test.pkl"), "wb" ),protocol=-1)
pickle.dump(y_test, open( os.path.join(diri,"y_test.pkl"), "wb" ),protocol=-1)
pickle.dump(X_train, open( os.path.join(diri,"X_train.pkl"), "wb" ),protocol=-1)
pickle.dump(y_train, open( os.path.join(diri,"y_train.pkl"), "wb" ),protocol=-1)
pickle.dump(class_weights, open( os.path.join(pickle_dir,"class_weights.pkl"), "wb" ),protocol=-1)

#splx=np.array_split(X_train,pklnum)
#sply=np.array_split(y_train,pklnum)
#
##for i in range(pklnum):
##    print 'shape set :',i, splx[i].shape, sply[i].shape
#print '-----------'
#for i in range(pklnum):
#    print 'work on subset :',i
#    diri=os.path.join(pickle_dir,str(i))
#    remove_folder(diri)
#    os.mkdir(diri)
#
#    print 'shape X_train :',splx[i].shape 
#    print 'shape y_train :',sply[i].shape
#
#    print('-' * 30)
#pickle.dump(splx[i], open( os.path.join(diri,"X_train.pkl"), "wb" ),protocol=-1)
#pickle.dump(sply[i], open( os.path.join(diri,"y_train.pkl"), "wb" ),protocol=-1)

debug=True
if debug:
#    for j in range(3):
#        diri=os.path.join(pickle_dir,str(j))
#        print 'set',j
        xt=  pickle.load(open( os.path.join(diri,"X_train.pkl"), "rb" ))
        yt= pickle.load(open( os.path.join(diri,"y_train.pkl"), "rb" ))
        xcol=30
        ycol=20
        for i in range(3):
            numtosee=i
            print 'numtosee',numtosee
            print 'xt', xt.shape
            print 'type xt', type(xt[0][0][0][0])
            print 'xt[0][0][0]',xt[numtosee][0][0]
            print 'xt[0][350][160]',xt[numtosee][ycol][xcol]
            print 'yt', yt.shape
            print 'yt[0][0][0]',yt[numtosee][0][0]
            print 'yt[0][350][160]',yt[numtosee][ycol][xcol]
            print 'xt min max', xt[numtosee].min(), xt[3].max()
            print 'yt min max',yt[numtosee].min(), yt[3].max()
            plt.figure(figsize = (5, 5))
            #    plt.subplot(1,3,1)
            #    plt.title('image')
            
            #    plt.imshow( np.asarray(crpim) )
            plt.subplot(1,2,1)
            plt.title(str(i)+'image')
            plt.imshow( normi(xt[numtosee][:,:,0]*10).astype(np.uint8) )
            plt.subplot(1,2,2)
            plt.title(str(i)+'label')
            plt.imshow( np.argmax(yt[numtosee],axis=2) )
            plt.show()
