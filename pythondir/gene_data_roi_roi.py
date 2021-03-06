# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017
Create Xtrain etc data for training 2nd step
use ROI data to generate dat: 1 pattern after another + patchassembly
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


nametopdummy='DUMMY' # name of top directory for dummy images with patches
toppatchdummy= 'TOPPATCH' #for dummy scan with patches
namesubdummy='lu_training' #for dummy scan with patches
nsubsubdummy='lu_f'  #for dummy scan with patches

nameHug='HUG'
#nameHug='CHU'
#nameHug='DUMMY'

#extension for output dir
#extendir='ILD_TXT'
#extendir='UIP'
extendirdummy=namesubdummy
#extendir='ILD9'
#extendir='S3'
#extendir='lu_predic'
#extendir='small1'

toppatch= 'TOPROI' #for scan classified ROI
extendir='ILD1'  #for scan classified ROI
#toppatch= 'TOPPATCH'

numberperset=80

extendir2=''
pklnum=2
pickel_dirsource_root='pickle'

##############################################################

sepextend2='ROI'
if len (extendir2)>0:
    sepextend2='_'

pickel_dirsource=pickel_dirsource_root+'_'+sepextend2+'_'+extendir


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
print 'work on :',patchtoppath , 'for scan data input '
patchpicklename='picklepatches.pkl'
roipicklepath = 'roipicklepatches'
picklepatches='picklepatches'
picklepathdir =os.path.join(patchtoppath,roipicklepath) # path scan classified by ROI
pathdummy =os.path.join(cwdtop,nametopdummy)
pathdummy =os.path.join(pathdummy,toppatchdummy+'_'+namesubdummy)
pathdummy =os.path.join(pathdummy,picklepatches)
pathdummy =os.path.join(pathdummy,nsubsubdummy) #path for dummy scan with patches

print 'path for dummy images with patches' ,pathdummy
lisdummy=os.listdir(pathdummy)
lenlisdummy=len(lisdummy)
print lenlisdummy

def get_class_weights(y):
    counter = collections.Counter(y)
    majority = max(counter.values())
#    for cls, count in counter.items():
#        print cls, count
    return  {cls: float(majority/count) for cls, count in counter.items()}

#remove_folder(pickle_dir)

def readclassesdummy(lisdummy,indexdummy,indexaug):

    patpick=os.path.join(pathdummy,lisdummy[indexdummy])
    readpkl=pickle.load(open(patpick, "rb"))
                                            
    scanr=readpkl[0][0]
    scan=geneaug(scanr,indexaug)
    maskr=readpkl[1][0]
    mask=geneaug(maskr,indexaug)
#    cv2.imshow(str(indexdummy)+'maskdummy',normi(mask))
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    scanm=norm(scan)

    return scanm, mask   



def readclasses(pat,namepat,indexpat,indexaug):

    patpick=os.path.join(picklepathdir,pat)
    patpick=os.path.join(patpick,namepat[indexpat])
    readpkl=pickle.load(open(patpick, "rb"))
                                            
    scanr=readpkl[0][0]
    scan=geneaug(scanr,indexaug)
    maskr=readpkl[1][0]
    mask=geneaug(maskr,indexaug)
#    cv2.imshow(pat+str(indexpat)+'mask',normi(mask))
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    scanm=norm(scan)

    return scanm, mask  

def numbclasses(y):
    y_train = np.array(y)
    uniquelbls = np.unique(y_train)
    for pat in uniquelbls:
        print  fidclass(pat,classif)
    
    nb_classes = int( uniquelbls.shape[0])
    print ('number of classes  in this data:', int(nb_classes)) 
    nb_classes = len( classif)
    print ('number of classes  in this set:', int(nb_classes)) 
    y_flatten=y_train.flatten()
    class_weights= get_class_weights(y_flatten)
#    class_weights[0]=class_weights[0]/200.0
#    class_weights[1]=class_weights[1]/100.0
    
    return int(nb_classes),class_weights

def readclasses2(num_classes,X_trainl,y_trainl):

    X_traini, X_testi, y_traini, y_testi = train_test_split(X_trainl,
                                        y_trainl,test_size=0.2, random_state=42)
    
    
    
    X_train = np.asarray(np.expand_dims(X_traini,3)) 

    X_test = np.asarray(np.expand_dims(X_testi,3))  


    y_train = np.array(y_traini)
    y_test = np.array(y_testi)
#    print X_train.shape
#    print y_train.shape
#    print y_train[3].min(),y_train[3].max()
#    o=normi(X_train[3])
#    x=normi(y_train[3])
#    print x.min(),x.max()
###            f=normi(tabroif)
#    cv2.imshow('X_train',o)
#    cv2.imshow('y_train',x)
###            cv2.imshow('tabroif',f)
#    cv2.imwrite('a.bmp',o)
#    cv2.imwrite('b.bmp',x)
#    cv2.imwrite('c.bmp',y_train[3])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    lytrain=y_train.shape[0]
    lytest=y_test.shape[0]
#    print lytrain,lytest
   
    ytrainr=np.zeros((lytrain,image_rows,image_cols,int(num_classes)),np.uint8)
    ytestr=np.zeros((lytest,image_rows, image_cols,int(num_classes)),np.uint8)
    
    
    for i in range (lytrain):
        for j in range (0,image_rows):
#            print 'y_train[i][j]',y_train[i][j]
#            print y_train[i][j].shape
            
            ytrainr[i][j] = np_utils.to_categorical(y_train[i][j], num_classes)
#            print 'ytrainr[i][j]',ytrainr[i][j]

    for i in range (lytest):
        for j in range (0,image_rows):
            ytestr[i][j] = np_utils.to_categorical(y_test[i][j], num_classes)
  
    """
    for i in range (lytrain):
#        yt=y_train[i].reshape(image_rows*image_cols)
        yt=y_train[i]

#        print yt.shape
        ytrainr[i]=np_utils.to_categorical(yt, num_classes)
    for i in range (lytest):
#        yt=y_train[i].reshape(image_rows*image_cols)
        yt=y_train[i]
        print 'yt.shape',yt.shape
        ytestr[i]=np_utils.to_categorical(yt, num_classes)
        print 'ytestr[i].shape',ytestr[i].shape
    
    print ytrainr.shape
    print ytestr.shape
#    ooo
    """
#    print 'ytestr[i].shape',ytestr[0].shape
#    print 'ytestr[i].shape',ytestr[0][100][100]
        
    
    return X_train, X_test, ytrainr, ytestr  
#    return X_train, X_test, y_train, y_test   


#start main
def geneaug(image,tt):
    if tt==0:
        imout=image
    elif tt==1:
    # 1 90 deg
        imout = np.rot90(image)
    elif tt==2:
    #2 180 deg
        imout = np.rot90(np.rot90(image))
    elif tt==3:
    #3 270 deg
        imout = np.rot90(np.rot90(np.rot90(image)))
    elif tt==4:
    #4 flip fimage left-right
            imout=np.fliplr(image)
    elif tt==5:
    #5 flip fimage left-right +rot 90
        imout = np.rot90(np.fliplr(image))
    elif tt==6:
    #6 flip fimage left-right +rot 180
        imout = np.rot90(np.rot90(np.fliplr(image)))
    elif tt==7:
    #7 flip fimage left-right +rot 270
        imout = np.rot90(np.rot90(np.rot90(np.fliplr(image))))
    elif tt==8:
    # 8 flip fimage up-down
        imout = imout=np.flipud(image)
    elif tt==9:
    #9 flip fimage up-down +rot90
        imout = np.rot90(np.flipud(image))
    elif tt==10:
    #10 flip fimage up-down +rot180
        imout = np.rot90(np.rot90(np.flipud(image)))
    elif tt==11:
    #11 flip fimage up-down +rot270
        imout = np.rot90(np.rot90(np.rot90(np.flipud(image))))

    return imout




listroi=[name for name in os.listdir(picklepathdir)]
numclass=len(listroi)
print'number of classes :', numclass
listscaninroi={}
for c in listroi:
    listscaninroi[c]=os.listdir(os.path.join(picklepathdir,c))

X_trainl={}
y_trainl={}
y_trainlist=[]

for i in range(pklnum):
    print 'set number :',i,' ',numberperset
    patch_list=[]
    label_list=[]
    
    for numgen in range(numberperset):    
    
        pat =listroi[numgen%numclass]
        numberscan=len(listscaninroi[pat])
        #         indexpat1=np.randomint(0,numberscan+1)
        indexpat = random.randint(0, numberscan-1)
        indexaug = random.randint(0, 11)
        indexdummy = random.randint(0, lenlisdummy-1)
        #         X_train,y_trainl=readclasses(pat,listscaninroi[pat],indexpat,indexpat)   
#        print pat,indexpat,indexaug
        scan,mask=readclasses(pat,listscaninroi[pat],indexpat,indexaug)  
        patch_list.append(scan)
        label_list.append(mask)
        if numgen%numclass==0:
            scan,mask=readclassesdummy(lisdummy,indexdummy,indexaug)  
            patch_list.append(scan)
            label_list.append(mask)
       
        
    X_trainl[i]=patch_list
    y_trainl[i]=label_list  
    y_trainlist=y_trainlist+y_trainl[i]



         
#         print indexpat,indexaug
#     pat_t_w_a=geneaug(patch_t_w,indexaug)
#     print len(listscaninroi[pat]),numscan
     
#     X_trainl[i],y_trainl[i]=readclasses(listscaninroi[c],numscan,numrot)   
for i in range(pklnum):
    print len(X_trainl[i]),len(y_trainl[i]),len(y_trainlist)
   
num_classes,class_weights=numbclasses(y_trainlist)
print "weights"
setvalue=[]
for key,value in class_weights.items():
   print key, fidclass (key,classif), value
   setvalue.append(key)
print('-' * 30)
print 'after adding for non existent :'
for numw in range(num_classes):
    if numw not in setvalue:
        class_weights[numw]=0.01
for key,value in class_weights.items():
   print key, fidclass (key,classif), value
print('-' * 30)

for i in range(pklnum):
    print 'work on subset :',i
    diri=os.path.join(pickle_dir,str(i))
    remove_folder(diri)
    os.mkdir(diri)
    X_train, X_test, y_train, y_test =readclasses2(num_classes,X_trainl[i],y_trainl[i])
    print 'shape X_train :',X_train.shape
    print 'shape X_test :',X_test.shape
    print 'shape y_train :',y_train.shape
    print 'shape y_test :',y_test.shape
    print('-' * 30)
    pickle.dump(X_train, open( os.path.join(diri,"X_train.pkl"), "wb" ),protocol=-1)
    pickle.dump(y_train, open( os.path.join(diri,"y_train.pkl"), "wb" ),protocol=-1)
    pickle.dump(X_test, open( os.path.join(diri,"X_test.pkl"), "wb" ),protocol=-1)
    pickle.dump(y_test, open( os.path.join(diri,"y_test.pkl"), "wb" ),protocol=-1)
pickle.dump(class_weights, open( os.path.join(pickle_dir,"class_weights.pkl"), "wb" ),protocol=-1)
debug=True
if debug:
    xt=  pickle.load(open( os.path.join(diri,"X_train.pkl"), "rb" ))
    yt=pickle.load(open( os.path.join(diri,"y_train.pkl"), "rb" ))
    xcol=30
    ycol=20
    for i in range(4):
        numtosee=i
        print 'numtosee',numtosee
        print 'xt', xt.shape
        print 'xt', xt[0][:,:,0].shape
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
