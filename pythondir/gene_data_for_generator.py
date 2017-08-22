# -*- coding: utf-8 -*-
"""
Created on Tue August 21 15:04:39 2017
Create validation data for training 
use ROI data to generate dat: 1 pattern after another + patchassembly
only one validation set for all training sets
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

#######################################################################################################

#pklnum=30 #number of sets (to not have too big databases)
#print 'number of set :',pklnum

#nametopdummy='DUMMY' # name of top directory for dummy images with patches
#toppatchdummy= 'TOPPATCH' #for dummy scan with patches
#namesubdummy='lu_training' #for dummy scan with patches
#nsubsubdummy='lu_f'  #for dummy scan with patches
#dummyinclude=False #to add patch roi after each set of pattern

nameHug='IMAGEDIR'

toppatch= 'TOPROI' #for scan classified ROI

#extendir='ILD_TXT'  #for scan classified ROI
#extendir='ILD0'  #for scan classified ROI
#extendir='0'  #for scan classified ROI
extendir='essai'  #for scan classified ROI

validationdir='V'

##############################################################
pickel_dirsource='pickle' #path for data fort training
#pickel_dirsource1='train' #path for data fort training

pickel_dirsourcenum='train_set' #extensioon for path for data for training

extendir2='3'

#sepextend2='ROI'
if len (extendir2)>0:
    extendir2='_'+extendir2
#path for cnn training data recording
pickle_dir=os.path.join(cwdtop,pickel_dirsource+'_'+pickel_dirsourcenum+extendir2)


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
##pathdummy =os.path.join(cwdtop,nametopdummy)
##pathdummy =os.path.join(pathdummy,toppatchdummy+'_'+namesubdummy)
#pathdummy =os.path.join(pathdummy,picklepatches)
#pathdummy =os.path.join(pathdummy,nsubsubdummy) #path for dummy scan with patches

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

def readclasses2(num_classes,X_testi,y_testi):

    X_test = np.asarray(np.expand_dims(X_testi,3))  
    y_test = np.array(y_testi)
    lytest=y_test.shape[0]
    ytestr=np.zeros((lytest,image_rows, image_cols,int(num_classes)),np.uint8)
    for i in range (lytest):
        for j in range (0,image_rows):
            ytestr[i][j] = np_utils.to_categorical(y_test[i][j], num_classes)
      
    return  X_test,  ytestr  



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


###########################################################"

listroi=[name for name in os.listdir(picklepathdir)]

numclass=len(listroi)
print '-----------'
print'number of classes in scan:', numclass
print '-----------'

listscaninroi={}

indexpatc={}
for j in listroi:
        indexpatc[j]=0

totalimages=0
maximage=0

for c in listroi:
    listscaninroi[c]=os.listdir(os.path.join(picklepathdir,c))
    numberscan=len(listscaninroi[c])
    if numberscan>maximage:
        maximage=numberscan
        ptmax=c
    totalimages+=numberscan
    
print 'number total of scan images:',totalimages
print 'maximum data in one pat:',maximage,' in ',ptmax
print '-----------'
setlist=[]
patch_list=[]
label_list=[]
    
for numgen in range(maximage*numclass):
        pat =listroi[numgen%numclass]
        numberscan=len(listscaninroi[pat])
        indexpatc[pat] =  indexpatc[pat]%numberscan
        indexpat=indexpatc[pat]
        indexpatc[pat]=indexpatc[pat]+1

        indexaug = random.randint(0, 11)
#        print numgen ,pat,indexpat,numberscan
        scan,mask=readclasses(pat,listscaninroi[pat],indexpat,indexaug)  
        elt=(scan,mask)
        setlist.append(elt)
#        label_list.append(mask)
    
print 'number of data',len(setlist)
dpc=len(setlist)/10
print 'length of test value for 10% :',dpc
print '-----------'

rsl=random.sample(setlist,dpc)

x_test=map(lambda x: x[0],rsl)
y_test=map(lambda x: x[1],rsl)

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

X_test, Y_test= readclasses2(num_classes,x_test,y_test)

print 'shape y_train :',X_test.shape
print 'shape y_test :',Y_test.shape
print '-----------'
diri=os.path.join(pickle_dir,validationdir)
remove_folder(diri)
os.mkdir(diri)
pickle.dump(X_test, open( os.path.join(diri,"X_test.pkl"), "wb" ),protocol=-1)
pickle.dump(Y_test, open( os.path.join(diri,"Y_test.pkl"), "wb" ),protocol=-1)
pickle.dump(class_weights, open( os.path.join(pickle_dir,"class_weights.pkl"), "wb" ),protocol=-1)

debug=True
if debug:
        print 'set',j
        xt=  pickle.load(open( os.path.join(diri,"X_test.pkl"), "rb" ))
        yt= pickle.load(open( os.path.join(diri,"Y_test.pkl"), "rb" ))
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
