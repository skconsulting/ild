# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017
parameter common for training

@author: sylvain
18 august 2017
version 1.0

"""
import cPickle as pickle
from numpy import argmax,amax
import os
import numpy as np
#from scipy.misc import bytescale
import shutil
import time
import keras
import theano
import random
import cv2

from keras import backend as K
K.set_image_dim_ordering('th')

print keras.__version__
print theano.__version__
print ' keras.backend.image_data_format :',keras.backend.image_data_format()
######################################################################
setdata='set1'
thrpatch = 0.95 #patch overlapp tolerance
######################################################
writeFile=False
medianblur=False
average3=False
median3=True
#"""
#MIN_BOUND = -1000.0
#MAX_BOUND = 400.0
#PIXEL_MEAN = 0.25

learning_rate=1e-4
#CHU
#limitHU=1700.0
#centerHU=-662.0

#minb=centerHU-(limitHU/2)
#maxb=centerHU+(limitHU/2)

#minb=-1024.0
#maxb=2000.0

minb=-1024.0
maxb=400.
PIXEL_MEAN = 0.2725

MIN_BOUND =minb
MAX_BOUND = maxb

#PIXEL_MEAN = 0.288702558038

#PIXEL_MEAN = 0.
print minb,maxb,PIXEL_MEAN

dimpavx=16
dimpavy=16
dimtabxref=512

pxy=float(dimpavx*dimpavy) #surface in pixel

avgPixelSpacing=0.734   # average pixel spacing in mm

surfelemp=avgPixelSpacing*avgPixelSpacing # for 1 pixel in mm2
surfelem= surfelemp*pxy/100 #surface of 1 patch in cm2

volelemp=avgPixelSpacing*avgPixelSpacing*avgPixelSpacing # for 1 pixel
volelem= volelemp*pxy/1000 #in ml, to multiply by slicepitch in mm

reservedword=['REPORT_SCORE']
augmentation=False#if True, roi of slice number +/- are added

modelname='CNN.h5'

jpegpath='jpegpath'

lungmask='lung'
lungmask1='lung_mask'
#directory to put  lung mask bmp
lungmaskbmp='bmp'
lungmaskbmp1='scan_bmp'
#lungimage='lungimage'

patchpicklename='picklepatches.pkl'#pickle for patches
patchesdirname = 'patches'#define the name of directory for patches
picklepath = 'picklepatches'#define the name of directory for pickle patch
imagedirname='patches_jpeg' #define the name for jpeg files
patchfile='patchfile' #name to put 

scan_bmp='scan_bmp'
transbmp='trans_bmp'
source='source'

#bgdir='bgdir3d'

typei='jpg'
typei1='bmp'
typei2='png' 

#volumeroifile='volumeroi'

perrorfile='genepatchlog.txt'
plabelfile='lislabel.txt'
#excluvisu=['healthy']
excluvisu=['']
locabg='locabg'

sroi='sroi'
sroid='sroi3d'


black=(0,0,0)
grey=(100,100,100)
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
highgrey=(240,240,240)

#cwd=os.getcwd()
#(cwdtop,tail)=os.path.split(cwd)
#dirpickle=os.path.join(cwdtop,path_pickle)
notToAug=['HC','reticulation','bronchiectasis','GGpret']
notToAug=[ 'consolidation',
        'HC',
#        'ground_glass',
        'healthy',
        'micronodules',
#        'reticulation',
        'bronchiectasis',
        'emphysema',
#        'GGpret'
        ]

usedclassifall = [
        'back_ground',
        'consolidation',
        'HC',
        'ground_glass',
        'emphysema',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        'HCpret',
        'HCpbro',
        'GGpbro',
        'GGpret',
        'bropret',
        ]
classifall ={
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
        'emphysema':10,
        'GGpret':11,
        'HCpret':12,
        'HCpbro':13,
        'GGpbro':14,
        'bropret':15,
        'lung':16
        }

derivedpatall=[
        'HCpret',
        'HCpbro',
        'GGpbro',
        'GGpret',
        'bropret']

layertokeep= [
        'bronchiectasis',
        ]
    
classifnotvisu=['healthy',]

if setdata=='set0':
    classif ={
        'consolidation':0,
        'HC':1,
        'ground_glass':2,
        'healthy':3,
        'micronodules':4,
        'reticulation':5,
        'air_trapping':6,
        'cysts':7,
        'bronchiectasis':8,
#        'emphysema':10,
        'GGpret':9,
        'lung':10
        }
    usedclassif = [
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        'GGpret'
        ]
    derivedpat=[
        'GGpret'
        ]
    
elif setdata=='set0p':
    classif ={
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
#        'emphysema':10,
        'GGpret':10,
        'lung':11
        }
    usedclassif = [
        'back_ground',
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        'GGpret'
        ]
    derivedpat=[
        'HCpret',
        'HCpbro',
        'GGpbro',
        'GGpret',
        'bropret'
        ]
elif setdata=='set1':
    classif ={
        'consolidation':0,
        'HC':1,
        'ground_glass':2,
        'healthy':3,
        'micronodules':4,
        'reticulation':5,
        'bronchiectasis':6,
        'emphysema':7,
        'GGpret':8,
        'lung':9
        }

    derivedpat=[
        'GGpret'
        ]
elif setdata=='setr':
    classif ={
        'reticulation':0,
        'ground_glass':1,
        'healthy':2,
        'GGpret':3,
        'lung':4
        }

    derivedpat=[
        'GGpret'
        ]
elif setdata=='set1p':
    classif ={
        'back_ground':0,
        'consolidation':1,
        'HC':2,
        'ground_glass':3,
        'healthy':4,
        'reticulation':5,
        'air_trapping':6,        
        'bronchiectasis':7,
        'GGpret':8,
        'lung':9
        }
    usedclassif = [
        'back_ground',
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'reticulation',
        'air_trapping',
        'bronchiectasis',
        'GGpret'
        ]
    derivedpat=[
        'HCpret',
        'HCpbro',
        'GGpbro',
        'GGpret',
        'bropret'
        ]
    
elif setdata=='set2':
    classif ={
        'consolidation':0,
        'HC':1,
        'ground_glass':2,
        'healthy':3,
        'reticulation':4,
        'air_trapping':5,
        'cysts':6,
        'bronchiectasis':7,
        'GGpret':8,
        'lung':9
        }
    usedclassif = [
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        'GGpret'
        ]
    derivedpat=[
        'HCpret',
        'HCpbro',
        'GGpbro',
        'GGpret',
        'bropret'
        ]    
    
elif setdata=='set2p':
    classif ={
        'back_ground':0,
        'consolidation':1,
        'HC':2,
        'ground_glass':3,
        'healthy':4,
        'reticulation':5,
        'air_trapping':6,
        'cysts':7,
        'bronchiectasis':8,
        'GGpret':9,
        'lung':10
        }
    usedclassif = [
        'back_ground',
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        'GGpret'
        ]
    derivedpat=[
        'HCpret',
        'HCpbro',
        'GGpbro',
        'GGpret',
        'bropret'
        ]    
    
    
elif setdata=='setall':
    classif ={
        'consolidation':0,
        'HC':1,
        'ground_glass':2,
        'healthy':3,
        'micronodules':4,
        'reticulation':5,
        'air_trapping':6,
        'cysts':7,
        'bronchiectasis':8,
        'emphysema':9,
        'bronchial_wall_thickening':10,
        'early_fibrosis':11,
        'increased_attenuation':12,
        'macronodules':13,
        'pcp':14,
        'peripheral_micronodules':15,
        'tuberculosis':16,
        'GGpret':17,
        'HCpret':18,
        'HCpbro':19,
        'GGpbro':20,
        'bropret':21,
        'lung':22
        }
    usedclassif = [
        'consolidation',
        'HC',
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
        'tuberculosis',
        'HCpret',
        'HCpbro',
        'GGpbro',
        'GGpret',
        'bropret'
        ]
    derivedpat=[
        'HCpret',
        'HCpbro',
        'GGpbro',
        'GGpret',
        'bropret'
        ]
    
elif setdata=='CHU':
    classif ={
        'consolidation':0,
        'HC':1,
        'ground_glass':2,
        'healthy':3,
        'reticulation':4,
        'bronchiectasis':5,
        'lung':6
        }
    
    usedclassif = [
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'reticulation',
        'bronchiectasis'
        ]
    
    derivedpat=[
        'HCpret',
        'HCpbro',
        'GGpbro',
        'GGpret',
        'bropret'
        ]
    
else:
    print 'error: not defined set'

classifc ={
    'back_ground':chatain,
    'consolidation':cyan,
    'HC':blue,
    'ground_glass':red,
    'healthy':darkgreen,
    'micronodules':green,
    'reticulation':yellow,
    'air_trapping':pink,
    'cysts':lightgreen,
    'bronchiectasis':orange,
    'emphysema':chatain,
    'GGpret': parme,
    'HCpret': white,
    'HCpbro': white,
    'GGpbro': white,
    'bropret': white,
     'lung': highgrey,
     'nolung': lowgreen,
     'bronchial_wall_thickening':white,
     'early_fibrosis':white,

     'increased_attenuation':white,
     'macronodules':white,
     'pcp':white,
     'peripheral_micronodules':white,
     'tuberculosis':white
 }
volcol={
    'consolidation':'boxMaterialCyan',
    'HC':'boxMaterialBlue',
    'ground_glass':'boxMaterialRed',
    'healthy':'boxMaterialGrey',
    'micronodules':'boxMaterialGreen',
    'reticulation':'boxMaterialYellow',
    'air_trapping':'boxMaterialPink',
    'cysts':'boxMaterialLightgreen',
     'bronchiectasis':'boxMaterialOrange',
     'emphysema':'boxMaterialChatain',
     'GGpret': 'boxMaterialParme',
     'lung': 'boxMaterialGrey',

     'bronchial_wall_thickening':'boxMaterialWhite',
     'early_fibrosis':'boxMaterialWhite',
     'increased_attenuation':'boxMaterialWhite',
     'macronodules':'boxMaterialWhite',
     'pcp':'boxMaterialWhite',
     'peripheral_micronodules':'boxMaterialWhite',
     'tuberculosis':'boxMaterialWhite'
  }

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
         time.sleep(1)
# Now the directory is empty of files


def normi(tabi):
     """ normalise patches"""
#     tabi2=bytescale(tabi, low=0, high=255)
     max_val=float(np.max(tabi))
     min_val=float(np.min(tabi))
     mm=max_val-min_val
     if mm ==0:
         mm=1
#     print 'tabi1',min_val, max_val,imageDepth/float(max_val)
     tabi2=(tabi-min_val)*(255/mm)
#     tabi2=np.clip(tabi2,0,255)
     tabi2=tabi2.astype('uint8')
     return tabi2
 
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
    
def normalize(image):
    image1= (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image1[image1>1] = 1.
    image1[image1<0] = 0.

    return image1

def zero_center(image):
    image1 = image - PIXEL_MEAN
    return image1

def norm(image):
    image1=normalize(image)
    image2=zero_center(image1)
    return image2

def maxproba(proba):
    """looks for max probability in result"""  
    im=argmax(proba)
    m=amax(proba)
    return im,m

#write the log file with label list
def genelabelloc(patchtoppath,plabelfile,jpegpath):
    print 'geneloc'
    eflabel=os.path.join(patchtoppath,plabelfile)
    mflabel=open(eflabel,"w")
    mflabel.write('label  _  localisation\n')
    mflabel.write('======================\n')
    categ=os.listdir(jpegpath)
    for f in categ:
        if f.find('.txt')>0 and f.find('_nbpat')>0:
            ends=f.find('.txt')
            debs=f.find('_')
#            slnc=f[debs:ends]
            debs=f.find('_',debs+1)
            sln=f[debs:ends]
            deb=f.find('_nbpat')
            slncc=f[0:deb]
            slncc=slncc+sln
            ffle=os.path.join(jpegpath,f)
            fr=open(ffle,'r')
            t=fr.read()
            fr.close()
            debs=t.find(':')
            ends=len(t)
            nump= t[debs+1:ends]
            mflabel.write(slncc+' number of patches: '+nump+'\n')
            
            listlabel={}
            for pat in usedclassifall:
                 listlabel[pat]=0
                
            for f1 in categ:
                
    #                print 'f1',f1
                if  f1.find(slncc+'_')==0 and f1.find('.txt')>0:

    #                    print 'f1found',f1
    
                    debl=f1.find('slice_')
                    debl1=f1.find('_',debl+1)
                    debl2=f1.find('_',debl1+1)
                    endl=f1.find('.txt')
                    posend=endl
                    while f1.find('_',posend)==-1:
                        posend-=1
                    debnumslice=posend+1
                    label=f1[debl2+1:debnumslice-1]
#                    print 'label',label
    
                    ffle1=os.path.join(jpegpath,f1)
    #                    print ffle1
                    fr1=open(ffle1,'r')
                    t1=fr1.read()
                    fr1.close()
                    debsp=t1.find(':')
                    endsp=  t1.find('\n')
                    npo=int(t1[debsp+1:endsp])
    #                print f1, label,npo,listlabel
                    if label in listlabel:
    
                        listlabel[label]=listlabel[label]+npo
                    else:
                        listlabel[label]=npo

#                    if label=='HCpret':
#                        print listlabel['HCpret']
#                        print f1
    #        listslice.append(sln)
            
    #        print listlabel
            for l in listlabel:
    #           if l !=labelbg+'_'+locabg:
                if listlabel[l]>0:
                    mflabel.write(l+' '+str(listlabel[l])+'\n')
            mflabel.write('---------------------'+'\n')   
    mflabel.close()
    
def totalpat(jpegpath) :
#   calculate number of patches
    ofilepwt = open(os.path.join(jpegpath,'totalnbpat.txt'), 'w')
    contenupatcht = os.listdir(jpegpath)
    #print(contenupatcht)
    npatcht=0
    for npp in contenupatcht:
    #    print('1',npp)
        if npp.find('.txt')>0 and npp.find('nbp')<0:
    #        print('2',npp)
            ofilep = open(os.path.join(jpegpath,npp), 'r')
            tp = ofilep.read()
    #        print( tp)
            ofilep.close()
            numpos2=tp.find('number')
            numposend2=len(tp)
            #tp.find('\n',numpos2)
            numposdeb2 = tp.find(':',numpos2)
            nump2=tp[numposdeb2+1:numposend2].strip()
    #        print(nump2)
            numpn2=int(nump2)
            npatcht=npatcht+numpn2
    #        print(npatch)
 
    ofilepwt.write('number of patches: '+str(npatcht))
    ofilepwt.close()
    
    
def totalnbpat (patchtoppath,picklepathdir):

    #print 'dirlabel',dirlabel
    #file for data pn patches
    filepwt1 = open(os.path.join(patchtoppath,'totalpat.txt'), 'w')
    dirlabel=os.walk( picklepathdir).next()[1]

    #print filepwt
    ntot=0;
    labellist=[]
    localist=[]
    labeldict={}
    labeldictref={}
    for pat in usedclassifall:
        labeldict[pat]=0
    for dirnam in dirlabel:
        dirloca=os.path.join(picklepathdir,dirnam)
    #    print ('dirloca', dirloca)
        listdirloca=os.listdir(dirloca)
        label=dirnam
    #    print ('dirname', dirname)
    
        loca=''
        if dirnam not in labellist:
                labellist.append(dirnam)
    #    print('label:',label)
        for dlo in listdirloca:
            loca=dlo
            if dlo not in localist:
                localist.append(dlo)
    #        print('localisation:',loca)
            if label=='' or loca =='':
                print('not found:',dirnam)
            subdir = os.path.join(dirloca,loca)
#            print 'subdir',subdir
            n=0
            listcwd=os.listdir(subdir)
#            print 'listcwd',listcwd
            for ff in listcwd:
                namtopp=ff.find('_',0)
                nametop=ff[0:namtopp]
                if  nametop not in labeldictref:
                    labeldictref[nametop]={}
                    for pat in usedclassifall:
                        labeldictref[nametop][pat]=0
                
                if ff.find('.pkl') >0 :
                    p=pickle.load(open(os.path.join(subdir,ff),'rb'))
                    lp=len(p)
                    n=n+lp
                    ntot=ntot+lp
                    labeldictref[nametop][label]+=lp
    #        print(label,loca,n)
            labeldict[label]+=n
            filepwt1.write('label: '+label+' localisation: '+loca+\
            ' number of patches: '+str(n)+'\n')
    filepwt1.write('-------------------------------------\n')
    filepwt1.write('total number of patches: '+str(ntot)+'\n')

    print('total number of patches: '+str(ntot))
    for pat in usedclassifall:
        if labeldict[pat]>0:
            filepwt1.write('label: '+pat+' : '+str(labeldict[pat])+'\n' )
#            print('label: '+pat+' : '+str(labeldict[pat]))
    
    filepwt1.write('-------------------------------------\n')
    for key ,value in labeldictref.items():
#        print key
        filepwt1.write(key+'\n')
        filepwt1.write('-\n')
        for k,v in value.items():
#            print k,v
            if v>0:
                filepwt1.write(k+' : '+str(v)+'\n')
        filepwt1.write('----------\n')
    filepwt1.close()

def geneshiftv(img,s):
    if s!=0:
        shap=img.shape[0]
        shi=int(shap*s/100.)
        tr=np.roll(img,shi,axis=0)
        if shi>0:
            remp=img[0]
            for i in range (shi):
                tr[i]=remp
        else:
            remp=img[-1]
            for i in range (-shi):
#                print i,img.shape[0]
                tr[img.shape[0]-i-1]=remp
    else:
        tr=img
    return tr

def geneshifth(img,s):
    if s!=0:
        shap=img.shape[1]
        shi=int(shap*s/100.)
        tr=np.roll(img,shi,axis=1)       
        if shi>0:
            remp=img[:,0]
            for i in range (shi):
                tr[:,i]=remp
        else:
            remp=img[:,-1]
            for i in range (-shi):
                tr[:,img.shape[1]-i-1]=remp    
    else:
                tr=img
    return tr

def generesize(img,r):
    if r !=0:
        shapx=img.shape[1]
        shapy=img.shape[0]  
        types=type(img[0][0])
        imgc=img.copy().astype('float64')
        imgr=cv2.resize(imgc,None,fx=(100+r)/100.,fy=(100+r)/100.,interpolation=cv2.INTER_LINEAR)  
        imgr=imgr.astype(types)
        newshapex=imgr.shape[1]
        newshapey=imgr.shape[0]

        valcor=(imgc[0][0] +imgc[0][shapx-1]+imgc[shapy-1][0]+imgc[shapy-1][shapx-1])/4.
    
        if newshapex>shapx:
            dx=int(newshapex-shapx)/2
            dy=int(newshapey-shapy)/2
            imgrf=imgr[dy:dy+shapy,dx:dx+shapx]
        else:
            dx=int(shapx-newshapex)/2
            dy=int(shapy-newshapey)/2
            imgrf=np.zeros((shapy,shapx),types)
            np.putmask(imgrf,imgrf==0,valcor)
            imgrf[dy:dy+newshapey,dx:dx+newshapex]=imgr
    else:
            imgrf=img
    return imgrf

def generot(image,tt):
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


def genescaleint(img,s):
    if s!=0:
        minint=-1.
        maxint=1.
        types=type(img[0][0])
        acts=s*(maxint-minint)/100.0
        imgc=img.copy().astype('float32')    
        imgr=imgc+acts
        imgr=np.clip(imgr,minint,maxint)
        imgr=imgr.astype(types)
    else:
        imgr=img
    return imgr


def genesmultint(img,s):
    if s!=0:
        minint=-1.
        maxint=1.
        
        types=type(img[0][0])
        acts=(100+s)/100.0
        imgc=img.copy().astype('float32')    
        imgr=imgc*acts
        imgr=np.clip(imgr,minint,maxint)
        imgr=imgr.astype(types)
    else:
        imgr=img
    return imgr


def geneaug(img,scaleint,multint,rotimg,resiz,shiftv,shifth):
    
    imgr=geneshifth(img,shifth)
    imgr=geneshiftv(imgr,shiftv)
    imgr=generot(imgr,rotimg)
    imgr=generesize(imgr,resiz)
    imgr=genesmultint(imgr,multint)
    imgr=genescaleint(imgr,scaleint)

    return imgr
    

def generandom(maxscaleint,maxmultint,maxrot,maxresize,maxshiftv,maxshifth,keepaenh):
    
    scaleint = random.randint(-100, 100)
    scaleint =keepaenh*maxscaleint*scaleint/100.
    
    multint = random.randint(-100, 100)
    multint =keepaenh*maxmultint*multint/100.
    
    rotimg = random.randint(0, maxrot)
    
    resiz = random.randint(-100,100)
    resiz =keepaenh*maxresize*resiz/100.
    
    shiftv = random.randint(-100, 100)
    shiftv =keepaenh*maxshiftv*shiftv/100.

    
    shifth = random.randint(-100, 100)
    shifth =keepaenh*maxshifth*shifth/100.

    return scaleint,multint,rotimg,resiz,shiftv,shifth



###############
    
