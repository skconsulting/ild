# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017

@author: sylvain
28 july 2017
version 1.0

"""
#import argparse
#from appJar import gui
#import cPickle as pickle
import cPickle as pickle
from numpy import argmax,amax
import os

from scipy.misc import bytescale
import shutil

import time


import keras
import theano

from keras import backend as K
K.set_image_dim_ordering('th')


print keras.__version__
print theano.__version__
print ' keras.backend.image_data_format :',keras.backend.image_data_format()

setdata='set1'

writeFile=False

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

thrpatch = 0.8#patch overlapp tolerance
dimpavx=16
dimpavy=16

pxy=float(dimpavx*dimpavy) #surface in pixel

avgPixelSpacing=0.734   # average pixel spacing in mm

surfelemp=avgPixelSpacing*avgPixelSpacing # for 1 pixel in mm2
surfelem= surfelemp*pxy/100 #surface of 1 patch in cm2

volelemp=avgPixelSpacing*avgPixelSpacing*avgPixelSpacing # for 1 pixel
volelem= volelemp*pxy/1000 #in ml, to multiply by slicepitch in mm

modelname='ILD_CNN_model.h5'
pathjs='../static'

datacrossn='datacross'
datafrontn='datafront'
path_data='data'
path_pickle='CNNparameters'
predictout='predicted_results'
predictout3d='predicted_results_3d'
predictout3dn='predicted_results_3dn'
predictout3d1='predicted_results_3dn1'
#predictout3dr='predicted_results_3dr'
predictoutmerge='predicted_results_merge'
dicompadirm='predict_dicom'
dicomcross='cross'
dicomfront='front'
dicomcross_merge='merge'
reportfile='report'
reportdir='report'


jpegpath='jpegpath'
jpegpath3d='jpegpath3d'
jpegpadirm='jpegpadirm'

lung_name='lung'
lung_namebmp='bmp'
lung_name_gen='lung'
#directory with lung mask dicom
lungmask='lung'
lungmask1='lung_mask'
#directory to put  lung mask bmp
lungmaskbmp='scan_bmp'
lungimage='lungimage'

patchpicklename='picklepatches.pkl'#pickle for patches
patchesdirname = 'patches'#define the name of directory for patches
picklepath = 'picklepatches'#define the name of directory for pickle patch
imagedirname='patches_jpeg' #define the name for jpeg files
patchfile='patchfile' #name to put 


scan_bmp='scan_bmp'
transbmp='trans_bmp'
source='source'

bgdir='bgdir3d'

typei='jpg'
typei1='bmp'
typei2='png' 

volumeroifile='volumeroi'

perrorfile='genepatchlog.txt'
plabelfile='lislabel.txt'
#excluvisu=['healthy']
excluvisu=['']

bmpname='scan_bmp'

sourcedcm = 'source'

sroi='sroi'
sroi3d='sroi3d'
volumeweb = 'volume.txt'
htmldir='html'
threeFileTxt='uip.txt'
threeFile='uip.html'
threeFilejs='world.js'

threeFileTxtMerge='uipMerge.txt'
threeFileMerge='uipMerge.html'
threeFilejsMerge='worldMerge.js'

threeFileTxt3d='uip3d.txt'
threeFile3d='uip3d.html'
threeFilejs3d='world3d.js'

threeFileTop0='uiptop0.html'
threeFileTop1='uiptop1.html'
threeFileTop2='uiptop2.html'
threeFileBot='uipbot.html'


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

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
dirpickle=os.path.join(cwdtop,path_pickle)

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
        'reticulation':4,
        'air_trapping':5,
        'bronchiectasis':6,
#        'emphysema':10,
        'GGpret':7,
        'lung':8
        }
    usedclassif = [
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
     tabi2=bytescale(tabi, low=0, high=255)
#     max_val=float(np.max(tabi))
#     min_val=float(np.min(tabi))
#     mm=max_val-min_val
#     if mm ==0:
#         mm=1
##     print 'tabi1',min_val, max_val,imageDepth/float(max_val)
#     tabi2=(tabi-min_val)*(255/mm)
#     tabi2=tabi2.astype('uint8')
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
            for pat in usedclassif:
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
    for pat in usedclassif:
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
    #        print 'subdir',subdir
            n=0
            listcwd=os.listdir(subdir)
    #        print 'listcwd',listcwd
            for ff in listcwd:
                if ff.find('.pkl') >0 :
                    p=pickle.load(open(os.path.join(subdir,ff),'rb'))
                    lp=len(p)
                    n=n+lp
                    ntot=ntot+lp
    #        print(label,loca,n)
            labeldict[label]+=n
            filepwt1.write('label: '+label+' localisation: '+loca+\
            ' number of patches: '+str(n)+'\n')
    filepwt1.write('-------------------------------------\n')
    filepwt1.write('total number of patches: '+str(ntot)+'\n')

    print('total number of patches: '+str(ntot))
    for pat in usedclassif:
        if labeldict[pat]>0:
            filepwt1.write('label: '+pat+' : '+str(labeldict[pat])+'\n' )
#            print('label: '+pat+' : '+str(labeldict[pat]))
    filepwt1.close()
    
###############
    
