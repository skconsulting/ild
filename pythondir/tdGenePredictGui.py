# coding: utf-8
#sylvain Kritter 04-Apr-2017
'''predict on lung scan front view and cross view
 '''
import os
import cv2
#import dircache
import sys
import shutil
from scipy import misc
import numpy as np

import cPickle as pickle
import dicom
from Tkinter import *
import scipy
import PIL
import cPickle as pickle
import random

#from PIL import Image as ImagePIL
from PIL import Image,ImageFont, ImageDraw
os.environ['KERAS_BACKEND'] = 'theano'
#from keras.models import model_from_json
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model
import cnn_model as CNN4
import time
from time import time as mytime

def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)

t0=mytime()
#####################################################################
#reservedparm=[ 'thrpatch','thrproba','thrprobaUIP','thrprobaMerge','picklein_file',
#                      'picklein_file_front','tdornot','threedpredictrequest',
#                      'onlyvisuaasked','cross','front','merge']
#define the working directory
#wbg=True
#globalHist=True
#contrast=False
#normiInternal =True
isGre=False

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25
writeFile=False
#HUG='predict_23d'
#HUG='predict_S106530'
#HUG='predict_S14740'
#HUG='predict_S15440'
#HUG='predict_S107260'

#thrpatch = 0.8 #threshold for pad acceptance (recovery)
#thrproba =0.6 #thresholm proba for generation of predicted images
#thrprobaUIP=0.6 #threshold proba for UIP
#thrprobaMerge=0.6 #threshold proba for UIP
#
#picklein_file = '../pickle_ex/pickle_ex61'
#picklein_file_front = '../pickle_ex/pickle_ex63'
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

source_name='source'
jpegpath='jpegpath'
jpegpath3d='jpegpath3d'
jpegpadirm='jpegpadirm'

lung_name='lung'
lung_namebmp='bmp'
lung_name_gen='lung_mask'
scan_bmp='scan_bmp'
transbmp='trans_bmp'
source='source'

sroid='sroi3d'
sroi='sroi'
bgdir='bgdir3d'

typei='jpg'
typeid='jpg' #can be png for 16b
typej='jpg'
typeiroi1='jpg'
typeiroi2='bmp'

excluvisu=['healthy']
#excluvisu=[,'healthy','consolidation','HC','ground_glass','micronodules','reticulation',
#           'air_trapping','cysts',  'bronchiectasis']
#excluvisu=[healthy','consolidation','micronodules','reticulation',
#           'air_trapping','cysts',  'bronchiectasis']
#excluvisuh=['bronchiectasis']
#excluvisu=[]
excluvisuh=[]
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


dimpavx=16
dimpavy=16
pxy=float(dimpavx*dimpavy)

#imageDepth=8191 #number of bits used on dicom images (2 **n) 13 bits
#imageDepth=255 #number of bits used on dicom images (2 **n) 13 bits
#patch overlapp tolerance

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
dirpickle=os.path.join(cwdtop,path_pickle)
#dirHUG=os.path.join(cwdtop,HUG)
#
#print dirHUG
#listHug= [ name for name in os.listdir(dirHUG) if os.path.isdir(os.path.join(dirHUG, name)) ]
#print listHug

font5 = ImageFont.truetype( 'arial.ttf', 5)
font10 = ImageFont.truetype( 'arial.ttf', 10)
font20 = ImageFont.truetype( 'arial.ttf', 20)

modelname='ILD_CNN_model.h5'
avgPixelSpacing=0.734   # average pixel spacing
###############################################################
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
        'GGpret':9
        }

clssifcount={
            'consolidation':0,
            'HC':0,
            'ground_glass':0,
            'healthy':0,
            'micronodules':0,
            'reticulation':0,
            'air_trapping':0,
            'cysts':0,
            'bronchiectasis':0,
            'GGpret':0,
            'bronchial_wall_thickening':0,
             'early_fibrosis':0,
             'emphysema':0,
             'increased_attenuation':0,
             'macronodules':0,
             'pcp':0,
             'peripheral_micronodules':0,
             'tuberculosis':0
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

     'bronchial_wall_thickening':'boxMaterialWhite',
     'early_fibrosis':'boxMaterialWhite',
     'increased_attenuation':'boxMaterialWhite',
     'macronodules':'boxMaterialWhite',
     'pcp':'boxMaterialWhite',
     'peripheral_micronodules':'boxMaterialWhite',
     'tuberculosis':'boxMaterialWhite'
  }

classifc ={
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



     'nolung': lowgreen,
     'bronchial_wall_thickening':white,
     'early_fibrosis':white,

     'increased_attenuation':white,
     'macronodules':white,
     'pcp':white,
     'peripheral_micronodules':white,
     'tuberculosis':white
 }
#pickle_dir=os.path.join(cwdtop,pickel_dirsource)
#pickle_dirToMerge=os.path.join(cwdtop,pickel_dirsourceToMerge)

#patch_dir=os.path.join(dirHUG,patch_dirsource)
#print patch_dir
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

def normi(tabi):
     """ normalise patches"""

     max_val=float(np.max(tabi))
     min_val=float(np.min(tabi))

     mm=max_val-min_val
     mm=max(mm,1.0)
#     print 'tabi1',min_val, max_val,imageDepth/float(max_val)
     tabi2=(tabi-min_val)*(255/mm)
     tabi2=tabi2.astype('uint8')
     return tabi2

def reshapeScan(tabscan,slnt,dimtabx):
    print 'reshape'
    tabres = np.zeros((dimtabx,slnt,dimtabx), np.int16)

    for i in range (0,dimtabx):
        for j in range (0,slnt):

            tabres[i][j]=tabscan[j][i]

    return tabres

def genebmp(fn,sou):
    """generate patches from dicom files"""
    global picklein_file
    print ('load dicom files in :',fn)
    (top,tail) =os.path.split(fn)
    lislnn=[]
    fmbmp=os.path.join(fn,sou)

    fmbmpbmp=os.path.join(fmbmp,scan_bmp)
    remove_folder(fmbmpbmp)
    os.mkdir(fmbmpbmp)

    listdcm=[name for name in  os.listdir(fmbmp) if name.lower().find('.dcm')>0]

    FilesDCM =(os.path.join(fmbmp,listdcm[0]))
    FilesDCM1 =(os.path.join(fmbmp,listdcm[1]))
    RefDs = dicom.read_file(FilesDCM)
    RefDs1 = dicom.read_file(FilesDCM1)
    patientPosition=RefDs.PatientPosition
    SliceThickness=RefDs.SliceThickness
    try:
            slicepitch = np.abs(RefDs.ImagePositionPatient[2] - RefDs1.ImagePositionPatient[2])
#            print RefDs.ImagePositionPatient[2]
#            print RefDs1.ImagePositionPatient[2]
    except:
            slicepitch = np.abs(RefDs.SliceLocation - RefDs1.SliceLocation)
#    try:
#            SliceSpacingB=RefDs.SpacingBetweenSlices
#    except AttributeError:
#             print "Oops! No Slice spacing..."
#             SliceSpacingB=0
#
#    print 'SliceThickness', SliceThickness
#    print 'SliceSpacingB', SliceSpacingB
#    slicepitch=float(SliceThickness)+float(SliceSpacingB)
#    print 'slice_thickness',slice_thickness
    print 'slice pitch in z :',slicepitch
#    ooo
    print patientPosition
    dsr= RefDs.pixel_array
    dsr = dsr.astype('int16')
    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing

#    imtowrite=normi(dsr)
#
#    cv2.imshow('b',imtowrite)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#
#    ooo
    imgresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
    dimtabx=imgresize.shape[0]
    dimtaby=imgresize.shape[1]
#    print dimtabx, dimtaby
    slnt=0
    for l in listdcm:

        FilesDCM =(os.path.join(fmbmp,l))
        RefDs = dicom.read_file(FilesDCM)
        slicenumber=int(RefDs.InstanceNumber)
        lislnn.append(slicenumber)
        if slicenumber> slnt:
            slnt=slicenumber

    print 'number of slices', slnt
    slnt=slnt+1
    tabscan = np.zeros((slnt,dimtabx,dimtaby), np.int16)
    for l in listdcm:
#        print l
        FilesDCM =(os.path.join(fmbmp,l))
        RefDs = dicom.read_file(FilesDCM)
        slicenumber=int(RefDs.InstanceNumber)

        dsr= RefDs.pixel_array
        dsr=dsr.astype('int16')
        dsr[dsr == -2000] = 0
        intercept = RefDs.RescaleIntercept
        slope = RefDs.RescaleSlope
        if slope != 1:
             dsr = slope * dsr.astype(np.float64)
             dsr = dsr.astype(np.int16)

        dsr += np.int16(intercept)
        dsr = dsr.astype('int16')
        imgresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)

        tabscan[slicenumber]=imgresize
        imtowrite=normi(imgresize)

        endnumslice=l.find('.dcm')
        imgcoreScan=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei
        bmpfile=os.path.join(fmbmpbmp,imgcoreScan)

        t2='Not for medical use'
        t1='Pt: '+tail
        t0='CONFIDENTIAL'
        t3='Scan: '+str(slicenumber)

        t4=time.asctime()
        (topw,tailw)=os.path.split(picklein_file)
        t5= tailw

        imtowrite=tagviews(imtowrite,t0,0,10,t1,0,20,t2,0,30,t3,0,40,t4,0,dimtaby-10,t5,0,dimtaby-20)
        scipy.misc.imsave(bmpfile,imtowrite)

    return tabscan,slnt,dimtabx,slicepitch,lislnn

def genebmplung(fn,lungname,slnt,dimtabx,dimtaby):
    """generate patches from dicom files"""

    print ('load dicom files for lung in :',fn)
    (top,tail) =os.path.split(fn)

    fmbmp=os.path.join(fn,lungname)
    fmbmpbmp=os.path.join(fmbmp,lung_namebmp)

    remove_folder(fmbmpbmp)

    os.mkdir(fmbmpbmp)

    listdcm=[name for name in  os.listdir(fmbmp) if name.lower().find('.dcm')>0]

    tabscan = np.zeros((slnt,dimtabx,dimtaby), np.uint8)

    for l in listdcm:
        FilesDCM =(os.path.join(fmbmp,l))
        RefDs = dicom.read_file(FilesDCM)

        dsr= RefDs.pixel_array
        dsr= dsr-dsr.min()
        c=float(255)/dsr.max()
        dsr=dsr*c
        dsr=dsr.astype('uint8')

        fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
        imgresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)

        slicenumber=int(RefDs.InstanceNumber)
        endnumslice=l.find('.dcm')

        imgcoreScan=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei
        bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
        scipy.misc.imsave(bmpfile,imgresize)

        tabscan[slicenumber]=imgresize
    return tabscan

def maxproba(proba):
    """looks for max probability in result"""
    lenp = len(proba)
    m=0
    for i in range(0,lenp):
        if proba[i]>m:
            m=proba[i]
            im=i
    return im,m

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

#        print proba[0]
def contour2(im,l,dimtabx,dimtaby):
    col=classifc[l]
#    print l
#    print 'dimtabx' , dimtabx
    vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
#    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(im,0,255,0)
    im2,contours0, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
    cv2.drawContours(vis,contours,-1,col,1,cv2.LINE_AA)
    return vis


def tagview(fig,label,x,y):
    """write text in image according to label and color"""
#    print ('write label :',label,' at: ', fig)
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    col=classifc[label]
    labnow=classif[label]
#    print (labnow, text)
   
    deltay=25*((labnow)%5)
#        deltax=175*((labnow-1)//5)
#        deltax=80*((labnow-1)//5)

#    print (x+deltax,y+deltay)
    draw.text((x, y+deltay),label,col,font=font10)
    imgn.save(fig)

def tagviewt(fig,text,x,y):
    """write simple text in image """
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    draw.text((x, y),text,white,font=font10)
    imgn.save(fig)


def tagviews (tab,t0,x0,y0,t1,x1,y1,t2,x2,y2,t3,x3,y3,t4,x4,y4,t5,x5,y5):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    col=white
#    imgn=Image.open(fig)
#    draw = ImageDraw.Draw(imgn)
#    if b:
#        draw.rectangle ([x1, y1,x1+100, y1+15],outline='black',fill='black')
#        draw.rectangle ([140, 0,dimtabx,75],outline='black',fill='black')
    viseg=cv2.putText(tab,t0,(x0, y0), font,0.3,col,1)
    viseg=cv2.putText(viseg,t1,(x1, y1), font,0.3,col,1)
    viseg=cv2.putText(viseg,t2,(x2, y2), font,0.3,col,1)
#    draw.text((x0, y0),t0,white,font=font10)
#    draw.text((x1, y1),t1,white,font=font10)
#    draw.text((x2, y2),t2,white,font=font10)
    viseg=cv2.putText(viseg,t3,(x3, y3), font,0.3,col,1)
    viseg=cv2.putText(viseg,t4,(x4, y4), font,0.3,col,1)
    viseg=cv2.putText(viseg,t5,(x5, y5), font,0.3,col,1)
#    draw.text((x3, y3),t3,white,font=font10)
#    draw.text((x4, y4),t4,white,font=font10)
#    draw.text((x5, y5),t5,white,font=font10)
#    imgn.save(fig)
#    viseg=cv2.putText(tab,label,(x, y+deltay), font,0.3,col,1)
    return viseg

def normalize(image):
    image1= (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image1[image1>1] = 1.
    image1[image1<0] = 0.
    return image1

def zero_center(image):
    image1 = image - PIXEL_MEAN
    return image1

def pavgene(dirf,dimtabx,dimtaby,tabscanScan,tabscanLung,slnt,jpegpath):
        """ generate patches from scan"""
        global thrpatch
        patch_list=[]
        (dptop,dptail)=os.path.split(dirf)
        print('generate patches on: ',dptail)
        jpegpathdir=os.path.join(dirf,jpegpath)
        remove_folder(jpegpathdir)
        os.mkdir(jpegpathdir)
        for img in range (0,slnt):

             tabfw = np.zeros((dimtabx,dimtaby,3), np.uint8)
             tablung = np.copy(tabscanLung[img])
             tabfrgb=np.copy(tablung)
             np.putmask(tablung,tablung>0,1)
             np.putmask(tabfrgb,tabfrgb>0,100)
             tabfrgb= cv2.cvtColor(tabfrgb,cv2.COLOR_GRAY2BGR)
             tabf=tabscanScan[img]
             nz= np.count_nonzero(tablung)
             if nz>0:

                 atabf = np.nonzero(tablung)
                #tab[y][x]  convention
                 xmin=atabf[1].min()
                 xmax=atabf[1].max()
                 ymin=atabf[0].min()
                 ymax=atabf[0].max()

                 i=xmin
                 while i < xmax:
                     j=ymin
                     while j<ymax:

                         tabpatch=tablung[j:j+dimpavy,i:i+dimpavx]
                         area= tabpatch.sum()
                         targ=float(area)/pxy

                         if targ>=thrpatch:
    #                        print i,j,targ,area
    #                        ooo
                            imgray = tabf[j:j+dimpavy,i:i+dimpavx]

                            imagemax= cv2.countNonZero(imgray)
                            min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(imgray)

                            if imagemax > 0 and min_val != max_val:
    #                            namepatch=patchpathf+'/p_'+str(slicenumber)+'_'+str(i)+'_'+str(j)+'.'+typei
                                imgray= normalize(imgray)
                                imgray=zero_center(imgray)

                                patch_list.append((img,i,j,imgray))
                                tablung[j:j+dimpavy,i:i+dimpavx]=0

                                x=0
                                while x < dimpavx:
                                    y=0
                                    while y < dimpavy:
                                        if y+j<dimtabx and x+i<dimtaby:
                                            tabfw[y+j][x+i]=[125,0,0]
                                        if x == 0 or x == dimpavx-1 :
                                            y+=1
                                        else:
                                            y+=dimpavy-1
                                    x+=1
                                j+=dimpavy-1
                         j+=1
                     i+=1

                 nameslijpeg='s_'+str(img)+'.'+typej
                 namepatchImage=os.path.join(jpegpathdir,nameslijpeg)
                 tabjpeg=cv2.add(tabfw,tabfrgb)
                 scipy.misc.imsave(namepatchImage,tabjpeg)

        return patch_list



def ILDCNNpredict(patch_list,model):
    print ('Predict started ....')
    dataset_list=[]
    for fil in patch_list:
              dataset_list.append(fil[3])
    X0=len(dataset_list)
    # adding a singleton dimension and rescale to [0,1]
    pa = np.asarray(np.expand_dims(dataset_list, 1))
    # look if the predict source is empty
    # predict and store  classification and probabilities if not empty
    if X0 > 0:
        proba = model.predict_proba(pa, batch_size=100)

    else:
        print (' no patch in selected slice')
        proba = ()
    print 'number of patches', len(pa)

    return proba

def wtebres(wridir,dirf,tab,dimtabx,slicepitch,lungm,ty):
    global picklein_file_front
    print 'generate front images from',ty
    (top,tail)=os.path.split(dirf)
    bgdirf=os.path.join(dirf,lungm)
    bgdirf=os.path.join(bgdirf,transbmp)
    remove_folder(bgdirf)
    os.mkdir(bgdirf)
#    print slicepitch
#    print avgPixelSpacing
    fxs=float(slicepitch/avgPixelSpacing )
    lislnn=[]
    print 'fxs',fxs

#    print tab[0].shape[0]
    ntd=int(round(fxs*tab[0].shape[0],0))
#    print  fxs,slicepitch,ntd,avgPixelSpacing,tab[0].shape[0],dimtabx

    if ty=='scan':
        tabres=np.zeros((dimtabx,ntd,dimtabx),np.int16)
    else:
        tabres=np.zeros((dimtabx,ntd,dimtabx),np.uint8)
    for i in range (0,dimtabx):
#        print i, tab[i].max()
        lislnn.append(i)
#        if tab[i].max()>0:
#            print tab[i].max()
#            print tab[i].shape
#            ooo

        imgresize=cv2.resize(tab[i],None,fx=1,fy=fxs,interpolation=cv2.INTER_LINEAR)
#            print i, imgresize.min(), imgresize.max(),imgresize.shape
#            ooo
#            print dimtabx,fxs,imgresize.shape,tab[i].shape

        if ty=='scan':
            typext=typeid
        else:
            typext=typei
        trscan='tr_'+str(i)+'.'+typext
        trscanbmp='tr_'+str(i)+'.'+typei
        if ty=='lung':
            namescan=os.path.join(bgdirf,trscanbmp)
            np.putmask(imgresize,imgresize>0,100)
            scipy.misc.imsave(namescan,imgresize)
            dimtabxn=imgresize.shape[0]
            dimtabyn=imgresize.shape[1]
        if ty=='scan':
            namescan=os.path.join(wridir,trscan)
            dimtabxn=imgresize.shape[0]
            dimtabyn=imgresize.shape[1]
            t2='Not for medical use'
            t1='Pt: '+tail
            t0='CONFIDENTIAL'
            t3='Scan: '+str(i)
            t4=time.asctime()
            (topw,tailw)=os.path.split(picklein_file_front)
            t5= tailw
            imgresize8=normi(imgresize)
            imgresize8r=tagviews(imgresize8,t0,0,10,t1,0,20,t2,(dimtabyn/2)-10,dimtabxn-10,t3,0,38,t4,0,dimtabxn-10,t5,0,dimtabxn-20)
            scipy.misc.imsave(namescan,imgresize8r)

        tabres[i]=imgresize
#        cv2.imwrite (trscan, tab[i],[int(cv2.IMWRITE_PNG_COMPRESSION),0])
    return dimtabxn,dimtabyn,tabres,lislnn


def modelCompilation(t,picklein_file,picklein_file_front):
    if t=='cross':
        modelpath = os.path.join(picklein_file, 'ILD_CNN_model.h5')
    if t=='front':
        modelpath = os.path.join(picklein_file_front, 'ILD_CNN_model.h5')

    if os.path.exists(modelpath):
        model = load_model(modelpath)
        model.compile(optimizer='Adam', loss=CNN4.get_Obj('ce'))
        return model
    else:
        print 'weight dos not exist',modelpath

def addpatch(col,lab, xt,yt,px,py,dimtabx,dimtaby):
    imgi = np.zeros((dimtabx,dimtaby,3), np.uint8)
    tablint=[(xt,yt),(xt,yt+py),(xt+px,yt+py),(xt+px,yt)]
    tabtxt=np.asarray(tablint)
    cv2.polylines(imgi,[tabtxt],True,col)
    cv2.fillPoly(imgi,[tabtxt],col)
    return imgi

def drawContour(imi,ll,dimtabx,dimtaby):
    vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
    for l in ll:
        col=classifc[l]
        masky=cv2.inRange(imi,col,col)
        outy=cv2.bitwise_and(imi,imi,mask=masky)
        imgray = cv2.cvtColor(outy,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,0,255,0)
        im2,contours0, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
        cv2.drawContours(vis,contours,-1,col,1)
    return vis

def tagviewn(tab,label,pro,nbr,x,y):
    """write text in image according to label and color"""
    col=classifc[label]
    font = cv2.FONT_HERSHEY_SIMPLEX
#    print col, label
    labnow=classif[label]

    deltax=130*((labnow)//10)
    deltay=11*((labnow)%10)
#    gro=-x*0.0027+1.2
    gro=0.3
    viseg=cv2.putText(tab,str(nbr)+' '+label+' '+pro,(x+deltax, y+deltay+10), font,gro,col,1)
    return viseg

def  visua(listelabelfinal,dirf,probaInput,patch_list,dimtabx,dimtaby,
           slnt,predictout,sroi,scan_bmp,sou,dcmf,dct,errorfile):
    global thrproba,picklein_file,picklein_file_front
    print('visualisation',scan_bmp)
    rsuid=random.randint(0,1000)

    sliceok=[]
    (dptop,dptail)=os.path.split(dirf)
    predictout_dir = os.path.join(dirf, predictout)
#    print predictout_dir
    remove_folder(predictout_dir)
    os.mkdir(predictout_dir)

    for i in range (0,len(classif)):
#        print 'visua dptail', topdir
        listelabelfinal[fidclass(i,classif)]=0
    #directory name with predict out dabasase, will be created in current directory
    preprob=probaInput

    dirpatientfdbsource1=os.path.join(dirf,sou)
    dirpatientfdbsource=os.path.join(dirpatientfdbsource1,scan_bmp)
    dirpatientfdbsroi=os.path.join(dirf,sroi)
    listbmpscan=os.listdir(dirpatientfdbsource)
    if dct:
        listdcm=[name for name in  os.listdir(dirpatientfdbsource1) if name.lower().find('.dcm')>0]


    listlabelf={}
    sroiE=False
    if os.path.exists(dirpatientfdbsroi):
        sroiE=True
        listbmpsroi=os.listdir(dirpatientfdbsroi)

    for img in listbmpscan:

        slicenumber= rsliceNum(img,'_','.'+typei)
        if dct:
            for imgdcm in listdcm:
                 FilesDCM =(os.path.join(dirpatientfdbsource1,imgdcm))
                 RefDs = dicom.read_file(FilesDCM)
                 slicenumberdicom=int(RefDs.InstanceNumber)
                 if slicenumberdicom==slicenumber:
                     dsr= RefDs.pixel_array
                     dsr= dsr-dsr.min()
                     c=float(255)/dsr.max()
                     dsr=dsr*c
                     dsr = dsr.astype('uint8')
                     dsr = cv2.cvtColor(dsr, cv2.COLOR_GRAY2RGB)

                     RefDs.PhotometricInterpretation='RGB'
                     RefDs.BitsAllocated=8
                     RefDs.SamplesPerPixel=3
                     RefDs.PlanarConfiguration=0
                     RefDs.HighBit=7
                     RefDs.BitsStored=8

                     RefDs.SeriesDescription='WithPredict'
                     RefDs.StudyInstanceUID                 =str(rsuid)
                     break


        imgt = np.zeros((dimtabx,dimtaby,3), np.uint8)
        listlabelaverage={}
        listlabel={}
        listlabelrec={}
        if sroiE:
            for imgsroi in listbmpsroi:
                slicenumbersroi=rsliceNum(imgsroi,'_','.'+typeiroi1)
                if slicenumbersroi <0:
                    slicenumbersroi=rsliceNum(imgsroi,'_','.'+typeiroi2)
                if slicenumbersroi==slicenumber:
                    imgc=os.path.join(dirpatientfdbsroi,imgsroi)
                    break
        else:
            imgc=os.path.join(dirpatientfdbsource,img)
        tablscan=cv2.imread(imgc,1)

        foundp=False
        for ll in range(0,len(patch_list)):
            slicename=patch_list[ll][0]
            xpat=patch_list[ll][1]
            ypat=patch_list[ll][2]
            proba=preprob[ll]
            prec, mprobai = maxproba(proba)
            mproba=round(mprobai,2)
            classlabel=fidclass(prec,classif)
            classcolor=classifc[classlabel]
#            print xpat,ypat,mprobai,slicename
            if slicenumber == slicename and classlabel not in excluvisu:
                    foundp=True
                    if classlabel in listlabel:
                        numl=listlabel[classlabel]
                        listlabel[classlabel]=numl+1
                    else:
                        listlabel[classlabel]=1
                    if classlabel in listlabelf:
                        nlt=listlabelf[classlabel]
                        listlabelf[classlabel]=nlt+1
                    else:
                        listlabelf[classlabel]=1
            if mprobai >=thrproba and slicenumber == slicename and classlabel not in excluvisu:
                    if slicenumber not in sliceok:
                        sliceok.append(slicenumber)
                    if classlabel in listlabelrec:
                        numl=listlabelrec[classlabel]
                        listlabelrec[classlabel]=numl+1
                        cur=listlabelaverage[classlabel]
                        averageproba= round((cur*numl+mproba)/(numl+1),2)
                        listlabelaverage[classlabel]=averageproba
                    else:
                        listlabelrec[classlabel]=1
                        listlabelaverage[classlabel]=mproba

                    imgi=addpatch(classcolor,classlabel,xpat,ypat,dimpavx,dimpavy,dimtabx,dimtaby)
                    imgt=cv2.add(imgt,imgi)

#            ill+=1
        tablscan = cv2.cvtColor(tablscan, cv2.COLOR_BGR2RGB)

        vis=drawContour(imgt,listlabel,dimtabx,dimtaby)
#        print slicenumber,dsr.shape, type(dsr[0][0][0]),type(vis[0][0][0]),vis.shape,type(tablscan[0][0][0]),tablscan.shape
#        ooo
#        print slicenumber
        imn=cv2.add(tablscan,vis)



        if foundp:
            for ll in listlabelrec:
                delx=int(dimtaby*0.6-120)
                imn=tagviewn(imn,ll,str(listlabelaverage[ll]),listlabelrec[ll],delx,0)
            t0='average probability'
        else:
                errorfile.write('no recognised label in: '+str(dptail)+' '+str (img)+'\n' )
                t0='no recognised label'
        t1=''
        t2=' '
        t3='For threshold: '+str(thrproba)+' :'
        t4=time.asctime()
        (topw,tailw)=os.path.split(picklein_file)
        t5= tailw
        imn=tagviews(imn,t0,0,50,t1,0,20,t2,0,30,t3,0,60,t4,0,dimtaby-10,t5,0,dimtaby-20)
        if dct:
            imncop=cv2.resize(imn,(dsr.shape[0],dsr.shape[1]))
            dsr=cv2.add(dsr,imncop)
            RefDs.PixelData=dsr
            FilesDCMW =(os.path.join(dcmf,imgdcm))
            RefDs.save_as(FilesDCMW)

        imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)
        predict_outFile=os.path.join( predictout_dir,img)
        cv2.imwrite(predict_outFile,imn)
        errorfile.write('\n'+'number of labels in :'+str(dptop)+' '+str(dptail)+str (img)+'\n' )
#    print listlabelf
    for classlabel in listlabelf:
          listelabelfinal[classlabel]=listlabelf[classlabel]
          print 'patient: ',dptail,', label:',classlabel,': ',listlabelf[classlabel]
          stringtw=str(classlabel)+': '+str(listlabelf[classlabel])+'\n'
#          print string
          errorfile.write(stringtw )
    return

def genethreef(dirpatientdb,patchPositions,probabilities_raw,slicepitch,dimtabx,dimtaby,dimpavx,lsn,v):
        """generate  voxels for 3d view"""
        print 'generate voxels for :',v
        global thrprobaUIP
        cwd=os.getcwd()
        pathjscomp=os.path.join(cwd,pathjs)
        pathjscompr=os.path.realpath(pathjscomp)

#bglist=listcl()
        (dptop,dptail)=os.path.split(dirpatientdb)
        pz=slicepitch/avgPixelSpacing
        htmldifr=os.path.join(dirpatientdb,htmldir)
        if not os.path.exists(htmldifr):
            os.mkdir(htmldifr)
        cwd=os.getcwd()
        souuip0=os.path.join(cwd,threeFileTop0)
        souuip1=os.path.join(cwd,threeFileTop1)
        souuip2=os.path.join(cwd,threeFileTop2)
#        print souuip
        if v =='cross':
            threeFilel=dptail+'_'+threeFile
            threeFileTxtl=dptail+'_'+threeFileTxt
#            jsfile=dptail+'_'+threeFilejs
            BGx=str(dimpavx)
            BGy=str(dimpavx)
            BGz=str(round(pz,3))

        if v =='merge':
            threeFilel=dptail+'_'+threeFileMerge
            threeFileTxtl=dptail+'_'+threeFileTxtMerge
#            jsfile=dptail+'_'+threeFilejsMerge
            BGx=str(dimpavx)
            BGy=str(dimpavx)
            BGz=str(round(pz,3))


        if v =='front':
            threeFilel=dptail+'_'+threeFile3d
            threeFileTxtl=dptail+'_'+threeFileTxt3d
#            jsfile=dptail+'_'+threeFilejs3d
            BGy=str(dimpavx)
            BGx=str(round(pz,3))
            BGz=str(dimpavx)

        desouip=os.path.join(htmldifr,threeFilel)
        shutil.copyfile(souuip0,desouip)


        volumefileT = open(os.path.join(htmldifr,threeFileTxtl), 'w')

        volumefile = open(os.path.join(htmldifr,threeFilel), 'a')
        volumefile.write( '<title> '+dptail+' '+v+' orbit </title> \n')
        volumefile.write( '<h1 id=patientname>'+dptail+' '+v+'  </h1> \n')

        vtop1=open(os.path.join(cwd,souuip1),'r')
        apptop1=vtop1.read()
        vtop1.close()
        volumefile.write(apptop1)

        volumefile.write( '<link rel="stylesheet" type="text/css" href="'+
                         pathjscompr+
                         '/css/aff.CSS"> \n')
        volumefile.write( '<script src="'+ pathjscompr+
                          '/js/three.js"></script> \n')
        volumefile.write( '<script src="'+ pathjscompr+
                          '/js/Detector.js"></script> \n')
        volumefile.write( '<script src="'+ pathjscompr+
                          '/js/CanvasRenderer.js"></script> \n')
        volumefile.write( '<script src="'+ pathjscompr+
                          '/js/Projector.js"></script> \n')
        volumefile.write( '<script src="'+ pathjscompr+
                          '/js/OrbitControls.js"></script> \n')
        volumefile.write( '<script src="'+ pathjscompr+
                          '/js/stats.min.js"></script> \n')
        volumefile.write( '<script src="'+ pathjscompr+
                          '/js/uip.js"></script> \n')
        volumefile.write( '<script src="'+ pathjscompr+
                          '/js/aff.js"></script> \n')
        volumefile.write( '<script src="'+ pathjscompr+
                          '/js/underscore-min.js"></script> \n')

        vtop2=open(os.path.join(cwd,souuip2),'r')
        apptop2=vtop2.read()
        vtop2.close()
        volumefile.write(apptop2)
#        volumefilejs = open(jsfilel, 'w')
#        jsfilel.replace("\\","/")
#        print jsfilel
#        volumefile.write( '<script src="'+jsfilel1+'"></script>\n)')
        volumefile.write( '<script> \n')

#        volumefilejs.write( 'function buildobj()	{\n')
        volumefile.write( 'function buildobj()	{\n')

        zxd=slicepitch*lsn/2/avgPixelSpacing
        zxd=pz*(lsn-1)/2
        volumefileT.write('camera.position.set(0 '+', -'+BGx+', 0 );\n')

        volumefileT.write( 'var boxGeometry = new THREE.BoxGeometry( '+BGx+' , '+\
        BGy+' , '+BGz+' ) ;\n\n')
        volumefileT.write('var voxels = [\n')
#        volumefilejs.write( 'var boxGeometry = new THREE.BoxGeometry( '+BGx+' , '+\
#        BGy+' , '+BGz+' ) ;\n')
#
        volumefile.write( 'var boxGeometry = new THREE.BoxGeometry( '+BGx+' , '+\
        BGy+' , '+BGz+' ) ;\n')

        print 'pz',pz
#        volumefilejs.write('camera.position.set(0 '+', -'+str(dimtaby)+', 0 );\n')
#        volumefilejs.write('controls.target.set(0 , 0 , 0 );\n')
#
        volumefile.write('camera.position.set(0 '+', -'+str(dimtaby)+', 0 );\n')
        volumefile.write('controls.target.set(0 , 0 , 0 );\n')

        for ll in range(0,len(patchPositions)):

            slicename=patchPositions[ll][0]
            if v =='cross' or v=='merge':
                 xpat=dimtaby/2-patchPositions[ll][1]
                 ypat=patchPositions[ll][2]-(dimtabx/2)
                 zpa=round((lsn-slicename)*pz-zxd,3)

            if v =='front':
                ypat=-round((lsn-slicename)*pz-zxd,3)
                xpat=(dimtaby/2)-patchPositions[ll][1]
                zpa=(dimtabx/2)-patchPositions[ll][2]

            proba=probabilities_raw[ll]
            prec, mprobai = maxproba(proba)
            mproba=round(mprobai,2)

            classlabel=fidclass(prec,classif)
            bm=volcol[classlabel]
            bmn = 'boxMesh'+str(ll)

            if ll ==0:
                volumefileT.write('{"x": '+str(xpat)+', "y": '+str(ypat)+', "z": '+str(zpa)\
                 +', "class": "'+classlabel+'", "proba": '+str(mproba)+' },\n')
            elif ll==1:
                 volumefileT.write('{"x": '+str(xpat)+', "y": '+str(ypat)+', "z": '+str(zpa)\
                 +', "class": "'+classlabel+'", "proba": '+str(mproba)+' }')
            else:
                 volumefileT.write(',\n{"x": '+str(xpat)+', "y": '+str(ypat)+', "z": '+str(zpa)\
                 +', "class": "'+classlabel+'", "proba": '+str(mproba)+' }')

#            volumefilejs.write( 'var newMaterial= '+bm+'.clone();\n');
            volumefile.write( 'var newMaterial= '+bm+'.clone();\n');
            if mproba>= thrprobaUIP-0.05: #for rounding
#                print '2', mproba
#                mproba=(mproba-thrprobaUIP)/thrprobaUIP
#                volumefilejs.write( 'newMaterial.opacity= '+str(mproba)+';\n');
                volumefile.write( 'newMaterial.opacity= '+str(mproba)+';\n');

            else:
#                volumefilejs.write( 'newMaterial.opacity= 0.1;\n');
#            volumefilejs.write( bmn+' = new THREE.Mesh(boxGeometry,newMaterial)\n');
#            volumefilejs.write( bmn+'.position.set('+str(xpat)+', '+str(ypat)+','+str(zpa)+');\n')
#            volumefilejs.write('scene.add('+bmn+');\n')

                volumefile.write( 'newMaterial.opacity= 0.1;\n');
            volumefile.write( bmn+' = new THREE.Mesh(boxGeometry,newMaterial)\n');
            volumefile.write( bmn+'.position.set('+str(xpat)+', '+str(ypat)+','+str(zpa)+');\n')
            volumefile.write('scene.add('+bmn+');\n')

        volumefileT.write('\n]\n')
        volumefile.write('}</script>\n\n')
        vtb=open(os.path.join(cwd,threeFileBot),'r')
        app=vtb.read()
        volumefile.write(app)

#        volumefilejs.write('\n}\n')



        volumefile.close()
        volumefileT.close()
        vtb.close()
#        volumefilejs.close()

def genecross(proba_cross,dirf,proba_front,patch_list_front,slnt,slicepitch,dimtabx,dimtaby,predictout):
    """generate cross view from front patches"""
    print 'genecross'
    global thrprobaUIP
    (dptop,dptail)=os.path.split(dirf)
#    predictout_dir = os.path.join(dirf, predictout)
#    print predictout_dir
#    remove_folder(predictout_dir)
#    os.mkdir(predictout_dir)
    plf=patch_list_front
    prf=proba_front
    plfd={}
    prfd={}
    tabpatch={}

    listp=[name for name in usedclassif if name not in excluvisu]
    print 'used patterns :',listp

    for i in usedclassif:
        plfd[i]=[]
        prfd[i]=[]

    probabg=createProba('healthy',0.1,proba_cross[0])

    for ll in range(0,len(plf)):
        proba=prf[ll]
        prec, mprobai = maxproba(proba)
        classlabel=fidclass(prec,classif)
        if mprobai> thrprobaUIP:
            plfd[classlabel].append(plf[ll])
            prfd[classlabel].append(prf[ll])
        else:
            plfd['healthy'].append(plf[ll])
            prfd['healthy'].append(probabg)

    for i in listp:
        tabres=np.zeros((dimtaby,dimtabx,dimtaby,3),np.uint8)
        tabpatch[i]=np.zeros((dimtaby,dimtabx,dimtaby,3),np.uint8)
        for ll in range(0,len(plfd[i])):
            pz=plfd[i][ll][0]
            py=plfd[i][ll][2]
            px=plfd[i][ll][1]
#            print i,pz,py,px,dimtabx,dimtaby
            tabres[pz,py:py+dimpavy,px:px+dimpavx]= classifc[i]
            tabpatch[i]=tabres
#            print tabpatch[i].shape

#        for scan in range(0,dimtaby):
#            if tabpatch[i][scan].max()>0:
#                pf=os.path.join(predictout_dir,i+'_'+str(scan)+'.'+typei)
#                cv2.imwrite(pf,tabpatch[i][scan])
#        cv2.imshow('tabpatch',tabpatch[i][200])
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    return tabpatch

def tagviewct(tab,label,x,y):
    """write text in image according to label and color"""

    col=classifc[label]
    labnow=classif[label]
#  
    deltay=10*(labnow)
    deltax=100*(labnow/5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,label,(x+deltax, y+deltay), font,0.3,col,1)
#    viseg = cv2.cvtColor(viseg,cv2.COLOR_RGB2BGR)
    return viseg



def reshapepatern(dirf,tabpx,dimtabxn,dimtaby,slnt,slicepitch,predictout,sou,dcmf):
    print 'reshape pattern'
    """reshape pattern table """
    imnc={}
    rsuid=random.randint(0,1000)
    (dptop,dptail)=os.path.split(dirf)

    predictout_dir = os.path.join(dirf, predictout)
    remove_folder(predictout_dir)
    os.mkdir(predictout_dir)

    listp=[name for name in usedclassif if name not in excluvisu]
    print 'used patterns :',listp
    fxs=float(avgPixelSpacing/slicepitch )

    dirpatientfdb1=os.path.join(dirf,sou)
    dirpatientfdb=os.path.join(dirpatientfdb1,scan_bmp)
    dirpatientfsdb=os.path.join(dirf,sroi)
    listbmpscan=os.listdir(dirpatientfdb)
    listdcm=[name for name in  os.listdir(dirpatientfdb1) if name.lower().find('.dcm')>0]

#    print dirpatientfdb
    tablscan=np.zeros((slnt,dimtaby,dimtaby,3),np.uint8)
    sroiE=False
    if os.path.exists(dirpatientfsdb):
        sroiE=True
        listbmpsroi=os.listdir(dirpatientfsdb)

    for img in listbmpscan:
        slicenumber= rsliceNum(img,'_','.'+typei)

        if sroiE:
            for imgsroi in listbmpsroi:
                slicenumbersroi=rsliceNum(imgsroi,'_','.'+typeiroi1)
                if slicenumbersroi <0:
                    slicenumbersroi=rsliceNum(imgsroi,'_','.'+typeiroi2)
                if slicenumbersroi==slicenumber:
                    imgc=os.path.join(dirpatientfsdb,imgsroi)
                    break
        else:
            imgc=os.path.join(dirpatientfdb,img)
        tablscan[slicenumber]=cv2.imread(imgc,1)
    tabresshape = np.zeros((slnt,dimtaby,dimtaby,3), np.uint8)
    impre = np.zeros((slnt,dimtaby,dimtaby,3), np.uint8)
    tabx={}
    for i in listp:
        tabx[i]=np.zeros((slnt,dimtaby,dimtaby), np.uint8)
        tabres=np.zeros((dimtaby,dimtabxn,dimtaby,3),np.uint8)
        tabrisize=np.zeros((dimtaby,slnt,dimtaby,3),np.uint8)

        for j in range(0,dimtaby):
                tabres[j]=tabpx[i][j]
                tabrisize[j]=cv2.resize( tabres[j],None,fx=1,fy=fxs,interpolation=cv2.INTER_LINEAR)

        for t in range (0,slnt):
            for u in range (0,dimtaby):
                tabresshape[t][u]=tabrisize[u][t]
            tabx[i][t]=cv2.cvtColor(tabresshape[t], cv2.COLOR_BGR2GRAY)
            np.putmask(tabx[i][t],tabx[i][t]>0,100)
            if tabresshape[t].max()>0:
                    tabresshape[t]=tagviewct(tabresshape[t],i,100,10)
            impre[t]=cv2.add(impre[t],tabresshape[t])

    for scan in range(0,slnt):
                pf=os.path.join(predictout_dir,'tr_'+str(scan)+'.'+typei)
                imcolor = cv2.cvtColor(impre[scan], cv2.COLOR_BGR2RGB)
                imn=cv2.add(imcolor,tablscan[scan])
                imnc[scan]=imn
                cv2.imwrite(pf,imn)

                for imgdcm in listdcm:
                 FilesDCM =(os.path.join(dirpatientfdb1,imgdcm))
                 RefDs = dicom.read_file(FilesDCM)
                 slicenumberdicom=int(RefDs.InstanceNumber)
                 if slicenumberdicom==scan:
                     dsr= RefDs.pixel_array
                     dsr= dsr-dsr.min()
                     c=float(255)/dsr.max()
                     dsr=dsr*c
                     dsr = dsr.astype('uint8')
                     dsr = cv2.cvtColor(dsr, cv2.COLOR_GRAY2RGB)
                     RefDs.PhotometricInterpretation='RGB'
                     RefDs.BitsAllocated=8
                     RefDs.SamplesPerPixel=3
                     RefDs.BitsStored=8
                     RefDs.StudyInstanceUID                 =str(rsuid)
                     RefDs.PlanarConfiguration=0
                     RefDs.HighBit=7
                     RefDs.SeriesDescription='WithPredict'

                     imncop=cv2.resize(imn,(dsr.shape[0],dsr.shape[1]))
                     dsr=cv2.add(dsr,imncop)
                     RefDs.PixelData=dsr
                     FilesDCMW =(os.path.join(dcmf,imgdcm))
                     RefDs.save_as(FilesDCMW)
                     break

    return tabx,imnc


def createProba(pat,pr,px):
    n=classif[pat]
    proba=[]
    l=len(px)
#    print n ,l
    for i in range(0,l):
        if i==n:
            proba.append(pr)
        else:
            proba.append(0)
    return proba


def pavf(proba_cross,dirf,pat,scan,tab,
         dimpavx,dimpavy,dimtabx,dimtaby,jpegpath,patch_list,proba_list,bg):
     global thrprobaUIP,thrprobaMerge
#     print 'pav merge', pat,scan
     tabfw=np.zeros((dimtabx,dimtaby,3),np.uint8)
     jpegpathdir=os.path.join(dirf,jpegpath)
     if not os.path.exists(jpegpathdir):
         os.mkdir(jpegpathdir)
     tabbc=np.copy(tab)
     kernele=np.ones((5,5),np.uint8)
#     kernelc=np.ones((3,3),np.uint8)
     kerneld=np.ones((5,5),np.uint8)

     dilatation = cv2.dilate(tabbc,kerneld,iterations = 1)
     tabbc = cv2.erode(dilatation,kernele,iterations = 1)

     tabr=np.copy(tabbc)
     tabr= cv2.cvtColor(tabr,cv2.COLOR_GRAY2BGR)
     atabf = np.nonzero(tab)
     pxy=float(dimpavx*dimpavy)
     np.putmask(tabbc,tabbc>0,1)
                #tab[y][x]  convention
     xmin=atabf[1].min()
     xmax=atabf[1].max()
     ymin=atabf[0].min()
     ymax=atabf[0].max()
     probadef=createProba(pat,thrprobaUIP+0.05,proba_cross[0])
     probadefbg=createProba(pat,0.1,proba_cross[0])

     i=xmin
     while i < xmax:
         j=ymin
         while j<ymax:

             tabpatch=tabbc[j:j+dimpavy,i:i+dimpavx]
             area= tabpatch.sum()
             targ=float(area)/pxy

             if targ>thrprobaMerge:
#                print i,j,targ,area
                patch_list.append((scan,i,j))
                if not bg:
                    proba_list.append(probadef)
                else:
                    proba_list.append(probadefbg)
                tabbc[j:j+dimpavy,i:i+dimpavx]=0
                x=0
                while x < dimpavx:
                        y=0
                        while y < dimpavy:
                            if y+j<dimtabx and x+i<dimtaby:
                                tabfw[y+j][x+i]=[125,0,0]
                            if x == 0 or x == dimpavx-1 :
                                y+=1
                            else:
                                y+=dimpavy-1
                        x+=1
#                j+=dimpavy-1
             j+=dimpavy
         i+=dimpavx

     nameslijpeg=pat+'s_'+str(scan)+'.'+typej
     namepatchImage=os.path.join(jpegpathdir,nameslijpeg)
     if os.path.exists(namepatchImage):
         tabr=cv2.imread(namepatchImage,1)
#     print tabfw.shape,tabr.shape
     tabjpeg=cv2.add(tabfw,tabr)
#     cv2.imshow('tabfw'+pat+str(scan),tabfw)
#     cv2.imshow('tabjpeg',tabjpeg)
#     cv2.imshow('tabr',tabr)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

     scipy.misc.imsave(namepatchImage,tabjpeg)
     return proba_list,patch_list

def mergeproba(dirf,prx,plx,tabx,slnt,dimtabx,dimtaby)  :
    global thrprobaUIP
    print "merge proba list"

    patch_list_merge=[]
    proba_merge=[]

    plxd={}
    prxd={}

    listp=[name for name in usedclassif if name not in excluvisu]
    print 'used patterns :',listp

    for i in usedclassif:

        plxd[i]=[]
        prxd[i]=[]
    probabg=createProba('healthy',0.1,prx[0])

    print 'sort plx'
    for ll in range(0,len(plx)):
        proba=prx[ll]
        prec, mprobai = maxproba(proba)
        classlabel=fidclass(prec,classif)
        if mprobai> thrprobaUIP:
            plxd[classlabel].append(plx[ll])
            prxd[classlabel].append(prx[ll])
        else:
            plxd['healthy'].append(plx[ll])
            prxd['healthy'].append(probabg)


    print 'merge pattern'
    for p in listp:
#        print p
        for i in range (1, slnt):
#            tab0=np.zeros((dimtabx,dimtaby),np.uint8)
            tab1=np.zeros((dimtabx,dimtaby),np.uint8)
            tab2=np.zeros((dimtabx,dimtaby),np.uint8)
            tab0=  tabx[p][i]
            for ll in range(0,len(plxd[p])):
                if plxd[p][ll][0]==i:
                    tab1[plxd[p][ll][2]:plxd[p][ll][2]+dimpavy,plxd[p][ll][1]:plxd[p][ll][1]+dimpavx]=100
            tab2=np.bitwise_and(tab0,tab1)
#            if i ==25 and p =='ground_glass':
#                 cv2.imshow('tab0',tab0)
#                 cv2.imshow('tab1',tab1)
#                 cv2.imshow('tab2',tab2)
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
#
            nz= np.count_nonzero(tab2)
            if nz>0:
                proba_merge,patch_list_merge=pavf(prx,dirf,p,i,tab2,dimpavx,dimpavy,
                dimtabx,dimtaby,jpegpadirm,patch_list_merge,proba_merge,False)
    print 'merge exclu'
    for p in excluvisu:
        print'excluvisu', p
        for i in range (1, slnt):
            tab2=np.zeros((dimtabx,dimtaby),np.uint8)
            for ll in range(0,len(plxd[p])):
                if plxd[p][ll][0]==i:
                    tab2[plxd[p][ll][2]:plxd[p][ll][2]+dimpavy,plxd[p][ll][1]:plxd[p][ll][1]+dimpavx]=100
            nz= np.count_nonzero(tab2)
            if nz>0:
                proba_merge,patch_list_merge=pavf(prx,dirf,p,i,tab2,dimpavx,dimpavy,
                dimtabx,dimtaby,jpegpadirm,patch_list_merge,proba_merge,True)

    print ' end excluvisu'
#                if len(pr)>0:
#                    proba_merge.append(pr)
#                    patch_list_merge.append(pl)
#    print patch_list_merge
    return proba_merge,patch_list_merge

def  calcMed (tabscanLung,lungSegment):
    '''calculate the median position in between left and right lung'''
#    print 'number of subpleural for : ',pat
#    print 'subpleural', ntp, pat
#    global lungSegment
    tabMed={}
    dimtabx=tabscanLung.shape[1]
    dimtaby=tabscanLung.shape[2]
    print dimtabx,dimtaby
#    print subErosion,avgPixelSpacing,subErosionPixel
#    lung_dir = os.path.join(ntp, lungmask)
#    lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)
#    lunglist = os.listdir(lung_bmp_dir)
    for slicename in lungSegment['allset']:

             imgngray = tabscanLung[slicename].copy()
             ke=5
             kernele=np.ones((ke,ke),np.uint8)
             kerneld=np.ones((ke,ke),np.uint8)

             erosion = cv2.erode(imgngray,kernele,iterations = 1)
             dilation = cv2.dilate(erosion,kerneld,iterations = 1)

             im2,contours0, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,\
                      cv2.CHAIN_APPROX_SIMPLE)
#             cv2.imshow('lung2',imgngray)
             xmed=np.zeros((2), np.uint16)
             xmaxi=np.zeros((2), np.uint16)
             xmini=np.zeros((2), np.uint16)
#             print len(contours0)
             if len(contours0)>1:
                 areaArray =[]
                 for i,c in enumerate(contours0):
                      area = cv2.contourArea(c)
                      areaArray.append(area)

    #first sort the array by area
                 sorteddata = sorted(zip(areaArray, contours0), key=lambda x: x[0], reverse=True)

    #find the nth largest contour [n-1][1], in this case 2
                 xmed= np.zeros(3, np.uint16)
                 xmini=np.zeros(3, np.uint16)
                 xmaxi=np.zeros(3, np.uint16)

                 firstlargestcontour = sorteddata[0][1]
                 visa = np.zeros((dimtabx,dimtaby,3), np.uint8)
                 cv2.drawContours(visa,firstlargestcontour,-1,white,1)
                 npvisa=np.array(visa)
                 atabf = np.nonzero(npvisa)
                 xmin=atabf[1].min()
                 xmax=atabf[1].max()
                 xmed[0]=(xmax+xmin)/2
                 xmini[0]=xmin
                 xmaxi[0]=xmax
#                 cv2.imshow('cont',visa)
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()


                 secondlargestcontour = sorteddata[1][1]
                 visa = np.zeros((dimtabx,dimtaby,3), np.uint8)
                 cv2.drawContours(visa,secondlargestcontour,-1,white,1)
                 npvisa=np.array(visa)
                 atabf = np.nonzero(npvisa)
                 xmin=atabf[1].min()
                 xmax=atabf[1].max()
                 xmed[1]=(xmax+xmin)/2
                 xmini[1]=xmin
                 xmaxi[1]=xmax
    #
#                 cv2.imshow('cont',visa)
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()

                 xmedf=0
    #             print n
                 ifinmin=0
                 for i in range (0,2):
    #                 print '3', i, xmed[i],xmedf
                     if xmed[i]>xmedf:

                         xmedf=xmed[i]
                         ifinmax=i
                     else:
                         ifinmin=i

                 xmedian=    (xmini[ifinmax]+xmaxi[ifinmin])/2
                 tabMed[slicename]=xmedian
             else:
                 xmedian=dimtaby/2

             if xmedian<0.75*dimtaby/2 or xmedian>1.25*dimtaby/2:
                 xmedian=dimtaby/2
             tabMed[slicename]=xmedian
#             print xmedian
#             tabm=np.zeros((dimtabx,dimtaby,3),np.uint8)
#             tabm[:,xmedian]=(0,125,0)
#
#             imgngrayc = cv2.cvtColor(imgngray,cv2.COLOR_GRAY2BGR)
#             cv2.imshow('image',cv2.add(imgngrayc,tabm) )
##             cv2.imshow('lung1',imgngray)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()

    return tabMed

def subpleural(dirf,tabscanLung,lungSegment,subErosion,crfr):
#def calcSupNp(preprob, posp, lungs, imscan, pat, midx, psp, dictSubP, dimtabx):
    '''calculate the number of pat in subpleural'''
    (top,tail)=os.path.split(dirf)
#    print 'number of subpleural for :',tail, 'pattern :', pat

    dimtabx=tabscanLung.shape[1]
    dimtaby=tabscanLung.shape[2]
#    slnt=tabscanLung.shape[0]
    subpleurmaskset={}
    for slicename in lungSegment['allset']:
        vis = np.zeros((dimtabx,dimtaby,3), np.uint8)

        imgngray = np.copy(tabscanLung[slicename])
        np.putmask(imgngray, imgngray == 1, 0)
        np.putmask(imgngray, imgngray > 0, 1)
    # subErosion=  in mm
    #avgPixelSpacing=0.734 in mm/ pixel
        subErosionPixel = int(round(2 * subErosion / avgPixelSpacing))
        kernele=np.ones((subErosionPixel,subErosionPixel),np.uint8)
        erosion = cv2.erode(imgngray,kernele,iterations = 1)

        ret, mask = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
#        mask_inv = np.bitwise_not(erosion)
        subpleurmask = np.bitwise_and(imgngray, mask_inv)
        subpleurmaskset[slicename]=subpleurmask
#        print subpleurmask.min(),subpleurmask.max()
        

        im2,contours0, hierarchy = cv2.findContours(subpleurmask,cv2.RETR_TREE,\
                      cv2.CHAIN_APPROX_SIMPLE)
        if crfr=='cross':
            sn=scan_bmp
        else:
            sn=transbmp
#    print "corectnumber",corectnumber
        pdirk = os.path.join(dirf,source_name)
        pdirk = os.path.join(pdirk,sn)
        listimage= [name for name in os.listdir(pdirk) if name.find('.'+typei)>0]
        for li in listimage:
            if rsliceNum(li,'_','.'+typei)==slicename:
                lfbg=os.path.join(pdirk,li)
                lf=os.path.join(pdirk,li)
                imscan = cv2.imread(lf)
                contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
                cv2.drawContours(vis,contours,-1,white,1)
                cv2.imwrite(lfbg,cv2.add(vis,imscan))
                break

    return subpleurmaskset

def calcSupNp(dirf, patch_list_cross_slice, pat, tabMed,
              dictSubP,dictP,thrprobaUIP,lungSegment,patch_list_cross_slice_sub):
    '''calculate the number of pat in subpleural'''
    (top,tail)=os.path.split(dirf)
#    print 'number of subpleural for :',tail, 'pattern :', pat

    for slicename in lungSegment['allset']:

        psp=posP(slicename, lungSegment)
#        ill = 0
        if len(patch_list_cross_slice[slicename])>0:
            for ll in range (0, len(patch_list_cross_slice[slicename])):
    
                xpat = patch_list_cross_slice[slicename][ll][0][0]
#                ypat = patch_list_cross_slice[slicename][ll][0][1]
                proba = patch_list_cross_slice[slicename][ll][1]
                prec, mprobai = maxproba(proba)
                classlabel = fidclass(prec, classif)
#                if pat == 'bronchiectasis' and slicename==14 and classlabel==pat:
#                    print slicename,pat,xpat,mprobai,classlabel
    
                if xpat >= tabMed[slicename]:
                    pospr = 0
                    pospl = 1
                else:
                    pospr = 1
                    pospl = 0
                if classlabel == pat and mprobai > thrprobaUIP:
                
                    dictP[pat][psp] = (
                    dictP[pat][psp][0] + pospl,
                    dictP[pat][psp][1] + pospr)
                    dictP[pat]['all'] = (dictP[pat]['all'][0] + pospl, dictP[pat]['all'][1] + pospr)
                    
        if len(patch_list_cross_slice_sub[slicename])>0:
            for ll in range (0, len(patch_list_cross_slice_sub[slicename])):
                xpat = patch_list_cross_slice_sub[slicename][ll][0][0]
#                ypat = patch_list_cross_slice_sub[slicename][ll][0][1]
                proba = patch_list_cross_slice_sub[slicename][ll][1]
                prec, mprobai = maxproba(proba)
                classlabel = fidclass(prec, classif)
#                if pat == 'bronchiectasis' and slicename==14 and classlabel==pat:
#                    print slicename,pat,xpat,mprobai,classlabel
                if classlabel == pat and mprobai > thrprobaUIP:
                    if xpat >= tabMed[slicename]:
                        pospr = 0
                        pospl = 1
                    else:
                        pospr = 1
                        pospl = 0
                    dictSubP[pat]['all'] = (
                            dictSubP[pat]['all'][0] + pospl,
                            dictSubP[pat]['all'][1] + pospr)
                    dictSubP[pat][psp] = (
                            dictSubP[pat][psp][0] + pospl,
                            dictSubP[pat][psp][1] + pospr)
#                ill += 1
#        if pat == 'bronchiectasis':
#             print 'tot',slicename, pat, dictP[pat]
#             print 'sub',slicename, pat, dictSubP[pat]
    return dictSubP,dictP


def cvsarea(p, f, de, dse, s, dc, wf):
    '''calculate area of patches related to total area'''
    dictint = {}
    d = de[p]
    ds = dse[p]

    llungloc = (('lowerset', 'lower'), ('middleset', 'middle'), ('upperset', 'upper'))
    llunglocsl = (('lowerset', 'left_sub_lower'), ('middleset', 'left_sub_middle'), ('upperset', 'left_sub_upper'))
    llunglocsr = (('lowerset', 'right_sub_lower'), ('middleset','right_sub_middle'), ('upperset', 'right_sub_upper'))
    llunglocl = (('lowerset', 'left_lower'), ('middleset', 'left_middle'), ('upperset', 'left_upper'))
    llunglocr = (('lowerset', 'right_lower'), ('middleset','right_middle'), ('upperset', 'right_upper'))
    if wf:
        f.write(p + ': ')
    for i in llungloc:
        st = s[i[0]][0] + s[i[0]][1]
        if st > 0:
            l = 100 * float(d[i[0]][0] + d[i[0]][1]) / st
            l = round(l, 2)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')
#        if p == 'bronchiectasis':
#             print 'surface',i,st
#             print dictint[i[1]]
    for i in llunglocsl:
        st = s[i[0]][0]
#        print 'cvsarea',s
#        print 'cvsarea i',i[0], i[1]
#        print 'cvsarea p d', p,d
#        oo
        if st > 0:
            l = 100 * float(ds[i[0]][0]) / st
            l = round(l, 2)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')
#        if p == 'bronchiectasis':
#             print 'surface sub left',i,st
#             print dictint[i[1]]

    for i in llunglocsr:
        st = s[i[0]][1]
        if st > 0:
            l = 100 * float(ds[i[0]][1]) / st
            l = round(l, 2)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')
#        if p == 'bronchiectasis':
#             print 'surface sub right',i,st
#             print dictint[i[1]]

    for i in llunglocl:
        st = s[i[0]][0]
        if st > 0:
            l = 100 * float(d[i[0]][0]) / st
            l = round(l, 2)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')
#        if p == 'bronchiectasis':
#             print 'surface left',i,st
#             print dictint[i[1]]
             
    for i in llunglocr:
        st = s[i[0]][1]
        if st > 0:
            l = 100 * float(d[i[0]][1]) / st
            l = round(l, 2)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')
#        if p == 'bronchiectasis':
#             print 'surface right',i,st
#             print dictint[i[1]]

    dc[p] = dictint
    if wf:
        f.write('\n')

    return dc


def writedic(p,v,d):
    v.write(p+ ' '+d[p]['lower']+' '+d[p]['middle']+' '+d[p]['upper']+' ')
    v.write(d[p]['left_sub_lower']+' '+d[p]['left_sub_middle']+' '+d[p]['left_sub_upper']+' ')
    v.write(d[p]['right_sub_lower']+' '+d[p]['right_sub_middle']+' '+d[p]['right_sub_upper']+' ')
    v.write(d[p]['left_lower']+' '+d[p]['left_middle']+' '+d[p]['left_upper']+' ')
    v.write(d[p]['right_lower']+' '+d[p]['right_middle']+' '+d[p]['right_upper']+'\n')


def posP(sln, lungSegment):
    '''define where is the slice number'''
    if sln in lungSegment['upperset']:
        psp = 'upperset'
    elif sln in lungSegment['middleset']:
        psp = 'middleset'
    else:
        psp = 'lowerset'
    return psp


def calculSurface(dirf,posp, midx,lungSegment,dictPS):
    (top,tail)=os.path.split(dirf)
#    print 'calculate surface for :',tail
    ill = 0
#    print posp[0]
#    oooo
    for ll in posp:
        xpat = ll[1]
        sln=ll[0]
#        print xpat,sln
        psp=posP(sln, lungSegment)

        if xpat >= midx[sln]:
            pospr = 0
            pospl = 1
        else:
            pospr = 1
            pospl = 0

        dictPS[psp] = (dictPS[psp][0] + pospl, dictPS[psp][1] + pospr)
        dictPS['all'] = (dictPS['all'][0] + pospl, dictPS['all'][1] + pospr)

        ill += 1
    return dictPS

def writedict(dirf, dx):
    (top,tail)=os.path.split(dirf)
    print 'write file  for :',tail
    ftw = os.path.join(dirf, str(dx) + '_' + volumeweb)
    volumefile = open(ftw,'w')
    volumefile.write(
        'patient UIP WEB: ' +
        str(tail) +
        ' ' +
        'patch_size: ' +
        str(dx) +
        '\n')
    volumefile.write('pattern   lower  middle  upper')
    volumefile.write('  left_sub_lower  left_sub_middle  left_sub_upper ')
    volumefile.write('  right_sub_lower  right_sub_middle  right_sub_upper ')
    volumefile.write('  left_lower  left_middle  left_upper ')
    volumefile.write(' right_lower  right_middle  right_upper\n')
    return volumefile

#def uipTree(pid, proba, posp, lungs, imscan, tabmedx,
#            psp, dictP, dictSubP, dictPS, dimtabx):
def uipTree(dirf,patch_list_cross_slice,lungSegment,tabMed,dictPS,
            dictP,dictSubP,dictSurf,thrprobaUIP,patch_list_cross_slice_sub):
    '''calculate the number of reticulation and HC in total and subpleural
    and diffuse micronodules'''
    (top,tail)=os.path.split(dirf)
#    print 'calculate volume in : ',tail

    
#    print '-------------------------------------------'
#    print 'surface total  by segment Left Right :'
#    print dictPS
#    print '-------------------------------------------'
    if writeFile:
         volumefile = writedict(dirf, dimpavx)
    else:
        volumefile = ''
    for pat in classif:
       
        dictSubP,dictP= calcSupNp(dirf, patch_list_cross_slice, pat, tabMed,
              dictSubP,dictP,thrprobaUIP,lungSegment,patch_list_cross_slice_sub)
#        if pat == 'bronchiectasis':
#            print ' volume total for:',pat
#            print dictP[pat]
#            print '-------------------------------------------'
#            print ' volume subpleural for:', pat
#            print dictSubP[pat]
#            print '-------------------------------------------'
#            print ' volume total for:',pat
#            print dictP[pat]
#            print '-------------------------------------------'
        dictSurf = cvsarea(
            pat,
            volumefile,
            dictP,
            dictSubP,
            dictPS,
            dictSurf,
            writeFile)
    if writeFile:
            volumefile.write('---------------------\n')
            volumefile.close()
#    print ' volume for:',tail
#    print dictSurf
#    print '-------------------------------------------'
    return dictP, dictSubP, dictSurf

def selectposition(lislnumber):
    upperset=[]
    middleset=[]
    lowerset=[]
    allset=[]
    lungs={}
    Nset=len(lislnumber)/3
    for scanumber in lislnumber:
            allset.append(scanumber)
            if scanumber < Nset:
                upperset.append(scanumber)
            elif scanumber < 2*Nset:
                middleset.append(scanumber)
            else:
                lowerset.append(scanumber)
            lungs['upperset']=upperset
            lungs['middleset']=middleset
            lungs['lowerset']=lowerset
            lungs['allset']=allset
    return lungs

def initdictP(d, p):
    d[p] = {}
    d[p]['upperset'] = (0, 0)
    d[p]['middleset'] = (0, 0)
    d[p]['lowerset'] = (0, 0)
    d[p]['all'] = (0, 0)
    return d

def genepatchlistslice(patch_list_cross,proba_cross,lissln,subpleurmask,thrpatch):
    res={}   
    ressub={}  
    dimtabx=subpleurmask[lissln[0]].shape[0]
    dimtaby=subpleurmask[lissln[0]].shape[1]
    for i in lissln:
        res[i]=[]
        ressub[i]=[]  
        ii=0
        for ll in patch_list_cross:
            xpat = ll[1]
            ypat = ll[2]
            sln= ll[0]
            if sln ==i:
                t=((xpat,ypat),proba_cross[ii])
                res[sln].append(t)                
                tabpatch = np.zeros((dimtabx, dimtaby), np.uint8)
                tabpatch[ypat:ypat + dimpavy, xpat:xpat + dimpavx] = 1
                tabsubpl = np.bitwise_and(subpleurmask[i], tabpatch)
                np.putmask(tabsubpl, tabsubpl > 0, 1)

                area = tabsubpl.sum()
                targ = float(area) / pxy    

                if targ > thrpatch:
                        ressub[i].append(t)  

            ii+=1
            
            
    return res,ressub

def predictrun(indata,path_patient):
        global thrpatch,thrproba,thrprobaMerge,thrprobaUIP,subErosion
        global  picklein_file,picklein_file_front
        td=False
#        print path_patient
#        print indata
#        ooo
        
        if indata['threedpredictrequest']=='Cross Only':
           td=False
        else:
            td=True
        thrproba=float(indata['thrproba'])
        thrprobaMerge=float(indata['thrprobaMerge'])
        thrprobaUIP=float(indata['thrprobaUIP'])
        thrpatch=float(indata['thrpatch'])
        picklein_filet=indata['picklein_file']
        picklein_file_frontt=indata['picklein_file_front']
        subErosion=indata['subErosion']
#        print thrproba,thrprobaMerge,thrprobaUIP
        listHug=[]
#        print indata
#        print path_patient
        listHugi=indata['lispatientselect']
#        print 'listHugi',listHugi
        for lpt in listHugi:
#             print lpt
             pos=lpt.find(' PREDICT!:')
#             print pos
             if pos >0:
                    listHug.append(lpt[0:pos])
             else:
                    pos=lpt.find(' noPREDICT!')
#                    print 'no predict',pos
                    listHug.append(lpt[0:pos])
#        listHug=['36']

        picklein_file =  os.path.join(dirpickle,picklein_filet)
        picklein_file_front =  os.path.join(dirpickle,picklein_file_frontt)

        dirHUG=os.path.join(cwdtop,path_patient)
#        print dirHUG
#        print listHug

        eferror=os.path.join(dirHUG,'gp3d.txt')
        for f in listHug:
            print 'work on patient',f
            lungSegment={}
            patch_list_cross_slice={}
            errorfile = open(eferror, 'a')
            listelabelfinal={}
            dirf=os.path.join(dirHUG,f)
#            """
            tabMed = {}  # dictionary with position of median between lung

            wridirsource=os.path.join(dirf,source)
            wridir=os.path.join(wridirsource,transbmp)
            remove_folder(wridir)
            os.mkdir(wridir)
#            """
            path_data_write=os.path.join(dirf,path_data)
#            """
#            remove_folder(path_data_write)
            if not os.path.exists(path_data_write):
                os.mkdir(path_data_write)

            jpegpathdir=os.path.join(dirf,jpegpadirm)
            remove_folder(jpegpathdir)
            os.mkdir(jpegpathdir)
#            """
            dicompathdir=os.path.join(dirf,dicompadirm)
#            """
            remove_folder(dicompathdir)
            os.mkdir(dicompathdir)

            dicompathdircross=os.path.join(dicompathdir,dicomcross)

            remove_folder(dicompathdircross)
            os.mkdir(dicompathdircross)
#            """
            dicompathdirfront=os.path.join(dicompathdir,dicomfront)
#            """
            remove_folder(dicompathdirfront)
            os.mkdir(dicompathdirfront)
#            """
            dicompathdirmerge=os.path.join(dicompathdir,dicomcross_merge)
#            """
            remove_folder(dicompathdirmerge)
            os.mkdir(dicompathdirmerge)

            
            tabscanScan,slnt,dimtabx,slicepitch,lissln=genebmp(dirf,source)
            lungSegment=selectposition(lissln)

#                print 'slnt',slnt
            datacross=(slnt,dimtabx,dimtabx,slicepitch,lissln)
            pickle.dump(datacross, open( os.path.join(path_data_write,datacrossn), "wb" ),protocol=-1)
            pickle.dump(tabscanScan, open( os.path.join(path_data_write,"tabscanScan"), "wb" ),protocol=-1)
            pickle.dump(lungSegment, open( os.path.join(path_data_write,"lungSegment"), "wb" ),protocol=-1)
#            """
            datacross= pickle.load( open( os.path.join(path_data_write,"datacross"), "rb" ))
            tabscanScan= pickle.load( open( os.path.join(path_data_write,"tabscanScan"), "rb" ))
            lungSegment= pickle.load( open( os.path.join(path_data_write,"lungSegment"), "rb" ))
        
            slnt=datacross[0]
            dimtabx=datacross[1]
            dimtaby=datacross[2]
            slicepitch=datacross[3]
            lissln=datacross[4]
#            """
#
            tabscanLung=genebmplung(dirf,lung_name_gen,slnt,dimtabx,dimtabx)
            subpleurmask=subpleural(dirf,tabscanLung,lungSegment,subErosion,'cross')
            pickle.dump(subpleurmask, open( os.path.join(path_data_write,"subpleurmask"), "wb" ),protocol=-1)
            """
            subpleurmask= pickle.load( open( os.path.join(path_data_write,"subpleurmask"), "rb" ))
            """
            patch_list_cross=pavgene(dirf,dimtabx,dimtabx,tabscanScan,tabscanLung,slnt,jpegpath)
            
            model=modelCompilation('cross',picklein_file,picklein_file_front)
            proba_cross=ILDCNNpredict(patch_list_cross,model)
            patch_list_cross_slice,patch_list_cross_slice_sub=genepatchlistslice(patch_list_cross,
                                                            proba_cross,lissln,subpleurmask,thrpatch)
            tabMed = calcMed(tabscanLung,lungSegment)

            pickle.dump(proba_cross, open( os.path.join(path_data_write,"proba_cross"), "wb" ),protocol=-1)
            pickle.dump(patch_list_cross_slice, open( os.path.join(path_data_write,"patch_list_cross_slice"), "wb" ),protocol=-1)
            pickle.dump(patch_list_cross_slice_sub, open( os.path.join(path_data_write,"patch_list_cross_slice_sub"), "wb" ),protocol=-1)
            pickle.dump(tabscanLung, open( os.path.join(path_data_write,"tabscanLung"), "wb" ),protocol=-1)
            pickle.dump(patch_list_cross, open( os.path.join(path_data_write,"patch_list_cross"), "wb" ),protocol=-1)
            pickle.dump(tabMed, open( os.path.join(path_data_write,"tabMed"), "wb" ),protocol=-1)
            """
            proba_cross= pickle.load( open( os.path.join(path_data_write,"proba_cross"),"rb" ))
            patch_list_cross_slice= pickle.load( open( os.path.join(path_data_write,"patch_list_cross_slice"), "rb" ))
            patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_write,"patch_list_cross_slice_sub"), "rb" ))
            tabscanLung= pickle.load( open( os.path.join(path_data_write,"tabscanLung"), "rb" ))
            patch_list_cross= pickle.load( open( os.path.join(path_data_write,"patch_list_cross"), "rb" ))
            tabMed= pickle.load( open( os.path.join(path_data_write,"tabMed"), "rb" ))
            """
            
            visua(listelabelfinal,dirf,proba_cross,patch_list_cross,dimtabx,
                  dimtabx,slnt,predictout,sroi,scan_bmp,source,dicompathdircross,True,errorfile)            
            genethreef(dirf,patch_list_cross,proba_cross,slicepitch,dimtabx,dimtabx,dimpavx,slnt,'cross')
#            """
    ###       cross
            if td:
#                """
                tabresScan=reshapeScan(tabscanScan,slnt,dimtabx)
                dimtabxn,dimtabyn,tabScan3d,lisslnfront=wtebres(wridir,dirf,tabresScan,dimtabx,slicepitch,lung_name_gen,'scan')
                tabresLung=reshapeScan(tabscanLung,slnt,dimtabx)               
                dimtabxn,dimtabyn,tabLung3d,a=wtebres(wridir,dirf,tabresLung,dimtabx,slicepitch,lung_name_gen,'lung')
                datafront=(dimtabx,dimtabxn,dimtabyn,slicepitch,lisslnfront)
                
                pickle.dump(datafront, open( os.path.join(path_data_write,datafrontn), "wb" ),protocol=-1)
                """
                datafront= pickle.load( open( os.path.join(path_data_write,"datafront"), "rb" ))               
                dimtabx=datafront[0]                
                dimtabxn=datafront[1]
                dimtabyn=datafront[2]
                slicepitch=datafront[3]
                lisslnfront=datafront[4]
                """
                patch_list_front=pavgene(dirf,dimtabxn,dimtabx,tabScan3d,tabLung3d,dimtabyn,jpegpath3d)
                model=modelCompilation('front',picklein_file,picklein_file_front)
                proba_front=ILDCNNpredict(patch_list_front,model)
                
                pickle.dump(proba_front, open( os.path.join(path_data_write,"proba_front"), "wb" ),protocol=-1)
                pickle.dump(patch_list_front, open( os.path.join(path_data_write,"patch_list_front"), "wb" ),protocol=-1)
                """
                proba_front=pickle.load(open( os.path.join(path_data_write,"proba_front"), "rb" ))
                patch_list_front=pickle.load(open( os.path.join(path_data_write,"patch_list_front"), "rb" ))
                """                
                visua(listelabelfinal,dirf,proba_front,patch_list_front,dimtabxn,dimtabx,
                      dimtabyn,predictout3d,sroid,transbmp,source,dicompathdirfront,False,errorfile)
                """
                proba_cross=pickle.load(open( os.path.join(path_data_write,"proba_cross"), "rb" ))
                patch_list_cross=pickle.load(open( os.path.join(path_data_write,"patch_list_cross"), "rb" ))
                proba_front=pickle.load(open( os.path.join(path_data_write,"proba_front"), "rb" ))
                patch_list_front=pickle.load(open( os.path.join(path_data_write,"patch_list_front"), "rb" ))
                """
                lungSegmentfront=selectposition(lisslnfront)
                subpleurmaskfront=subpleural(dirf,tabLung3d,lungSegmentfront,subErosion,'front')
                patch_list_front_slice,patch_list_front_slice_sub=genepatchlistslice(patch_list_front,
                                                            proba_front,lisslnfront,subpleurmaskfront,thrpatch)
                tabMedfront = calcMed(tabLung3d,lungSegmentfront)
                
                pickle.dump(tabMedfront, open( os.path.join(path_data_write,"tabMedfront"), "wb" ),protocol=-1)
                pickle.dump(patch_list_front_slice, open( os.path.join(path_data_write,"patch_list_front_slice"), "wb" ),protocol=-1)
                pickle.dump(patch_list_front_slice_sub, open( os.path.join(path_data_write,"patch_list_front_slice_sub"), "wb" ),protocol=-1)
                pickle.dump(lungSegmentfront, open( os.path.join(path_data_write,"lungSegmentfront"), "wb" ),protocol=-1)
                """
                tabMedfront=pickle.load(open( os.path.join(path_data_write,"tabMedfront"), "rb" ))
                patch_list_front_slice=pickle.load(open( os.path.join(path_data_write,"patch_list_front_slice"), "rb" ))
                patch_list_front_slice_sub=pickle.load(open( os.path.join(path_data_write,"patch_list_front_slice_sub"), "rb" ))
                lungSegmentfront=pickle.load(open( os.path.join(path_data_write,"lungSegmentfront"), "rb" ))
                """
                genethreef(dirf,patch_list_front,proba_front,avgPixelSpacing,dimtabxn,dimtabyn,dimpavx,dimtabx,'front')
                tabpx=genecross(proba_cross,dirf,proba_front,patch_list_front,slnt,slicepitch,dimtabxn,dimtabyn,predictout3dn)
        #        pickle.dump(tabpx, open( os.path.join(path_data_write,"tabpx"), "wb" ),protocol=-1)
        #        tabpx=pickle.load(open( os.path.join(path_data_write,"tabpx"), "rb" ),protocol=-1)
                tabx,tabfromfront=reshapepatern(dirf,tabpx,dimtabxn,dimtabx,slnt,slicepitch,predictout3d1,source,dicompathdirfront)
                pickle.dump(tabfromfront, open( os.path.join(path_data_write,"tabfromfront"), "wb" ),protocol=-1)
        #        tabx=pickle.load(open( os.path.join(path_data_write,"tabx"), "rb" ),protocol=-1)
#                    print 'before merge proba'
                proba_merge,patch_list_merge=mergeproba(dirf,proba_cross,patch_list_cross,tabx,slnt,dimtabx,dimtabx)
                patch_list_merge_slice,patch_list_merge_slice_sub=genepatchlistslice(patch_list_merge,
                                                            proba_merge,lissln,subpleurmask,thrpatch)
                
                pickle.dump(proba_merge, open( os.path.join(path_data_write,"proba_merge"), "wb" ),protocol=-1)
                pickle.dump(patch_list_merge, open( os.path.join(path_data_write,"patch_list_merge"), "wb" ),protocol=-1)
                pickle.dump(patch_list_merge_slice, open( os.path.join(path_data_write,"patch_list_merge_slice"), "wb" ),protocol=-1)
                pickle.dump(patch_list_merge_slice_sub, open( os.path.join(path_data_write,"patch_list_merge_slice_sub"), "wb" ),protocol=-1)
                """
                proba_merge=pickle.load(open( os.path.join(path_data_write,"proba_merge"), "rb" ))
                patch_list_merge=pickle.load(open( os.path.join(path_data_write,"patch_list_merge"), "rb" ))
                patch_list_merge_slice=pickle.load(open( os.path.join(path_data_write,"patch_list_merge_slice"), "rb" ))
                patch_list_merge_slice_sub=pickle.load(open( os.path.join(path_data_write,"patch_list_merge_slice_sub"), "rb" ))
                """                                              
                visua(listelabelfinal,dirf,proba_merge,patch_list_merge,dimtabx,dimtabx
                      ,slnt,predictoutmerge,sroi,scan_bmp,source,dicompathdirmerge,True,errorfile)
                genethreef(dirf,patch_list_merge,proba_merge,slicepitch,dimtabx,dimtabx,dimpavx,slnt,'merge')

            errorfile.write('completed :'+f)
            errorfile.close()


#errorfile.close()
print "predict time:",round(mytime()-t0,3),"s"

#ILDCNNpredict(bglist)