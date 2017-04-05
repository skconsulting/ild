# coding: utf-8
'''create patches from patient on front view, including ROI when overlapping
    sylvain kritter 6 february 2017
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
from PIL import Image, ImageFont, ImageDraw

def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)

#####################################################################
#define the working directory
toppatch= 'TOPPATCH'
    
#extension for output dir
#extendir='2'
extendir='3d162'

#alreadyDone =['S1830','S14740','S15440','S28200','S106530','S107260','S139430','S145210']
alreadyDone =['S106530','S107260','S139430','S145210','S14740','S15440','S1830','S28200','S335940','S359750']
alreadyDone =[]

normiInternal=True
globalHist=True #use histogram equalization on full image
globalHistInternal=True #use internal for global histogram when True otherwise opencv

isGre=True
HUG='CHU'   
#HUG='HUG'
#subDir='ILDt1'
#subDir='UIPtt'
subDir='UIP'
#subDir='UIP_S14740'
#subDir='UIP_106530'

scan_bmp='scan_bmp'
transbmp='trans_bmp'
sroid='sroi3d'
bgdir='bgdir3d'

typei='jpg'
typeroi='jpg'
typeid='png' #can be png for 16b
typej='jpg'

source_name='source'
lung_name='lung'
labelbg='back_ground'
locabg='td_CHUG'

reserved=['bgdir','sroi','sroi1','bgdir3d','sroi3d']
notclas=['lung','source','B70f']

dimpavx=16
dimpavy=16
pxy=float(dimpavx*dimpavy)

imageDepth=8191 #number of bits used on dicom images (2 **n) 13 bits
#imageDepth=255 #number of bits used on dicom images (2 **n) 13 bits
#patch overlapp tolerance
thrpatch = 0.8

patchesdirnametop = toppatch+'_'+extendir
#define the name of directory for patches
patchesdirname = 'patches'
#define the name of directory for normalised patches
patchesNormdirname = 'patches_norm'
#define the name for jpeg files
imagedirname='patches_jpeg'

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
dirHUG=os.path.join(cwdtop,HUG)
patchtoppath=os.path.join(dirHUG,patchesdirnametop)

dirHUG=os.path.join(dirHUG,subDir)

listHug= [ name for name in os.listdir(dirHUG) if os.path.isdir(os.path.join(dirHUG, name)) and \
            name not in alreadyDone]
print listHug

font5 = ImageFont.truetype( 'arial.ttf', 5)
font10 = ImageFont.truetype( 'arial.ttf', 10)
font20 = ImageFont.truetype( 'arial.ttf', 20)
#create patch and jpeg directory
patchpath=os.path.join(patchtoppath,patchesdirname)
#create patch and jpeg directory
patchNormpath=os.path.join(patchtoppath,patchesNormdirname)
#print patchpath
#define the name for jpeg files
jpegpath=os.path.join(patchtoppath,imagedirname)

if not os.path.isdir(patchtoppath):
    os.mkdir(patchtoppath)   

#remove_folder(patchpath)
if not os.path.isdir(patchpath):
    os.mkdir(patchpath)   

#remove_folder(patchNormpath)
if not os.path.isdir(patchNormpath):
    os.mkdir(patchNormpath)  

#remove_folder(jpegpath)
if not os.path.isdir(jpegpath):
    os.mkdir(jpegpath)   
    
eferror=os.path.join(patchtoppath,'genepatcherrortop3d.txt')

avgPixelSpacing=0.734   # average pixel spacing
###############################################################
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
        'emphysema':10,
        'HCpret':11,
        'HCpbro':12,
        'GGpbro':13,
        'GGpret':14,
        'bropret':15,
        
         'bronchial_wall_thickening':16,
         'early_fibrosis':17,
         'increased_attenuation':18,
         'macronodules':19,
         'pcp':20,
         'peripheral_micronodules':21,
         'tuberculosis':22
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
        'emphysema',
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

reserved=reserved+derivedpat   
        
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
col1= (142,180,227)
col2=(155,215,204)
col3=(214,226,144)
col4=(234,136,222)
col5=(218,163,152)


classifc ={
'back_ground':darkgreen,
#'consolidation':red,
'consolidation':cyan,
'HC':blue,
'ground_glass':red,
#'ground_glass':yellow,
'healthy':darkgreen,
#'micronodules':cyan,
'micronodules':green,
#'reticulation':purple,
'reticulation':yellow,
'air_trapping':pink,
'cysts':lightgreen,
 'bronchiectasis':orange,
 'HCpret': col1,
 'HCpbro': col2,
 'GGpbro': col3,
 'GGpret': col4,
 'bropret': col5,
 'nolung': lowgreen,
 'bronchial_wall_thickening':white,
 'early_fibrosis':white,
 'emphysema':white,
 'increased_attenuation':white,
 'macronodules':white,
 'pcp':white,
 'peripheral_micronodules':white,
 'tuberculosis':white
 }


def rsliceNum(s,c,e):
    endnumslice=s.find(e)
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

     tabi2=(tabi-min_val)*(imageDepth/mm)
     tabi2=tabi2.astype('uint16')

     return tabi2
     
def reshapeScan(tabscan,slnt,dimtabx):
    print 'reshapescan '
    tabres = np.zeros((dimtabx,slnt,dimtabx), np.uint16)
    for i in range (0,dimtabx):
        for j in range (0,slnt):
            tabres[i][j]=tabscan[j][i]
    return tabres
    
def genepara(namedirtopcf): 
    dirFileP = os.path.join(namedirtopcf, 'source')
        #list dcm files
    fileList =[name for name in  os.listdir(dirFileP) if ".dcm" in name.lower()]
    slnt=0
    for filename in fileList:
        FilesDCM =(os.path.join(dirFileP,filename))         
        RefDs = dicom.read_file(FilesDCM) 
        scanNumber=int(RefDs.InstanceNumber)
        if scanNumber>slnt:
            slnt=scanNumber
    slnt=slnt+1
    SliceThickness=RefDs.SliceThickness
    try:
            SliceSpacingB=RefDs. SpacingBetweenSlices
    except AttributeError:
             print "Oops! No Slice spacing..."
             SliceSpacingB=0
    
    slicepitch=float(SliceThickness)+float(SliceSpacingB)
    print SliceThickness
    print SliceSpacingB
    print 'slice pitch in z :',slicepitch
    
    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing 
    dsr= RefDs.pixel_array
    dsr= dsr-dsr.min() 
    dsr=dsr.astype('uint16')
    dsrresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
    dimtabx=int(dsrresize.shape[0])
    dimtaby=int(dsrresize.shape[1])  
    return dimtabx,dimtaby,slnt  ,slicepitch          

def tagviews(tab,text,x,y):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,text,(x, y), font,0.3,white,1)
    return viseg
    
    
def genebmp(dirName, sou,slnt,dx,dy):
    """generate patches from dicom files and sroi"""
    
    if sou=='source':
        tabres=np.zeros((slnt,dx,dy),np.uint16)
        dsrresize=np.zeros((dx,dy),np.uint16)
    else:
        tabres=np.zeros((slnt,dx,dy),np.uint8)
        dsrresize=np.zeros((dx,dy),np.uint8)
        
    dirFileP = os.path.join(dirName, sou)

    (top,tail)=os.path.split(dirName)
    print ('generate image in :',tail, 'directory :',sou)
    fileList =[name for name in  os.listdir(dirFileP) if ".dcm" in name.lower()]
    
    for filename in fileList:
                FilesDCM =(os.path.join(dirFileP,filename))         
                RefDs = dicom.read_file(FilesDCM) 
                dsr= RefDs.pixel_array
                dsr= dsr-dsr.min() 
                dsr=dsr.astype('uint16')
                if dsr.max()>0:
                    if sou !='source' :
                        c=float(255)/dsr.max()
                        dsr=dsr*c     
#                        dsr=dsr.astype('uint8')
                        dsr = dsr.astype('uint8')
                        
                    #resize the dicom to have always the same pixel/mm
                    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing   
                    dsrresize1=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
#                    print type(dsrresize1[0][0]),sou
                    if sou=='source' :
                        if globalHist:
                            if globalHistInternal:
                                dsrresize = normi(dsrresize1) 
                                
                            else:
                                dsrresize= cv2.normalize(dsrresize1, 0, imageDepth,cv2.NORM_MINMAX);
                        else:
                            dsrresize=dsrresize1 
                    else:
                            dsrresize=dsrresize1                    
                    scanNumber=int(RefDs.InstanceNumber)
                   
                    if sou == 'lung':
                        np.putmask(dsrresize,dsrresize>0,100)
                        
                    elif sou !='source':                        
                        np.putmask(dsrresize,dsrresize==1,0)
                        np.putmask(dsrresize,dsrresize>0,100)
                  
                    tabres[scanNumber]=  dsrresize

    return tabres        
 

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
def contour2(im,l,dx,dy):  
    col=classifc[l]
    vis = np.zeros((dx,dy,3), np.uint8)
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

           
def pavs (dirName,pat,dx,dy):
    """ generate patches from ROI"""
    ntotpat=0 
    
    tabf=np.zeros((dx,dy),np.uint8)
    _tabsroi=np.zeros((dx,dy,3),np.uint8)
    _tabbg = np.zeros((dx,dy),np.uint8)
    _tabscan = np.zeros((dx,dy),np.uint16)
    
    (top,tail)=os.path.split(dirName)
    print 'pav :',tail,'pattern :',pat
           
    nampadir=os.path.join(patchpath,pat)
    nampadirl=os.path.join(nampadir,locabg)
    if not os.path.exists(nampadir):
         os.mkdir(nampadir)
         os.mkdir(nampadirl)
    nampaNormdir=os.path.join(patchNormpath,pat)
    nampadirNorml=os.path.join(nampaNormdir,locabg)
    if not os.path.exists(nampaNormdir):
         os.mkdir(nampaNormdir)           
         os.mkdir(nampadirNorml)                     

    for scannumb in range (0,dy):
       tabp = np.zeros((dx, dy), dtype=np.uint8)
       tabf=np.copy(tabroipat3d[pat][scannumb])

       tabfc=np.copy(tabf)
       nbp=0
       if tabf.max()>0:    
           vis=contour2(tabf,pat,dx,dy)
           if vis.sum()>0:              
                _tabsroi = np.copy(tabsroi3d[scannumb])             
                imn=cv2.add(vis,_tabsroi)
                imn=tagview(imn,pat,0,20)
                tabsroi3d[scannumb]=imn
                imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)
                
                sroifile='tr_'+str(scannumb)+'.'+typeroi                   
                filenamesroi=os.path.join(sroidir,sroifile)
                cv2.imwrite(filenamesroi,imn)
                
                bgfile='bg_'+str(scannumb)+'.'+typeroi
                namebg=os.path.join(bgdirf,bgfile)

                _tabbg = np.copy(tabsbg3d[scannumb]) 

                np.putmask(_tabbg,_tabbg>0,100)
                np.putmask(tabf,tabf>0,100)

                mask=cv2.bitwise_not(tabf)

                outy=cv2.bitwise_and(_tabbg,mask)
                tabsbg3d[scannumb]=outy   

                cv2.imwrite(namebg,outy)
                atabf = np.nonzero(tabf)

                xmin=atabf[1].min()
                xmax=atabf[1].max()
                ymin=atabf[0].min()
                ymax=atabf[0].max()

                np.putmask(tabf,tabf>0,1)
                _tabscan=tabscan3d[scannumb]            
                 
                i=xmin
                while i <= xmax:
                    j=ymin
                    while j<=ymax:
                        tabpatch=tabf[j:j+dimpavy,i:i+dimpavx]
                       
                        area= tabpatch.sum()  
                        targ=float(area)/pxy
                        
                        if targ >thrpatch:                               
                            imgray = _tabscan[j:j+dimpavy,i:i+dimpavx]   
                            imagemax= cv2.countNonZero(imgray)
                            min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(imgray)
                            
                            if imagemax > 0 and max_val - min_val>20:   
                                nbp+=1                                   
                                nampa=os.path.join(nampadirl,tail+'_'+str(i)+'_'+str(j)+'_'+str(scannumb)+'.'+typeid )
                                cv2.imwrite (nampa, imgray,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
                                if normiInternal:
                                    tabi2 = normi(imgray)                                   
                                else:
                                    tabi2= cv2.normalize(imgray, 0, imageDepth,cv2.NORM_MINMAX); 
                                nampa=os.path.join(nampadirNorml,tail+'_'+str(i)+'_'+str(j)+'_'+str(scannumb)+'.'+typeid )
                                cv2.imwrite (nampa, tabi2,[int(cv2.IMWRITE_PNG_COMPRESSION),0])                             
                                x=0
                                #we draw the rectange
                                while x < dimpavx:
                                    y=0
                                    while y < dimpavy:
                                        tabp[y+j][x+i]=150
                                        if x == 0 or x == dimpavx-1 :
                                            y+=1
                                        else:
                                            y+=dimpavy-1
                                    x+=1
                                #we cancel the source
                                tabf[j:j+dimpavy,i:i+dimpavx]=0
                                j+=dimpavy-1                    
                        j+=1
                    i+=1                                                
                    
       if nbp>0:
             tabfc =tabfc+tabp
             ntotpat=ntotpat+nbp
             if scannumb not in listsliceok:
                    listsliceok.append(scannumb) 
             stw=tail+'_slice_'+str(scannumb)+'_'+pat+'_'+locabg+'_'+str(nbp)
             stww=stw+'.txt'
             flw=os.path.join(jpegpath,stww)
             mfl=open(flw,"w")
             mfl.write('#number of patches: '+str(nbp)+'\n')
             mfl.close()
             stww=stw+'.'+typej
             flw=os.path.join(jpegpath,stww)
             scipy.misc.imsave(flw, tabfc)

    return ntotpat

def pavbg (dirName,dx,dy):
    """ generate patches back-ground from ROI"""
    ntotpat=0
    tabf=np.zeros((dx,dy,),np.uint8)
 
    _tabscan = np.zeros((dx,dy),np.uint16)
    
    (top,tail)=os.path.split(dirName)
    print 'pavbg :',tail,'pattern : back_ground'
 
    labelbg='back_ground'       
    nampadir=os.path.join(patchpath,labelbg)
    nampadirl=os.path.join(nampadir,locabg)
    if not os.path.exists(nampadir):
         os.mkdir(nampadir)
         os.mkdir(nampadirl)
    nampaNormdir=os.path.join(patchNormpath,labelbg)
    nampadirNorml=os.path.join(nampaNormdir,locabg)
    if not os.path.exists(nampaNormdir):
         os.mkdir(nampaNormdir)           
         os.mkdir(nampadirNorml) 
                    
    scanauto=[name for name in range (0,dy) if name in listsliceok]     
    for scannumb in scanauto:
            
           tabp = np.zeros((dx, dy), dtype=np.uint8)    
           tabf=np.copy(tabsbg3d[scannumb])
           tabfc=np.copy(tabf)
           nbp=0
           if tabf.max()>0:    
                    
                    atabf = np.nonzero(tabf)
    
                    xmin=atabf[1].min()
                    xmax=atabf[1].max()
                    ymin=atabf[0].min()
                    ymax=atabf[0].max()
    
                    np.putmask(tabf,tabf>0,1)
                    _tabscan=tabscan3d[scannumb]                 
                     
                    i=xmin
                    while i <= xmax:
                        j=ymin
                        while j<=ymax:
                            tabpatch=tabf[j:j+dimpavy,i:i+dimpavx]
                            area= tabpatch.sum()  
                            targ=float(area)/pxy
                            
                            if targ >thrpatch:                               
                                imgray = _tabscan[j:j+dimpavy,i:i+dimpavx]   
                                imagemax= cv2.countNonZero(imgray)
                                min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(imgray)
    
                                if imagemax > 0 and max_val - min_val>20:                               
                                    nbp+=1                                            
                                    nampa=os.path.join(nampadirl,tail+'_'+str(i)+'_'+str(j)+'_'+str(scannumb)+'.'+typeid )
                                    cv2.imwrite (nampa, imgray,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
                                    if normiInternal:
                                        tabi2 = normi(imgray) 
                                    else:
                                        tabi2= cv2.normalize(imgray, 0, imageDepth,cv2.NORM_MINMAX);                             
                                    nampa=os.path.join(nampadirNorml,tail+'_'+str(i)+'_'+str(j)+'_'+str(scannumb)+'.'+typeid )                                    
                                    cv2.imwrite (nampa, tabi2,[int(cv2.IMWRITE_PNG_COMPRESSION),0])

                                    x=0
                                    #we draw the rectange
                                    while x < dimpavx:
                                        y=0
                                        while y < dimpavy:
                                            tabp[y+j][x+i]=150
                                            if x == 0 or x == dimpavx-1 :
                                                y+=1
                                            else:
                                                y+=dimpavy-1
                                        x+=1
                                    #we cancel the source
                                    tabf[j:j+dimpavy,i:i+dimpavx]=0
                                    j+=dimpavy-1                    
                            j+=1
                        i+=1                                                
                        
           if nbp>0:
                 tabfc =tabfc+tabp
                 ntotpat=ntotpat+nbp
                 stw=tail+'_slice_'+str(scannumb)+'_'+labelbg+'_'+locabg+'_'+str(nbp)
                 stww=stw+'.txt'
                 flw=os.path.join(jpegpath,stww)
                 mfl=open(flw,"w")
                 mfl.write('#number of patches: '+str(nbp)+'\n')
                 mfl.close()
                 stww=stw+'.'+typej
                 flw=os.path.join(jpegpath,stww)
                 scipy.misc.imsave(flw, tabfc)

    return ntotpat
    

def wtebres(dirf,tab,dx,slicepitch,fxs,dxd):
    
    (top,tail1)=os.path.split(dirf)
    print 'wtebres' ,tail1
    (top1,tail)=os.path.split(top)
    if tail1=='source':
        tabres=np.zeros((dx,dxd,dx),np.uint16)
    else:
        tabres=np.zeros((dx,dxd,dx),np.uint8)
    tabsroi=np.zeros((dx,dxd,dx,3),np.uint8)
    tabsbg=np.zeros((dx,dxd,dx),np.uint8)
    
    imgresize=np.zeros((dx,dxd,dx),np.uint8)
    
    wridir=os.path.join(dirf,transbmp)
    remove_folder(wridir)
    os.mkdir(wridir)
    
    bgdirf=os.path.join(top,bgdir)
    sroidir=os.path.join(top,sroid)

    for i in range (0,dx):
        if tab[i].max()>1:

            imgresize=cv2.resize(tab[i],None,fx=1,fy=fxs,interpolation=cv2.INTER_LINEAR)
#            print imgresize.shape

            trcore=tail1+'_'+str(i)+'.'
            trscan=trcore+typeid
            trscanbmp=trcore+typei
            trscanroi= 'tr_'+str(i)+'.'+typeroi
            
           
            if tail1=='lung':
                namescan=os.path.join(wridir,trscanbmp)
                cv2.imwrite (namescan, imgresize) 
                
                trbg=os.path.join(bgdirf,trscanroi)
                tabres[i]=imgresize
                tabsbg[i]=imgresize
                cv2.imwrite (trbg, imgresize)   

               
            if tail1=='source':                
                trscan=os.path.join(wridir,trscan)
                tabres[i]=imgresize
                cv2.imwrite (trscan, imgresize,[int(cv2.IMWRITE_PNG_COMPRESSION),0])  
                
                namescan=os.path.join(sroidir,trscanroi)                        
                textw='n: '+tail+' scan: '+str(i)
#                print imgresize.min(),imgresize.max()
                c=255.0/imgresize.max()
                imgresize=imgresize*c
                imgresize=imgresize.astype('uint8')
#                print imgresize.min(),imgresize.max()
                imgresize=cv2.cvtColor(imgresize,cv2.COLOR_GRAY2BGR)
                tagviews(imgresize,textw,0,20)
                tabsroi[i]=imgresize     
                cv2.imwrite (namescan, imgresize) 
            else:
                trscan=os.path.join(wridir,trscanbmp)
                tabres[i]=  imgresize
                cv2.imwrite (trscan, imgresize)
                 
    return tabres,tabsroi,tabsbg

def calnewpat(dirName,pat,slnt,dimtabx,dimtaby):
    print 'new pattern : ',pat

    tab=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    if pat=='HCpret':
        pat1='HC'
        pat2='reticulation'

    elif pat=='HCpbro':
        pat1='HC'
        pat2='bronchiectasis'

    elif pat=='GGpbro':
        pat1='ground_glass'
        pat2='bronchiectasis'

    elif pat == 'GGpret':
        pat1='ground_glass'
        pat2='reticulation'

    elif pat=='bropret':
        pat1='bronchiectasis'
        pat2='reticulation'

    tab1=np.copy(tabroipat[pat1])
    tab2=np.copy(tabroipat[pat2])
    tab3=np.copy(tabroipat[pat])   
    
    nm=False
    
    for i in range (0,slnt):
        tab3[i]=np.bitwise_and(tab1[i],tab2[i])
        if tab3[i].max()>0:
          
            tab[i]=np.bitwise_not(tab3[i])
            tab1[i]=np.bitwise_and(tab1[i],tab[i])
            tabroipat[pat1][i]= tab1[i]
            tab2[i]=np.bitwise_and(tab2[i],tab[i])
            tabroipat[pat2][i]= tab2[i]
            nm=True

    if nm:
            npd=os.path.join(dirName,pat)    
            remove_folder(npd)
            os.mkdir(npd)  
            npd=os.path.join(npd,transbmp)    
            remove_folder(npd)
            os.mkdir(npd) 
 
            for i in range (0,slnt):
                 if tab3[i].max()>0:
                    naf3=pat+'_'+str(i)+'.'+typei
                    npdn3=os.path.join(npd,naf3)
                    
                    cv2.imwrite(npdn3,tab3[i])
                    
                    naf2=pat2+'_'+str(i)+'.'+typei
                    npd2=os.path.join(dirName,pat2)
                    npd2=os.path.join(npd2,transbmp)
                    npdn2=os.path.join(npd2,naf2)

                    cv2.imwrite(npdn2,tab2[i])
                    
                    naf1=pat1+'_'+str(i)+'.'+typei  
                    npd1=os.path.join(dirName,pat1)
                    npd1=os.path.join(npd1,transbmp)
                    npdn1=os.path.join(npd1,naf1)                   
                    cv2.imwrite(npdn1,tab1[i])
    return tab3      
        

            
for f in listHug:
    print f
    errorfile = open(eferror, 'a')
    dirf=os.path.join(dirHUG,f)

    sroidir=os.path.join(dirf,sroid)
    remove_folder(sroidir)
    os.mkdir(sroidir)
    
    bgdirf=os.path.join(dirf,bgdir)
    remove_folder(bgdirf)
    os.mkdir(bgdirf)   
    
    listsliceok=[]
    tabroipat={} 
    tabroipat3d={}    

    if isGre:
        
        dimtabx,dimtaby,slnt,slicepitch = genepara(dirf)
        tabscan =np.zeros((slnt,dimtabx,dimtaby),np.uint16)
        tabslung =np.zeros((slnt,dimtabx,dimtaby),np.uint8)
            
        fxs=float(slicepitch/avgPixelSpacing )
        dimtabxd=int(round(fxs*slnt,0))
#        print 'dimtabxd', dimtabxd
        dimtabyd=dimtaby
        
        for i in usedclassif:
            tabroipat[i]=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
            tabroipat3d[i]=np.zeros((dimtabx,dimtabxd,dimtabyd),np.uint8)
       
        tabscan3d =np.zeros((dimtabx,dimtabxd,dimtabyd),np.uint16)
        tabslung3d =np.zeros((dimtabx,dimtabxd,dimtabyd),np.uint8)
        tabsbg3d =np.zeros((dimtabx,dimtabxd,dimtabyd),np.uint8)
        tabsroi3d =np.zeros((dimtabx,dimtabxd,dimtabyd,3),np.uint8)
        tabres=np.zeros((dimtabx,slnt,dimtabyd),np.uint16)
        
        dirg=os.path.join(dirf,source_name)
        tabscan=genebmp(dirf, source_name,slnt,dimtabx,dimtaby)       
        tabres=reshapeScan(tabscan,slnt,dimtabx) 
        
        tabscan3d,tabsroi3d,a=wtebres(dirg,tabres,dimtabx,slicepitch,fxs,dimtabxd)

        dirg=os.path.join(dirf,lung_name)
        tabslung=genebmp(dirf, lung_name,slnt,dimtabx,dimtaby)   
        tabres=reshapeScan(tabslung,slnt,dimtabx)
        tablung3,a,tabsbg3d=wtebres(dirg,tabres,dimtabx,slicepitch,fxs,dimtabxd)
        
        contenudir = [name for name in os.listdir(dirf) if name in usedclassif and name not in derivedpat]

        for g in contenudir:           
            tabroipat[g]=genebmp(dirf, g,slnt,dimtabx,dimtaby)

        for i in derivedpat:
           tabroipat[i]=calnewpat(dirf,i,slnt,dimtabx,dimtaby) 

        contenudir = [name for name in os.listdir(dirf) if name in usedclassif]
        for g in contenudir:
            print g
            if  tabroipat[g].max()>0:
                dirg=os.path.join(dirf,g)
                tabres=reshapeScan(tabroipat[g],slnt,dimtabx)
                tabroipat3d[g],a,a=wtebres(dirg,tabres,dimtabx,slicepitch,fxs,dimtabxd)
                nbp=pavs(dirf,g,dimtabxd,dimtabyd)
 
        pavbg(dirf,dimtabxd,dimtabyd)
    else:
       print 'is not gre'
    errorfile.write('completed :'+f+'\n')
    errorfile.close()  
errorfile.close()        