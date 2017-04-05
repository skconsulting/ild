# coding: utf-8
'''create patches from patient on front view, no overlapping ROI
    sylvain kritter 06 february 2017
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
#extendir='0'
extendir='0'
reservedDir=['bgdir','sroi','sroi1','bgdir3d','sroi3d']
#alreadyDone =['S1830','S14740','S15440','S28200','S106530','S107260','S139430','S145210']
alreadyDone =['S106530','S107260','S139430','S145210','S14740','S15440','S1830','S28200','S335940','S359750']
alreadyDone=[]

notclas=['lung','source','B70f']

isGre=True
HUG='CHU'   
#HUG='HUG'
#subDir='ILDt'
subDir='UIPt'
#subDir='UIP'
#subDir='UIP_S14740'

scan_bmp='scan_bmp'
transbmp='trans_bmp'
sroid='sroi3d'
bgdir='bgdir3d'

typei='bmp'
typeroi='jpg'
typeid='png' #can be png for 16b
typej='jpg'

source_name='source'
lung_name='lung'
labelbg='back_ground'
locabg='td_CHUG'

dimpavx=16
dimpavy=16
pxy=float(dimpavx*dimpavy)
#imageDepth=65535 #number of bits used on dicom images (2 **n) 13 bits
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

    
    
#picklein_file = '../pickle_ex/pickle_ex59'
#modelname='ILD_CNN_model.h5'
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
            'bronchial_wall_thickening':10,
             'early_fibrosis':11,
             'emphysema':12,
             'increased_attenuation':13,
             'macronodules':14,
             'pcp':15,
             'peripheral_micronodules':16,
             'tuberculosis':17      
            }

clssifcount={
            'back_ground':0,
            'consolidation':0,
            'HC':0,
            'ground_glass':0,
            'healthy':0,
            'micronodules':0,
            'reticulation':0,
            'air_trapping':0,
            'cysts':0,
            'bronchiectasis':0,
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
        'emphysema'
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
#pickle_dir=os.path.join(cwdtop,pickel_dirsource) 
#pickle_dirToMerge=os.path.join(cwdtop,pickel_dirsourceToMerge)  

#patch_dir=os.path.join(dirHUG,patch_dirsource)
#print patch_dir
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
#     print 'tabi1',min_val, max_val,imageDepth/float(max_val)
     tabi2=(tabi-min_val)*(imageDepth/mm)
     tabi2=tabi2.astype('uint16')

     return tabi2
     
def reshapeScan(tabscan,slnt,dimtabx,slicepitch):

    tabres = np.zeros((dimtabx,slnt,dimtabx), np.uint16)

    for i in range (0,dimtabx):
        for j in range (0,slnt):
            tabres[i][j]=tabscan[j][i]

    return tabres
        
 
def genebmp(fn):
    """generate patches from dicom files"""
    
    print ('load dicom files in :',fn)
    (top,tail) =os.path.split(fn)
#    sroidir=os.path.join(top,sroi)
    #directory for patches
    fmbmp=os.path.join(fn,scan_bmp)
    remove_folder(fmbmp)
    os.mkdir(fmbmp)
    listdcm=[name for name in  os.listdir(fn) if name.lower().find('.dcm')>0]
#    print listdcm
   
    FilesDCM =(os.path.join(fn,listdcm[0]))  
    #           
    RefDs = dicom.read_file(FilesDCM)
    #    print RefDs
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
    imgresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
    dimtabx=imgresize.shape[0]
    dimtaby=imgresize.shape[1]
    print dimtabx, dimtaby
    slnt=0
    for l in listdcm:
        
        FilesDCM =(os.path.join(fn,l))  
        RefDs = dicom.read_file(FilesDCM)
        slicenumber=int(RefDs.InstanceNumber)
        if slicenumber> slnt:
            slnt=slicenumber
    slnt=slnt+1
    print 'number of slices', slnt-1
    tabscan = np.zeros((slnt,dimtabx,dimtaby), np.uint16)
    for l in listdcm:
#        print l
        FilesDCM =(os.path.join(fn,l))  
    #           
        RefDs = dicom.read_file(FilesDCM)
  
        dsr= RefDs.pixel_array
        dsr= dsr-dsr.min()
        dsr=dsr.astype('uint16')
##        min_val=np.min(dsr)
        max_val=np.max(dsr)
        if max_val>10:
##        print 'dsr orig',min_val, max_val
           
    #        c=float(imageDepth)/dsr.max()
    #        dsr=dsr*c
            if tail !='source':
                c=float(255)/dsr.max()
                dsr=dsr*c     
#                        if imageDepth <256:
                dsr=dsr.astype('uint8')
                np.putmask(dsr,dsr==1,0)
            
           
            slicenumber=int(RefDs.InstanceNumber)
            endnumslice=l.find('.dcm') 
            imgresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
#         
#            np.putmask(imgresize,imgresize==1,0)
#            np.putmask(imgresize,imgresize>1,255)
            
    #        cv2.imwrite(bmpfile, imgresize)
            imgcoreScan=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei
    #            print 'imgcore' , imgcore
            bmpfile=os.path.join(fmbmp,imgcoreScan)
            scipy.misc.imsave(bmpfile,imgresize)
            if tail =='source':
                imgresize = normi(imgresize) 
#            if tail =='lung':
#                imgresize = normi(imgresize) 
            tabscan[slicenumber]=imgresize
    return tabscan,slnt,dimtabx,slicepitch


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
    if label == 'back_ground':
        x=0
        y=0        
#        deltax=0
        deltay=60
    else:        
        deltay=25*((labnow-1)%5)
#        deltax=175*((labnow-1)//5)
#        deltax=80*((labnow-1)//5)

#    print (x+deltax,y+deltay)
    draw.text((x, y+deltay),label,col,font=font10)
    imgn.save(fig)

def tagviews(fig,text,x,y):
    """write simple text in image """
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    draw.text((x, y),text,white,font=font10)
    imgn.save(fig)


           
def pavs (dirName,dimtabx,dimtaby):
    """ generate patches from ROI"""
    print 'pav :',dirName
    ntotpat=0
    
#    bgdirf = os.path.join(dirName, bgdir)
    (top,tail)=os.path.split(dirName)
    (top1,tail1)=os.path.split(top)
    bgdirf = os.path.join(top, bgdir)   
    
    nampadir=os.path.join(patchpath,tail)
#    print nampadir
    nampadirl=os.path.join(nampadir,locabg)
    if not os.path.exists(nampadir):
         os.mkdir(nampadir)
         os.mkdir(nampadirl)
    nampaNormdir=os.path.join(patchNormpath,tail)
    nampadirNorml=os.path.join(nampaNormdir,locabg)
    if not os.path.exists(nampaNormdir):
         os.mkdir(nampaNormdir)
         
         os.mkdir(nampadirNorml)                     
   
    dirFilePs = os.path.join(dirName, transbmp)
    #dirFilePs: directory with 3d scan of pattern
    #list file in pattern directory
    listmroi = os.listdir(dirFilePs)
    imgsp = os.path.join(top, source_name)
    imgsp_dir = os.path.join(imgsp,transbmp)
    #imgsp_dir: source file direc
    sroidir = os.path.join(top, sroid)
#    listsroi =os.listdir(sroidir)
    for filename in listmroi:
   
       #files in pattern
         slicenumber= rsliceNum(filename,'_','.'+typei)
#             print 'filename de listmroi', filename,slicenumber
         tabp = np.zeros((dimtabx, dimtaby), dtype='i')
         filenamec = os.path.join(dirFilePs, filename)
         #filenamec: file name full for pattern         
         origscanroi = Image.open(filenamec,'r') 
         #mask image
         origscanroi = origscanroi.convert('L')
         tabf=np.array(origscanroi)   
         np.putmask(tabf,tabf==1,0)
         tabfcopy=np.copy(tabf)
         np.putmask(tabfcopy,tabfcopy>0,100)
#                         cv2.imshow('tabf',tabf) 
#                         cv2.waitKey(0)    
#                         cv2.destroyAllWindows()
         # create ROI as overlay
         vis=contour2(tabf,tail,dimtabx,dimtaby)
         
         if vis.sum()>0:
             endnumslice=filename.find(typei)
             imgcoredeb=filename[0:endnumslice]
             filenameroi=imgcoredeb+typeroi
#             print filenameroi

             filenamesroi = os.path.join(sroidir, filenameroi)
             oriroi = Image.open(filenamesroi)
             imroi= oriroi.convert('RGB')           
             tablroi = np.array(imroi)
#                             print vis.shape,tablroi.shape
             imn=cv2.add(vis,tablroi)
             imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)
             cv2.imwrite(filenamesroi,imn)
             tagview(filenamesroi,tail,0,100)

#
# 
             namebg=os.path.join(bgdirf,filename)            
             origscan = cv2.imread(namebg,0)             
             tabscanlung=np.array(origscan)
             
             np.putmask(tabscanlung,tabscanlung>0,255)
             
             np.putmask(tabf,tabf>0,255)             
             mask=cv2.bitwise_not(tabf)
             outy=cv2.bitwise_and(tabscanlung,mask)
             cv2.imwrite(namebg,outy)    
    #                     print 'start pav'
             atabf = np.nonzero(tabf)
             imagemax=tabf.sum()
                 
             if imagemax>0:
            #tab[y][x]  convention
                 xmin=atabf[1].min()
                 xmax=atabf[1].max()
                 ymin=atabf[0].min()
                 ymax=atabf[0].max()
             else:
                 xmin=0
                 xmax=2
                 ymin=0
                 ymax=2

             np.putmask(tabf,tabf>0,1)

             
             endnumslice=filename.find('.'+typei) 
             imgcoreScan=filename[0:endnumslice]+'.'+typeid
             imgscanp = os.path.join(imgsp_dir, imgcoreScan)

             origbmp = Image.open(imgscanp)
          
             i=xmin
             nbp=0
             while i <= xmax:
                j=ymin
                while j<=ymax:
                    tabpatch=tabf[j:j+dimpavy,i:i+dimpavx]
                    area= tabpatch.sum()  
                    targ=float(area)/pxy
                    
                    if targ >thrpatch:
                                         
                        crorig = origbmp.crop((i, j, i+dimpavx, j+dimpavy))
                        imagemax=crorig.getbbox()
                        nim= np.count_nonzero(crorig)
                        if nim>0:
                             min_val=np.min(crorig)
                             max_val=np.max(crorig)         
                        else:
                             min_val=0
                             max_val=0                              
                        if imagemax!=None and max_val-min_val>20:   
                            
                            imgray =np.array(crorig)
                            imgray=imgray.astype('uint16')
                            nbp+=1                                   
                            nampa=os.path.join(nampadirl,tail+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typeid )
    #                                    crorig.save(nampa) 
                            cv2.imwrite (nampa, imgray,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
    #                                    print 'imgray', imgray.min(), imgray.max()

                            tabi2 = normi(imgray) 

                            nampa=os.path.join(nampadirNorml,tail+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typeid )                                    
    #                                    scipy.misc.imsave(nampa, tabi2)
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
    #                                if dirFile not in labelEnh:
                            tabf[j:j+dimpavy,i:i+dimpavx]=0
    #                                else:
    #                                     tabf[j:j+dimpavy/2,i:i+dimpavx/2]=0                          
                    j+=1
                i+=1
                 
             
             tabfcopy =tabfcopy+tabp
             ntotpat=ntotpat+nbp
    
             if nbp>0:
                 if slicenumber not in listsliceok:
                            listsliceok.append(slicenumber) 
                 stw=tail1+'_slice_'+str(slicenumber)+'_'+tail+'_'+locabg+'_'+str(nbp)
                 stww=stw+'.txt'
                 flw=os.path.join(jpegpath,stww)
                 mfl=open(flw,"w")
                 mfl.write('#number of patches: '+str(nbp)+'\n')
                 mfl.close()
                 stww=stw+'.'+typej
                 flw=os.path.join(jpegpath,stww)
                 scipy.misc.imsave(flw, tabfcopy)
             else:
                 tw=tail1+'_slice_'+str(slicenumber)+'_'+tail+'_'+locabg
                 print tw,' :no patch'

    return ntotpat
    
def pavbg (dirName,dimtabx,dimtaby):
    """ generate patches from ROI"""
    print 'pav background:',dirName
    ntotpat=0
#    bgdirf = os.path.join(dirName, bgdir)
    (top,tail)=os.path.split(dirName)
#    (top1,tail1)=os.path.split(top)
    bgdirf = os.path.join(dirName, bgdir)   
    labelbg='back_ground'
    nampadir=os.path.join(patchpath,labelbg)
#    print nampadir
    nampadirl=os.path.join(nampadir,locabg)
    if not os.path.exists(nampadir):
         os.mkdir(nampadir)
         os.mkdir(nampadirl)
    nampaNormdir=os.path.join(patchNormpath,labelbg)
    nampadirNorml=os.path.join(nampaNormdir,locabg)
    if not os.path.exists(nampaNormdir):
         os.mkdir(nampaNormdir)
         
         os.mkdir(nampadirNorml)                     
   
    #list file in bgdr  directory
    listmroi = os.listdir(bgdirf)
    imgsp = os.path.join(dirName, source_name)
    imgsp_dir = os.path.join(imgsp,transbmp)
  
    for filename in listmroi:
       #files in pattern
         slicenumber= rsliceNum(filename,'_','.'+typei)
#         print 'filename de listmroi', filename,slicenumber
         if slicenumber in listsliceok:
#                 ooo
             tabp = np.zeros((dimtabx, dimtaby), dtype='i')
    
    
             namebg=os.path.join(bgdirf,filename)            
             imgsource = cv2.imread(namebg,0)             
             tabf=np.array(imgsource)
             np.putmask(tabf,tabf==1,0)
             tabc=np.copy(tabf)             
    #                     print 'start pav'
             atabf = np.nonzero(tabf)
             imagemax=tabf.sum()
                 
             if imagemax>0:
            #tab[y][x]  convention
                 xmin=atabf[1].min()
                 xmax=atabf[1].max()
                 ymin=atabf[0].min()
                 ymax=atabf[0].max()
            
                 np.putmask(tabf,tabf>0,1)
                 endnumslice=filename.find('.'+typei) 
                 imgcoreScan=filename[0:endnumslice]+'.'+typeid
                 imgscanp = os.path.join(imgsp_dir, imgcoreScan)
        #             print imgscanp
        #             ooo
                 origbmp = Image.open(imgscanp)     
                 
                 i=xmin
                 nbp=0
                 while i <= xmax:
                    j=ymin
                    while j<=ymax:
                        tabpatch=tabf[j:j+dimpavy,i:i+dimpavx]
                        area= tabpatch.sum()  
                        targ=float(area)/pxy
                        
                        if targ >thrpatch:
        #                                    print area,pxy, i,j                                          
                            crorig = origbmp.crop((i, j, i+dimpavx, j+dimpavy))
                            
                             #detect black pixels
                             #imagemax=(crorig.getextrema())
                            imagemax=crorig.getbbox()
                            nim= np.count_nonzero(crorig)
                            if nim>0:
                                 min_val=np.min(crorig)
                                 max_val=np.max(crorig)         
                            else:
                                 min_val=0
                                 max_val=0                              
                            if imagemax!=None and max_val-min_val>20:   
                                
                                imgray =np.array(crorig)
                                imgray=imgray.astype('uint16')
                                nbp+=1                                   
                                nampa=os.path.join(nampadirl,labelbg+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typeid )
        #                                    crorig.save(nampa) 
                                cv2.imwrite (nampa, imgray,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
        #                                    print 'imgray', imgray.min(), imgray.max()
                                tabi2 = normi(imgray) 
    
                                nampa=os.path.join(nampadirNorml,labelbg+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typeid )                                    
        #                                    scipy.misc.imsave(nampa, tabi2)
                                cv2.imwrite (nampa, tabi2,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
                                
                            #                print('pavage',i,j)  

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
        #                                                      
                        j+=1
                    i+=1
                                      
                 tabc =tabc+tabp
                 ntotpat=ntotpat+nbp
        
                 if nbp>0:
                  
                     stw=tail+'_slice_'+str(slicenumber)+'_'+labelbg+'_'+locabg+'_'+str(nbp)
                     stww=stw+'.txt'
                     flw=os.path.join(jpegpath,stww)
                     mfl=open(flw,"w")
                     mfl.write('#number of patches: '+str(nbp)+'\n')
                     mfl.close()
                     stww=stw+'.'+typej
                     flw=os.path.join(jpegpath,stww)
                     scipy.misc.imsave(flw, tabc)
                 else:
                     tw=tail+'_slice_'+str(slicenumber)+'_'+labelbg+'_'+locabg
                     print tw,' :no patch'
        #                     errorfile.write(tw+' :no patch\n')
        #                     print 'end pav'

    return ntotpat


def wtebres(dirf,tab,dimtabx,slicepitch):
    wridir=os.path.join(dirf,transbmp)
    remove_folder(wridir)
    os.mkdir(wridir)
    (top,tail1)=os.path.split(dirf)
    (top1,tail)=os.path.split(top)
    bgdirf=os.path.join(top,bgdir)
    for i in range (0,dimtabx):
        if tab[i].max()>10:
#            print tab[i].max()
#            print tab[i].shape
            fxs=float(slicepitch/avgPixelSpacing )
            imgresize=cv2.resize(tab[i],None,fx=1,fy=fxs,interpolation=cv2.INTER_LINEAR)
#            print imgresize.shape
            trcore='tr_'+str(i)+'.'
            trscan=trcore+typeid
            trscanbmp=trcore+typei
            trscanroi=trcore+typeroi
            if tail1=='lung':
                namescan=os.path.join(bgdirf,trscanbmp)
                scipy.misc.imsave(namescan,imgresize) 
               
            if tail1=='source':
#                print imgresize.min(),imgresize.max()
                
                trscan=os.path.join(wridir,trscan)
                cv2.imwrite (trscan, imgresize,[int(cv2.IMWRITE_PNG_COMPRESSION),0])           
                namescan=os.path.join(sroidir,trscanroi)                        
                textw='n: '+tail+' scan: '+str(i)
#                print imgresize.min(),imgresize.max()
                c=255.0/imgresize.max()
                imgresize=imgresize*c
                imgresize=imgresize.astype('uint8')
#                print imgresize.min(),imgresize.max()
                tablscan=cv2.cvtColor(imgresize,cv2.COLOR_GRAY2BGR)
                scipy.misc.imsave(namescan, tablscan)
                tagviews(namescan,textw,0,20)
            else:
                trscan=os.path.join(wridir,trscanbmp)

                scipy.misc.imsave(trscan,imgresize)                   
            dimtabxn=imgresize.shape[0]
            dimtabyn=imgresize.shape[1]
#        cv2.imwrite (trscan, tab[i],[int(cv2.IMWRITE_PNG_COMPRESSION),0])
    return dimtabxn,dimtabyn
            
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
    if isGre:
        dirg=os.path.join(dirf,source_name)
        tabscan,slnt,dimtabx,slicepitch=genebmp(dirg)
        print tabscan.shape,tabscan.max()
        tabres=reshapeScan(tabscan,slnt,dimtabx,slicepitch)
        print tabres.shape,tabres.max()
        ooo
        dimtabx,dimtaby=wtebres(dirg,tabres,dimtabx,slicepitch)

        dirg=os.path.join(dirf,lung_name)
        tabscan,slnt,dimtabx,slicepitch=genebmp(dirg)
        tabres=reshapeScan(tabscan,slnt,dimtabx,slicepitch)
        dimtabx,dimtaby=wtebres(dirg,tabres,dimtabx,slicepitch)
        listdirf=[name for name in os.listdir(dirf) if name in usedclassif]
        print listdirf
        for g in listdirf:           
            
            dirg=os.path.join(dirf,g)
            print g
            
            tabscan,slnt,dimtabx,slicepitch=genebmp(dirg)
            tabres=reshapeScan(tabscan,slnt,dimtabx,slicepitch)
            dimtabx,dimtaby=wtebres(dirg,tabres,dimtabx,slicepitch)
            pavs(dirg,dimtabx,dimtaby)
        pavbg(dirf,dimtabx,dimtaby)
    else:
        tabscan,slnt,dimtabx,slicepitch=genebmp(dirf)
        tabres=reshapeScan(tabscan,slnt,dimtabx,slicepitch)
        wtebres(dirf,tabres,dimtabx,slicepitch)
        pavs(dirg,dimtabx,dimtaby)
        pavbg(dirg,dimtabx,dimtaby)
    errorfile.write('completed :'+f+'\n')
    errorfile.close()  
errorfile.close()        
#bglist=listcl()
#ILDCNNpredict(bglist)