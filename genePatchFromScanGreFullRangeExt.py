# coding: utf-8
#Sylvain Kritter 6 february 2017 
"""Top file to generate patches from DICOM database internal equalization from CHU Grenoble
include new patterns when patterns are super imposed, cross view"""
import os
import sys
#import png
import numpy as np
import shutil
import scipy as sp
import scipy.misc
import dicom
import PIL
from PIL import Image, ImageFont, ImageDraw
import cv2
import matplotlib.pyplot as plt    
#general parameters and file, directory names
#######################################################
#customisation part for datataprep
#global directory for scan file
namedirHUG = 'CHU'
#subdir for roi in text
subHUG='UIP'
#subHUG='UIP_106530'
#subHUG='UIPt'
#subHUG='UIP_S14740'

alreadyDone =[]

toppatch= 'TOPPATCH'  
#extension for output dir
extendir='16_set1_13b'
#extendir='1'

#image  patch format
typei='jpg' #can be jpg
typeid='png' #can be png for 16 bits
#typeid='bmp' #can be jpg
typej='jpg' #can be jpg
typeroi='jpg' #can be jpg

imageDepth=8191 #number of bits used on dicom images (2 **n) 13 bits
#imageDepth=255 #number of bits used on dicom images (2 **n) 13 bits

#normalization internal procedure or openCV
normiInternal=True
globalHist=True #use histogram equalization on full image
globalHistInternal=True #use internal for global histogram when True otherwise opencv
#patch overlapp tolerance
thrpatch = 0.8
#labelEnh=('consolidation','reticulation,air_trapping','bronchiectasis','cysts')
labelEnh=()
# average pxixel spacing
avgPixelSpacing=0.734

dimpavx =16
dimpavy = 16
    
########################################################################
######################  end ############################################
########################################################################
patchesdirnametop = toppatch+'_'+extendir
#define the name of directory for patches
patchesdirname = 'patches'
#define the name of directory for normalised patches
patchesNormdirname = 'patches_norm'
#define the name for jpeg files
imagedirname='patches_jpeg'
#define name for image directory in patient directory 
bmpname='scan_bmp'
#directory with lung mask dicom
lungmask='lung'
#directory to put  lung mask bmp
lungmaskbmp='scan_bmp'
#directory name with scan with roi
sroi='sroi'
#directory name with scan with roi
bgdir='bgdir'
bgdirw='bgdirw'
reserved=['bgdir','sroi','sroi1','sroi3d','bgdir3d']
notclas=['lung','source','B70f']
#full path names
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
#path for HUG dicom patient parent directory
path_HUG=os.path.join(cwdtop,namedirHUG)
##directory name with patient databases
namedirtopc =os.path.join(path_HUG,subHUG)
if not os.path.exists(namedirtopc):
    print('directory ',namedirtopc, ' does not exist!') 

font5 = ImageFont.truetype( 'arial.ttf', 5)
font10 = ImageFont.truetype( 'arial.ttf', 10)
font20 = ImageFont.truetype( 'arial.ttf', 20)
labelbg='back_ground'
locabg='anywhere_CHUG'

#end general part
#########################################################
#log files
##error file
patchtoppath=os.path.join(path_HUG,patchesdirnametop)
#create patch and jpeg directory
patchpath=os.path.join(patchtoppath,patchesdirname)
#create patch and jpeg directory
patchNormpath=os.path.join(patchtoppath,patchesNormdirname)
#print patchpath
#define the name for jpeg files
jpegpath=os.path.join(patchtoppath,imagedirname)
#print jpegpath

#patchpath = final/patches
#remove_folder(patchtoppath)
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

eferror=os.path.join(patchtoppath,'genepatcherrortop.txt')
errorfile = open(eferror, 'w')
#filetowrite=os.path.join(namedirtopc,'lislabel.txt')
eflabel=os.path.join(patchtoppath,'lislabel.txt')
mflabel=open(eflabel,"w")
#end customisation part for datataprep
#######################################################
#color of labels
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

roitab={}

 

def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)
#end log files
def fidclass(numero):
    """return class from number"""
    found=False
    for cle, valeur in classif.items():
        
        if valeur == numero:
            found=True
            return cle
      
    if not found:
        return 'unknown'

def rotateImage(image, angle):
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def rsliceNum(s,c,e):
    endnumslice=s.find(e)
    posend=endnumslice
    while s.find(c,posend)==-1:
        posend-=1
    debnumslice=posend+1
    return int((s[debnumslice:endnumslice])) 

def tagviewct(tab,label,x,y):
    """write text in image according to label and color"""

    col=classifc[label]
    labnow=classif[label]
#    print (labnow, label)
    if label == 'back_ground':
        x=0
        y=0        
        deltay=60
    else:        
        deltay=10*(labnow-1)
        deltax=100*(labnow/5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,label,(x+deltax, y+deltay), font,0.3,col,1)
#    viseg = cv2.cvtColor(viseg,cv2.COLOR_RGB2BGR)
    return viseg
    
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
    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing 
    dsr= RefDs.pixel_array
    dsr= dsr-dsr.min() 
    dsr=dsr.astype('uint16')
    dsrresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
    dimtabx=int(dsrresize.shape[0])
    dimtaby=int(dsrresize.shape[1])  
    return dimtabx,dimtaby,slnt            
    
    
def genebmp(dirName, sou):
    """generate patches from dicom files and sroi"""
    print ('generate  bmp files from dicom files in :',dirName, 'directory :',sou)
    if sou =='source':
        tabscan=np.zeros((slnt,dimtabx,dimtaby),np.uint16)
    else:
        tabscan=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    dirFileP = os.path.join(dirName, sou)
    dirFilePbmp=os.path.join(dirFileP,bmpname)
    remove_folder(dirFilePbmp)
    os.mkdir(dirFilePbmp)
    (top,tail)=os.path.split(dirName)
        #list dcm files
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
                        dsr = dsr.astype('uint8')
                        np.putmask(dsr,dsr==1,0)
                    #resize the dicom to have always the same pixel/mm
                    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing   
                    dsrresize1=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)

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
                    endnumslice=filename.find('.dcm')
                    imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)+'.'
                    imgcore=imgcoredeb+typei
                    bmpfile=os.path.join(dirFilePbmp,imgcore)
                                          
                    if sou =='source':
                        imgcored=imgcoredeb+typeid                   
                        bmpfiled=os.path.join(dirFilePbmp,imgcored)                 
                        cv2.imwrite (bmpfiled, dsrresize,[int(cv2.IMWRITE_PNG_COMPRESSION),0])  
                        tabscan[scanNumber-1]=dsrresize
                        imgcoresroi='sroi_'+str(scanNumber)+'.'+typeroi
                        bmpfileroi=os.path.join(sroidir,imgcoresroi)
                        textw='n: '+tail+' scan: '+str(scanNumber)
                        c=float(255)/dsrresize.max()
                        dsrresizer=dsrresize*c     
#                        if imageDepth <256:
                        
                        dsrresizer=dsrresizer.astype(np.uint8)
                        dsrresizer=cv2.cvtColor(dsrresizer,cv2.COLOR_GRAY2BGR)
                        dsrresizer=tagviews(dsrresizer,textw,0,20) 
                        cv2.imwrite (bmpfileroi, dsrresizer)
                        tabsroi[scanNumber-1]=dsrresizer
                    elif sou == 'lung':
                        cv2.imwrite (bmpfile, dsrresize)
                        bgfile=os.path.join(bgdirf,imgcore)
                        dsrresizer=np.copy(dsrresize)
                        np.putmask(dsrresizer,dsrresizer>0,100)
                        tabscan[scanNumber-1]=dsrresizer
                        tabbg[scanNumber-1]=dsrresize
                        cv2.imwrite (bgfile, dsrresizer)
                    else:
                        dsrresizer=np.copy(dsrresize)
                        np.putmask(dsrresizer,dsrresizer>0,100)
                        cv2.imwrite (bmpfile, dsrresizer)
                        tabscan[scanNumber-1]=dsrresizer

    return tabscan,tabsroi,tabbg 

def reptfulle(tabc,dx,dy):
    imgi = np.zeros((dx,dy,3), np.uint8)
    cv2.polylines(imgi,[tabc],True,(1,1,1)) 
    cv2.fillPoly(imgi,[tabc],(1,1,1))
    tabzi = np.array(imgi)
    tabz = tabzi[:, :,1]   
    return tabz, imgi

def normi(img):
     tabi1=img-img.min()
     maxt=float(tabi1.max())
     if maxt==0:
         maxt=1
     tabi2=tabi1*(imageDepth/maxt)
     if imageDepth<256:
         tabi2=tabi2.astype('uint8')
     else:
         tabi2=tabi2.astype('uint16') 
     return tabi2


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

def tagviews(tab,text,x,y):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,text,(x, y), font,0.3,white,1)
    return viseg

def pavbg(dirName,slnt,dimtabx,dimtaby):
    print('generate back-ground for :',dirName)
    """ generate patches from ROI"""
    (top,tail)=os.path.split(dirName)
   
    nampadir=os.path.join(patchpath,labelbg)
    if not os.path.exists(nampadir):
                 os.mkdir(nampadir)
                 
    nampadirl=os.path.join(nampadir,locabg)
    if not os.path.exists(nampadirl):
                 os.mkdir(nampadirl)
                 
    nampaNormdir=os.path.join(patchNormpath,labelbg)
    if not os.path.exists(nampaNormdir):
                 os.mkdir(nampaNormdir)
                 
    nampadirNorml=os.path.join(nampaNormdir,locabg)
    if not os.path.exists(nampadirNorml):
                 os.mkdir(nampadirNorml)

    pxy=float(dimpavx*dimpavy) 
    slnl=[name for name in range (0,slnt) if name in listsliceok]
    for slicenumber in slnl:
        tabp = np.zeros((dimtabx,dimtaby),np.uint8)
        nbp=0
        _tabscan=np.copy(tabscan[slicenumber])
        _tabbg=np.copy(tabbg[slicenumber])
        tabbgc=np.copy(_tabbg)
#        if slicenumber==132:
#            cv2.imshow('tabbgc',tabbgc) 
#    ###
#            cv2.waitKey(0)    
#            cv2.destroyAllWindows()  
        nz= np.count_nonzero(_tabbg)
        if nz>0:
                atabf = np.nonzero(_tabbg)
                #tab[y][x]  convention
                xmin=atabf[1].min()
                xmax=atabf[1].max()
                ymin=atabf[0].min()
                ymax=atabf[0].max()
                np.putmask(_tabbg,_tabbg>0,1)
   
                i=xmin
                while i <= xmax:
                    j=ymin
                    while j<=ymax:
#                        if i%10==0 and j%10==0:
#                         print(i,j)                                
                        tabpatch=_tabbg[j:j+dimpavy,i:i+dimpavx]
                        area= tabpatch.sum()
#                            print area  
                        if float(area)/pxy >thrpatch:

                             imgray = _tabscan[j:j+dimpavy,i:i+dimpavx]                            
                             imagemax= cv2.countNonZero(imgray)
                             min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(imgray)

                             if imagemax > 0 and max_val - min_val>20:                                                             
                                
                                    nbp+=1
                                    nampa=os.path.join(nampadirl,tail+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typeid )   
                                    cv2.imwrite (nampa, imgray,[int(cv2.IMWRITE_PNG_COMPRESSION),0])

                                    if normiInternal:
                                        tabi2 = normi(imgray) 
                                    else:
                                        tabi2= cv2.normalize(imgray, 0, imageDepth,cv2.NORM_MINMAX);
                                    nampa=os.path.join(nampadirNorml,tail+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typeid )                                    

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
                                    _tabbg[j:j+dimpavy,i:i+dimpavx]=0  
                                    j+=j+dimpavy-1
                        j+=1
                    i+=1    
                

                tabpw =tabbgc+tabp
                stw=tail+'_slice_'+str(slicenumber)+'_'+labelbg+'_'+locabg+'_'+str(nbp)
                stww=stw+'.'+typej
                flw=os.path.join(jpegpath,stww)
                scipy.misc.imsave(flw, tabpw) 
                stww=stw+'.txt'
                flw=os.path.join(jpegpath,stww)
                mfl=open(flw,"w")
#        mfl=open(jpegpath+'/'+f+'_'+slicenumber+'.txt',"w")
                mfl.write('#number of patches: '+str(nbp)+'\n')
                mfl.close()
    #                  print 'end pav'                  
                

def contour2(im,l):  
    col=classifc[l]
    vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
#    im=im.astype('uint8')
    ret,thresh = cv2.threshold(im,0,255,0)
    im2,contours0, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
    cv2.drawContours(vis,contours,-1,col,1,cv2.LINE_AA)
    return vis
    
def pavs (dirName,pat,slnt,dimtabx,dimtaby):
    """ generate patches from ROI"""
    print 'pav :',dirName,'pattern :',pat
    ntotpat=0    
    tabf=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    _tabsroi=np.zeros((slnt,dimtabx,dimtaby,3),np.uint8)
    _tabbg = np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    _tabscan = np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    (top,tail)=os.path.split(dirName)
    
    pxy=float(dimpavx*dimpavy)
            
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

    for scannumb in range (0,slnt):
       tabp = np.zeros((dimtabx, dimtaby), dtype='i')
       tabf=np.copy(tabroipat[pat][scannumb])
#       print type(tabf[0][0])
#       ooo
#       tabf = tabf.astype(np.uint8)
       tabfc=np.copy(tabf)
       nbp=0
       if tabf.max()>0:    
           vis=contour2(tabf,pat)
           if vis.sum()>0:

                _tabsroi = np.copy(tabsroi[scannumb])
                imn=cv2.add(vis,_tabsroi)
                imn=tagview(imn,pat,0,100)
                tabsroi[scannumb]=imn
                imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)
                sroifile='sroi_'+str(scannumb)+'.'+typeroi
                filenamesroi=os.path.join(sroidir,sroifile)
                cv2.imwrite(filenamesroi,imn)
                 
                bgfile='bg_'+str(scannumb)+'.'+typeroi
                namebg=os.path.join(bgdirf,bgfile)

                _tabbg = np.copy(tabbg[scannumb]) 


                np.putmask(_tabbg,_tabbg>0,100)
                np.putmask(tabf,tabf>0,100)

                mask=cv2.bitwise_not(tabf)

                outy=cv2.bitwise_and(_tabbg,mask)
                tabbg[scannumb]=outy   
#                if scannumb==132:
#                    cv2.imshow('outy',outy)
#                    cv2.imshow('mask',mask)
#                    cv2.imshow('_tabbg',_tabbg)
##    ###
#                    cv2.waitKey(0)    
#                    cv2.destroyAllWindows()  
                cv2.imwrite(namebg,outy)
                atabf = np.nonzero(tabf)

                xmin=atabf[1].min()
                xmax=atabf[1].max()
                ymin=atabf[0].min()
                ymax=atabf[0].max()

                np.putmask(tabf,tabf>0,1)
                _tabscan=tabscan[scannumb]                 
                 
                i=xmin
                while i <= xmax:
                    j=ymin
                    while j<=ymax:
                        tabpatch=tabf[j:j+dimpavy,i:i+dimpavx]
                        area= tabpatch.sum()  
                        targ=float(area)/pxy
                        
                        if targ >thrpatch:                               
                            imgray = _tabscan[j:j+dimpavy,i:i+dimpavx]   
#                            print '_tabscan', _tabscan.min(),_tabscan.max(),scannumb
#                            imgray=imgray.astype('uint16')
                            imagemax= cv2.countNonZero(imgray)
                            min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(imgray)

                            if imagemax > 0 and max_val - min_val>20:                               
                                nbp+=1                                   
                                nampa=os.path.join(nampadirl,tail+'_'+str(scannumb)+'_'+str(nbp)+'.'+typeid )
                                cv2.imwrite (nampa, imgray,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
                                if normiInternal:
#                                    print 'norminternal',imgray.min(),imgray.max(),  i, j
                                    tabi2 = normi(imgray) 
#                                    print 'norminternal',tabi2.min(),tabi2.max(),  i, j
#                                    ooo
                                else:
                                    tabi2= cv2.normalize(imgray, 0, imageDepth,cv2.NORM_MINMAX); 
                        
                                nampa=os.path.join(nampadirNorml,tail+'_'+str(scannumb)+'_'+str(nbp)+'.'+typeid )
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
#       else:
#                 tw=tail+'_slice_'+str(scannumb)+'_'+pat+'_'+locabg
#                 print tw,' :no patch'
#                 errorfile.write(tw+' :no patch\n')
    return ntotpat
    
    
def calnewpat(dirName,pat,slnt,dimtabx,dimtaby):
    print 'new pattern : ',pat
#    (top,tail)=os.path.split(dirName)
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
            npd=os.path.join(namedirtopcf,pat)    
            remove_folder(npd)
            os.mkdir(npd)    
            for i in range (0,slnt):
                 if tab3[i].max()>0:
                    naf3=pat+'_'+str(i)+'.'+typei
                    npdn3=os.path.join(npd,naf3)
                    cv2.imwrite(npdn3,tab3[i])
                    
                    naf2=pat2+'_'+str(i)+'.'+typei
                    npd2=os.path.join(namedirtopcf,pat2)
                    npd2=os.path.join(npd2,bmpname)
                    npdn2=os.path.join(npd2,naf2)
                    cv2.imwrite(npdn2,tab2[i])
                    
                    naf1=pat1+'_'+str(i)+'.'+typei  
                    npd1=os.path.join(namedirtopcf,pat1)
                    npd1=os.path.join(npd1,bmpname)
                    npdn1=os.path.join(npd1,naf1)                   
                    cv2.imwrite(npdn1,tab1[i])
    return tab3      
        

listdirc= [ name for name in os.listdir(namedirtopc) if os.path.isdir(os.path.join(namedirtopc, name)) and \
            name not in alreadyDone]

npat=0
#listdirc=('S335940',)
for f in listdirc:
#    f = 'S335940'
    print('work on:',f)
    
    nbpf=0
    listsliceok=[]
    tabroipat={}    
#    namedirtopcf=namedirtopc+'/'+f
    namedirtopcf=os.path.join(namedirtopc,f) 
   
    if os.path.isdir(namedirtopcf):   
        sroidir=os.path.join(namedirtopcf,sroi)
        remove_folder(sroidir)
        os.mkdir(sroidir)
        bgdirf = os.path.join(namedirtopcf, bgdir)
        remove_folder(bgdirf)    
        os.mkdir(bgdirf)

    dimtabx,dimtaby,slnt = genepara(namedirtopcf)

    tabsroi=np.zeros((slnt,dimtabx,dimtaby,3),np.uint8)
    tabbg =np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    tabscan =np.zeros((slnt,dimtabx,dimtaby),np.uint16)
    tabslung =np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    
    tabscan,tabsroi,tabbg=genebmp(namedirtopcf, 'source')
    tabslung,a,a=genebmp(namedirtopcf, 'lung')
    
    for i in usedclassif:
        tabroipat[i]=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
        
    contenudir = [name for name in os.listdir(namedirtopcf) if name in usedclassif and name not in derivedpat]
    for i in contenudir:
        tabroipat[i],tabsroi,tabbg=genebmp(namedirtopcf, i)
  
    for i in derivedpat:
        tabroipat[i]=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
        tabroipat[i]=calnewpat(namedirtopcf,i,slnt,dimtabx,dimtaby)
    contenudir = [name for name in os.listdir(namedirtopcf) if name in usedclassif]   
    for i in contenudir:
        nbp=pavs(namedirtopcf,i,slnt,dimtabx,dimtaby)
    pavbg(namedirtopcf,slnt,dimtabx,dimtaby)

    nbpf=nbpf+nbp

    ofilepw = open(jpegpath+'/nbpat_'+f+'.txt', 'w')
    ofilepw.write('number of patches: '+str(nbpf))
    ofilepw.close()
    errorfile.write('completed :'+f+'\n')
#    
    
#################################################################    
#   calculate number of patches
contenupatcht = os.listdir(jpegpath) 
#print(contenupatcht)
npatcht=0
for npp in contenupatcht:
#    print('1',npp)
    if npp.find('.txt')>0 and npp.find('nbp')<0:
#        print('2',npp)
        ofilep = open(jpegpath+'/'+npp, 'r')
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
ofilepwt = open(jpegpath+'/totalnbpat.txt', 'w')
ofilepwt.write('number of patches: '+str(npatcht))
ofilepwt.close()
#mf.write('================================\n')
#mf.write('number of datasets:'+str(npat)+'\n')
#mf.close()
#################################################################
#data statistics on paches
#nametopc=os.path.join(cwd,namedirtop)
dirlabel=os.walk( patchpath).next()[1]
#file for data pn patches
filepwt = open(namedirtopc+'totalnbpat.txt', 'w')
ntot=0;

labellist=[]
localist=[]

for dirnam in dirlabel:
    dirloca=os.path.join(patchpath,dirnam)
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
#    print(subdir)
        n=0
        listcwd=os.listdir(subdir)
        for ff in listcwd:
            if ff.find(typei) >0 :
                n+=1
                ntot+=1
#        print(label,loca,n) 
        filepwt.write('label: '+label+' localisation: '+loca+\
        ' number of patches: '+str(n)+'\n')
filepwt.close() 
#listslice=[]
#write the log file with label list
mflabel.write('label  _  localisation\n')
mflabel.write('======================\n')
categ=os.listdir(jpegpath)
for f in categ:
#    print f
    if f.find('.txt')>0 and f.find('nb')==0:
        ends=f.find('.txt')
        debs=f.find('_')
        sln=f[debs+1:ends]
#        print sln
        listlabel={}
        
        for f1 in categ:
#                print f1
                if  f1.find(sln+'_')==0 and f1.find('.txt')>0:
#                    print f1
                    debl=f1.find('slice_')
                    debl1=f1.find('_',debl+1)
                    debl2=f1.find('_',debl1+1)
                    endl=f1.find('.txt')
                    posend=endl
                    while f1.find('_',posend)==-1:
                        posend-=1
                    debnumslice=posend+1
                    label=f1[debl2+1:debnumslice-1]
#                    print label
                    ffle1=os.path.join(jpegpath,f1)
                    fr1=open(ffle1,'r')
                    t1=fr1.read()
                    fr1.close()
                    debsp=t1.find(':')
                    endsp=  t1.find('\n')
                    npo=int(t1[debsp+1:endsp])
#                    print label,npo,listlabel
                    if label in listlabel:
                        
                        listlabel[label]=listlabel[label]+npo
                    else:
                        listlabel[label]=npo
#        listslice.append(sln)
        ffle=os.path.join(jpegpath,f)
        fr=open(ffle,'r')
        t=fr.read()
        fr.close()
        debs=t.find(':')
        ends=len(t)
        nump= t[debs+1:ends]
        mflabel.write(sln+' number of patches: '+nump+'\n')
#        print listlabel
        for l in listlabel:
#           print l,l.find(labelbg)
           if l.find(labelbg)<0:
#             print l
             mflabel.write(l+' '+str(listlabel[l])+'\n')
        mflabel.write('---------------------'+'\n')
mflabel.close()

##########################################################
errorfile.write('completed')
errorfile.close()
#print listslice
print('completed')