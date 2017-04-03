# coding: utf-8
#Sylvain Kritter 12 decembre 2016
"""generate roi for segmentation"""
import os
import numpy as np
import shutil
import scipy.misc
import dicom
import PIL
from PIL import Image, ImageFont, ImageDraw
import cv2
#import matplotlib.pyplot as plt    
#general parameters and file, directory names
#######################################################
#customisation part for datataprep
#global directory for scan file
#namedirHUG = 'HUG'
namedirHUG = 'CHU'
#subdir for roi in text
#subHUG='ILD_TXT' #HUG
subHUG='UIPt' #CHU

########################################################################
######################  end ############################################
########################################################################
#define the name of directory with patchfile txt
patchfile = 'patchfile'

bmpname='scan_bmp'
#directory with lung mask dicom
lungmask='lung_mask'
#directory to put  lung mask bmp
lungmaskbmp='bmp'
#directory name with scan with roi for segmentation
sroiseg='sroiseg'

#full path names
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
#path for HUG dicom patient parent directory
path_HUG=os.path.join(cwdtop,namedirHUG)
##directory name with patient databases
namedirtopc =os.path.join(path_HUG,subHUG)
if not os.path.exists(namedirtopc):
    print('directory ',namedirtopc, ' does not exist!') 

#end dataprep part
#########################################################
# general
#image  patch format
typei='bmp' #can be jpg
#patch size

font5 = ImageFont.truetype( 'arial.ttf', 5)
font10 = ImageFont.truetype( 'arial.ttf', 10)
font20 = ImageFont.truetype( 'arial.ttf', 20)

#end general part
#########################################################
#log files
##error file

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

usedclassif = [
        'consolidation',
        'fibrosis',
        'ground_glass',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis'
        ]
classif ={
        'consolidation':0,
        'fibrosis':1,
        'ground_glass':2,
        'healthy':3,
        'micronodules':4,
        'reticulation':5,
        'air_trapping':6,
        'cysts':7,
        'bronchiectasis':8,
        
         'bronchial_wall_thickening':9,
         'early_fibrosis':10,
         'emphysema':11,
         'increased_attenuation':12,
         'macronodules':13,
         'pcp':14,
         'peripheral_micronodules':15,
         'tuberculosis':16
        }
 
classifcseg ={
'consolidation':20,
'fibrosis':30,
'ground_glass':40,
'healthy':50,
'micronodules':60,
'reticulation':70,
'air_trapping':80,
'cysts':90,
 'bronchiectasis':100,
 
 'bronchial_wall_thickening':110,
 'early_fibrosis':120,
 'emphysema':130,
 'increased_attenuation':140,
 'macronodules':150,
 'pcp':160,
 'peripheral_micronodules':170,
 'tuberculosis':180
 } 
#print namedirtopc
#print jpegpath

def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)

def rsliceNum(s,c,e):
    ''' look for  afile according to slice number'''
    #s: file name, c: delimiter for snumber, e: end of file extension
    endnumslice=s.find(e)
    posend=endnumslice
    while s.find(c,posend)==-1:
        posend-=1
    debnumslice=posend+1
    return int((s[debnumslice:endnumslice])) 
#


def genebmp(dirName):
    """generate patches from dicom files and sroi"""
#    print ('generate  bmp files from dicom files in :',f)
    global  dimtabx,dimtaby
    #directory for patches
    (top,tail)=os.path.split(dirName)
    if tail=='source' or tail =='B70f':
        (top,tail)=os.path.split(top)
        
    fileList = [name for name in os.listdir(dirName) if ".dcm" in name.lower()]
    mins=1000
    maxs=0
    for filename in fileList:
#        print(filename)
            FilesDCM =(os.path.join(dirName,filename))            
            ds = dicom.read_file(FilesDCM)
            print >> volumefile,ds
            ooo
                
             
            dsh=int(ds.HighBit)
            dshs=int(ds.BitsStored)
#            print dsh,dshs
          
            dsr= ds.pixel_array   
            if dsr.min()< mins:
                mins=dsr.min()
            if dsr.max()> maxs:
                maxs=dsr.max()
                
#            print 'scan', dsr.min(), dsr.max()
            dsr= dsr-dsr.min()
            c=255.0/dsr.max()
            dsr=dsr*c
            dsr=dsr.astype('uint8')


    print >> volumefile,'scanmax',tail, mins, maxs,dsh,dshs

            #    print   ,dictSurf
            #special sroiseg
                 
        



def reptfulle(tabc,dx,dy):
    imgi = np.zeros((dx,dy,3), np.uint8)
    cv2.polylines(imgi,[tabc],True,(1,1,1)) 
    cv2.fillPoly(imgi,[tabc],(1,1,1))
    tabzi = np.array(imgi)
    tabz = tabzi[:, :,1]   
    return tabz, imgi
    

def tagviewb(fig,label,x,y):
    """write text in image according to label and color"""
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    col=white
    extseg=str(classifcseg[label])
    labnow=classif[label]
#    print (labnow, text)
    if label == 'back_ground':
        x=0
        y=0        
        deltay=60
    else:        
        deltay=25*((labnow-1)%5)
    draw.text((x, y+deltay),label+' '+extseg,col,font=font10)
    imgn.save(fig)

def tagviews(fig,text,x,y):
    """write simple text in image """
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    draw.text((x, y),text,white,font=font10)
    imgn.save(fig)


def contour2(im,l):  

    viseg=np.zeros((dimtabx,dimtaby,3), np.uint8)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,0,255,0)
    im2,contours0, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
    colseg=classifcseg[l]

    cv2.drawContours(viseg,contours,-1,(colseg,colseg,colseg),-1,cv2.LINE_AA)
#    cv2.fillPoly(viseg,pts=[contours],col)
    return viseg
   
###
    
def pavs (imgi,tab,dx,dy,namedirtopcf, iln,f,label,loca,typei):
    """ generate patches from ROI"""
#    print 'pavement'
    
    viseg=contour2(imgi,label)
   
    patchpathc=os.path.join(namedirtopcf,bmpname)
    patchpathc=os.path.join(namedirtopcf,sroiseg)
    contenujpg = os.listdir(patchpathc)
    print patchpathc

    debnumslice=iln.find('_')+1
    endnumslice=iln.find('_',debnumslice)
    slicenumber=int(iln[debnumslice:endnumslice])

    for  n in contenujpg:           
#        print n, typei
        sliceNumberN=rsliceNum(n,'_','.'+typei)
        if sliceNumberN==slicenumber:
            namescanseg=os.path.join(sroidirseg,n)
            orignseg = Image.open(namescanseg)
            tablscanseg = np.array(orignseg)
            imnseg=cv2.add(viseg,tablscanseg)
            cv2.imwrite(namescanseg,imnseg)
            tagviewb (namescanseg,label,0,100)
            break
            

    return 


def fileext(namefile,curdir):

    ofi = open(namefile, 'r')
    t = ofi.read()
    #print( t)
    ofi.close()
#
#    print('number of slice:',nslice)
    nset=0
#    print('number of countour:',numbercon)
    spapos=t.find('SpacingX')
    coefposend=t.find('\n',spapos)
    coefposdeb = t.find(' ',spapos)
    coef=t[coefposdeb:coefposend]
    coefi=float(coef)
#    print('coef',coefi)
#    
    labpos=t.find('label')
    while (labpos !=-1):
#        print('boucle label')
        labposend=t.find('\n',labpos)
        labposdeb = t.find(' ',labpos)
        
        label=t[labposdeb:labposend].strip()
        if label.find('/')>0:
            label=label.replace('/','_')
    
#        print('label',label)
        locapos=t.find('loca',labpos)
        locaposend=t.find('\n',locapos)
        locaposdeb = t.find(' ',locapos)
        loca=t[locaposdeb:locaposend].strip()
#        print(loca)
        if loca.find('/')>0:
            loca=loca.replace('/','_')

        condslap=True
        slapos=t.find('slice',labpos)
        while (condslap==True):
#            print('boucle slice')

            slaposend=t.find('\n',slapos)
            slaposdeb=t.find(' ',slapos)
            slice=t[slaposdeb:slaposend].strip()
#            print('slice:',slice)

            nbpoint=0
            nbppos=t.find('nb_point',slapos)     
            conend=True
            while (conend):
                nset=nset+1
                nbpoint=nbpoint+1
                nbposend=t.find('\n',nbppos)
                tabposdeb=nbposend+1
                
                slaposnext=t.find('slice',slapos+1)
                nbpposnext=t.find('nb_point',nbppos+1)
                labposnext=t.find('label',labpos+1)
                #last contour in file
                if nbpposnext==-1:
                    tabposend=len(t)-1
                else:
                    tabposend=nbpposnext-1
                #minimum between next contour and next slice
                if (slaposnext >0  and nbpposnext >0):
                     tabposend=min(nbpposnext,slaposnext)-1 
                #minimum between next contour and next label
                if (labposnext>0 and labposnext<nbpposnext):
                    tabposend=labposnext-1
#                    
                nametab=curdir+'/'+patchfile+'/slice_'+str(slice)+'_'+str(label)+\
                '_'+str(loca)+'_'+str(nbpoint)+'.txt'
    #            print(nametab)
#
                mf=open(nametab,"w")
                mf.write('#label: '+label+'\n')
                mf.write('#localisation: '+loca+'\n')
                mf.write(t[tabposdeb:tabposend])
                mf.close()
                nbppos=nbpposnext 
                #condition of loop contour
                if (slaposnext >1 and slaposnext <nbpposnext) or\
                   (labposnext >1 and labposnext <nbpposnext) or\
                   nbpposnext ==-1:
                    conend=False
            slapos=t.find('slice',slapos+1)
            labposnext=t.find('label',labpos+1)
            #condition of loop slice
            if slapos ==-1 or\
            (labposnext >1 and labposnext < slapos ):
                condslap = False
        labpos=t.find('label',labpos+1)
#    print('total number of contour',nset,'in:' , namefile)
    return(coefi)


listdirc= (os.listdir(namedirtopc))
volumefile = open('scanmaxl.txt', 'a')

for f in listdirc:
    #f = 35
    print('work on:',f)

    nbpf=0
    namedirtopcf=os.path.join(namedirtopc,f) 
    so=os.path.join(namedirtopcf,'source')
    re=os.path.join(namedirtopcf,'reticulation')
    B7=os.path.join(namedirtopcf,'B70f')
    if os.path.exists(so):
        print so
        genebmp(re)
    elif os.path.exists(B7):
        print B7
        genebmp(B7)
    elif os.path.exists(re):
        print re
        genebmp(re)
    else:
         genebmp(namedirtopcf)
  

   

volumefile.close()    

print('completed')