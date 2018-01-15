# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:03:33 2017

@author: sylvain
generate data from dicom images for segmentation roi , include normalization

-1st step

include generation per pattern
"""
#from __future__ import print_function

from param_pix import cwdtop,lungmask,lungmask1,lung_namebmp,image_rows,image_cols,typei1
from param_pix import sroi,source,scan_bmp ,derivedpat,layertokeep
from param_pix import limitHU,centerHU,yellow
from param_pix import remove_folder,normi,rsliceNum,norm,colorimage
from param_pix import classifc,classif,MAX_BOUND

import cPickle as pickle
import cv2
import dicom
import numpy as np
import os
import datetime
import sys
import time
from skimage import measure, morphology
from skimage.segmentation import clear_border
from skimage.morphology import  disk, binary_erosion, binary_closing
from skimage.filters import roberts
from skimage.measure import label,regionprops
from scipy import ndimage as ndi
from itertools import product

#path for images source
nametop='SOURCE_IMAGE'
#nameHug='HUG'
#nameHug='CHU2new'
#nameHug='CHU'
nameHug='CHU2'
#nameHug='REFVAL'
#subHUG='ILD6'
#subHUG='ILD_TXT'
#subHUG='ILD208'
#subHUG='UIP4'
#subHUG='UIP6'
subHUG='UIP'
#subHUG='TR18'


#path for image dir for CNN

imagedir='IMAGEDIR'
#toppatch= 'TOPVAL'
toppatch= 'TOPROI'

extendir='0'
#extendir='ILD6'

###############################################################

path_TOP=os.path.join(cwdtop,nametop)
path_HUG=os.path.join(path_TOP,nameHug)

path_IMAGE=os.path.join(cwdtop,imagedir)
if not os.path.exists(path_IMAGE):
    os.mkdir(path_IMAGE)
#path_HUG=os.path.join(nameHug,namsubHug)
namedirtopc =os.path.join(path_HUG,subHUG)


patchesdirnametop = toppatch+'_'+extendir
patchtoppath=os.path.join(path_IMAGE,patchesdirnametop) #path for data recording 1st step
patchpicklename='picklepatches.pkl'
picklepath = 'picklepatches'
roipicklepath = 'roipicklepatches'
picklepathdir =os.path.join(patchtoppath,picklepath)
roipicklepathdir =os.path.join(patchtoppath,roipicklepath)

if not os.path.isdir(patchtoppath):
    os.mkdir(patchtoppath)

if not os.path.isdir(picklepathdir):
    os.mkdir(picklepathdir)
if not os.path.isdir(roipicklepathdir):
    os.mkdir(roipicklepathdir)

#patchtoppath=os.path.join(path_HUG,patchesdirnametop)

def genepara(fileList,namedir):
    print 'gene parametres'
    listsln=[]
#    fileList =[name for name in  os.listdir(namedir) if ".dcm" in name.lower()]
    slnt=0
    for filename in fileList:
        FilesDCM =(os.path.join(namedir,filename))
        RefDs = dicom.read_file(FilesDCM,force=True)
        scanNumber=int(RefDs.InstanceNumber)
        listsln.append(scanNumber)
        if scanNumber>slnt:
            slnt=scanNumber
    print 'number of slices', slnt
    slnt=slnt+1
    return slnt,listsln

#def tagviews(tab,text,x,y):
#    """write simple text in image """
#    font = cv2.FONT_HERSHEY_SIMPLEX
#    viseg=cv2.putText(tab,text,(x, y), font,0.3,white,1)
#    return viseg
def tagviews (tab,t0,x0,y0,t1,x1,y1,t2,x2,y2,t3,x3,y3,t4,x4,y4,t5,x5,y5,t6,x6,y6):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_PLAIN
    col=yellow
    sizef=0.8
    sizefs=0.7
    viseg=cv2.putText(tab,t0,(x0, y0), font,sizef,col,1)
    viseg=cv2.putText(viseg,t1,(x1, y1), font,sizef,col,1)
    viseg=cv2.putText(viseg,t2,(x2, y2), font,sizefs,col,1)

    viseg=cv2.putText(viseg,t3,(x3, y3), font,sizef,col,1)
    viseg=cv2.putText(viseg,t4,(x4, y4), font,sizefs,col,1)
    viseg=cv2.putText(viseg,t5,(x5, y5), font,sizef,col,1)
    viseg=cv2.putText(viseg,t6,(x6, y6), font,sizef,col,1)
    return viseg
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, slnt ,srd,fill_lung_structures=True):
    print 'start generation'

    binary_image = np.array(image > -350, dtype=np.int8)+1 # initial 320 350 
    labels = measure.label(binary_image)

    ls0=labels.shape[0]-1
    ls1=labels.shape[1]-1
    ls2=labels.shape[2]-1

    for i,j,k in product(range (0,4), range (0,4),range(0,4)):

        im=int(i/3.*ls0)
        jm=int(j/3.*ls1)
        km=int(k/3.*ls2)
        if im*jm*km==0:
            background_label=labels[im,jm,km]
            binary_image[background_label == labels] = 2

    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    if srd: 
        ke=5
        kernele=np.ones((ke,ke),np.uint8)
        kerneld=np.ones((ke,ke),np.uint8)
    
        for i in range (image.shape[0]):
            binary_image[i]= cv2.dilate(binary_image[i].astype('uint8'),kerneld,iterations = 5)
            binary_image[i] = cv2.erode(binary_image[i].astype('uint8'),kernele,iterations = 5)

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    labels = measure.label(binary_image[slnt/2]) # Different labels are displayed in different colors
    regions = measure.regionprops(labels)
    numlung=0
    for prop in regions:

        if prop.area>5000:
            numlung+=1

    areas = [r.area for r in regionprops(labels)]
    areassorted=sorted(areas,reverse=True)
    if len(areassorted)>0:
        if numlung==2 or areassorted[0]>50000:
            ok=True
            print 'successful generation'
        else:
            ok=False
            print 'NOT successful generation'          
    else:
        ok=False
        print 'NOT successful generation'
 
    return binary_image,ok


def segment_lung_maskold(image, fill_lung_structures=True):

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -350, dtype=np.int8)+1#init 320
#    binary_image = clear_border(binary_image)
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
#    print labels.shape[0],labels.shape[1],labels.shape[2]
#    background_label = labels[0,0,0]
#    bg={} 
    ls0=labels.shape[0]-1
    ls1=labels.shape[1]-1
    ls2=labels.shape[2]-1
    for i in range (0,8):
#        print  'i:',i
#        print (i/4)%2, (i/2)%2, i%2
        for j in range (1,3):
#            print 'j:',j
#            print labels[(i/4)%2*ls0,(i/2)%2*ls1,i%2*ls2/j]
#            print (i/4)%2*ls0,(i/2)%2*ls1,i%2*ls2/j
#            print (i/4)%2*ls0,(i/2)%2*ls1/j,i%2*ls2
#            print (i/4)%2*ls0/j,(i/2)%2*ls1,i%2*ls2
            background_label=labels[(i/4)%2*ls0,(i/2)%2*ls1,i%2*ls2/j]
            binary_image[background_label == labels] = 2
            background_label=labels[(i/4)%2*ls0,(i/2)%2*ls1/j,i%2*ls2]
            binary_image[background_label == labels] = 2
            background_label=labels[(i/4)%2*ls0/j,(i/2)%2*ls1,i%2*ls2]
            binary_image[background_label == labels] = 2  
#    for i in range (0,8):
#        binary_image[background_label == labels] = 2
#        background_label = labels[labels.shape[0]-1,labels.shape[1]-1,labels.shape[2]-1]

    #Fill the air around the person
#    binary_image[background_label == labels] = 2


    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1


    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    return binary_image

def morph(imgt,k):

    img=imgt.astype('uint8')
    img[img>0]=100
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
#    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # <- to remove speckles...
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
#    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    return img


  
def genebmp(fn,sou,nosource,centerHU, limitHU, tabscanName,tabscanroi,dimtabx,dimtaby):
    """generate patches from dicom files"""
    global picklein_file
    (top,tail) =os.path.split(fn)
    print ('load scan dicom files in:' ,tail)
    lislnn=[]
   
    fmbmp=os.path.join(fn,sou)
    fmbmpbmp=os.path.join(fmbmp,scan_bmp)
    remove_folder(fmbmpbmp)
    os.mkdir(fmbmpbmp)
    
    if nosource:
        fmbmp=fn

    listdcm=[name for name in  os.listdir(fmbmp) if name.lower().find('.dcm')>0]

    FilesDCM =(os.path.join(fmbmp,listdcm[0]))
    FilesDCM1 =(os.path.join(fmbmp,listdcm[1]))
    RefDs = dicom.read_file(FilesDCM,force=True)
    RefDs1 = dicom.read_file(FilesDCM1,force=True)
    patientPosition=RefDs.PatientPosition
#    SliceThickness=RefDs.SliceThickness
    try:
            slicepitch = np.abs(RefDs.ImagePositionPatient[2] - RefDs1.ImagePositionPatient[2])
    except:
            slicepitch = np.abs(RefDs.SliceLocation - RefDs1.SliceLocation)

    print 'slice pitch in z :',slicepitch
#    ooo
    print 'patient position :',patientPosition
    lbHU=centerHU-limitHU/2
    lhHU=centerHU+limitHU/2
#    dsr= RefDs.pixel_array
    
    PixelSpacing=float(RefDs.PixelSpacing[0])
#    if dsr.shape[0]!= dimtabx:
#        dsr = dsr.astype('float32')
#        dsr=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)
#    print dimtabx, dimtaby
#    dsr = dsr.astype('int16')
    slnt=0
    for l in listdcm:

        FilesDCM =(os.path.join(fmbmp,l))
        RefDs = dicom.read_file(FilesDCM,force=True)
        slicenumber=int(RefDs.InstanceNumber)
        lislnn.append(slicenumber)
        if slicenumber> slnt:
            slnt=slicenumber

    print 'number of slices', slnt
    slnt=slnt+1
    tabscan = np.zeros((slnt,dimtaby,dimtabx),np.int16)
#    for i in range(slnt):
#        tabscan[i] = []
#    tabscan = np.zeros((slnt,dimtabx,dimtaby), np.int16)
    for l in listdcm:
#        print l
        FilesDCM =(os.path.join(fmbmp,l))
        RefDs = dicom.read_file(FilesDCM,force=True)
        slicenumber=int(RefDs.InstanceNumber)
#        print l, slicenumber

        dsr= RefDs.pixel_array
        dsr=dsr.astype('int16')
        dsr[dsr == -2000] = 0
        intercept = RefDs.RescaleIntercept
        slope = RefDs.RescaleSlope
        if slope != 1:
             dsr = slope * dsr.astype(np.float64)
             dsr = dsr.astype(np.int16)

        dsr += np.int16(intercept)
       
        if dsr.shape[0]!= dimtabx:
             dsr = dsr.astype('float32')
             dsr=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)
        dsr = dsr.astype('int16')
        
        endnumslice=l.find('.dcm')
        imgcoreScan=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei1  
                
        tabscan[slicenumber]=dsr.copy()
        np.putmask(dsr,dsr<lbHU,lbHU)
        np.putmask(dsr,dsr>lhHU,lhHU)
              
        imtowrite=normi(dsr)
        imtowrite = cv2.cvtColor(imtowrite, cv2.COLOR_GRAY2RGB)

#        bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
        tabscanName[slicenumber]=imgcoreScan
#        (topw,tailw)=os.path.split(picklein_file)
        t2='Prototype '
        t1=''
        t0='CONFIDENTIAL'
        t3='Scan: '+str(slicenumber)

        t4=time.asctime()
        t5='CenterHU: '+str(int(centerHU))
        t6='LimitHU: +/-' +str(int(limitHU/2))
                
        anoted_image=tagviews(imtowrite,t0,0,10,t1,0,dimtaby-20,t2,0,20,
                     t3,0,dimtaby-30,t4,0,dimtaby-10,t5,0,dimtaby-40,t6,0,dimtaby-50)        
        tabscanroi[slicenumber]=anoted_image

    return tabscan,slnt,slicepitch,lislnn,tabscanroi,tabscanName,PixelSpacing

def get_segmented_lungs(im):

    binary = im < -320
    cleared = clear_border(binary) 
    cleared=morph(cleared,5)
    label_image = label(cleared)
  
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0  
    selem = disk(2)
    binary = binary_erosion(binary, selem)
 
    selem = disk(10)
    binary = binary_closing(binary, selem)
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
 
    get_high_vals = binary == 0
    im[get_high_vals] = 0
  
    binary = morphology.dilation(binary,np.ones([5,5]))
    return binary


def genebmplung(fn,lungname,slnt,dimtabx,dimtaby,tabscanScan,listsln,tabscanName):
    """generate patches from dicom files"""

    tabrange={}
    tabrange['min']=100000000
    tabrange['max']=0
    (top,tail) =os.path.split(fn)
    print ('load lung segmented dicom files in :',tail)
    fmbmp=os.path.join(fn,lungname)
#    if not os.path.exists(fmbmp):
#        os.mkdir(fmbmp)       
    fmbmpbmp=os.path.join(fmbmp,lung_namebmp)
    if not os.path.exists(fmbmpbmp):
        os.mkdir(fmbmpbmp)

    listdcm=[name for name in  os.listdir(fmbmp) if name.lower().find('.dcm')>0]
#    listbmp= os.listdir(fmbmpbmp) 
    listbmp=[name for name in  os.listdir(fmbmpbmp) if name.lower().find(typei1)>0]

    tabscan = np.zeros((slnt,dimtaby,dimtabx), np.uint8)
    if len(listbmp)>0:
        print 'lung scan exists in bmp'
        for img in listbmp:
            slicenumber= rsliceNum(img,'_','.'+typei1)
            imr=cv2.imread(os.path.join(fmbmpbmp,img),0) 

            if imr.max()>0: 
                imr=cv2.resize(imr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)
                np.putmask(imr,imr>0,100)          
                tabscan[slicenumber]=imr
    else:
        print 'NO lung scan exists in bmp'

    if len(listbmp)<slnt-1: 
        print 'not all lung in bmp'
        if len(listdcm)>0  :
            print 'lung scan exists in dcm'           
            for l in listdcm:
                FilesDCM =(os.path.join(fmbmp,l))
                RefDs = dicom.read_file(FilesDCM,force=True)    
                dsr= RefDs.pixel_array
                dsr=normi(dsr)    
                imgresize=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)
                np.putmask(imgresize,imgresize>0,100)
        
                try:
                    slicenumber=int(RefDs.InstanceNumber)
    
                    imgcoreScan=tabscanName[slicenumber]
                    bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
                    if tabscan[slicenumber].max()==0:
                        tabscan[slicenumber]=imgresize
                        colorlung=colorimage(imgresize,(100,100,100))
                        cv2.imwrite(bmpfile,colorlung)  
                except:
                            print 'no scan name equiv to lung'
    
        else:
                print 'no lung scan in dcm'
                tabscan1 = np.zeros((slnt,dimtaby,dimtabx), np.int16)
                
                srd=False
                segmented_lungs_fill,ok = segment_lung_mask(tabscanScan,slnt, srd,True)
                if ok== False:
                    srd=True
                    print 'use modified algorihm'
                    segmented_lungs_fill,ok = segment_lung_mask(tabscanScan,slnt, srd,True)
#                    segmented_lungs_fill,ok = segment_lung_mask(tabscanScan,slnt, True)
                if ok== False:
                    print 'use 2nd algorihm'
                    segmented_lungs_fill=np.zeros((slnt,dimtabx,dimtabx), np.uint8)
                    for i in listsln:
                        segmented_lungs_fill[i]=get_segmented_lungs(tabscanScan[i])
                
                
                
    
#                segmented_lungs_fill = segment_lung_mask(tabscanScan, True)
    #            print segmented_lungs_fill.shape
                for i in listsln:
                    
                    tabscan1[i]=morph(segmented_lungs_fill[i],13)
    #                if i ==200:
    #                    cv2.imshow(str(i),normi(tabscan1[i]))
    
                    imgcoreScan=tabscanName[i]
                    bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
                    if tabscan[i].max()==0:
                        tabscan[i]=tabscan1[i]
                        colorlung=colorimage(tabscan[i],(100,100,100))
                        cv2.imwrite(bmpfile,colorlung)
    else:
        print 'all lung in bmp'
                        
    
    for sli in listsln:
        cpt=np.copy(tabscan[sli])
        np.putmask(cpt,cpt>0,1)
        area=cpt.sum()
        if area >0:
            if sli> tabrange['max']:
                tabrange['max']=sli
            if sli< tabrange['min']:
                tabrange['min']=sli
                
    return tabscan,tabrange
 
 

def contour2(im,l):
    col=classifc[l]
    vis = np.zeros((image_rows,image_cols,3), np.uint8)
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

 
    deltay=40+10*labnow

    viseg=cv2.putText(tab,label,(x, y+deltay), font,0.3,col,1)
    return viseg

def peparescan(numslice,tabs,tabl,datascan):
    
    tabslung=tabl.copy()
    scan=tabs.copy()
    if scan.max() ==0:
        print 'error',numslice
        sys.exit()
    np.putmask(tabslung,tabslung>0,255)
    taba=cv2.bitwise_and(scan,scan,mask=tabslung)
#    np.putmask(tabslung,tabslung>0,1)
    
    tablc=tabslung.astype(np.int16)
    np.putmask(tablc,tablc==0,MAX_BOUND)#MAX_BOUND is the label for outside lung area
    np.putmask(tablc,tablc==255,0)
        
#    tabab=cv2.bitwise_or(taba,tablc) 
    tabab=taba+tablc
#    print 'aa',np.average(tabab)
    tababn=norm(tabab)
#    print np.average(tababn)
    datascan[numslice]=tababn
    return datascan,np.average(tababn)
    

def preparroi(namedirtopcf,datascan,tabroi,tabroipat,slnroi):
    (top,tail)=os.path.split(namedirtopcf)

    pathpicklepat=os.path.join(picklepathdir,nameHug+' _'+tail)
    if not os.path.exists (pathpicklepat):
                os.mkdir(pathpicklepat)
    
    for num in slnroi:
        patchpicklenamepatient=str(num)+'_'+patchpicklename        

        pathpicklepatfile=os.path.join(pathpicklepat,patchpicklenamepatient)
            
        patpickle=(datascan[num],tabroi[num])

        pickle.dump(patpickle, open(pathpicklepatfile, "wb"),protocol=-1)
        nh=True
        for pat in tabroipat[num]:
            if pat !='healthy':
                nh=False
        for pat in tabroipat[num]:
             if pat !='healthy' or  (pat =='healthy' and nh):
                 listslicef[tail][num][pat]+=1
                 roipathpicklepat=os.path.join(roipicklepathdir,pat)   
                 if not os.path.exists (roipathpicklepat):
                    os.mkdir(roipathpicklepat)
    
                 roipathpicklepatfile=os.path.join(roipathpicklepat,nameHug+'_'+tail+'_'+patchpicklenamepatient)
                 pickle.dump(patpickle, open(roipathpicklepatfile, "wb"),protocol=-1)
 
                     

def drawcontours2(im,pat,dimtabx,dimtaby):
#    print 'contour',pat
    imgray = np.copy(im)
    ret,thresh = cv2.threshold(imgray,10,255,0)
    _,contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2 = np.zeros((dimtabx,dimtaby,3), np.uint8)
    cv2.drawContours(im2,contours,-1,classifc[pat],2)
#    im2=cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
    return im2

 

def calnewpat(pat,slnroi,tabroipat,tabroi):
    print 'new pattern : ',pat

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

    for i in slnroi:
        tab1=np.copy(tabroipat[pat1][i])
        np.putmask(tab1,tab1>0, 255)
        tab2=np.copy(tabroipat[pat2][i])
        np.putmask(tab2,tab2>0, 255)
        tab3=np.copy(tabroipat[pat][i])
        np.putmask(tab3,tab3>0, 255)
        taball=np.bitwise_or(tab2,tab1) 
        taball=np.bitwise_or(taball,tab3)
        np.putmask(taball, taball> 0, 255) 
        taballnot=np.bitwise_not(taball)


        tab=np.bitwise_and(tab1,tab2)        
        if tab.max()>0:     
            tab3=np.bitwise_or(tab3,tab)
            tabn=np.bitwise_not(tab3)      
            tab1=np.bitwise_and(tab1,tabn)
            np.putmask(tab1, tab1> 0, classif[pat1])
            
            tab2=np.bitwise_and(tab2,tabn)
            np.putmask(tab2, tab2> 0, classif[pat2])  
            
            np.putmask(tab, tab> 0, classif[pat])            

            tabroi[i]=np.bitwise_and(tabroi[i],taballnot)             
            tabroi[i]=np.bitwise_or(tabroi[i],tab1) 
            tabroi[i]=np.bitwise_or(tabroi[i],tab2) 
            tabroi[i]=np.bitwise_or(tabroi[i],tab) 

    return tabroi
def tagviewct(tab,label,x,y):
    """write text in image according to label and color"""

    col=classifc[label]

    labnow=classif[label]
#  
    deltay=10*(labnow%5)
    deltax=100*(labnow/5)
    font = cv2.FONT_HERSHEY_PLAIN
    viseg=cv2.putText(tab,label,(x+deltax, y+deltay), font,0.7,col,1)
#    viseg = cv2.cvtColor(viseg,cv2.COLOR_RGB2BGR)
    return viseg

def generoi(dirf,tabroi,dimtabx,dimtaby,slnroi,tabscanName,dirroit,tabscanroi,tabscanLung,PixelSpacing,slnt):
    (top,tail)=os.path.split(dirf)

    tabroipat={}    
    listroi={}
    for pat in classif:
#        print pat,classif[pat]
        if pat !='back_ground':
            tabroipat[pat]=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
            pathroi=os.path.join(dirf,pat)
            if os.path.exists(pathroi):
                lroi=[name for name in os.listdir(pathroi) if name.find('.'+typei1)>0  ]   
#                print pat,'lroi',lroi,pathroi
                for s in lroi:
                    numslice=rsliceNum(s,'_','.'+typei1)                    
                    img=cv2.imread(os.path.join(pathroi,s),0)
                    img=cv2.resize(img,(dimtabx,dimtabx),interpolation=cv2.INTER_LINEAR)                
                    np.putmask(img, img > 0, classif[pat])
                    tabroipat[pat][numslice]=img  
#                    if numslice==3:
#                        cv2.imshow(pat+str(numslice),normi(img))
                    if numslice not in slnroi:
                        slnroi.append(numslice)  
#    print slnroi
    for numslice in slnroi:
        for pat in classif:
            if pat !='back_ground':
                tab=np.copy(tabroipat[pat][numslice])
                np.putmask(tabroi[numslice], tab > 0, 0)           
                tabroi[numslice]+=tab   
    for pat in derivedpat:    
            tabroi=calnewpat(pat,slnroi,tabroipat,tabroi)

    for numslice in slnroi:
        for pat in layertokeep:
            tab=np.copy(tabroipat[pat][numslice])
            np.putmask(tabroi[numslice], tab > 0, 0)       
            tabroi[numslice]+=tab
    for numslice in slnroi:
        maskLung=tabscanLung[numslice].copy()
        np.putmask(maskLung,maskLung>0,255)

        
        maskRoi=tabroi[numslice].copy() 
        maskRoi1=tabroi[numslice].copy()
        np.putmask(maskRoi1,maskRoi1>0,255)
        
        if maskLung.max()==0 and maskRoi.max()!=0:
            print ('no lung for: '+str(numslice))
        tabroi[numslice]=np.bitwise_and(maskRoi,maskLung)   

        maskRoi1Not=np.bitwise_not(maskRoi1)       
        tablung=np.bitwise_and(maskLung, maskRoi1Not)
        np.putmask(tablung,tablung>0,classif['healthy']) 
        tabroi[numslice]=np.bitwise_or(tabroi[numslice],tablung)
#    
#    cv2.imshow('3',normi(tabroi[3]))
    slnroi.sort()
    volumeroi={}
#    print PixelSpacing
    surfelemp=PixelSpacing*PixelSpacing # for 1 pixel in mm2
    surfelem= surfelemp/100 #surface of 1 pixel in cm2
    for numslice in slnroi:
            listroi[numslice]=[]
            imgcoreScan=tabscanName[numslice]
            roibmpfile=os.path.join(dirroit,imgcoreScan)
            anoted_image=tabscanroi[numslice]
            
            volumeroi[numslice]={}
            for pat in classif:
                if pat !='back_ground':
#                if pat =='ground_glass':

                    img=np.copy(tabroi[numslice])
                    if img.max()>0:                    
#                        if classif[pat]>0:
                        np.putmask(img, img !=classif[pat], 0)
                        np.putmask(img, img ==classif[pat], 1)
#                        else:
#                            np.putmask(img, img !=0, 2)
#                            np.putmask(img, img ==0, 1)
#                            np.putmask(img, img ==2, 0)
                        
                        area= img.sum()* surfelem
                        volumeroi[numslice][pat]=area  
                        if area>0:
#                            print numslice, pat,area
                            if pat not in listroi[numslice] and area>0.1:
                                listroi[numslice].append(pat)
#                            img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

                            np.putmask(img, img >0, 100)
                            ctkey=drawcontours2(img,pat,dimtabx,dimtaby) 
                            ctkeym=ctkey.copy()
                            ctkeym=cv2.cvtColor(ctkeym,cv2.COLOR_RGB2GRAY)
                            ctkeym=cv2.cvtColor(ctkeym,cv2.COLOR_GRAY2RGB)                       
                            np.putmask(anoted_image, ctkeym >0, 0)
                            anoted_image=cv2.add(anoted_image,ctkey)

                            anoted_image=tagviewct(anoted_image,pat,200,10)                                           
                    else:
                        volumeroi[numslice][pat]=0    
            anoted_image=cv2.cvtColor(anoted_image,cv2.COLOR_RGB2BGR)
            cv2.imwrite(roibmpfile,anoted_image)
#    print listroi
    return tabroi,volumeroi,slnroi,listroi


##############################################################################
listdirc= (os.listdir(namedirtopc))
listpat=[]
listslicetot={}
listpatf={}
listslicef={}
totalnumperpat={}
slntdict={}

for pat in classif:
    totalnumperpat[pat]=0
print 'directory work: ',namedirtopc
avg=0
nump=0
for f in listdirc:
    print '-----------------'
    print('work on:',f)
    print '-----------------'
    listslicef[f]={}
    dirf=os.path.join(namedirtopc,f)
    fmbmp=os.path.join(dirf,lungmask1)
    if os.path.exists(fmbmp):
        lungmaski=lungmask1
    else:
        fmbmp=os.path.join(dirf,lungmask)
        if os.path.exists(fmbmp):
            lungmaski=lungmask
        else:
            os.mkdir(fmbmp)
            lungmaski=lungmask
    
    wridirsource=os.path.join(dirf,source)
    listdcm= [name for name in os.listdir(dirf) if name.find('.dcm')>0]
    nosource=False
    if len(listdcm)>0:
        nosource=True
        if not os.path.exists(wridirsource):
            os.mkdir(wridirsource)
#    numsliceok=[]
    listslicetot[f]=0
    sroidir=os.path.join(dirf,sroi)
    if os.path.exists(sroidir):
        remove_folder(sroidir)
    os.mkdir(sroidir)
    tabscanName={}
    tabscanroi={}
    tabscanScan,slnt,slicepitch,lissln,tabscanroi,tabscanName,PixelSpacing=genebmp(dirf,
                        source,nosource, centerHU, limitHU,tabscanName,tabscanroi,image_rows,image_cols)
    tabscanLung,tabrange=genebmplung(dirf,lungmaski,slnt,image_rows,image_cols,tabscanScan,lissln,tabscanName)
    slnroi=[]
    tabroi=np.zeros((slnt,image_rows,image_cols), np.uint8) 
    tabroi,volumeroi,slnroi,listroi=generoi(dirf,tabroi,image_rows,image_cols,slnroi,
                                            tabscanName,sroidir,tabscanroi,tabscanLung,PixelSpacing,slnt)
    datascan={}
    
    for numslice in slnroi:
        datascan,avgp=peparescan(numslice,tabscanScan[numslice],tabscanLung[numslice],datascan)
#        print numslice,np.mean(datascan[numslice])
        avg+=avgp
        nump+=1
    if nump>0:
        print 'calcul average pixel :',f,avg,nump,1.0*avg/nump
    

    slntdict[f]=slnt
    for slic in range(slnt):
        listslicef[f][slic]={}
        for pat in classif:
            listslicef[f][slic][pat]=0
#
    listpatf[f]=[]

    for num,value in listroi.items():
#                print num,value
                for p in value:
                    if p not in listpat:
                        listpat.append(p)
                    if p not in listpatf[f]:
                        listpatf[f].append(p)

    preparroi(dirf,datascan,tabroi,listroi,slnroi)
    
    listslicetot[f]=len(slnroi)
    print 'number of different images :',len(slnroi)
    for i in range(slnt):
#        print i, tabroipat[i]
        for pat in classif:
             if listslicef[f][i][pat] !=0:
                 print  f,i, pat, listslicef[f][i][pat]
                 totalnumperpat[pat]+=1
print 'total calcul average pixel :',avgp,nump,1.0*avg/nump 
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'

pathpatfile=os.path.join(patchtoppath,'listpat.txt')
filetw = open(pathpatfile,"a")      
              
print '-----------------------------'
print('TOP DIR: '+nameHug)

print 'list of patterns :',listpat
print '-----------------------------'

filetw.write( '-----------------------------\n')
filetw.write('started ' +nameHug+' '+subHUG+' at :'+todayn)
filetw.write('total calcul average pixel: ' +str(avgp)+' number of images: '+str(nump)+' average: '+str(1.0*avg/nump)+'\n')            
filetw.write('list of patterns :'+str(listpat)+'\n')       
filetw.write( '-----------------------------\n')

totsln=0
npattot={}
for pat in classif:
        npattot[pat]=0
      
for f in listdirc:
    npat={}
    for pat in classif:
        npat[pat]=0
    print 'patient :',f
    filetw.write('patient :'+f+'\n')
    print 'number of diferent images :',listslicetot[f]
    filetw.write('number of different  images :'+str(listslicetot[f]) +'\n')
    print 'list of patterns :',listpatf[f]
    
    filetw.write('list of patterns :'+str(listpatf[f]) +'\n')

#    filetw.write( '-----------------------------\n')
    for s in range(slntdict[f]):
        for pat in classif:
            if listslicef[f][s][pat] !=0:
                npat[pat]+=1
                npattot[pat]+=1
      
    totsln=totsln+listslicetot[f]
    
    for pat in classif:
         if npat[pat]!=0:
             print 'number of images for :', pat,' :',  npat[pat] 
             filetw.write('number of images for :'+ pat+' :'+  str(npat[pat])+'\n')             
    filetw.write('--------------------\n')
    print '-----------------------------'  
    
print 'number total of different images',totsln
filetw.write('number total of different images: '+str(totsln) +'\n')
totimages=0
for pat in classif:
         if npattot[pat]!=0:
             print 'number of images for :', pat,' :',  npattot[pat] 
             filetw.write('number of images for :'+ pat+' :'+  str(npattot[pat])+'\n' )
             totimages+=npattot[pat]
             
print ' total number of images :',totimages
filetw.write('total number of images :'+str(totimages)+'\n')
             

filetw.write('--------------------\n')
filetw.close()

