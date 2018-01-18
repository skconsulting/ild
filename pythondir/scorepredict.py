# coding: utf-8
#sylvain Kritter 04-Apr-2017
'''predict on lung scan front view and cross view
@author: sylvain Kritter 

version 1.5
6 September 2017
'''
#from param_pix_p import *
from param_pix_s import scan_bmp,avgPixelSpacing,dimpavx,dimpavy,dirpickleArch,modelArch,surfelemp
from param_pix_s import typei,typei1,excluvisu
from param_pix_s import white,yellow,red

from param_pix_s import lung_namebmp,jpegpath,lungmask,lungmask1
from param_pix_s import fidclass,pxy
#from param_pix_p import classifc,classif,excluvisu,usedclassif
from param_pix_s import classifc


from param_pix_s import  source
from param_pix_s import  transbmp
from param_pix_s import sroi
from param_pix_s import jpegpath3d
from param_pix_s import jpegpadirm,source_name,path_data,dirpickle,cwdtop

from param_pix_s import remove_folder,normi,rsliceNum,norm,maxproba
from param_pix_s import classifdict,usedclassifdict,oldFormat,derivedpatdict,layertokeep

#import scipy
import time
from time import time as mytime
import numpy as np
#from numpy import argmax,amax
import os
import cv2
import dicom
import copy
#import sys
from skimage import measure, morphology
from skimage.segmentation import clear_border
from skimage.morphology import  disk, binary_erosion, binary_closing
from skimage.filters import roberts
from skimage.measure import label,regionprops
from scipy import ndimage as ndi
from itertools import product
import cPickle as pickle
#import matplotlib.pyplot as plt

#import keras
from keras.models import load_model
from keras.models import model_from_json
#from keras.optimizers import Adam


moderesize=True # True means linear resize
t0=mytime()

def reshapeScanl(tabscan):
    print 'reshape lung'
    tabres=np.moveaxis(tabscan,0,1)
    return tabres



def genebmp(fn,sou,nosource,centerHU, limitHU, tabscanName,tabscanroi):
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
    try:
            slicepitch = np.abs(RefDs.ImagePositionPatient[2] - RefDs1.ImagePositionPatient[2])
    except:
            slicepitch = np.abs(RefDs.SliceLocation - RefDs1.SliceLocation)
    print 'slice pitch in z :',slicepitch
    print 'patient position :',patientPosition
    lbHU=centerHU-limitHU/2
    lhHU=centerHU+limitHU/2
    dsr= RefDs.pixel_array

    dsr = dsr.astype('float32')
    if moderesize:
        print 'resize'
        fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
        imgresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
        dimtabx=imgresize.shape[0]
        dimtaby=imgresize.shape[1]
    else:
        print 'no resize'
        dimtabx=dsr.shape[0]
        dimtaby=dimtabx
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
    tabscan = np.zeros((slnt,dimtabx,dimtaby),np.int16)

    for l in listdcm:
#        print l
        FilesDCM =(os.path.join(fmbmp,l))
        RefDs = dicom.read_file(FilesDCM,force=True)
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
        
        if moderesize:
            dsr = dsr.astype('float32')
            dsr=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)
            dsr=dsr.astype('int16')
      
        endnumslice=l.find('.dcm')
        imgcoreScan=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei1  
        tabscan[slicenumber]=dsr.copy()
        np.putmask(dsr,dsr<lbHU,lbHU)
        np.putmask(dsr,dsr>lhHU,lhHU)

        imtowrite=normi(dsr)
        imtowrite = cv2.cvtColor(imtowrite, cv2.COLOR_GRAY2RGB)

        tabscanName[slicenumber]=imgcoreScan
        t2='Prototype '
        t1='Patient: '+tail
        t0='CONFIDENTIAL'
        t3='Scan: '+str(slicenumber)
        t4=time.asctime()
        t5='CenterHU: '+str(int(centerHU))
        t6='LimitHU: +/-' +str(int(limitHU/2))   
        anoted_image=tagviews(imtowrite,
                              t0,dimtabx-150,dimtaby-10,
                              t1,0,dimtaby-21,
                              t2,dimtabx-150,dimtaby-20,
                              t3,0,dimtaby-32,
                              t4,0,dimtaby-10,
                              t5,0,dimtaby-43,
                              t6,0,dimtaby-54)     
           
        tabscanroi[slicenumber]=anoted_image

    return tabscan,slnt,dimtabx,slicepitch,lislnn,tabscanroi,tabscanName

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

def morph(imgt,k):

    img=imgt.astype('uint8')
    img[img>0]=classif['lung']+1
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

def colorimage(image,color):
#    im=image.copy()
    im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    np.putmask(im,im>0,color)
    return im


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
#    kernel = np.ones((4,4),np.uint8)
    (top,tail) =os.path.split(fn)
    print ('load lung segmented dicom files in :',tail)
    fmbmp=os.path.join(fn,lungname)    
    fmbmpbmp=os.path.join(fmbmp,lung_namebmp)
    if not os.path.exists(fmbmpbmp):
        os.mkdir(fmbmpbmp)
        
#    print listsln
    listslnCopy=copy.copy(listsln)
#    print listslnCopy
    listdcm=[name for name in  os.listdir(fmbmp) if name.lower().find('.dcm')>0]
    listbmp=[name for name in  os.listdir(fmbmpbmp) if name.lower().find(typei1)>0]  
    tabscan = np.zeros((slnt,dimtabx,dimtaby), np.uint8)
    if len(listbmp)>0:
        print 'lung scan exists in bmp'
        for img in listbmp:
            slicenumber= rsliceNum(img,'_','.'+typei1)
            if slicenumber>0:       
                    imr=cv2.imread(os.path.join(fmbmpbmp,img),0) 
                    imr=cv2.resize(imr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)  
                    np.putmask(imr,imr>0,classif['lung']+1)                                  
#                    dilation = cv2.dilate(imr,kernel,iterations = 1)
                    tabscan[slicenumber]=imr
#                    if slicenumber==12:
#                        pickle.dump(imr, open('scorelung.pkl', "wb" ),protocol=-1)
                    listslnCopy.remove(slicenumber)
    if len(listslnCopy)>0:
        print 'not all lung in bmp'
        if len(listdcm)>0:  
            print 'lung scan exists in dcm'
               
            for l in listdcm:
                FilesDCM =(os.path.join(fmbmp,l))
                RefDs = dicom.read_file(FilesDCM,force=True)
        
                dsr= RefDs.pixel_array
                dsr=normi(dsr)
        
                fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
                imgresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
                np.putmask(imgresize,imgresize>0,classif['lung']+1)    
                slicenumber=int(RefDs.InstanceNumber)
                imgcoreScan=tabscanName[slicenumber]
                bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
                if tabscan[slicenumber].max()==0:
#                    dilation = cv2.dilate(imgresize,kernel,iterations = 1)
                    tabscan[slicenumber]=imgresize
                    colorlung=colorimage(imgresize,classifc['lung'])
                    cv2.imwrite(bmpfile,colorlung)  
                    
                    
        else:
                print 'no lung scan in dcm'
                tabscan1 = np.zeros((slnt,dimtabx,dimtaby), np.int16)
                srd=False
                segmented_lungs_fill,ok = segment_lung_mask(tabscanScan,slnt, srd,True)
                if ok== False:
                    srd=True
                    segmented_lungs_fill,ok = segment_lung_mask(tabscanScan,slnt, srd,True)
#                    segmented_lungs_fill,ok = segment_lung_mask(tabscanScan,slnt, True)
                if ok== False:
                    print 'use 2nd algorihm'
                    segmented_lungs_fill=np.zeros((slnt,dimtabx,dimtabx), np.uint8)
                    for i in listsln:
                        segmented_lungs_fill[i]=get_segmented_lungs(tabscanScan[i])
    #            tabscanlung = np.zeros((slnt,dimtabx,dimtaby), np.uint8)
    #            print tabscanScan.shape
#                segmented_lungs_fill = segment_lung_mask(tabscanScan, True)
    #            print segmented_lungs_fill.shape
                for i in listsln:
    #                tabscan[i]=normi(tabscan[i])
                    tabscan1[i]=morph(segmented_lungs_fill[i],13)
    #                imgcoreScan='lung_'+str(i)+'.'+typei
                    imgcoreScan=tabscanName[i]
                    bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
                    if tabscan[i].max()==0:
#                        dilation = cv2.dilate(tabscan1[i],kernel,iterations = 1)
                        tabscan[i]=tabscan1[i]
                        colorlung=colorimage(tabscan[i],classifc['lung'])
                        cv2.imwrite(bmpfile,colorlung)
    else:
        print 'all lung in bmp'
    for sli in listsln:
        cpt=np.copy(tabscan[sli])
        np.putmask(cpt,cpt>0,1)
        area=cpt.sum()
        if area >pxy:
            if sli> tabrange['max']:
                tabrange['max']=sli
            if sli< tabrange['min']:
                tabrange['min']=sli                
    return tabscan,tabrange



def tagviews (tab,t0,x0,y0,t1,x1,y1,t2,x2,y2,t3,x3,y3,t4,x4,y4,t5,x5,y5,t6,x6,y6):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_PLAIN
    col=red
    size=0.5
    sizes=0.4

    viseg=cv2.putText(tab,t0,(x0, y0), font,sizes,col,1)
    viseg=cv2.putText(viseg,t1,(x1, y1), font,size,col,1)
    viseg=cv2.putText(viseg,t2,(x2, y2), font,sizes,col,1)

    viseg=cv2.putText(viseg,t3,(x3, y3), font,size,col,1)
    viseg=cv2.putText(viseg,t4,(x4, y4), font,size,col,1)
    viseg=cv2.putText(viseg,t5,(x5, y5), font,size,col,1)
    viseg=cv2.putText(viseg,t6,(x6, y6), font,size,col,1)

    return viseg


def pavgene(dirf,dimtabx,dimtaby,tabscanScan,tabscanLung,slnt,jpegpath,listsln):
   
        """ generate patches from scan"""
        global thrpatch

        tpav=mytime()
        
        patch_list=[]
        (dptop,dptail)=os.path.split(dirf)
        print('generate patches on: ',dptail)
        jpegpathdir=os.path.join(dirf,jpegpath)
        remove_folder(jpegpathdir)
        os.mkdir(jpegpathdir)
        tabscan = np.zeros((slnt,dimtabx,dimtaby), np.int16)
#        for i in listsln:
                
        for img in listsln:
             tabscan[img]=tabscanScan[img]
#             if img==12:
#                        pickle.dump(tabscan[img], open('scorepav.pkl', "wb" ),protocol=-1)


             tabfw = np.zeros((dimtabx,dimtaby,3), np.uint8)
             tablung = np.copy(tabscanLung[img])
             tabfrgb=np.copy(tablung)
             np.putmask(tablung,tablung>0,1)
             np.putmask(tabfrgb,tabfrgb>0,100)
             tabfrgb= cv2.cvtColor(tabfrgb,cv2.COLOR_GRAY2BGR)
             tabf=norm(tabscan[img])
#             print nz1
             nz= tablung.max()

             
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

                            min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(imgray)

                            if  min_val != max_val:
#                                imgray= norm(imgray)
                                patch_list.append((img,i,j,imgray))
                                tablung[j:j+dimpavy,i:i+dimpavx]=0
                                cv2.rectangle(tabfw,(i,j),(i+dimpavx,j+dimpavy),yellow,0)
                                j+=dimpavy-1

                         j+=1
                     i+=1

                 nameslijpeg='s_'+str(img)+'.'+typei
                 namepatchImage=os.path.join(jpegpathdir,nameslijpeg)
                 tabjpeg=cv2.add(tabfw,tabfrgb)
                 cv2.imwrite(namepatchImage,tabjpeg)
        print "pav time:",round(mytime()-tpav,3),"s"

        return patch_list

def pavgenefront(dirf,dimtabx,dimtaby,tabscanScan,tabscanLung,slnt,jpegpath):
        """ generate patches from scan"""
        global thrpatch
        tpav=mytime()        
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
             tabf=norm(tabscanScan[img])
#             print nz1
             nz= tablung.max()             

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
                            imgray = tabf[j:j+dimpavy,i:i+dimpavx]

                            min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(imgray)
                            if  min_val != max_val:
#                                imgray= norm(imgray)
                                patch_list.append((img,i,j,imgray))
                                tablung[j:j+dimpavy,i:i+dimpavx]=0
                                cv2.rectangle(tabfw,(i,j),(i+dimpavx,j+dimpavy),yellow,0)
                                j+=dimpavy-1

                         j+=1
                     i+=1

                 nameslijpeg='s_'+str(img)+'.'+typei
                 namepatchImage=os.path.join(jpegpathdir,nameslijpeg)
                 tabjpeg=cv2.add(tabfw,tabfrgb)
                 cv2.imwrite(namepatchImage,tabjpeg)
        print "pav time:",round(mytime()-tpav,3),"s"
        return patch_list


def ILDCNNpredict(patch_list,model):
    print ('Predict started ....')
    dataset_list=[]
    for fil in patch_list:
              dataset_list.append(fil[3])
    X0=len(dataset_list)
    if X0 > 0:      
        pa = np.expand_dims(dataset_list, 1)
        proba = model.predict_proba(pa, batch_size=500,verbose=1)
        pru= np.unique(np.argmax(proba,axis=1))
        print pru
#        print proba.shape
#        for i in range(proba.shape[0]):
#            if np.argmax(proba[i])>4:
#                print proba[i]
#                print np.argmax(proba[i])
##            print np.unique(np.argmax(proba[i]))
##            print np.unique(np.argmax(proba,axis=0))
#        sys.exit(1)
        

    else:
        print (' no patch in selected slice')
        proba = []
    print 'number of patches', len(pa)
    print (' predicted patterns:')
    for i in range(len(pru)):
        print  pru[i], usedclassif[pru[i]]
#    sys.exit(1)

    return proba

def wtebres(wridir,dirf,tab,dimtabx,slicepitch,lungm,ty,centerHU,limitHU):
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

    ntd=int(round(fxs*tab[0].shape[0],0))

    if ty=='scan':
        tabres=np.zeros((dimtabx,ntd,dimtabx),np.int16)
    else:
        tabres=np.zeros((dimtabx,ntd,dimtabx),np.uint8)
    for i in range (0,dimtabx):
#        print i, tab[i].max()
        lislnn.append(i)

        imgresize=cv2.resize(tab[i],None,fx=1,fy=fxs,interpolation=cv2.INTER_LINEAR)

        if ty=='scan':
            typext=typei1
        else:
            typext=typei
        trscan='tr_'+str(i)+'.'+typext
        trscanbmp='tr_'+str(i)+'.'+typei
        if ty=='lung':
            namescan=os.path.join(bgdirf,trscanbmp)
            np.putmask(imgresize,imgresize>0,100)
            cv2.imwrite(namescan,imgresize)
            dimtabxn=imgresize.shape[0]
            dimtabyn=imgresize.shape[1]
        if ty=='scan':
            namescan=os.path.join(wridir,trscan)
            dimtabxn=imgresize.shape[0]
            dimtabyn=imgresize.shape[1]
            
            
            imgresize8=normi(imgresize)
            (topw,tailw)=os.path.split(picklein_file_front)
            t2='Prototype '
            t1='param :'+tailw
            t0='CONFIDENTIAL'
            t3='Scan: '+str(i)
    
            t4=time.asctime()
            t5='CenterHU: '+str(int(centerHU))
            t6='LimitHU: +/-' +str(int(limitHU))
            
            
            anoted_image=tagviews(imgresize8,t0,dimtabxn-300,dimtabyn-10,t1,0,dimtabyn-20,t2,dimtabx-350,dimtabyn-10,
                         t3,0,dimtabyn-30,t4,0,dimtabyn-10,t5,0,dimtabyn-40,t6,0,dimtabyn-50)
                    
            
            cv2.imwrite(namescan,anoted_image)

        tabres[i]=imgresize
#        cv2.imwrite (trscan, tab[i],[int(cv2.IMWRITE_PNG_COMPRESSION),0])
    return dimtabxn,dimtabyn,tabres,lislnn


def modelCompilation(t,picklein_file,picklein_file_front,setdata):
    
    print 'model compilation',t
    
    if oldFormat== False:
        dirpickleArchs=os.path.join(dirpickleArch,setdata)
        dirpickleArchsc=os.path.join(dirpickleArchs,modelArch)
    
        json_string=pickle.load( open(dirpickleArchsc, "rb"))
        model = model_from_json(json_string)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#        model.compile()


    if t=='cross':
        lismodel=os.listdir(picklein_file)
        
        modelpath = os.path.join(picklein_file, lismodel[0])
#        print modelpath
    if t=='front':
        lismodel=os.listdir(picklein_file_front)
        modelpath = os.path.join(picklein_file_front, lismodel[0])

    if os.path.exists(modelpath):
        if oldFormat:
            model = load_model(modelpath)
        else:
            model.load_weights(modelpath)  
        
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

def drawcontours2(im,pat,dimtabx,dimtaby):
#    print 'contour',pat
    imgray = np.copy(im)
    ret,thresh = cv2.threshold(imgray,10,255,0)
    _,contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2 = np.zeros((dimtabx,dimtaby,3), np.uint8)
    cv2.drawContours(im2,contours,-1,classifc[pat],1)
#    im2=cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
    return im2

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


def genecrossfromfront(proba_front,patch_list_front,dimtabx,lissln,dimtabxn,slnt):
    pl=[]
    pr=[]
    maxhight=1.0*(slnt-1)/dimtabxn

    print 'maxhight',maxhight
#    tabhight={}
#    pitchslice=maxhight/slnt
#    for i in range (0,slnt):
#        tabhight[i]=pitchslice*(i)
#        print i, tabhight[i]
    def findsln(y):
        return max(1,int(round(y*maxhight,0)))
#        for i in range (0,slnt):
#            if y+(dimpavx/2)<tabhight[i]:
#                return i
#        return i
#    maxx=0
#    maxy=0
#    maxz=0
    for ll in range(0,len(patch_list_front)):
        proba=proba_front[ll]
        x=patch_list_front[ll][1]
        y=patch_list_front[ll][2]
        z=patch_list_front[ll][0]
#        if x>maxx:
#            maxx=x
#        if y>maxy:
#            maxy=y
#        if z>maxz:
#            maxz=z

        sln=findsln(y+(dimpavx/2))
#        print y,sln
        t=(sln,x,z)
#        print sln,z,x
#        print type(z),type(x)
        pl.append(t)
        pr.append(proba)
#    print maxx,maxy,maxz
    return pr,pl
    





def tagviewct(tab,label,x,y):
    """write text in image according to label and color"""

    col=classifc[label]
#    print label,col
    labnow=classif[label]
#  
    deltay=10*(labnow%5)
    deltax=100*(labnow/5)
#    print label, deltax,deltay
    font = cv2.FONT_HERSHEY_PLAIN
    viseg=cv2.putText(tab,label,(x+deltax, y+deltay), font,0.7,col,1)
#    viseg = cv2.cvtColor(viseg,cv2.COLOR_RGB2BGR)
    return viseg




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


def  mergeproba(list_cross,from_front,slnt,dimtabx,dimtaby):
    print "merge proba list"
    volpat={}
    for sln in range(slnt):
        volpat[sln]={}
        for pat in usedclassif:
            volpat[sln][pat]=np.zeros((dimtabx,dimtaby), np.uint8)
    patch_list_merge=[]
    proba_merge=[]
#    frontpat={}
    print 'fill table'
    for sln in from_front:
#        frontpat[sln]={}
#        for pat in usedclassif:
#            frontpat[sln][pat]=[]

        for ii in range(0,len(from_front[sln])):
                (xpat,ypat)=from_front[sln][ii][0]
                proba=from_front[sln][ii][1]
                prec, mproba = maxproba(proba)   
                volpat[sln][fidclass(prec,classif)][ypat:ypat+dimpavy,xpat:xpat+dimpavx]=1
#                t=((xpat,ypat),proba)
#                frontpat[sln][fidclass(prec,classif)].append(t)
    
    print 'scan'
    for sln in list_cross:
#        print sln
        for jj in range(0,len(list_cross[sln])):
#            print list_cross[jj]
            (xpat,ypat)=list_cross[sln][jj][0]
            proba=list_cross[sln][jj][1]
            prec, mproba = maxproba(proba)  
            pat=fidclass(prec,classif)
            tab1=np.zeros((dimtabx,dimtaby),np.uint8)
            tab1[ypat:ypat+dimpavy,xpat:xpat+dimpavx]=1
#            if sln==12 and pat=='ground_glass':
#                print xpat,ypat
#            try:
#                for ii in range(0,len(frontpat[sln][pat])):
#                    (xpat1,ypat1)=frontpat[sln][pat][ii][0]
#                    tab2=np.zeros((dimtabx,dimtaby),np.uint8)
#                    tab2[ypat1:ypat1+dimpavy,xpat1:xpat1+dimpavx]=255
            tab2=volpat[sln][pat]

            tab3=np.bitwise_and(tab1,tab2)
            nz= np.count_nonzero(tab3)
            if nz>pxy/2:
                patch_list_merge.append((int(sln),xpat,ypat))
                proba_merge.append(proba)
#                break
#            except:
#                continue
                                     
    return proba_merge,patch_list_merge

def  calcMed (tabscanLung,lisslnfront):
    '''calculate the median position in between left and right lung'''
#    print 'number of subpleural for : ',pat
#    print 'subpleural', ntp, pat
#    global lungSegment
    tabMed={}
    dimtabx=tabscanLung.shape[1]
    dimtaby=tabscanLung.shape[2]

    for slicename in lisslnfront:

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

def subpleural(dirf,tabscanLung,lissln,subErosion,crfr):
#def calcSupNp(preprob, posp, lungs, imscan, pat, midx, psp, dictSubP, dimtabx):
    '''calculate the number of pat in subpleural'''
    (top,tail)=os.path.split(dirf)
#    print 'number of subpleural for :',tail, 'pattern :', pat

    dimtabx=tabscanLung.shape[1]
    dimtaby=tabscanLung.shape[2]
#    slnt=tabscanLung.shape[0]
    subpleurmaskset={}
    for slicename in lissln:
        vis = np.zeros((dimtabx,dimtaby,3), np.uint8)

        imgngray = np.copy(tabscanLung[slicename])
#        np.putmask(imgngray, imgngray == 1, 0)
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


def selectposition(lislnumber,tabrange):
    minr=tabrange['min']
    maxr=tabrange['max']
    minsln=10000
    maxsln=0
    for i in lislnumber:
        if i<minsln:
            minsln=i
        if i>maxsln:
            maxsln=i
    bmax=(maxr+minr)/2
    bmin=(minr+bmax)/2
    upperset=[]
    middleset=[]
    lowerset=[]
    allset=[]
    lungs={}
#    Nset=len(lislnumber)/3
    for scanumber in lislnumber:
            allset.append(scanumber)
            if scanumber < bmin:
                upperset.append(scanumber)
            elif scanumber < bmax:
                middleset.append(scanumber)
            else:
                lowerset.append(scanumber)
            lungs['upperset']=upperset
            lungs['middleset']=middleset
            lungs['lowerset']=lowerset
            lungs['allset']=allset
    return lungs



def genepatchlistslice(patch_list_cross,proba_cross,lissln,dimtabx,dimtaby):
#    print 'start genepatchilist'
    res={}   
    for i in lissln:
        res[i]=[]
        ii=0
        for ll in patch_list_cross:
            xpat = ll[1]
            ypat = ll[2]
            sln= ll[0]
            if sln ==i:
                t=((xpat,ypat),proba_cross[ii])
                res[sln].append(t)                

            ii+=1                   
    return res

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
            np.putmask(tab1, tab1> 0, classif[pat1]+1)
            
            tab2=np.bitwise_and(tab2,tabn)
            np.putmask(tab2, tab2> 0, classif[pat2]+1)  
            
            np.putmask(tab, tab> 0, classif[pat]+1)            

            tabroi[i]=np.bitwise_and(tabroi[i],taballnot)             
            tabroi[i]=np.bitwise_or(tabroi[i],tab1) 
            tabroi[i]=np.bitwise_or(tabroi[i],tab2) 
            tabroi[i]=np.bitwise_or(tabroi[i],tab) 

    return tabroi


def generoi(dirf,tabroi,dimtabx,dimtaby,slnroi,tabscanName,dirroit,tabscanroi,tabscanLung,slnroidir,slnt,fer):
    (top,tail)=os.path.split(dirf)
    tabroipat={}    
    listroi={}
    for pat in usedclassif:
        tabroipat[pat]=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
        pathroi=os.path.join(dirf,pat)
        if os.path.exists(pathroi):
            lroi=[name for name in os.listdir(pathroi) if name.find('.'+typei1)>0  ]      
            for s in lroi:
                numslice=rsliceNum(s,'_','.'+typei1)                    
                img=cv2.imread(os.path.join(pathroi,s),0)
                img=cv2.resize(img,(dimtabx,dimtabx),interpolation=cv2.INTER_LINEAR)                
                np.putmask(img, img > 0, classif[pat]+1)
                tabroipat[pat][numslice]=img     
                if numslice not in slnroi:
                    slnroi.append(numslice)  
    
    
    for numslice in slnroi:
        for pat in usedclassif:
            tab=np.copy(tabroipat[pat][numslice])
            np.putmask(tabroi[numslice], tab > 0, 0)
#            np.putmask(tab, tab > 0, classif[pat]+1)            
            tabroi[numslice]+=tab
#            if tab.max()>0:
#                cv2.imshow(str(numslice)+' '+pat+' tabo',normi(tabroi[numslice])) 
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()     
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
            fer.write('no lung for: '+str(numslice)+'\n')
            print ('no lung for: '+str(numslice))
        tabroi[numslice]=np.bitwise_and(maskRoi,maskLung)   
        
        maskRoi1Not=np.bitwise_not(maskRoi1)       
        tablung=np.bitwise_and(maskLung, maskRoi1Not)
        np.putmask(tablung,tablung>0,classif['healthy']+1) 
        tabroi[numslice]=np.bitwise_or(tabroi[numslice],tablung)
        
#        if tabroi[numslice].max()>0:
#                cv2.imshow(str(numslice)+' tabo',normi(tabroi[numslice])) 
#                cv2.imshow(str(numslice)+' maskLung',normi(maskLung)) 
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()    
#    cv2.imwrite('a225.bmp',10*tabroi[12])
    slnroi.sort()
    volumeroi={}
    for numslice in slnroi:
            imgcoreScan=tabscanName[numslice]
            roibmpfile=os.path.join(dirroit,imgcoreScan)
            anoted_image=tabscanroi[numslice]
            listroi[numslice]=[]
#            anoted_image=cv2.cvtColor(anoted_image,cv2.COLOR_BGR2RGB)
            
            volumeroi[numslice]={}
            for pat in classif:
                
                img=np.copy(tabroi[numslice])
                if img.max()>0:                    
                   
                    np.putmask(img, img !=classif[pat]+1, 0)
                    np.putmask(img, img ==classif[pat]+1, 1)
                    
                    area= img.sum()* surfelemp /100
                    volumeroi[numslice][pat]=area  
                    
                    if area>0:
                        
                        if pat not in listroi[numslice] and area>1:
                            listroi[numslice].append(pat)
                        if pat not in excluvisu:
#                            print pat
                            np.putmask(img, img >0, 100)
#                                ctkey=drawcontours2(img,pat,dimtabx,dimtaby)
#                                ctkeym=ctkey.copy()
#                                ctkeym=cv2.cvtColor(ctkeym,cv2.COLOR_RGB2GRAY)
#                                ctkeym=cv2.cvtColor(ctkeym,cv2.COLOR_GRAY2RGB)                       
#                                np.putmask(anoted_image, ctkeym >0, 0)
    #                        anoted_image=cv2.add(anoted_image,ctkey)
                            
                            
    #                        imgcolor=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)  
                            colorlung=colorimage(img,classifc[pat])
    
                            anoted_image=cv2.addWeighted(anoted_image,1,colorlung,0.4,0)
    
                            anoted_image=tagviewct(anoted_image,pat,200,10)  
#                                tabscanroi[numslice]=anoted_image
                                               
                else:
                    volumeroi[numslice][pat]=0 
            anoted_image= cv2.cvtColor(anoted_image,cv2.COLOR_RGB2BGR)
#            print roibmpfile
            cv2.imwrite(roibmpfile,anoted_image)   
#    print volumeroi[12]  
    slnroidir[tail]=len(slnroi)
    return tabroi,volumeroi,slnroi,slnroidir,listroi,tabscanroi
        

def predictrun(indata,path_patient):
        global thrpatch,thrproba,thrprobaUIP,subErosion
        global  picklein_file,picklein_file_front,classif,usedclassif,derivedpat
        td=False
        
        print '-------------------'
        if indata['threedpredictrequest']=='Cross Only':
           td=False           
           print 'CROSS PREDICT'
        else:
            td=True
            print 'CROSS +FRONT PREDICT'
        thrproba=float(indata['thrproba'])
        thrpatch=float(indata['thrpatch'])
        picklein_filet=indata['picklein_file']
        centerHU=indata['centerHU']
        limitHU=indata['limitHU']
        ForceGenerate=indata['ForceGenerate']
        
        #define set data and associated patterns
        if oldFormat==False:
            posund=picklein_filet.find('_')
            setref=picklein_filet[0:posund]
        else:
            setref='set0'
            
        picklein_file_front=indata['picklein_file_front']
        if oldFormat==False:
            posund=picklein_file_front.find('_')
            setref2=picklein_file_front[0:posund]
        else:
            setref2='set0'
        if setref!=setref2:
            print 'ERROR NOT SAME PATTERN SET  FOR CROSS AND FRONT'
            return 'ERROR NOT SAME PATTERN SET  FOR CROSS AND FRONT'            
            
        classif=classifdict[setref]
        usedclassif=usedclassifdict[setref]
        derivedpat=derivedpatdict[setref]

        picklein_file_frontt=indata['picklein_file_front']

        wvisu=indata['Fast']
        if wvisu:
            print 'no record of predict images on disk'
        else:
            print 'record of predict images on disk'

        listHug=[]
        listHugi=indata['lispatientselect']
        for lpt in listHugi:
             pos=lpt.find(' PREDICT!:')
             if pos >0:
                    listHug.append(lpt[0:pos])
             else:
                    pos=lpt.find(' noPREDICT!')
                    if pos >0:
                        listHug.append(lpt[0:pos])
                    else:
                        listHug.append(lpt)

        picklein_file =  os.path.join(dirpickle,picklein_filet)
        picklein_file_front =  os.path.join(dirpickle,picklein_file_frontt)

        dirHUG=os.path.join(cwdtop,path_patient)
        slnroidir={}
        fer=open(os.path.join(dirHUG,'report.txt'),'a')
        modelcross=modelCompilation('cross',picklein_file,picklein_file_front,setref)
        for f in listHug:
            print '------------------'
            print 'work on patient',f,' set:',setref,picklein_filet,'thrproba:',thrproba,'thrpatch',thrpatch
            patch_list_cross_slice={}
            
            dirf=os.path.join(dirHUG,f)
            wridirsource=os.path.join(dirf,source)

            listdcm= [name for name in os.listdir(dirf) if name.find('.dcm')>0]
            nosource=False
            if len(listdcm)>0:
                nosource=True
                if not os.path.exists(wridirsource):
                    os.mkdir(wridirsource)

            path_data_write=os.path.join(dirf,path_data)

            if not os.path.exists(path_data_write):
                os.mkdir(path_data_write)
            wridir=os.path.join(wridirsource,transbmp)
            remove_folder(wridir)
            os.mkdir(wridir)

            eferror=os.path.join(path_data_write,'log.txt')
            errorfile = open(eferror, "w")
#            """
            jpegpathdir=os.path.join(dirf,jpegpadirm)
            remove_folder(jpegpathdir)
            os.mkdir(jpegpathdir)
                      
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
            dirroi=os.path.join(dirf,sroi)
            if not os.path.exists(dirroi):
                os.mkdir(dirroi)
            
            print '------------------'
            print 'START PREDICT CROSS'
            print '------------------'
            crosscompleted=False
            pickle.dump(crosscompleted, open( os.path.join(path_data_write,"crosscompleteds"), "wb" ),protocol=-1)
            frontcompleted=False
            pickle.dump(frontcompleted, open( os.path.join(path_data_write,"frontcompleteds"), "wb" ),protocol=-1)
#            print 'source',source
#            return ''
            tabscanName={}
            tabscanroi={}

            centerHU1=0
            limitHU1=0
            if os.path.exists(os.path.join(path_data_write,'centerHUs')):
                centerHU1=pickle.load( open(os.path.join(path_data_write,'centerHUs'), "rb" ))
            if os.path.exists(os.path.join(path_data_write,'limitHUs')):
                limitHU1=pickle.load( open(os.path.join(path_data_write,'limitHUs'), "rb" ))
        
            if centerHU1==centerHU and limitHU1==limitHU and not(ForceGenerate):
                print 'no need to regenerate'
                tabscanScan=pickle.load( open(os.path.join(path_data_write,'tabscanScans'), "rb" ))
                tabscanName=pickle.load( open(os.path.join(path_data_write,'tabscanNames'), "rb" ))
                tabscanroi=pickle.load( open(os.path.join(path_data_write,'tabscanrois'), "rb" ))
                tabscanLung=pickle.load( open(os.path.join(path_data_write,'tabscanLungs'), "rb" ))
                tabrange=pickle.load( open(os.path.join(path_data_write,'tabranges'), "rb" ))
                datacross=pickle.load( open(os.path.join(path_data_write,'datacrosss'), "rb" ))

                slnt=datacross[0]
                dimtabx=datacross[1]
                lissln=datacross[4]
                slicepitch=datacross[3]
                print 'end load'
            else:
                print 'generate'
                
                tabscanScan,slnt,dimtabx,slicepitch,lissln,tabscanroi,tabscanName=genebmp(dirf,
                        source,nosource, centerHU, limitHU,tabscanName,tabscanroi)               
                tabscanLung,tabrange=genebmplung(dirf,lungmaski,slnt,dimtabx,dimtabx,tabscanScan,lissln,tabscanName)
                datacross=(slnt,dimtabx,dimtabx,slicepitch,lissln,setref, thrproba, thrpatch)
                
                pickle.dump(centerHU, open(os.path.join(path_data_write,'centerHUs'), "wb" ),protocol=-1) 
                pickle.dump(limitHU, open(os.path.join(path_data_write,'limitHUs'), "wb" ),protocol=-1)                 
                pickle.dump(tabscanScan, open(os.path.join(path_data_write,'tabscanScans'), "wb" ),protocol=-1) 
                pickle.dump(tabscanroi, open(os.path.join(path_data_write,'tabscanrois'), "wb" ),protocol=-1) 
                pickle.dump(tabscanName, open(os.path.join(path_data_write,'tabscanNames'), "wb" ),protocol=-1)                
                pickle.dump(tabscanLung, open( os.path.join(path_data_write,"tabscanLungs"), "wb" ),protocol=-1)
                pickle.dump(tabrange, open( os.path.join(path_data_write,"tabranges"), "wb" ),protocol=-1)
                pickle.dump(datacross, open( os.path.join(path_data_write,'datacrosss'), "wb" ),protocol=-1)
            
            datacross=(slnt,dimtabx,dimtabx,slicepitch,lissln,setref, thrproba, thrpatch)
            pickle.dump(datacross, open( os.path.join(path_data_write,'datacrosss'), "wb" ),protocol=-1)

            slnroi=[]
            tabroi=np.zeros((slnt,dimtabx,dimtabx), np.uint8) 
            tabroi,volumeroi,slnroi,slnroidir,listroi,tabscanroi=generoi(dirf,tabroi,dimtabx,dimtabx,slnroi,     
                    tabscanName,dirroi,tabscanroi,tabscanLung,slnroidir,slnt,fer)
#            print listroi
#            return ' '
            pickle.dump(volumeroi, open(os.path.join(path_data_write,'volumerois'), "wb" ),protocol=-1)
            pickle.dump(listroi, open(os.path.join(path_data_write,'listrois'), "wb" ),protocol=-1)           
            pickle.dump(tabroi, open( os.path.join(path_data_write,"tabrois"), "wb" ),protocol=-1)
            pickle.dump(slnroi, open( os.path.join(path_data_write,"slnrois"), "wb" ),protocol=-1)
            pickle.dump(tabscanroi, open(os.path.join(path_data_write,'tabscanrois'), "wb" ),protocol=-1)
            """
            slnt=datacross[0]
            dimtabx=datacross[1]
            dimtaby=datacross[2]
            slicepitch=datacross[3]
            lissln=datacross[4]
            setref=datacross[5]
            thrproba=datacross[6]
            thrpatch=datacross[7]
            """
            
            regene=True
            if os.path.exists(os.path.join(path_data_write,"thrpatchs")):
                    thrpatch1= pickle.load( open( os.path.join(path_data_write,"thrpatchs"), "rb" ))
                    if thrpatch == thrpatch1:
                        if os.path.exists(os.path.join(path_data_write,"patch_list_crosss")):
                            regene=False   
            if os.path.exists(os.path.join(path_data_write,"slnrois")):
                    slnroi1= pickle.load( open( os.path.join(path_data_write,"slnrois"), "rb" ))
                    if slnroi1 == slnroi:
                        if os.path.exists(os.path.join(path_data_write,"patch_list_crosss")):
                            regene=False   
                                              
            if regene or ForceGenerate:
                print 'regenerate patch list'
                patch_list_cross=pavgene(dirf,dimtabx,dimtabx,tabscanScan,tabscanLung,slnt,jpegpath,slnroi)
                pickle.dump(patch_list_cross, open( os.path.join(path_data_write,"patch_list_crosss"), "wb" ),protocol=-1)
                pickle.dump(thrpatch, open( os.path.join(path_data_write,"thrpatchs"), "wb" ),protocol=-1)
            else:
                print 'no need to regenerate patch list'
                patch_list_cross= pickle.load( open( os.path.join(path_data_write,"patch_list_crosss"), "rb" ))                
            
            
            proba_cross=ILDCNNpredict(patch_list_cross,modelcross)
            patch_list_cross_slice=genepatchlistslice(patch_list_cross,
                                                            proba_cross,lissln,dimtabx,dimtabx)

            crosscompleted=True

            pickle.dump(patch_list_cross_slice, open( os.path.join(path_data_write,"patch_list_cross_slices"), "wb" ),protocol=-1)
            pickle.dump(crosscompleted, open( os.path.join(path_data_write,"crosscompleteds"), "wb" ),protocol=-1)
#            """
    ###       cross
            if td:
#                """
                print 'START PREDICT FRONT'
                print '------------------'
                frontcompleted=False
                pickle.dump(frontcompleted, open( os.path.join(path_data_write,"frontcompleteds"), "wb" ),protocol=-1)
                tabresScan=reshapeScanl(tabscanScan)
                dimtabxn,dimtabyn,tabScan3d,lisslnfront=wtebres(wridir,dirf,tabresScan,
                                                                dimtabx,slicepitch,lungmaski,'scan',centerHU,limitHU)
                tabresLung=reshapeScanl(tabscanLung)   
                dimtabxn,dimtabyn,tabLung3d,a=wtebres(wridir,dirf,tabresLung,dimtabx,slicepitch,
                                                      lungmaski,'lung',centerHU,limitHU)

                datafront=(dimtabx,dimtabxn,dimtabyn,slicepitch,lisslnfront)
                
                pickle.dump(datafront, open( os.path.join(path_data_write,'datafronts'), "wb" ),protocol=-1)
                pickle.dump(tabLung3d, open( os.path.join(path_data_write,"tabLung3ds"), "wb" ),protocol=-1)
                """
                datafront= pickle.load( open( os.path.join(path_data_write,"datafront"), "rb" ))               
                dimtabx=datafront[0]                
                dimtabxn=datafront[1]
                dimtabyn=datafront[2]
                slicepitch=datafront[3]
                lisslnfront=datafront[4]
                """
                regene=True
                if os.path.exists(os.path.join(path_data_write,"thrpatch")):
                    thrpatch1= pickle.load( open( os.path.join(path_data_write,"thrpatchs"), "rb" ))
                    if thrpatch == thrpatch1:
                        if os.path.exists(os.path.join(path_data_write,"patch_list_front")):
                            regene=False

                            patch_list_front= pickle.load( open( os.path.join(path_data_write,"patch_list_fronts"), "rb" ))
                if regene or ForceGenerate:
                    print 'regenerate patch list'
                    patch_list_front=pavgenefront(dirf,dimtabxn,dimtabx,tabScan3d,tabLung3d,dimtabyn,jpegpath3d)
                    pickle.dump(patch_list_front, open( os.path.join(path_data_write,"patch_list_fronts"), "wb" ),protocol=-1)
                else:
                    print 'no need to regenerate patch list'
                
                modelfront=modelCompilation('front',picklein_file,picklein_file_front,setref)

                proba_front=ILDCNNpredict(patch_list_front,modelfront)

                patch_list_front_slice=genepatchlistslice(patch_list_front,
                                                            proba_front,lisslnfront,dimtabxn,dimtabyn)

                pickle.dump(patch_list_front_slice, open( os.path.join(path_data_write,"patch_list_front_slices"), "wb" ),protocol=-1)                
                proba_cross_from_front,patch_list_cross_from_front= genecrossfromfront(proba_front,patch_list_front,
                                                                   dimtabx,lissln,dimtabxn,slnt)
                
                patch_list_cross_slice_from_front=genepatchlistslice(patch_list_cross_from_front,
                                                            proba_cross_from_front,lissln,dimtabx,dimtabx)
                
                pickle.dump(patch_list_cross_slice_from_front, open( os.path.join(path_data_write,"patch_list_cross_slice_from_fronts"), "wb" ),protocol=-1)

                proba_merge,patch_list_merge=mergeproba(patch_list_cross_slice,
                                                        patch_list_cross_slice_from_front,slnt,dimtabx,dimtabx)
                patch_list_merge_slice=genepatchlistslice(patch_list_merge,
                                                            proba_merge,lissln,dimtabx,dimtabx)

                pickle.dump(patch_list_merge_slice, open( os.path.join(path_data_write,"patch_list_merge_slices"), "wb" ),protocol=-1)
                frontcompleted=True
                pickle.dump(frontcompleted, open( os.path.join(path_data_write,"frontcompleteds"), "wb" ),protocol=-1)
            errorfile.write('completed :'+f)
            errorfile.close()
            
            print 'PREDICT  COMPLETED  for ',f,' set ',setref,'thrproba',thrproba,'thrpatch',thrpatch
            print '------------------'
        
#        print slnroidir
        slnroidirtot=0
        for i in listHug:
            slnroidirtot+=slnroidir[i]
        print 'number of images with roi:',slnroidirtot
        print "predict time:",round(mytime()-t0,3),"s"
        fer.write('----\n')
        fer.close()
        return''

#ILDCNNpredict(bglist)