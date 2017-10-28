# coding: utf-8
#sylvain Kritter 04-Apr-2017
'''predict on lung scan front view and cross view
version 1.5
06 Sep 2017
'''
#from param_pix_p import *
from param_pix_p import scan_bmp,avgPixelSpacing,dimpavx,dimpavy,volumeroifilep,dirpickleArch,modelArch,surfelemp
from param_pix_p import typei,typei1,typei2
from param_pix_p import white,yellow,red

from param_pix_p import lung_namebmp,jpegpath,lungmask,lungmask1
from param_pix_p import datacrossn,pathjs,fidclass,pxy
#from param_pix_p import classifc,classif,excluvisu,usedclassif
from param_pix_p import classifc,excluvisu


from param_pix_p import threeFileTop0,threeFileTop1,threeFileTop2,htmldir,source,dicomcross_merge
from param_pix_p import dicomfront,dicomcross,threeFile,threeFile3d,threeFileTxt,transbmp,threeFileMerge
from param_pix_p import threeFileTxtMerge,volcol,sroi,sroi3d,threeFileTxt3d,threeFileBot
from param_pix_p import predictout3d,jpegpath3d,predictout
from param_pix_p import jpegpadirm,dicompadirm,source_name,datafrontn,path_data,dirpickle,cwdtop

from param_pix_p import remove_folder,normi,rsliceNum,norm,maxproba
from param_pix_p import classifdict,usedclassifdict,oldFormat,derivedpatdict,layertokeep

#import copy
import time
from time import time as mytime
import numpy as np
#from numpy import argmax,amax
import os
import cv2
import dicom
import random

import shutil
from skimage import measure, morphology
from skimage.segmentation import clear_border
from skimage.morphology import  disk, binary_erosion, binary_closing
from skimage.filters import roberts
from skimage.measure import label,regionprops
from scipy import ndimage as ndi
from itertools import product
import cPickle as pickle

#import keras
from keras.models import load_model
from keras.models import model_from_json
#from keras.optimizers import Adam



t0=mytime()

def reshapeScan(tabscanScan,slnt,lissln,dimtabx,dimtaby):
    print 'reshape'
    tabscan = np.zeros((slnt,dimtabx,dimtaby), np.int16)
    for i in lissln:
                tabscan[i]=tabscanScan[i]

#    print tabscan.shape
    tabres=np.moveaxis(tabscan,0,1)
    return tabres

def reshapeScanl(tabscan):
    print 'reshape lung'
    tabres=np.moveaxis(tabscan,0,1)
    return tabres

def genebmp(fn,sou,nosource,centerHU, limitHU,tabscanroi,tabscanName={}):
    """generate patches from dicom files"""
    global picklein_file
    (top,tail) =os.path.split(fn)
    print ('load scan dicom files in:' ,tail)
    lislnn=[]
    fmbmp=os.path.join(fn,sou)
    fmbmproi=os.path.join(fn,sroi)
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
    dsr= RefDs.pixel_array
    dsr = dsr.astype('int16')
    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing

    imgresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
    dimtabx=imgresize.shape[0]
    dimtaby=imgresize.shape[1]
#    print dimtabx, dimtaby
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
#    tabscan = np.zeros((slnt,dimtabx,dimtaby), np.int16)
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
        
        dsr = dsr.astype('float32')
#        imgresize1=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
        imgresize1=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)

        imgresize = imgresize1.astype('int16')
        
        endnumslice=l.find('.dcm')
        imgcoreScan=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei1

        tabscanName[slicenumber]=imgcoreScan        
        tabscan[slicenumber]=imgresize.copy()

        np.putmask(imgresize,imgresize<lbHU,lbHU)
        np.putmask(imgresize,imgresize>lhHU,lhHU)
        imtowrite=normi(imgresize)

        imtowrite = cv2.cvtColor(imtowrite, cv2.COLOR_GRAY2RGB)
        imtowrite = cv2.cvtColor(imtowrite, cv2.COLOR_BGR2RGB)

        bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
        bmpfileroi=os.path.join(fmbmproi,imgcoreScan)
        (topw,tailw)=os.path.split(picklein_file)
        t2='Prototype '
        t1='param :'+tailw
        t0='CONFIDENTIAL'
        t3='Scan: '+str(slicenumber)

        t4=time.asctime()
        t5='CenterHU: '+str(int(centerHU))
        t6='LimitHU: +/-' +str(int(limitHU/2))
                
        anoted_image=tagviews(imtowrite,
                              t0,dimtabx-100,dimtaby-10,
                              t1,0,dimtaby-21,
                              t2,dimtabx-100,dimtaby-20,
                              t3,0,dimtaby-32,
                              t4,0,dimtaby-10,
                              t5,0,dimtaby-43,
                              t6,0,dimtaby-54) 
               
        tabscanroi[slicenumber]=anoted_image.copy()
        anoted_image=cv2.cvtColor(anoted_image,cv2.COLOR_BGR2RGB)
        cv2.imwrite(bmpfile,anoted_image)   
        cv2.imwrite(bmpfileroi,anoted_image)
        

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
    listbmp=[name for name in  os.listdir(fmbmpbmp) if name.lower().find(typei1)>0]  
    tabscan = np.zeros((slnt,dimtabx,dimtaby), np.uint8)
    if len(listbmp)>0:
        print 'lung scan exists in bmp'
        for img in listbmp:
            slicenumber= rsliceNum(img,'_','.'+typei1)
       
            imr=cv2.imread(os.path.join(fmbmpbmp,img),0) 
            imr=cv2.resize(imr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)  
            np.putmask(imr,imr>0,classif['lung']+1)
            tabscan[slicenumber]=imr
#            if slicenumber==12:
##            pickle.dump(dsr, open('predict.pkl', "wb" ),protocol=-1)
##            cv2.imwrite('predict.bmp',imtowrite)
##            print imtowrite.shape
#                scorer=pickle.load(open('C:/ProgramData/MedikEye/Score/modulepython/scorelung.pkl',"rb"))
#                diff=scorer-imr
    #            cv2.imwrite('diff.bmp',diff)
#                print diff.max()
#                print diff.min()
#                print diff.shape
    if len(listbmp)<slnt-1: 
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
                        segmented_lungs_fill[i]=get_segmented_lungs(tabscanScan[i], False)
    #            tabscanlung = np.zeros((slnt,dimtabx,dimtaby), np.uint8)
    #            for i in listsln:
    #                tabscan1[i]=tabscanScan[i]
    #            segmented_lungs_fill = segment_lung_mask(tabscan1, True)
    #            print segmented_lungs_fill.shape
                for i in listsln:
    #                tabscan[i]=normi(tabscan[i])
                    tabscan1[i]=morph(segmented_lungs_fill[i],13)
    #                imgcoreScan='lung_'+str(i)+'.'+typei
                    imgcoreScan=tabscanName[i]
                    bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
                    if tabscan[i].max()==0:
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
    size=1
    sizes=0.8

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
###            pickle.dump(dsr, open('predict.pkl', "wb" ),protocol=-1)
#                print tabscan[img].min(),tabscan[img].max()
##                cv2.imwrite('predict.bmp',normi(tabscan[img]))
##                print tabscanScan[img][0]
###            print imtowrite.shape
#                scorer=pickle.load(open('C:/ProgramData/MedikEye/Score/modulepython/scorepav.pkl',"rb"))
#                print scorer.min(),scorer.max()
#                diff=scorer-tabscan[img]
#    #            cv2.imwrite('diff.bmp',diff)
#                print diff.max()
#                print diff.min()
#                print diff.shape

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

#                            imagemax= cv2.countNonZero(imgray)
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
    # adding a singleton dimension and rescale to [0,1]
    pa = np.asarray(np.expand_dims(dataset_list, 1))
    # look if the predict source is empty
    # predict and store  classification and probabilities if not empty
    if X0 > 0:
        proba = model.predict_proba(pa, batch_size=500,verbose=1)

    else:
        print (' no patch in selected slice')
        proba = ()
    print 'number of patches', len(pa)

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
            t6='LimitHU: +/-' +str(int(limitHU/2))
            
            anoted_image=tagviews(imgresize8,t0,dimtabxn-100,dimtabyn-10,t1,0,dimtabyn-20,t2,dimtabxn-100,dimtabyn-20,
                         t3,0,dimtabyn-30,t4,0,dimtabyn-10,t5,0,dimtabyn-40,t6,0,dimtabyn-50) 
                    
#            t1='Pt: '+tail
            
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

def colorimage(image,color):
#    im=image.copy()
    im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    np.putmask(im,im>0,color)
    return im


def  visua(listelabelfinal,dirf,patch_list,dimtabx,dimtaby,
           slnt,predictout,sroi,scan_bmp,sou,dcmf,dct,errorfile,nosource,typevi):
    global thrproba,picklein_file,picklein_file_front
    tv=mytime()
    print('visualisation',typevi)
    rsuid=random.randint(0,1000)

    sliceok=[]
    (dptop,dptail)=os.path.split(dirf)
    predictout_dir = os.path.join(dirf, predictout)
#    print predictout_dir
    remove_folder(predictout_dir)
    os.mkdir(predictout_dir)

    for i in range (0,len(usedclassif)):
#        print 'visua dptail', topdir
        listelabelfinal[fidclass(i,classif)]=0
    #directory name with predict out dabasase, will be created in current directory
  
    dirpatientfdbsource1=os.path.join(dirf,sou)
         
    dirpatientfdbsource=os.path.join(dirpatientfdbsource1,scan_bmp)
    dirpatientfdbsroi=os.path.join(dirf,sroi)
    listbmpscan=os.listdir(dirpatientfdbsource)
    if dct:
        if nosource:
            dirpatientfdbsource1=dirf
        listdcm=[name for name in  os.listdir(dirpatientfdbsource1) if name.lower().find('.dcm')>0]

    listlabelf={}
    sroiE=False
    if os.path.exists(dirpatientfdbsroi):
        sroiE=True
        listbmpsroi=os.listdir(dirpatientfdbsroi)

    for img in listbmpscan:

        slicenumber= rsliceNum(img,'_','.'+typei)
        if slicenumber <0:
                    slicenumber=rsliceNum(img,'_','.'+typei1)
                    if slicenumber <0:
                          slicenumber=rsliceNum(img,'_','.'+typei2)
        if dct:
            for imgdcm in listdcm:
                 FilesDCM =(os.path.join(dirpatientfdbsource1,imgdcm))
                 RefDs = dicom.read_file(FilesDCM,force=True)
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
        foundroi=False
#        tablscan=np.zeros((dimtabx,dimtaby,3), np.uint8)
        if sroiE:
            for imgsroi in listbmpsroi:
                slicenumbersroi=rsliceNum(imgsroi,'_','.'+typei)
                if slicenumbersroi <0:
                    slicenumbersroi=rsliceNum(imgsroi,'_','.'+typei1)
                    if slicenumbersroi <0:
                         slicenumbersroi=rsliceNum(imgsroi,'_','.'+typei2)
                if slicenumbersroi==slicenumber:
                    imgc=os.path.join(dirpatientfdbsroi,imgsroi)
                    tablscan=cv2.imread(imgc,1)
                    foundroi=True
                    break
            if not foundroi:
                 imgc=os.path.join(dirpatientfdbsource,img)
                 tablscan=cv2.imread(imgc,1)
#                 tablscan=np.zeros((dimtabx,dimtaby,3), np.uint8)
        else:
            imgc=os.path.join(dirpatientfdbsource,img)
            tablscan=cv2.imread(imgc,1)
        tablscan=cv2.resize(tablscan,(dimtaby,dimtabx),interpolation=cv2.INTER_LINEAR)
        foundp=False
#        print slicenumber
        pn=patch_list[slicenumber]
        for ll in range(0,len(pn)):
#            slicename=patch_list[ll][0]
            xpat=pn[ll][0][0]
            ypat=pn[ll][0][1]
            proba=pn[ll][1]

            prec, mprobai = maxproba(proba)
            mproba=round(mprobai,2)
            classlabel=fidclass(prec,classif)
            classcolor=classifc[classlabel]
#            print xpat,ypat,mprobai,slicename
            if  classlabel not in excluvisu:
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
            if mprobai >=thrproba and classlabel not in excluvisu:
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

        tablscan = cv2.cvtColor(tablscan, cv2.COLOR_BGR2RGB)

        vis=drawContour(imgt,listlabel,dimtabx,dimtaby)
#        print tablscan.shape,vis.shape
        imn=cv2.add(tablscan,vis)

        if foundp:
            for ll in listlabelrec:
                delx=int(dimtaby*0.6-120)
                imn=tagviewn(imn,ll,str(listlabelaverage[ll]),listlabelrec[ll],delx,0)
            t0='average probability'
        else:
#                errorfile.write('no recognised label in: '+str(dptail)+' '+str (img)+'\n' )
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
#        errorfile.write('\n'+'number of labels in :'+str(dptop)+' '+str(dptail)+str (img)+'\n' )
#    print listlabelf
    errorfile.write('type :'+typevi+'\n')
    for classlabel in listlabelf:
          listelabelfinal[classlabel]=listlabelf[classlabel]
          print 'patient: ',dptail,', label:',classlabel,': ',listlabelf[classlabel]
          stringtw=dptail+': '+str(classlabel)+': '+str(listlabelf[classlabel])+'\n'
#          print string
          errorfile.write(stringtw )
    print "visua time:",round(mytime()-tv,3),"s"
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
#            BGx=str(dimpavx)
#            BGy=str(dimpavx)
#            BGz=str(round(pz,3))

        if v =='merge':
            threeFilel=dptail+'_'+threeFileMerge
            threeFileTxtl=dptail+'_'+threeFileTxtMerge
#            jsfile=dptail+'_'+threeFilejsMerge
#            BGx=str(dimpavx)
#            BGy=str(dimpavx)
#            BGz=str(round(pz,3))


        if v =='front':
            threeFilel=dptail+'_'+threeFile3d
            threeFileTxtl=dptail+'_'+threeFileTxt3d
#            jsfile=dptail+'_'+threeFilejs3d
#            BGy=str(dimpavx)
#            BGx=str(round(pz,3))
#            BGz=str(dimpavx)

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
#        volumefileT.write('camera.position.set(0 '+', -'+BGx+', 0 );\n')
        volumefileT.write('camera.position.set(0 '+', -3, 0 );\n')


#        volumefileT.write( 'var boxGeometry = new THREE.BoxGeometry( '+BGx+' , '+\
#        BGy+' , '+BGz+' ) ;\n\n')
        
        volumefileT.write( 'var boxGeometry = new THREE.BoxGeometry( 3,3,3)  ;\n\n')
        
        volumefileT.write('var voxels = [\n')
#        volumefilejs.write( 'var boxGeometry = new THREE.BoxGeometry( '+BGx+' , '+\
#        BGy+' , '+BGz+' ) ;\n')
#
#        volumefile.write( 'var boxGeometry = new THREE.BoxGeometry( '+BGx+' , '+\
#        BGy+' , '+BGz+' ) ;\n')
        volumefile.write( 'var boxGeometry = new THREE.BoxGeometry( 3,3,3 ) ;\n')

#        print 'pz',pz
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
    



def genecross(proba_cross,proba_front,patch_list_front,dimtabx,dimtaby):
    """generate cross view from front patches"""
    print 'genecross'
    global thrprobaUIP
#    (dptop,dptail)=os.path.split(dirf)

    plf=patch_list_front
    prf=proba_front
    plfd={}
    prfd={}
    tabpatch={}

    listp=[name for name in usedclassif if name not in excluvisu]
#    print 'used patterns :',listp

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
    deltay=10*(labnow%5)
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
#    print 'used patterns :',listp
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
        if slicenumber <0:
                    slicenumber= rsliceNum(img,'_','.'+typei1)
                    if slicenumber <0:
                        slicenumber= rsliceNum(img,'_','.'+typei2)

        if sroiE:
            for imgsroi in listbmpsroi:
                slicenumbersroi=rsliceNum(imgsroi,'_','.'+typei)
                if slicenumbersroi <0:
                    slicenumbersroi=rsliceNum(imgsroi,'_','.'+typei1)
                    if slicenumbersroi <0:
                        slicenumbersroi=rsliceNum(imgsroi,'_','.'+typei2)
                if slicenumbersroi==slicenumber:
                    imgc=os.path.join(dirpatientfsdb,imgsroi)
                    break
        else:
            imgc=os.path.join(dirpatientfdb,img)
            
        tabint=cv2.imread(imgc,1)  
        tabint=cv2.resize(tabint,(dimtaby,dimtaby),interpolation=cv2.INTER_LINEAR)
            
        tablscan[slicenumber]=tabint
        
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
                pf=os.path.join(predictout_dir,'tr_'+str(scan)+'.'+typei1)
                imcolor = cv2.cvtColor(impre[scan], cv2.COLOR_BGR2RGB)
                imn=cv2.add(imcolor,tablscan[scan])
                imnc[scan]=imn
                cv2.imwrite(pf,imn)

                for imgdcm in listdcm:
                 FilesDCM =(os.path.join(dirpatientfdb1,imgdcm))
                 RefDs = dicom.read_file(FilesDCM,force=True)
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



def genepatchlistslice(patch_list_cross,proba_cross,lissln,subpleurmask,thrpatch):
#    print 'start genepatchilist'
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

def drawcontours2(im,pat,dimtabx,dimtaby):
#    print 'contour',pat
    imgray = np.copy(im)
    ret,thresh = cv2.threshold(imgray,10,255,0)
    _,contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2 = np.zeros((dimtabx,dimtaby,3), np.uint8)
    cv2.drawContours(im2,contours,-1,classifc[pat],1)
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

#        if i == 12 :
#            if tab1.max()>0:
#        cv2.imshow(pat1+'tab1',normi(tab1))
#        cv2.imshow(pat2+'tab2',normi(tab2))
#        cv2.imshow(pat+'tab3',normi(tab3))
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()

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
            
#            if i == 12 :
#            cv2.imshow('tabf',normi(tabroi[i])) 
#            cv2.imwrite('a.bmp',10*(tabroi[i])) 
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()

    return tabroi

def generoi(dirf,tabroi,dimtabx,tabscanLung,slnroi,dirroit,tabscanroi,tabscanName,slnt):
    tabroipat={}    
#    listroi={}
    for pat in usedclassif:
        tabroipat[pat]=np.zeros((slnt,dimtabx,dimtabx),np.uint8)
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
            tabroi[numslice]+=tab

    for pat in derivedpat:    
            tabroi=calnewpat(pat,slnroi,tabroipat,tabroi)
    for numslice in slnroi:
        for pat in layertokeep:
            tab=np.copy(tabroipat[pat][numslice])
            np.putmask(tabroi[numslice], tab > 0, 0)        
            tabroi[numslice]+=tab
            
    for numslice in slnroi:
        tablung=np.copy(tabscanLung[numslice])        
        maskRoi=tabroi[numslice].copy()
        if tablung.max()==0 and maskRoi.max()!=0:
            print ('no lung for: '+str(numslice))
        tabxorig=tabroi[numslice].copy()
        np.putmask(maskRoi,maskRoi>0,255)
        maskRoin=np.bitwise_not(maskRoi)        
        tablung=np.bitwise_and(tablung, maskRoin)
        np.putmask(tablung,tablung>0,classif['healthy']+1)
        tabroi[numslice]=np.bitwise_or(tabxorig,tablung)
       
    slnroi.sort()               
    volumeroi={}
    for numslice in slnroi:
            imgcoreScan=tabscanName[numslice]
            roibmpfile=os.path.join(dirroit,imgcoreScan)
            anoted_image=tabscanroi[numslice]
#            anoted_image=cv2.cvtColor(anoted_image,cv2.COLOR_BGR2RGB)
            volumeroi[numslice]={}
            for pat in classif:
                img=np.copy(tabroi[numslice])
                if img.max()>0:                    
                    np.putmask(img, img !=classif[pat]+1, 0)
                    np.putmask(img, img ==classif[pat]+1, 1)
                    area= img.sum()* surfelemp  
                    volumeroi[numslice][pat]=area 
                    if area>0:
           
                        np.putmask(img, img >0, 100)
                        ctkey=drawcontours2(img,pat,dimtabx,dimtabx)
                        ctkeym=ctkey.copy()
                        ctkeym=cv2.cvtColor(ctkeym,cv2.COLOR_RGB2GRAY)
                        ctkeym=cv2.cvtColor(ctkeym,cv2.COLOR_GRAY2RGB)                       
                        np.putmask(anoted_image, ctkeym >0, 0)
                        anoted_image=cv2.add(anoted_image,ctkey)
                        anoted_image=tagviewct(anoted_image,pat,200,10) 
                        
                else:
                    volumeroi[numslice][pat]=0
            anoted_image= cv2.cvtColor(anoted_image,cv2.COLOR_RGB2BGR)  
#            print roibmpfile
            cv2.imwrite(roibmpfile,anoted_image)
    return tabroi,volumeroi,slnroi
        

def predictrun(indata,path_patient):
        global thrpatch,thrproba,thrprobaUIP,subErosion
        global  picklein_file,picklein_file_front,classif,usedclassif,derivedpat
        td=False
#        time.sleep(10)
#        return 'rrr'
#        print path_patient
#        print indata
#        ooo
        print '-------------------'
        if indata['threedpredictrequest']=='Cross Only':
           td=False
           
           print 'CROSS PREDICT'
        else:
            td=True
            print 'CROSS +FRONT PREDICT'
        thrproba=float(indata['thrproba'])
        thrprobaUIP=float(indata['thrprobaUIP'])
        thrpatch=float(indata['thrpatch'])
        picklein_filet=indata['picklein_file']
        centerHU=indata['centerHU']
        limitHU=indata['limitHU']
        
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
        subErosion=indata['subErosion']
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
                    listHug.append(lpt[0:pos])
        picklein_file =  os.path.join(dirpickle,picklein_filet)
        picklein_file_front =  os.path.join(dirpickle,picklein_file_frontt)

        dirHUG=os.path.join(cwdtop,path_patient)

        for f in listHug:
            print '------------------'
            print 'work on patient',f, 'set',setref,'thrproba',thrproba,'thrpatch',thrpatch
            lungSegment={}
            patch_list_cross_slice={}
            
            listelabelfinal={}
            dirf=os.path.join(dirHUG,f)
            wridirsource=os.path.join(dirf,source)
           
            listdcm= [name for name in os.listdir(dirf) if name.find('.dcm')>0]
            nosource=False
            if len(listdcm)>0:
                nosource=True
                if not os.path.exists(wridirsource):
                    os.mkdir(wridirsource)
           
            tabMed = {}  # dictionary with position of median between lung
        
            wridir=os.path.join(wridirsource,transbmp)
            remove_folder(wridir)
            os.mkdir(wridir)
#            """
            path_data_write=os.path.join(dirf,path_data)
            
#            remove_folder(path_data_write)
            if not os.path.exists(path_data_write):
                os.mkdir(path_data_write)

            eferror=os.path.join(path_data_write,'log.txt')
            errorfile = open(eferror, "w")
#            """
            jpegpathdir=os.path.join(dirf,jpegpadirm)
            remove_folder(jpegpathdir)
            os.mkdir(jpegpathdir)
#            """
            dicompathdir=os.path.join(dirf,dicompadirm)
            
            dirroit=os.path.join(dirf,sroi)
            if not os.path.exists(dirroit):
                os.mkdir(dirroit)
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
#            path_data_writefile=os.path.join(path_data_write,volumeroifilep)
                       
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

            print '------------------'
            print 'START PREDICT CROSS'
            print '------------------'
            crosscompleted=False
            pickle.dump(crosscompleted, open( os.path.join(path_data_write,"crosscompleted"), "wb" ),protocol=-1)
            print 'source',source

            tabscanroi={}
#            return ''
            tabscanName={}
            tabscanScan,slnt,dimtabx,slicepitch,lissln,tabscanroi,tabscanName=genebmp(dirf,source,nosource, 
                                                        centerHU, limitHU,tabscanroi,tabscanName)
            tabscanLung,tabrange=genebmplung(dirf,lungmaski,slnt,dimtabx,dimtabx,tabscanScan,lissln,tabscanName)
            lungSegment=selectposition(lissln,tabrange)
            datacross=(slnt,dimtabx,dimtabx,slicepitch,lissln,setref)
            
            pickle.dump(datacross, open( os.path.join(path_data_write,datacrossn), "wb" ),protocol=-1)
            pickle.dump(tabscanLung, open( os.path.join(path_data_write,"tabscanLung"), "wb" ),protocol=-1)
            pickle.dump(lungSegment, open( os.path.join(path_data_write,"lungSegment"), "wb" ),protocol=-1)

            slnroi=[]
            tabroi=np.zeros((slnt,dimtabx,dimtabx), np.uint8) 
            tabroi,volumeroi,slnroi=generoi(dirf,tabroi,dimtabx,tabscanLung,
                                            slnroi,dirroit,tabscanroi,tabscanName,slnt)
            pickle.dump(volumeroi, open(os.path.join(path_data_write,volumeroifilep), "wb" ),protocol=-1)
            pickle.dump(tabroi, open( os.path.join(path_data_write,"tabroi"), "wb" ),protocol=-1)
            pickle.dump(slnroi, open( os.path.join(path_data_write,"slnroi"), "wb" ),protocol=-1)
                     
            """        
            slnt=datacross[0]
            dimtabx=datacross[1]
            dimtaby=datacross[2]
            slicepitch=datacross[3]
            lissln=datacross[4]
#            """          
            subpleurmask=subpleural(dirf,tabscanLung,lissln,subErosion,'cross')

            regene=True
            if os.path.exists(os.path.join(path_data_write,"thrpatch")):
                    thrpatch1= pickle.load( open( os.path.join(path_data_write,"thrpatch"), "rb" ))
                    if thrpatch == thrpatch1:
                        if os.path.exists(os.path.join(path_data_write,"p_patch_list_cross")):
                            regene=False
                            print 'no need to regenerate patch list'
                            patch_list_cross= pickle.load( open( os.path.join(path_data_write,"p_patch_list_cross"), "rb" ))
                  
            if regene:
                patch_list_cross=pavgene(dirf,dimtabx,dimtabx,tabscanScan,tabscanLung,slnt,jpegpath,lissln)
                pickle.dump(patch_list_cross, open( os.path.join(path_data_write,"patch_list_cross"), "wb" ),protocol=-1)
                pickle.dump(thrpatch, open( os.path.join(path_data_write,"thrpatch"), "wb" ),protocol=-1)

            
            modelcross=modelCompilation('cross',picklein_file,picklein_file_front,setref)
            proba_cross=ILDCNNpredict(patch_list_cross,modelcross)
            patch_list_cross_slice,patch_list_cross_slice_sub=genepatchlistslice(patch_list_cross,
                                                            proba_cross,lissln,subpleurmask,thrpatch)
            tabMed = calcMed(tabscanLung,lissln)


            pickle.dump(patch_list_cross_slice, open( os.path.join(path_data_write,"patch_list_cross_slice"), "wb" ),protocol=-1)
            pickle.dump(patch_list_cross_slice_sub, open( os.path.join(path_data_write,"patch_list_cross_slice_sub"), "wb" ),protocol=-1)

            pickle.dump(tabMed, open( os.path.join(path_data_write,"tabMed"), "wb" ),protocol=-1)

            if not wvisu:
                visua(listelabelfinal,dirf,patch_list_cross_slice,dimtabx,
                  dimtabx,slnt,predictout,sroi,scan_bmp,source,dicompathdircross,True,errorfile,nosource,'cross')            
            genethreef(dirf,patch_list_cross,proba_cross,slicepitch,dimtabx,dimtabx,dimpavx,slnt,'cross')
            crosscompleted=True
            pickle.dump(crosscompleted, open( os.path.join(path_data_write,"crosscompleted"), "wb" ),protocol=-1)
#            """
    ###       cross
            if td:
#                """
                print 'START PREDICT FRONT'
                print '------------------'
                frontcompleted=False
                pickle.dump(frontcompleted, open( os.path.join(path_data_write,"frontcompleted"), "wb" ),protocol=-1)
                tabresScan=reshapeScanl(tabscanScan)
                dimtabxn,dimtabyn,tabScan3d,lisslnfront=wtebres(wridir,dirf,tabresScan,
                                                                dimtabx,slicepitch,lungmaski,'scan',centerHU,limitHU)
                tabresLung=reshapeScanl(tabscanLung)   
                dimtabxn,dimtabyn,tabLung3d,a=wtebres(wridir,dirf,tabresLung,dimtabx,slicepitch,
                                                      lungmaski,'lung',centerHU,limitHU)

                datafront=(dimtabx,dimtabxn,dimtabyn,slicepitch,lisslnfront)
                
                pickle.dump(datafront, open( os.path.join(path_data_write,datafrontn), "wb" ),protocol=-1)
                pickle.dump(tabLung3d, open( os.path.join(path_data_write,"tabLung3d"), "wb" ),protocol=-1)
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
                    thrpatch1= pickle.load( open( os.path.join(path_data_write,"thrpatch"), "rb" ))
                    if thrpatch == thrpatch1:
                        if os.path.exists(os.path.join(path_data_write,"p_patch_list_front")):
                            regene=False
                            print 'no need to regenerate patch list'
                            patch_list_front= pickle.load( open( os.path.join(path_data_write,"p_patch_list_front"), "rb" ))
                if regene:
                    patch_list_front=pavgenefront(dirf,dimtabxn,dimtabx,tabScan3d,tabLung3d,dimtabyn,jpegpath3d)
                    pickle.dump(patch_list_front, open( os.path.join(path_data_write,"patch_list_front"), "wb" ),protocol=-1)
                
                modelfront=modelCompilation('front',picklein_file,picklein_file_front,setref)

                proba_front=ILDCNNpredict(patch_list_front,modelfront)
                
                subpleurmaskfront=subpleural(dirf,tabLung3d,lisslnfront,subErosion,'front')
                patch_list_front_slice,patch_list_front_slice_sub=genepatchlistslice(patch_list_front,
                                                            proba_front,lisslnfront,subpleurmaskfront,thrpatch)

                if not wvisu:
                    visua(listelabelfinal,dirf,patch_list_front_slice,dimtabxn,dimtabx,
                      dimtabyn,predictout3d,sroi3d,transbmp,source,dicompathdirfront,False,errorfile,nosource,'front')
#                tabMedfront = calcMed(tabLung3d,lisslnfront)
                
#                pickle.dump(tabMedfront, open( os.path.join(path_data_write,"tabMedfront"), "wb" ),protocol=-1)
                pickle.dump(patch_list_front_slice, open( os.path.join(path_data_write,"patch_list_front_slice"), "wb" ),protocol=-1)
                pickle.dump(patch_list_front_slice_sub, open( os.path.join(path_data_write,"patch_list_front_slice_sub"), "wb" ),protocol=-1)
#                pickle.dump(lungSegmentfront, open( os.path.join(path_data_write,"lungSegmentfront"), "wb" ),protocol=-1)
                """
                tabMedfront=pickle.load(open( os.path.join(path_data_write,"tabMedfront"), "rb" ))
                patch_list_front_slice=pickle.load(open( os.path.join(path_data_write,"patch_list_front_slice"), "rb" ))
                patch_list_front_slice_sub=pickle.load(open( os.path.join(path_data_write,"patch_list_front_slice_sub"), "rb" ))
                lungSegmentfront=pickle.load(open( os.path.join(path_data_write,"lungSegmentfront"), "rb" ))
                """
                genethreef(dirf,patch_list_front,proba_front,avgPixelSpacing,dimtabxn,dimtabyn,dimpavx,dimtabx,'front')
#                tabpx=genecross(proba_cross,proba_front,patch_list_front,dimtabxn,dimtabyn)
                
                
                proba_cross_from_front,patch_list_cross_from_front= genecrossfromfront(proba_front,patch_list_front,
                                                                   dimtabx,lissln,dimtabxn,slnt)
                
                patch_list_cross_slice_from_front,patch_list_cross_slice_sub_from_front=genepatchlistslice(patch_list_cross_from_front,
                                                            proba_cross_from_front,lissln,subpleurmask,thrpatch)
                
                pickle.dump(patch_list_cross_slice_from_front, open( os.path.join(path_data_write,"patch_list_cross_slice_from_front"), "wb" ),protocol=-1)
                pickle.dump(patch_list_cross_slice_sub_from_front, open( os.path.join(path_data_write,"patch_list_cross_slice_sub_from_front"), "wb" ),protocol=-1)

                
                
                """
        #        pickle.dump(tabpx, open( os.path.join(path_data_write,"tabpx"), "wb" ),protocol=-1)
        #        tabpx=pickle.load(open( os.path.join(path_data_write,"tabpx"), "rb" ),protocol=-1)
                tabx,tabfromfront=reshapepatern(dirf,tabpx,dimtabxn,dimtabx,slnt,slicepitch,predictout3d1,source,dicompathdirfront)
#                pickle.dump(tabfromfront, open( os.path.join(path_data_write,"tabfromfront"), "wb" ),protocol=-1)
        #        tabx=pickle.load(open( os.path.join(path_data_write,"tabx"), "rb" ),protocol=-1)
#                    print 'before merge proba'
                """
#                print '0',patch_list_cross_slice
#                print 'len',len(patch_list_cross_slice)
                proba_merge,patch_list_merge=mergeproba(patch_list_cross_slice,
                                                        patch_list_cross_slice_from_front,slnt,dimtabx,dimtabx)
                patch_list_merge_slice,patch_list_merge_slice_sub=genepatchlistslice(patch_list_merge,
                                                            proba_merge,lissln,subpleurmask,thrpatch)
                
                
#                pickle.dump(proba_merge, open( os.path.join(path_data_write,"proba_merge"), "wb" ),protocol=-1)
#                pickle.dump(patch_list_merge, open( os.path.join(path_data_write,"patch_list_merge"), "wb" ),protocol=-1)
                pickle.dump(patch_list_merge_slice, open( os.path.join(path_data_write,"patch_list_merge_slice"), "wb" ),protocol=-1)
                pickle.dump(patch_list_merge_slice_sub, open( os.path.join(path_data_write,"patch_list_merge_slice_sub"), "wb" ),protocol=-1)
            
                """
                proba_merge=pickle.load(open( os.path.join(path_data_write,"proba_merge"), "rb" ))
                patch_list_merge=pickle.load(open( os.path.join(path_data_write,"patch_list_merge"), "rb" ))
                patch_list_merge_slice=pickle.load(open( os.path.join(path_data_write,"patch_list_merge_slice"), "rb" ))
                patch_list_merge_slice_sub=pickle.load(open( os.path.join(path_data_write,"patch_list_merge_slice_sub"), "rb" ))
                """                                              
                """
                if not wvisu:
                    visua(listelabelfinal,dirf,patch_list_merge_slice,dimtabx,dimtabx
                      ,slnt,predictoutmerge,sroi,scan_bmp,source,dicompathdirmerge,True,errorfile,nosource,'merge')
                genethreef(dirf,patch_list_merge,proba_merge,slicepitch,dimtabx,dimtabx,dimpavx,slnt,'merge')
                """
            errorfile.write('completed :'+f)
            errorfile.close()
            frontcompleted=False
            pickle.dump(frontcompleted, open( os.path.join(path_data_write,"frontcompleted"), "wb" ),protocol=-1)
            print 'PREDICT  COMPLETED  for ',f,'thrproba',thrproba,'thrpatch',thrpatch
            print '------------------'
        return''


#errorfile.close()
        print "predict time:",round(mytime()-t0,3),"s"

#ILDCNNpredict(bglist)