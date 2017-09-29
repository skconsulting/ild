# coding: utf-8
#sylvain Kritter 04-Apr-2017
'''predict on lung scan front view and cross view
@author: sylvain Kritter 

version 1.5
6 September 2017
'''
#from param_pix_p import *
from param_pix_s2 import scan_bmp,dimpavx,dimpavy,dirpickleArch,modelArch
from param_pix_s2 import typei,typei1
from param_pix_s2 import white,yellow

from param_pix_s2 import lung_namebmp,lungmask,lungmask1
from param_pix_s2 import fidclass,pxy
#from param_pix_p import classifc,classif,excluvisu,usedclassif
from param_pix_s2 import classifc


from param_pix_s2 import  source, dimtabx,dimtaby
from param_pix_s2 import  transbmp
from param_pix_s2 import sroi
from param_pix_s2 import jpegpath3d
from param_pix_s2 import jpegpadirm,source_name,path_data,dirpickle,cwdtop

from param_pix_s2 import remove_folder,normi,rsliceNum,norm,maxproba
from param_pix_s2 import classifdict,usedclassifdict,oldFormat

#from param_pix2 import get_model

import time
from time import time as mytime
import numpy as np
#from numpy import argmax,amax
import os
import cv2
import dicom

from skimage import measure
import cPickle as pickle
#import matplotlib.pyplot as plt
#import keras
from keras.models import load_model
from keras.models import model_from_json
#from keras import backend as K
#K.set_image_dim_ordering('tf')
#from keras.optimizers import Adam



t0=mytime()

#def reshapeScan(tabscanScan,slnt,lissln,dimtabx,dimtaby):
#    print 'reshape'
#    tabscan = np.zeros((slnt,dimtabx,dimtaby), np.int16)
#    for i in lissln:
#                tabscan[i]=tabscanScan[i][1]
#
##    print tabscan.shape
#    tabres=np.moveaxis(tabscan,0,1)
#    return tabres

def reshapeScanl(tabscan):
    print 'reshape lung'
    tabres=np.moveaxis(tabscan,0,1)
    return tabres

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
    dsr= RefDs.pixel_array
    dsr = dsr.astype('int16')
    PixelSpacing=float(RefDs.PixelSpacing[0])

    imgresize=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)
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
    tabscan = np.zeros((slnt,dimtaby,dimtabx),np.int16)
#    for i in range(slnt):
#        tabscan[i] = []
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
        dsr = dsr.astype('int16')
        
        imgresize=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)
        
        
        endnumslice=l.find('.dcm')
        imgcoreScan=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei1  
        
        
        tabscan[slicenumber]=imgresize
        np.putmask(imgresize,imgresize<lbHU,lbHU)
        np.putmask(imgresize,imgresize>lhHU,lhHU)
        
        
        imtowrite=normi(imgresize)
        imtowrite = cv2.cvtColor(imtowrite, cv2.COLOR_GRAY2RGB)

#        bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
        tabscanName[slicenumber]=imgcoreScan
        (topw,tailw)=os.path.split(picklein_file)
        t2='Prototype '
        t1='param :'+tailw
        t0='CONFIDENTIAL'
        t3='Scan: '+str(slicenumber)

        t4=time.asctime()
        t5='CenterHU: '+str(int(centerHU))
        t6='LimitHU: +/-' +str(int(limitHU/2))
                
        anoted_image=tagviews(imtowrite,t0,0,10,t1,0,dimtaby-20,t2,0,20,
                     t3,0,dimtaby-30,t4,0,dimtaby-10,t5,0,dimtaby-40,t6,0,dimtaby-50)        
        tabscanroi[slicenumber]=anoted_image

    return tabscan,slnt,slicepitch,lislnn,tabscanroi,tabscanName,PixelSpacing

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -350, dtype=np.int8)+1#init 320, or 350
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
    img[img>0]=classif['lung']
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
    listbmp= os.listdir(fmbmpbmp) 
    tabscan = np.zeros((slnt,dimtaby,dimtabx), np.uint8)
    if len(listbmp)>0:
        print 'lung scan exists in bmp'
        for img in listbmp:
            slicenumber= rsliceNum(img,'_','.'+typei1)
            imr=cv2.imread(os.path.join(fmbmpbmp,img),0) 
            if imr.max()>0: 
                imr=cv2.resize(imr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)
                np.putmask(imr,imr>0,classif['lung']+1)
           
                tabscan[slicenumber]=imr
    
    if len(listdcm)>0:  
        print 'lung scan exists in dcm'           
        for l in listdcm:
            FilesDCM =(os.path.join(fmbmp,l))
            RefDs = dicom.read_file(FilesDCM,force=True)    
            dsr= RefDs.pixel_array
            dsr=normi(dsr)    
            imgresize=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)
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
            tabscan1 = np.zeros((slnt,dimtaby,dimtabx), np.int16)

            segmented_lungs_fill = segment_lung_mask(tabscanScan, True)
#            print segmented_lungs_fill.shape
            for i in listsln:
                
                tabscan1[i]=morph(segmented_lungs_fill[i],13)
#                if i ==200:
#                    cv2.imshow(str(i),normi(tabscan1[i]))

                imgcoreScan=tabscanName[i]
                bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
                if tabscan[i].max()==0:
                    tabscan[i]=tabscan1[i]
                    colorlung=colorimage(tabscan[i],classifc['lung'])
                    cv2.imwrite(bmpfile,colorlung)
  
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    col=yellow

    viseg=cv2.putText(tab,t0,(x0, y0), font,0.3,col,1)
    viseg=cv2.putText(viseg,t1,(x1, y1), font,0.4,col,1)
    viseg=cv2.putText(viseg,t2,(x2, y2), font,0.3,col,1)

    viseg=cv2.putText(viseg,t3,(x3, y3), font,0.4,col,1)
    viseg=cv2.putText(viseg,t4,(x4, y4), font,0.4,col,1)
    viseg=cv2.putText(viseg,t5,(x5, y5), font,0.4,col,1)
    viseg=cv2.putText(viseg,t6,(x6, y6), font,0.4,col,1)

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


def ILDCNNpredict(X_predict,model):
    print ('Predict started ....')
    
    
    X0=X_predict.shape[0]
    # adding a singleton dimension and rescale to [0,1
    if X0 > 0:
#        proba = model.predict_proba(pa, batch_size=500,verbose=1)
        proba = model.predict(X_predict, verbose=1,batch_size=1)

    else:
        print (' no data in selected slice')
        proba = []
    print 'number of  predicted', proba.shape[0]

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
    fxs=float(slicepitch/PixelSpacing )
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
                    
#            t1='Pt: '+tail
            
#            imgresize8r=tagviews(imgresize8,t0,0,10,t1,0,20,t2,(dimtabyn/2)-10,dimtabxn-10,t3,0,38,t4,0,dimtabxn-10,t5,0,dimtabxn-20)
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
    im2=cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
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
    labnow=classif[label]
#  
    deltay=10*(labnow)
    deltax=100*(labnow/5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,label,(x+deltax, y+deltay), font,0.3,col,1)
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
        subErosionPixel = int(round(2 * subErosion / PixelSpacing))
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
                tabpatch = np.zeros((dimtabx, dimtaby), np.uint8)
    
                tabpatch[ypat:ypat + dimpavy, xpat:xpat + dimpavx] = 1
#                tabsubpl = np.bitwise_and(subpleurmask[i], tabpatch)
#                np.putmask(tabsubpl, tabsubpl > 0, 1)
#
#                area = tabsubpl.sum()
#                targ = float(area) / pxy    
#
#                if targ > thrpatch:
#                        ressub[i].append(t)  
            ii+=1                   
    return res

def generoi(dirf,tabroi,dimtabx,dimtaby,slnroi,tabscanName,dirroit,tabscanroi,tabscanLung,PixelSpacing):
    (top,tail)=os.path.split(dirf)

    for pat in usedclassif:
        if pat !='back_ground':
#        print pat
            pathroi=os.path.join(dirf,pat)
            if os.path.exists(pathroi):
        
                lroi=[name for name in os.listdir(pathroi) if name.find('.'+typei1)>0 ]            
                for s in lroi:
                    numslice=rsliceNum(s,'_','.'+typei1)
              
                    img=cv2.imread(os.path.join(pathroi,s),0)
    #                print os.path.join(pathroi,s)
    #                print img.shape
                    img=cv2.resize(img,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)
                    np.putmask(tabroi[numslice], img > 0, 0)
    #                if classif[pat]>0:
                    np.putmask(img, img > 0, classif[pat])
    #                else:
    #                    np.putmask(img, img > 0, classif['lung'])
                    tablung=np.copy(tabscanLung[numslice])
                    np.putmask(tablung,tablung>0,255)                      
                    img=np.bitwise_and(tablung, img)  
                    tabroi[numslice]+=img
                    np.putmask(tablung,tablung>0,classif['healthy']) 
                    tabroii=np.copy(tabroi[numslice])
                    np.putmask(tabroii,tabroii>0,255) 
                    mask=np.bitwise_not(tabroii)
                    img=np.bitwise_and(tablung, mask)
                    tabroi[numslice]=np.bitwise_or(img,tabroi[numslice])
                    
#                    
#                    
#                    cv2.imshow(str(numslice)+pat,normi(tabroi[numslice]))
#                    cv2.waitKey(0)
#                    cv2.destroyAllWindows()
                    if numslice not in slnroi:
                        slnroi.append(numslice)  
#    cv2.imshow('e',normi(tabroi[13]))
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
    
    slnroi.sort()
    volumeroi={}

    surfelemp=PixelSpacing*PixelSpacing # for 1 pixel in mm2
    surfelem= surfelemp/100 #surface of 1 pixel in cm2
    for numslice in slnroi:
            imgcoreScan=tabscanName[numslice]
            roibmpfile=os.path.join(dirroit,imgcoreScan)
            anoted_image=tabscanroi[numslice]
            anoted_image=cv2.cvtColor(anoted_image,cv2.COLOR_BGR2RGB)
            
            volumeroi[numslice]={}
            for pat in classif:
                img=np.copy(tabroi[numslice])
                if img.max()>0:                    
                    if classif[pat]>0:
                        np.putmask(img, img !=classif[pat], 0)
                        np.putmask(img, img ==classif[pat], 1)
                    else:
                        np.putmask(img, img !=0, 2)
                        np.putmask(img, img ==0, 1)
                        np.putmask(img, img ==2, 0)
    #                if  numslice==225:
    #                    cv2.imshow(pat+'143',normi(img))
                    
                    area= img.sum()* surfelem /100
                    volumeroi[numslice][pat]=area  
                    if area>0:
#                        print 'write'+pat+roibmpfile
                        np.putmask(img, img >0, 100)
                        ctkey=drawcontours2(img,pat,dimtabx,dimtaby)    
                        anoted_image=cv2.add(anoted_image,ctkey)
                        cv2.imwrite(roibmpfile,anoted_image)
                                       
                else:
                    volumeroi[numslice][pat]=0                    
                      
    return tabroi,volumeroi,slnroi

def preparscan(tabscan,tabslung,numsliceok):
#    (top,tail)=os.path.split(namedirtopcf)
    xtrains={}
    scan_list=[]
    i=-1
    for num in numsliceok:
        i+=1
        tabl=tabslung[num].copy()
        scan=tabscan[num].copy()

#        print scan.min(),scan.max()
        np.putmask(tabl,tabl>0,255)
        taba=cv2.bitwise_and(scan,scan,mask=tabl)
        np.putmask(tabl,tabl>0,1)
        tablc=tabl.astype(np.int16)

        np.putmask(tablc,tablc==0,-1000)
        np.putmask(tablc,tablc==1,0)
#        taba=cv2.bitwise_and(scan,scan,mask=tabl)
        
        tabab=cv2.bitwise_or(taba,tablc) 
        tababn=norm(tabab)
#        if num==12:
#            print type(taba[0][0]),taba.min(),taba.max()
#            print type(tablc[0][0]),tablc.min(),tablc.max()
#            cv2.imwrite('scan.bmp',normi(scan))
#            cv2.imwrite('tabl.bmp',normi(tabl))
#            cv2.imwrite('taba.bmp',normi(taba))
#            cv2.imwrite('tablc.bmp',normi(tablc))
#            cv2.imwrite('tabab.bmp',normi(tabab))
#            cv2.imwrite('tababn.bmp',normi(tababn))
#            ooo
        
#        cv2.imshow('tababn',normi(tababn))
###            cv2.imshow('tabroif',f)
###            cv2.imwrite('a.bmp',o)
###            cv2.imwrite('b.bmp',roif)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        xtrains[num]=i
        scan_list.append(tababn)
        
#        print tabab.min(),tabab.max()

    X_train = np.asarray(np.expand_dims(scan_list,3))
#    print len(scan_list)
    return X_train,xtrains



def predictrun(indata,path_patient):
        global thrproba,thrprobaUIP,subErosion,PixelSpacing
        global  picklein_file,picklein_file_front,classif,usedclassif
        td=False

        print '-------------------'
        if indata['threedpredictrequest']=='Cross Only':
           td=False           
           print 'CROSS PREDICT'
        else:
            td=True
            print 'CROSS +FRONT PREDICT'
        thrproba=float(indata['thrproba'])
 
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

        picklein_file_frontt=indata['picklein_file_front']
#        subErosion=indata['subErosion']
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
#        model=loadmodel()
        modelcross=modelCompilation('cross',picklein_file,picklein_file_front,setref)

        if indata['threedpredictrequest']!='Cross Only':
            modelfront=modelCompilation('front',picklein_file,picklein_file_front,setref)
            
        for f in listHug:
            print '------------------'
            print 'work on patient',f,' set:',setref,picklein_filet,'thrproba:',thrproba
            patch_list_cross_slice={}
            
#            listelabelfinal={}
            dirf=os.path.join(dirHUG,f)
            wridirsource=os.path.join(dirf,source)

            listdcm= [name for name in os.listdir(dirf) if name.find('.dcm')>0]
            nosource=False
            if len(listdcm)>0:
                nosource=True
                if not os.path.exists(wridirsource):
                    os.mkdir(wridirsource)
           
            path_data_write=os.path.join(dirf,path_data)
            
#            remove_folder(path_data_write)
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
#            """                     
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
            pickle.dump(crosscompleted, open( os.path.join(path_data_write,"crosscompleteds2"), "wb" ),protocol=-1)
            frontcompleted=False
            pickle.dump(frontcompleted, open( os.path.join(path_data_write,"frontcompleteds2"), "wb" ),protocol=-1)
            print 'source',source
#            return ''
            tabscanName={}
            tabscanroi={}

            centerHU1=0
            limitHU1=0
            if os.path.exists(os.path.join(path_data_write,'centerHUs2')):
                centerHU1=pickle.load( open(os.path.join(path_data_write,'centerHUs2'), "rb" ))
            if os.path.exists(os.path.join(path_data_write,'limitHUs2')):
                limitHU1=pickle.load( open(os.path.join(path_data_write,'limitHUs2'), "rb" ))
        
            if centerHU1==centerHU and limitHU1==limitHU and not(ForceGenerate):
                print 'no need to regenerate'
                datacross=pickle.load(open( os.path.join(path_data_write,'datacrosss2'), "rb" ))
                slnt= datacross[0]
#                dimtaby=datacross[2]
                slicepitch=datacross[1]
                lissln=datacross[2]
                PixelSpacing=datacross[5]
                
                tabscanScan=pickle.load( open(os.path.join(path_data_write,'tabscanScans2'), "rb" ))
                tabscanName=pickle.load( open(os.path.join(path_data_write,'tabscanNames2'), "rb" ))
                volumeroi=pickle.load( open(os.path.join(path_data_write,'volumerois2'), "rb" ))
                tabscanroi=pickle.load( open(os.path.join(path_data_write,'tabscanrois2'), "rb" ))
                tabscanLung=pickle.load( open(os.path.join(path_data_write,'tabscanLungs2'), "rb" ))
                tabrange=pickle.load( open(os.path.join(path_data_write,'tabranges2'), "rb" ))
                tabroi=pickle.load( open(os.path.join(path_data_write,'tabrois2'), "rb" ))
                slnroi=pickle.load( open(os.path.join(path_data_write,'slnrois2'), "rb" ))
                print 'end load'
                
            else:
                print 'generate'               
                tabscanScan,slnt,slicepitch,lissln,tabscanroi,tabscanName,PixelSpacing=genebmp(dirf,
                        source,nosource, centerHU, limitHU,tabscanName,tabscanroi,dimtabx,dimtaby)
                tabscanLung,tabrange=genebmplung(dirf,lungmaski,slnt,dimtabx,dimtabx,tabscanScan,lissln,tabscanName)
                slnroi=[]
                tabroi=np.zeros((slnt,dimtaby,dimtabx), np.uint8) 
                tabroi,volumeroi,slnroi=generoi(dirf,tabroi,dimtabx,dimtabx,slnroi,
                                            tabscanName,dirroi,tabscanroi,tabscanLung,PixelSpacing)                
                datacross=(slnt,slicepitch,lissln,setref, thrproba,PixelSpacing)
                pickle.dump(centerHU, open(os.path.join(path_data_write,'centerHUs2'), "wb" ),protocol=-1)
                pickle.dump(limitHU, open(os.path.join(path_data_write,'limitHUs2'), "wb" ),protocol=-1) 
                
                pickle.dump(tabscanScan, open(os.path.join(path_data_write,'tabscanScans2'), "wb" ),protocol=-1) 
                pickle.dump(tabscanroi, open(os.path.join(path_data_write,'tabscanrois2'), "wb" ),protocol=-1) 
                pickle.dump(tabscanName, open(os.path.join(path_data_write,'tabscanNames2'), "wb" ),protocol=-1) 
                pickle.dump(volumeroi, open(os.path.join(path_data_write,'volumerois2'), "wb" ),protocol=-1)
                pickle.dump(tabscanLung, open( os.path.join(path_data_write,"tabscanLungs2"), "wb" ),protocol=-1)
                pickle.dump(tabrange, open( os.path.join(path_data_write,"tabranges2"), "wb" ),protocol=-1)

                pickle.dump(tabroi, open( os.path.join(path_data_write,"tabrois2"), "wb" ),protocol=-1)
                pickle.dump(slnroi, open( os.path.join(path_data_write,"slnrois2"), "wb" ),protocol=-1)
                pickle.dump(datacross, open( os.path.join(path_data_write,'datacrosss2'), "wb" ),protocol=-1)
            datacross=(slnt,slicepitch,lissln,setref, thrproba,PixelSpacing)
            pickle.dump(datacross, open( os.path.join(path_data_write,'datacrosss2'), "wb" ),protocol=-1)
            
            regene=True
            if os.path.exists(os.path.join(path_data_write,"X_predicts2")):
                try:
                    picklein_filet2=pickle.load(open( os.path.join(path_data_write,"picklein_filets2"), "rb" ))
                    if   picklein_filet2==picklein_filet:                    
                        regene=False 
                    else:
                        regene=True 
                except:
                    regene=True
                
                                      

            if regene or ForceGenerate:
                print 'regenerate patch list'
                
                X_predict,xtrains=preparscan(tabscanScan,tabscanLung,slnroi)
                proba_cross=ILDCNNpredict(X_predict,modelcross)
                pickle.dump(X_predict, open( os.path.join(path_data_write,"X_predicts2"), "wb" ),protocol=-1)
                pickle.dump(proba_cross, open( os.path.join(path_data_write,"proba_crosss2"), "wb" ),protocol=-1)
                pickle.dump(xtrains, open( os.path.join(path_data_write,"xtrainss2"), "wb" ),protocol=-1)
                pickle.dump(picklein_filet, open( os.path.join(path_data_write,'picklein_filets2'), "wb" ),protocol=-1)

            else:
                print 'no need to regenerate scan data'
                proba_cross=pickle.load(open( os.path.join(path_data_write,"proba_crosss2"), "rb" ))
                X_predict=pickle.load(open( os.path.join(path_data_write,"X_predicts2"), "rb" ))
                xtrains=pickle.load(open( os.path.join(path_data_write,"xtrainss2"), "rb" ))
                
#                pickle.dump(proba_cross, open( os.path.join(path_data_write,"proba_crosss2"), "wb" ),protocol=-1)
 
            crosscompleted=True
            pickle.dump(crosscompleted, open( os.path.join(path_data_write,"crosscompleteds2"), "wb" ),protocol=-1)
#            """
    ###       cross
            if td:
#                """
                print 'START PREDICT FRONT'
                print '------------------'
                return ' '
                frontcompleted=False
                pickle.dump(frontcompleted, open( os.path.join(path_data_write,"frontcompleteds2"), "wb" ),protocol=-1)
                tabresScan=reshapeScanl(tabscanScan)
                dimtabxn,dimtabyn,tabScan3d,lisslnfront=wtebres(wridir,dirf,tabresScan,
                                                                dimtabx,slicepitch,lungmaski,'scan',centerHU,limitHU)
                tabresLung=reshapeScanl(tabscanLung)   
                dimtabxn,dimtabyn,tabLung3d,a=wtebres(wridir,dirf,tabresLung,dimtabx,slicepitch,
                                                      lungmaski,'lung',centerHU,limitHU)

                datafront=(dimtabx,dimtabxn,dimtabyn,slicepitch,lisslnfront)
                
                pickle.dump(datafront, open( os.path.join(path_data_write,'datafronts2'), "wb" ),protocol=-1)
                pickle.dump(tabLung3d, open( os.path.join(path_data_write,"tabLung3ds2"), "wb" ),protocol=-1)
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
                    thrpatch1= pickle.load( open( os.path.join(path_data_write,"thrpatchs2"), "rb" ))
                    if thrpatch == thrpatch1:
                        if os.path.exists(os.path.join(path_data_write,"patch_list_fronts2")):
                            regene=False

                            patch_list_front= pickle.load( open( os.path.join(path_data_write,"patch_list_fronts2"), "rb" ))
                if regene or ForceGenerate:
                    print 'regenerate patch list'

                    patch_list_front=pavgenefront(dirf,dimtabxn,dimtabx,tabScan3d,tabLung3d,dimtabyn,jpegpath3d)
                    pickle.dump(patch_list_front, open( os.path.join(path_data_write,"patch_list_fronts2"), "wb" ),protocol=-1)
                else:
                    print 'no need to regenerate patch list'
                

                proba_front=ILDCNNpredict(patch_list_front,modelfront)
                
                patch_list_front_slice=genepatchlistslice(patch_list_front,
                                                            proba_front,lisslnfront,dimtabxn,dimtabyn)
                pickle.dump(patch_list_front_slice, open( os.path.join(path_data_write,"patch_list_front_slices2"), "wb" ),protocol=-1)                
                
                proba_cross_from_front,patch_list_cross_from_front= genecrossfromfront(proba_front,patch_list_front,
                                                                   dimtabx,lissln,dimtabxn,slnt)               
                patch_list_cross_slice_from_front=genepatchlistslice(patch_list_cross_from_front,
                                                            proba_cross_from_front,lissln,dimtabx,dimtabx)
                
                pickle.dump(patch_list_cross_slice_from_front, open( os.path.join(path_data_write,"patch_list_cross_slice_from_fronts2"), "wb" ),protocol=-1)

                proba_merge,patch_list_merge=mergeproba(patch_list_cross_slice,
                                                        patch_list_cross_slice_from_front,slnt,dimtabx,dimtabx)
                patch_list_merge_slice=genepatchlistslice(patch_list_merge,
                                                            proba_merge,lissln,dimtabx,dimtabx)
                pickle.dump(patch_list_merge_slice, open( os.path.join(path_data_write,"patch_list_merge_slices2"), "wb" ),protocol=-1)

                frontcompleted=True
                pickle.dump(frontcompleted, open( os.path.join(path_data_write,"frontcompleteds2"), "wb" ),protocol=-1)
            errorfile.write('completed :'+f)
            errorfile.close()
            
            print 'PREDICT  COMPLETED  for ',f,' set ',setref,'thrproba',thrproba,'thrpatch'
            print '------------------'
        return''


#errorfile.close()
        print "predict time:",round(mytime()-t0,3),"s"

#ILDCNNpredict(bglist)