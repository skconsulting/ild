# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:03:33 2017

@author: sylvain
generate data from dicom images for segmentation roi 

-1st step

include generation per pattern
"""
#from __future__ import print_function

from param_pix import cwdtop,bmpname,lungmask,lungmask1,lungmaskbmp,image_rows,image_cols,typei,typei1
from param_pix import sroi,sourcedcm
from param_pix import white
from param_pix import remove_folder,normi,rsliceNum,norm
from param_pix import classifc,classif,usedclassif

import cPickle as pickle
import cv2
import dicom
import numpy as np
import os
import datetime
import sys
from skimage import measure

#path for images source
nametop='SOURCE_IMAGE'
#nameHug='HUG'
nameHug='CHU'
#nameHug='CHU2'
#nameHug='REFVAL'
#subHUG='ILD94'
#subHUG='ILD_TXT'
#subHUG='ILDS14740'

#subHUG='UIP6'
subHUG='UIP'

#path for image dir for CNN

imagedir='IMAGEDIR'
toppatch= 'TOPROI'
extendir='4'
#extendir='3'

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

def tagviews(tab,text,x,y):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,text,(x, y), font,0.3,white,1)
    return viseg

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


 
def genebmp(dirName,fileList,slnt,hug,tabscanName):
    """generate patches from dicom files and sroi"""
    print ('generate  bmp files from dicom files in :',f)
    (top,tail)=os.path.split(dirName)
    remdir = os.path.join(dirName, 'bgdir')
    remove_folder(remdir)
    remdir = os.path.join(dirName, 'bmp')
    remove_folder(remdir)

    if hug:

        bmp_dir = os.path.join(dirName, sourcedcm)
        if not os.path.exists(bmp_dir):
            os.mkdir(bmp_dir)        
        bmp_dir = os.path.join(bmp_dir, bmpname)
        remove_folder(bmp_dir)
        os.mkdir(bmp_dir)
        
    else:
        (top,tail)=os.path.split(dirName)
        bmp_dir = os.path.join(top, sourcedcm)
        if not os.path.exists(bmp_dir):
            os.mkdir(bmp_dir)        
        bmp_dir = os.path.join(bmp_dir, bmpname)
        remove_folder(bmp_dir)
        os.mkdir(bmp_dir)
        
    tabscan=np.zeros((slnt,image_rows,image_cols),np.int16)

    tabsroi=np.zeros((slnt,image_rows,image_cols,3),np.uint8)
    for filename in fileList:

            FilesDCM =(os.path.join(dirName,filename))
            RefDs = dicom.read_file(FilesDCM,force=True)
            dsr= RefDs.pixel_array
            dsr=dsr.astype('int16')

            scanNumber=int(RefDs.InstanceNumber)
            endnumslice=filename.find('.dcm')
            imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)+'.'
            dsr[dsr == -2000] = 0
            intercept = RefDs.RescaleIntercept

            slope = RefDs.RescaleSlope
            if slope != 1:
                dsr = slope * dsr.astype(np.float64)
                dsr = dsr.astype(np.int16)

            dsr += np.int16(intercept)
            dsr = dsr.astype('int16')

            if dsr.shape[0] != image_cols:
                print 'resize'
                dsr=cv2.resize(dsr,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)

            dsrforimage=normi(dsr)
            tabscan[scanNumber]=dsr

            imgcored=imgcoredeb+typei
            imgcoredbmp=imgcoredeb+typei1
#            print filename, scanNumber
            tabscanName[scanNumber]=imgcoredbmp
            bmpfiled=os.path.join(bmp_dir,imgcored)
#            imgcoresroi='sroi_'+str(scanNumber)+'.'+typei
            bmpfileroi=os.path.join(sroidir,imgcoredbmp)
#            print imgcoresroi,bmpfileroi
            textw='n: '+tail+' scan: '+str(scanNumber)

            cv2.imwrite (bmpfiled, dsrforimage,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
            dsrforimage=cv2.cvtColor(dsrforimage,cv2.COLOR_GRAY2BGR)
            dsrforimage=tagviews(dsrforimage,textw,0,20)
            cv2.imwrite (bmpfileroi, dsrforimage)
            tabsroi[scanNumber]=dsrforimage



    return tabscan,tabsroi,tabscanName
 
def genebmplung(dirName,slnt,hug,tabscanName,tabscanScan,listsln):
    """generate patches from dicom files and sroi"""
    print ('generate lung files from dicom files in :',f)
    (top,tail)=os.path.split(dirName)
    
    if hug:
        lung_dir = os.path.join(dirName, lungmask1)
        if not os.path.exists(lung_dir):
            lung_dir = os.path.join(dirName, lungmask)        
    else:
        (top,tail)=os.path.split(dirName)
        lung_dir = os.path.join(top, lungmask1)
        if not os.path.exists(lung_dir):
            lung_dir = os.path.join(top, lungmask)
    if not os.path.exists(lung_dir):
        os.mkdir(lung_dir)   
    lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)   
    if not os.path.exists(lung_bmp_dir):
        os.mkdir(lung_bmp_dir)
        
    lung_bmp_dirbmp = os.path.join(lung_dir,bmpname)
    remove_folder(lung_bmp_dirbmp)
    
    lunglist = [name for name in os.listdir(lung_dir) if ".dcm" in name.lower()]

    tabslung=np.zeros((slnt,image_rows,image_cols),np.uint8)

    listbmp= os.listdir(lung_bmp_dir) 

    if len(listbmp)>0:
        print 'lung scan exists in bmp'
        for img in listbmp:
            slicenumber= rsliceNum(img,'_','.'+typei1)
            if slicenumber>0:
        
                    imr=cv2.imread(os.path.join(lung_bmp_dir,img),0) 
                    imr=cv2.resize(imr,(image_rows,image_cols),interpolation=cv2.INTER_LINEAR)  
                    np.putmask(imr,imr>0,1)
                    tabslung[slicenumber]=imr


    if len(lunglist)>0:
        for lungfile in lunglist:
#             print(lungfile)
#             if ".dcm" in lungfile.lower():  # check whether the file's DICOM
                FilesDCM =(os.path.join(lung_dir,lungfile))
                RefDs = dicom.read_file(FilesDCM,force=True)
                dsr= RefDs.pixel_array
                dsr=dsr.astype('int16')
#                fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
#                print 'fxs',fxs
                scanNumber=int(RefDs.InstanceNumber)
                if tabslung[scanNumber].max()==0 and scanNumber in listsln:
                    bmpfile=os.path.join(lung_bmp_dir,tabscanName[scanNumber])
                    dsr=normi(dsr)
                    dsrresize=cv2.resize(dsr,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)
                    if dsrresize.max()>0:
                        cv2.imwrite (bmpfile, dsrresize)
    #                np.putmask(dsrresize,dsrresize==1,0)
                        np.putmask(dsrresize,dsrresize>0,1)
                        tabslung[scanNumber]=dsrresize
    else:
        print 'no lung scan in dcm'
        tabscan1 = np.zeros((slnt,image_rows,image_cols), np.int16)
        segmented_lungs_fill = segment_lung_mask(tabscanScan, True)
        for i in range(1,slnt):
            if tabslung[i].max()==0 and i in listsln:
                tabscan1[i]=morph(segmented_lungs_fill[i],13)
                bmpfile=os.path.join(lung_bmp_dir,tabscanName[i])
                if tabscan1[i].max()>0:
                    cv2.imwrite (bmpfile, tabscan1[i])
                    np.putmask(tabscan1[i],tabscan1[i]>0,1)
                    tabslung[i]=tabscan1[i]
                
    return tabslung

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

    np.putmask(tabslung,tabslung>0,255)
    taba=cv2.bitwise_and(scan,scan,mask=tabslung)
    np.putmask(tabslung,tabslung>0,1)
    
    tablc=tabslung.astype(np.int16)
    np.putmask(tablc,tablc==0,-1000)
    np.putmask(tablc,tablc==1,0)
        
    tabab=cv2.bitwise_or(taba,tablc) 
    tababn=norm(tabab)
    datascan[numslice]=tababn
    return datascan
    

def preparroi(namedirtopcf,datascan,tabroi):
    (top,tail)=os.path.split(namedirtopcf)

    pathpicklepat=os.path.join(picklepathdir,nameHug+' _'+tail)
    if not os.path.exists (pathpicklepat):
                os.mkdir(pathpicklepat)
    
    for num in numsliceok:
        patchpicklenamepatient=str(num)+'_'+patchpicklename        

        pathpicklepatfile=os.path.join(pathpicklepat,patchpicklenamepatient)
            
        patpickle=(datascan[num],tabroi[num])

        pickle.dump(patpickle, open(pathpicklepatfile, "wb"),protocol=-1)
        for pat in tabroipat[num]:
             listslicef[tail][num][pat]+=1
             roipathpicklepat=os.path.join(roipicklepathdir,pat)   
             if not os.path.exists (roipathpicklepat):
                os.mkdir(roipathpicklepat)

             roipathpicklepatfile=os.path.join(roipathpicklepat,nameHug+' _'+tail+'_'+patchpicklenamepatient)
             pickle.dump(patpickle, open(roipathpicklepatfile, "wb"),protocol=-1)

def drawcontours2(im,pat,dimtabx,dimtaby):
#    print 'contour',pat
    imgray = np.copy(im)
    ret,thresh = cv2.threshold(imgray,10,255,0)
    _,contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2 = np.zeros((dimtabx,dimtaby,3), np.uint8)
    cv2.drawContours(im2,contours,-1,classifc[pat],2)
    im2=cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
    return im2

def create_test_data(namedirtopcf,pat,tabscan,tabsroi,tabslung,datascan,tabscanName):
    
    (top,tail)=os.path.split(namedirtopcf)
    print 'create test data for :', tail, 'pattern :',pat
    pathpat=os.path.join(namedirtopcf,pat)

    list_image=[name for name in os.listdir(pathpat) if name.find('.'+typei1)>0] 
#    if len(list_image)==0:
#        list_image=[name for name in os.listdir(pathpat) if name.find('.'+typei)>0] 
#  
    if len(list_image)>0:

            for l in list_image:
                pos=l.find('.'+typei1)      
                ext=l[pos:len(l)]
                numslice=rsliceNum(l,'_',ext)
#                print numslice,tabroipat[numslice]
                if pat not in tabroipat[numslice]:
                    tabroipat[numslice].append(pat)                    
                if numslice not in numsliceok:
                    numsliceok.append(numslice)
                    
                    datascan=peparescan(numslice,tabscan[numslice],tabslung[numslice],datascan)
                    tabroi[numslice]=np.zeros((tabscan.shape[1],tabscan.shape[2]), np.uint8)
#                print numslice,tabroipat[numslice]
    #            tabl=tabslung[numslice].copy()
    #            np.putmask(tabl,tabl>0,1)
    
                newroi = cv2.imread(os.path.join(pathpat, l), 0) 
                
                if newroi.max()==0:
                    print pathpat,l
                    print newroi.shape
                    print newroi.max(),newroi.min()
                    print 'error image empty'
                    sys.exit()
                img=cv2.resize(newroi,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)

                
                np.putmask(tabroi[numslice], img > 0, 0)
    #                if classif[pat]>0:
                np.putmask(img, img > 0, classif[pat])
#                else:
#                    np.putmask(img, img > 0, classif['lung'])
                tablung=np.copy(tabslung[numslice])
                np.putmask(tablung,tablung>0,255)                      
                img=np.bitwise_and(tablung, img)  
                tabroi[numslice]+=img
                np.putmask(tablung,tablung>0,classif['healthy']) 
                tabroii=np.copy(tabroi[numslice])
                np.putmask(tabroii,tabroii>0,255) 
                mask=np.bitwise_not(tabroii)
                img=np.bitwise_and(tablung, mask)
                tabroif=np.bitwise_or(img,tabroi[numslice])
                tabroi[numslice]=tabroif
                           
#            
                    
    return tabroi,datascan

def genesroi(numsliceok,tabroi,tabsroi,tabscanName):
    for sln in numsliceok:
        anoted_image=tabsroi[sln]
#        print anoted_image.shape
        anoted_image=cv2.cvtColor(anoted_image,cv2.COLOR_BGR2RGB)
        imgcoreScan=tabscanName[sln]
        roibmpfile=os.path.join(sroidir,imgcoreScan)
        for pat in usedclassif:
            newroic=tabroi[sln].copy()            
#            newroic1=tabroi[sln].copy()
           
            np.putmask(newroic,newroic!=classif[pat],0)
            np.putmask(newroic, newroic==classif[pat], 100)
            if newroic.max()>0:
                
                ctkey=drawcontours2(newroic,pat,image_rows,image_cols)   
                
                anoted_image=cv2.add(anoted_image,ctkey)

        cv2.imwrite(roibmpfile,anoted_image)

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
print 'work on: ',namedirtopc
for f in listdirc:
    print '-----------------'
    print('work on:',f)
    print '-----------------'
    listslicef[f]={}
        
    numsliceok=[]
    listslicetot[f]=0
    namedirtopcf=os.path.join(namedirtopc,f)
    sroidir=os.path.join(namedirtopcf,sroi)
    if os.path.exists(sroidir):
        remove_folder(sroidir)
    os.mkdir(sroidir)
    tabscanName={}
    contenudir = [name for name in os.listdir(namedirtopcf) if name.find('.dcm')>0]
    if len(contenudir)>0:
        slnt,listsln = genepara(contenudir,namedirtopcf)

        tabscan,tabsroi,tabscanName=genebmp(namedirtopcf,contenudir,slnt,True,tabscanName)
        tabslung=genebmplung(namedirtopcf,slnt,True,tabscanName,tabscan,listsln)
    else:
          namedirtopcfs=os.path.join(namedirtopcf,sourcedcm)
          contenudir = [name for name in os.listdir(namedirtopcfs) if name.find('.dcm')>0]
          slnt,listsln = genepara(contenudir,namedirtopcfs)
          tabscan,tabsroi,tabscanName=genebmp(namedirtopcfs,contenudir,slnt,False,tabscanName)
          tabslung=genebmplung(namedirtopcfs,slnt,False,tabscanName,tabscan,listsln)

    slntdict[f]=slnt
    for slic in range(slnt):
        listslicef[f][slic]={}
        for pat in classif:
            listslicef[f][slic][pat]=0
    contenupat = [name for name in os.listdir(namedirtopcf) if name in usedclassif]

    datascan={}
    datamask={}
    tabroi={}
    tabroipat={}
    listpatf[f]=contenupat
    
    for i in range(slnt):
        tabroipat[i]=[]

    for pat in usedclassif:
        if pat in contenupat:
            print 'work on :',pat
            if pat not in listpat:
                listpat.append(pat)
            tabroi,datascan=create_test_data(namedirtopcf,pat,tabscan,tabsroi,tabslung,datascan,tabscanName)


    genesroi(numsliceok,tabroi,tabsroi,tabscanName)
    preparroi(namedirtopcf,datascan,tabroi)
    
    listslicetot[f]=len(numsliceok)
    print 'number of different images :',len(numsliceok)
    for i in range(slnt):
#        print i, tabroipat[i]
        for pat in classif:
             if listslicef[f][i][pat] !=0:
                 print  f,i, pat, listslicef[f][i][pat]
                 totalnumperpat[pat]+=1
                 
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'

pathpatfile=os.path.join(patchtoppath,'listpat.txt')
filetw = open(pathpatfile,"a")                       
print '-----------------------------'
print('TOP DIR: '+nameHug)

print 'list of patterns :',listpat
print '-----------------------------'

filetw.write('started ' +nameHug+' '+subHUG+' at :'+todayn)
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

