# coding: utf-8
#sylvain Kritter 04-Apr-2017
'''predict on lung scan front view and cross view
@author: sylvain Kritter 

version 1.5
6 September 2017
'''
#from param_pix_p import *
from param_pix_c import scan_bmp,avgPixelSpacing,dimpavx,dimpavy,dirpickleArch,modelArch,surfelemp
from param_pix_c import typei,typei1
from param_pix_c import white,yellow,red

from param_pix_c import lung_namebmp,jpegpath,lungmask,lungmask1
from param_pix_c import fidclass,pxy
#from param_pix_p import classifc,classif,excluvisu,usedclassif
from param_pix_c import classifc


from param_pix_c import  source
from param_pix_c import  transbmp
from param_pix_c import sroi
from param_pix_c import jpegpath3d
from param_pix_c import jpegpadirm,source_name,path_data,dirpickle,cwdtop

from param_pix_c import remove_folder,normi,rsliceNum,norm,maxproba
from param_pix_c import classifdict,usedclassifdict,oldFormat,derivedpatdict,layertokeep

#import scipy
import time
from time import time as mytime
import numpy as np
#from numpy import argmax,amax
import os
import cv2
import dicom
import copy
from skimage import measure
import cPickle as pickle


#from keras.optimizers import Adam


moderesize=True # True means linear resize
t0=mytime()

def reshapeScanl(tabscan):
    print 'reshape lung'
    tabres=np.moveaxis(tabscan,0,1)
    return tabres



def genebmp(fn,sou,nosource,tabscanName,tabscanroi):
    """generate patches from dicom files"""
    global picklein_file
    (top,tail) =os.path.split(fn)
    print ('load scan dicom files in:' ,tail)
    lislnn=[]
   
    fmbmp=os.path.join(fn,sou)
    if nosource:
        fmbmp=fn
    print fmbmp
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
  
    dsr= RefDs.pixel_array

    dsr = dsr.astype('float32')
    if moderesize:
        print 'resize'
        fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
        imgresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
#    imgresize = scipy.ndimage.interpolation.zoom(dsr, fxs, mode='nearest')
#    dsr = dsr.astype('int16')
        dimtabx=imgresize.shape[0]
        dimtaby=imgresize.shape[1]
    else:
        print 'no resize'
        dimtabx=dsr.shape[0]
        dimtaby=dimtabx
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
        if moderesize:
            dsr = dsr.astype('float32')
            imgresize1=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)
            imgresize=imgresize1.astype('int16')
#        imgresize = scipy.ndimage.interpolation.zoom(dsr, fxs, mode='nearest')
        else:
            imgresize=dsr
        

        endnumslice=l.find('.dcm')
        imgcoreScan=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei1  
        
        tabscan[slicenumber]=imgresize.copy()
#        if slicenumber==12:
#            print '1',tabscan[slicenumber].min(),tabscan[slicenumber].max()
        
     
        imtowrite=normi(imgresize)
        imtowrite = cv2.cvtColor(imtowrite, cv2.COLOR_GRAY2RGB)
#        if slicenumber==12:
#            print '2',tabscan[slicenumber].min(),tabscan[slicenumber].max()
#        if slicenumber==12:
#            pickle.dump(imgresize, open('score.pkl', "wb" ),protocol=-1)
#            cv2.imwrite('score.bmp',imtowrite)
#        bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
        tabscanName[slicenumber]=imgcoreScan
#        (topw,tailw)=os.path.split(picklein_file)
        t2='Prototype '
        t1='Patient: '+tail
        t0='CONFIDENTIAL'
        t3='Scan: '+str(slicenumber)

        t4=time.asctime()
        t5=''
        t6=''
    
        anoted_image=tagviews(imtowrite,
                              t0,dimtabx-100,dimtaby-10,
                              t1,0,dimtaby-21,
                              t2,dimtabx-100,dimtaby-20,
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
    listbmp= os.listdir(fmbmpbmp) 
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
    #            tabscanlung = np.zeros((slnt,dimtabx,dimtaby), np.uint8)
    #            print tabscanScan.shape
                segmented_lungs_fill = segment_lung_mask(tabscanScan, True)
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
    cv2.drawContours(im2,contours,-1,classifc[pat],2)
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


def generoi(dirf,tabroi,dimtabx,dimtaby,slnroi,tabscanroi,tabscanLung,slnroidir,slnt,tabscanName,dirroit):
    
    (top,tail)=os.path.split(dirf)
    print 'generate roi table',tail
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
#            fer.write('no lung for: '+str(numslice)+'\n')
            print ('no lung for: '+str(numslice))
        tabroi[numslice]=np.bitwise_and(maskRoi,maskLung)   

        maskRoi1Not=np.bitwise_not(maskRoi1)       
        tablung=np.bitwise_and(maskLung, maskRoi1Not)
        np.putmask(tablung,tablung>0,classif['healthy']+1) 
        tabroi[numslice]=np.bitwise_or(tabroi[numslice],tablung)
        
#        if tabroi[numslice].max()>0:
#                cv2.imshow(str(numslice)+' tabo',normi(tabroi[numslice])) 
#                cv2.imshow(str(numslice)+' mask',normi(mask)) 
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
                        np.putmask(img, img >0, 100)
                        ctkey=drawcontours2(img,pat,dimtabx,dimtaby) 
                        anoted_image=tagviewct(anoted_image,pat,200,10)
                        anoted_image=cv2.add(anoted_image,ctkey)

                                               
                else:
                    volumeroi[numslice][pat]=0 
            anoted_image= cv2.cvtColor(anoted_image,cv2.COLOR_RGB2BGR)
#            print roibmpfile
            cv2.imwrite(roibmpfile,anoted_image)   
#    print volumeroi[12]  
    slnroidir[tail]=len(slnroi)
    return tabroi,volumeroi,slnroi,slnroidir,listroi
        

def predictrun(indata,dirHUGref,dirHUGcomp):
        global  classif,usedclassif,derivedpat

        listHugref=[]
        listHugrefi=indata['lispatientselectref']
        for lpt in listHugrefi:
             pos=lpt.find(' PREDICT!:')
             if pos >0:
                    listHugref.append(lpt[0:pos])
             else:
                    pos=lpt.find(' noPREDICT!')
                    if pos >0:
                        listHugref.append(lpt[0:pos])
                    else:
                        listHugref.append(lpt)
        listHugcomp=[]                   
        listHugcompi=indata['lispatientselectcomp']
        for lpt in listHugcompi:
             pos=lpt.find(' PREDICT!:')
             if pos >0:
                    listHugcomp.append(lpt[0:pos])
             else:
                    pos=lpt.find(' noPREDICT!')
                    if pos >0:
                        listHugcomp.append(lpt[0:pos])
                    else:
                        listHugcomp.append(lpt)
               

        print dirHUGref
        print dirHUGcomp
        print listHugref
        print listHugcomp
        posu=listHugref[0].find('_')
        rootref=listHugref[0][0:posu]
        posu=listHugcomp[0].find('_')
        rootcomp=listHugcomp[0][0:posu]

        setref='set0'
        classif=classifdict[setref]
        usedclassif=usedclassifdict[setref]
        derivedpat=derivedpatdict[setref]
 
    
        slnroidirref={}
        slnroidircomp={}   
    
        for f in listHugref:
            print '------------------'
            print 'work on patient',f
            
            dirfref=os.path.join(dirHUGref,f)
            posu=f.find('_')
            namref=f[posu+1:]
            fcomp=rootcomp+'_'+namref
            
            dirfcomp=os.path.join(dirHUGcomp,fcomp)
                        
            path_data_write_ref=os.path.join(dirfref,path_data)
            path_data_write_comp=os.path.join(dirfcomp,path_data)
            
            tabscanName={}
            tabscanroi={}

            
            listdcm= [name for name in os.listdir(dirfref) if name.find('.dcm')>0]
            nosource=False
            if len(listdcm)>0:
                    nosource=True
            fmbmp=os.path.join(dirfref,lungmask1)
            if os.path.exists(fmbmp):
                    lungmaski=lungmask1
            else:
                    fmbmp=os.path.join(dirfref,lungmask)
                    if os.path.exists(fmbmp):
                        lungmaski=lungmask
            dirroit=os.path.join(dirfref,sroi)
            tabscanScan,slnt,dimtabx,slicepitch,lissln,tabscanroi,tabscanName=genebmp(dirfref,
                        source,nosource, tabscanName,tabscanroi)
            tabscanLungref,tabrange=genebmplung(dirfref,lungmaski,slnt,dimtabx,dimtabx,tabscanScan,lissln,tabscanName)
            slnroiref=[]
            tabroiref=np.zeros((slnt,dimtabx,dimtabx), np.uint8) 
            tabroiref,volumeroiref,slnroiref,slnroidirref,listroiref=generoi(dirfref,tabroiref,dimtabx,dimtabx,slnroiref,     
                    tabscanroi,tabscanLungref,slnroidirref,slnt,tabscanName,dirroit)
            datacrossref=(slnt,dimtabx,lissln,setref,rootcomp)
            pickle.dump(tabscanLungref, open( os.path.join(path_data_write_ref,"tabscanLungref"), "wb" ),protocol=-1)
            pickle.dump(tabroiref, open( os.path.join(path_data_write_ref,"tabroiref"), "wb" ),protocol=-1)
            pickle.dump(slnroiref, open( os.path.join(path_data_write_ref,"slnroiref"), "wb" ),protocol=-1)
            pickle.dump(datacrossref, open( os.path.join(path_data_write_ref,"datacrossref"), "wb" ),protocol=-1)
            pickle.dump(volumeroiref, open( os.path.join(path_data_write_ref,"volumeroiref"), "wb" ),protocol=-1)
            pickle.dump(listroiref, open( os.path.join(path_data_write_ref,"listroiref"), "wb" ),protocol=-1)
            
            dirroit=os.path.join(dirfcomp,sroi)
            slnroicomp=[]
            tabroicomp=np.zeros((slnt,dimtabx,dimtabx), np.uint8) 
            tabroicomp,volumeroicomp,slnroicomp,slnroidircomp,listroicomp=generoi(dirfcomp,tabroicomp,dimtabx,dimtabx,slnroicomp,     
                    tabscanroi,tabscanLungref,slnroidircomp,slnt,tabscanName,dirroit)
            pickle.dump(tabroicomp, open( os.path.join(path_data_write_comp,"tabroicomp"), "wb" ),protocol=-1)
            pickle.dump(slnroicomp, open( os.path.join(path_data_write_comp,"slnroicomp"), "wb" ),protocol=-1)
            pickle.dump(volumeroicomp, open( os.path.join(path_data_write_comp,"volumeroicomp"), "wb" ),protocol=-1)
                
        
        
        slnroidirtot=0
#        print 'slnroidirref',slnroidirref
        for i in listHugref:
            slnroidirtot+=slnroidirref[i]
        print 'number of images with roi in ref:',slnroidirtot
        slnroidirtot=0
        for i in listHugcomp:
            slnroidirtot+=slnroidircomp[i]
        print 'number of images with roi in comp:',slnroidirtot
        print "predict time:",round(mytime()-t0,3),"s"
#        fer.write('----\n')
#        fer.close()
        return''

#ILDCNNpredict(bglist)