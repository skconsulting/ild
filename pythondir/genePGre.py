# coding: utf-8
#Sylvain Kritter 4 aout 2017
"""Top file to generate patches from DICOM database HU method and pixle out from CHU Grenoble
include new patterns when patterns are super imposed, cross view
it is for cross view only
version 1.0
S. Kritter

"""

#from param_pix_t import *

from param_pix_t import classif,derivedpat,usedclassif,classifc
from param_pix_t import dimpavx,dimpavy,typei,typei1,avgPixelSpacing,thrpatch,perrorfile,plabelfile,setdata,pxy
from param_pix_t import remove_folder,normi,genelabelloc,totalpat,totalnbpat
from param_pix_t import white
from param_pix_t import patchpicklename,scan_bmp,lungmask,lungmask1,sroi,patchesdirname
from param_pix_t import imagedirname,picklepath,source,lungmaskbmp
import os
#import sys
#import png
import numpy as np
import datetime
#import scipy as sp
import scipy.misc
import dicom
#import PIL
import cv2
#import matplotlib.pyplot as plt
import cPickle as pickle
#general parameters and file, directory names
#######################################################
#customisation part for datataprep
#global directory for scan file
topdir='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool'
namedirHUG = 'CHU'
#subdir for roi in text
subHUG='UIP'
#subHUG='UIP_106530'
#subHUG='UIP0'
#subHUG='UIP_S14740'

toppatch= 'TOPPATCH'
#extension for output dir
extendir='set0'
#extendir='essai1'
alreadyDone =['S4550','S106530','S107260','S139430','S145210','S14740','S15440','S1830','S28200','S335940','S359750']
alreadyDone =[]

#labelEnh=('consolidation','reticulation,air_trapping','bronchiectasis','cysts')
labelEnh=()
locabg='anywhere_CHUG'

########################################################################
######################  end ############################################
########################################################################

patchesdirnametop = 'th'+str(round(thrpatch,1))+'_'+toppatch+'_'+extendir
print 'name of directory for patches :', patchesdirnametop

#full path names
#cwd=os.getcwd()
#(cwdtop,tail)=os.path.split(cwd)
cwdtop=topdir
#path for HUG dicom patient parent directory
path_HUG=os.path.join(cwdtop,namedirHUG)
##directory name with patient databases
namedirtopc =os.path.join(path_HUG,subHUG)
if not os.path.exists(namedirtopc):
    print('directory ',namedirtopc, ' does not exist!')


#end general part
#########################################################
#log files
##error file
patchtoppath=os.path.join(cwdtop,patchesdirnametop)
#create patch and jpeg directory
patchpath=os.path.join(patchtoppath,patchesdirname)
#path for patch pickle
picklepathdir =os.path.join(patchtoppath,picklepath)
#print 'picklepathdir',picklepathdir

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

if not os.path.isdir(picklepathdir):
    os.mkdir(picklepathdir)

#remove_folder(jpegpath)
if not os.path.isdir(jpegpath):
    os.mkdir(jpegpath)

eferror=os.path.join(patchtoppath,perrorfile)
errorfile = open(eferror, 'a')
errorfile.write('---------------\n')
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
errorfile.write('started ' +namedirHUG+' '+subHUG+' at :'+todayn)
errorfile.write('using pattern set: ' +setdata+'\n')
for pat in usedclassif:
    errorfile.write(pat+'\n')
errorfile.write('--------------------------------\n')
errorfile.close()
#filetowrite=os.path.join(namedirtopc,'lislabel.txt')

#end customisation part for datataprep
#######################################################


roitab={}

def genepara(namedirtopcf):
    dirFileP = os.path.join(namedirtopcf, source)
        #list dcm files
    fileList =[name for name in  os.listdir(dirFileP) if ".dcm" in name.lower()]
    slnt=0
    for filename in fileList:
        FilesDCM =(os.path.join(dirFileP,filename))
        RefDs = dicom.read_file(FilesDCM)
        scanNumber=int(RefDs.InstanceNumber)
        if scanNumber>slnt:
            slnt=scanNumber
    print 'number of slices', slnt
    slnt=slnt+1
    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
    dsr= RefDs.pixel_array
    dsr= dsr-dsr.min()
    dsr=dsr.astype('uint16')
    dsrresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
    dimtabx=int(dsrresize.shape[0])
    dimtaby=int(dsrresize.shape[1])
    return dimtabx,dimtaby,slnt

def tagviews(tab,text,x,y):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,text,(x, y), font,0.3,white,1)
    return viseg


def genebmp(dirName, sou,tabscanName):
    """generate patches from dicom files and sroi"""
#    print ('generate  bmp files from dicom files in :',dirName, 'directory :',sou)
    dirFileP = os.path.join(dirName, sou)
    if sou ==source:
        tabscan=np.zeros((slnt,dimtabx,dimtaby),np.int16)
        dirFilePbmp=os.path.join(dirFileP,scan_bmp)
        remove_folder(dirFilePbmp)
        os.mkdir(dirFilePbmp)
    elif sou == lungmask or sou == lungmask1 :
        dirFilePbmp=os.path.join(dirFileP,lungmaskbmp)
        tabscan=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
        remove_folder(dirFilePbmp)
        os.mkdir(dirFilePbmp)
    else:
        dirFilePbmp=dirFileP
        tabscan=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
        if not os.path.exists(dirFilePbmp):
            os.mkdir(dirFilePbmp)
                
    (top,tail)=os.path.split(dirName)
        #list dcm files
    fileList =[name for name in  os.listdir(dirFileP) if ".dcm" in name.lower()]

    for filename in fileList:
                FilesDCM =(os.path.join(dirFileP,filename))
                RefDs = dicom.read_file(FilesDCM)
                dsr= RefDs.pixel_array
                dsr=dsr.astype('int16')
                fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
#                print 'fxs',fxs
                scanNumber=int(RefDs.InstanceNumber)
                endnumslice=filename.find('.dcm')
                imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)
                    
#                imgcore=imgcoredeb+typei
#                bmpfile=os.path.join(dirFilePbmp,imgcore)

                if dsr.max()>0:
                    if sou==source :
                        dsr[dsr == -2000] = 0
                        intercept = RefDs.RescaleIntercept
                        slope = RefDs.RescaleSlope
                        if slope != 1:
                            dsr = slope * dsr.astype(np.float64)
                            dsr = dsr.astype(np.int16)

                        dsr += np.int16(intercept)
                        dsr = dsr.astype('int16')
#                        print dsr.min(),dsr.max(),dsr.shape
                        dsr=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
                        dsrforimage=normi(dsr)

#
                        tabscan[scanNumber]=dsr
                                           

                        tabscanName[scanNumber]=imgcoredeb

                        imgcored=imgcoredeb+'.'+typei
                        bmpfiled=os.path.join(dirFilePbmp,imgcored)
#                        imgcoresroi=imgcoredeb+'.'+typei

                        bmpfileroi=os.path.join(sroidir,imgcored)
#                        print imgcoresroi,bmpfileroi
                        textw='n: '+tail+' scan: '+str(scanNumber)

                        cv2.imwrite (bmpfiled, dsrforimage)
                        dsrforimage=cv2.cvtColor(dsrforimage,cv2.COLOR_GRAY2BGR)
                        dsrforimage=tagviews(dsrforimage,textw,2,20)
                        cv2.imwrite (bmpfileroi, dsrforimage)
                        tabsroi[scanNumber]=dsrforimage
                    else:
#                        print 'dssresize dsr 1' ,sou, dsr.min(),dsr.max()
                        dsr=normi(dsr)
                        dsrresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
#                        print 'dssresize' ,sou, dsrresize.min(),dsrresize.max()
#                        print 'dssresize dsr 2' ,sou, dsr.min(),dsr.max()
#                        dsrforimage=normi(dsrresize)
                        
                        imgcored=tabscanName[scanNumber]+'.'+typei1
                        bmpfiled=os.path.join(dirFilePbmp,imgcored)
                        imgc=colorimage(dsrresize,classifc[sou])
                        cv2.imwrite (bmpfiled, imgc)
                        dsrresizer=np.copy(dsrresize)
                        np.putmask(dsrresizer,dsrresizer==1,0)
                        np.putmask(dsrresizer,dsrresizer>0,100)
                        tabscan[scanNumber]=dsrresizer
#    print sou, tabscan.min(),tabscan.max()
    return tabscan,tabsroi,tabscanName

def reptfulle(tabc,dx,dy):
    imgi = np.zeros((dx,dy,3), np.uint8)
    cv2.polylines(imgi,[tabc],True,(1,1,1))
    cv2.fillPoly(imgi,[tabc],(1,1,1))
    tabzi = np.array(imgi)
    tabz = tabzi[:, :,1]
    return tabz, imgi


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

def pavs (dirName,pat,slnt,dimtabx,dimtaby,tabscanName):
    """ generate patches from ROI"""
    print 'pav :',dirName,'pattern :',pat
    ntotpat=0
    tabf=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    _tabsroi=np.zeros((slnt,dimtabx,dimtaby,3),np.uint8)
#    _tabbg = np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    _tabscan = np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    patpickle=[]
    (top,tail)=os.path.split(dirName)


    nampadir=os.path.join(patchpath,pat)
    nampadirl=os.path.join(nampadir,locabg)
    if not os.path.exists(nampadir):
         os.mkdir(nampadir)
    if not os.path.exists(nampadirl):
         os.mkdir(nampadirl)

    pathpicklepat=os.path.join(picklepathdir,pat)
#    print pathpicklepat
    pathpicklepatl=os.path.join(pathpicklepat,locabg)

#    patchpicklenamepatient=namedirHUG+'_'+tail+'_'+patchpicklename
    patchpicklenamepatient=namedirHUG+'_'+subHUG+'_'+tail+'_'+patchpicklename

    pathpicklepatfile=os.path.join(pathpicklepatl,patchpicklenamepatient)
    if not os.path.exists(pathpicklepat):
         os.mkdir(pathpicklepat)
    if not os.path.exists(pathpicklepatl):
         os.mkdir(pathpicklepatl)
    if os.path.exists(pathpicklepatfile):
        os.remove(pathpicklepatfile)

    for scannumb in range (0,slnt):
       tabp = np.zeros((dimtabx, dimtaby), dtype='i')
       tabf=np.copy(tabroipat[pat][scannumb])

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
                sroifile=tabscanName[scannumb]+'.'+typei
                filenamesroi=os.path.join(sroidir,sroifile)
                cv2.imwrite(filenamesroi,imn)

                atabf = np.nonzero(tabf)

                xmin=atabf[1].min()
                xmax=atabf[1].max()
                ymin=atabf[0].min()
                ymax=atabf[0].max()

                np.putmask(tabf,tabf>0,1)
                _tabscan=tabscan[scannumb]
#                imgray8b=normi(_tabscan)

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

                            if imagemax > 0 and max_val - min_val>2:
                                nbp+=1
                                patpickle.append(imgray)
#                                imgraytowrite = normi(imgray)
#                                nampa=os.path.join(nampadirl,namedirHUG+'_'+tail+'_'+str(scannumb)+
#                                                   '_'+str(i)+'_'+str(j)+'_'+str(nbp)+'.'+typei )                                
#                                cv2.imwrite (nampa, imgraytowrite)
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
#             print tabfc.shape,tabp.shape
             tabfc =tabfc+tabp
             ntotpat=ntotpat+nbp
             if scannumb not in listsliceok:
                    listsliceok.append(scannumb)
             stw=namedirHUG+'_'+tail+'_slice_'+str(scannumb)+'_'+pat+'_'+locabg+'_'+str(nbp)
             stww=stw+'.txt'
             flw=os.path.join(jpegpath,stww)
             mfl=open(flw,"w")
             mfl.write('#number of patches: '+str(nbp)+'\n')
             mfl.close()
             stww=stw+'.'+typei
             flw=os.path.join(jpegpath,stww)
             scipy.misc.imsave(flw, tabfc)
#             print 'pathpicklepatfile',pathpicklepatfile
#             print patpickle
             pickle.dump(patpickle, open(pathpicklepatfile, "wb"),protocol=-1)
#             aaa=pickle.load( open(pathpicklepatfile, "rb"))
#             print len(aaa)
#             for i in range(0,len(aaa)):
#                 print aaa[i].min(), aaa[i].max()
#                 imgray8b=normi(aaa[i])
#                 print aaa[i]
#                 print imgray8b
#                 nampa=os.path.join(nampadirl,tail+'_r_'+str(i)+'.'+typeid )
#                 cv2.imwrite (nampa, imgray8b,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
#cv2.imshow('aa',imgray8b)

#       else:
#                 tw=tail+'_slice_'+str(scannumb)+'_'+pat+'_'+locabg
#                 print tw,' :no patch'
#                 errorfile.write(tw+' :no patch\n')
    return ntotpat

def colorimage(image,color):
#    im=image.copy()
    im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    np.putmask(im,im>0,color)

    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    return im


def calnewpat(dirName,pat,slnt,dimtabx,dimtaby,tabscanName):
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
#        if i == 145 and pat=='bropret':
#            cv2.imshow('tab1',tab1[i])
#            cv2.imshow('tab2',tab2[i])
#            cv2.imshow('tab3',tab3[i])

        tab3[i]=np.bitwise_and(tab1[i],tab2[i])
        if tab3[i].max()>0:

            tab[i]=np.bitwise_not(tab3[i])
            tab1[i]=np.bitwise_and(tab1[i],tab[i])
            tabroipat[pat1][i]= tab1[i]
            tab2[i]=np.bitwise_and(tab2[i],tab[i])
            tabroipat[pat2][i]= tab2[i]
            nm=True
#            cv2.imshow('tab11',tab1[i])
#            cv2.imshow('tab21',tab2[i])
#            cv2.imshow('tab31',tab3[i])
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()

    if nm:
            npd=os.path.join(namedirtopcf,pat)
            remove_folder(npd)
            os.mkdir(npd)
#            npdbmp=os.path.join(npd,scan_bmp)
#            os.mkdir(npdbmp)
            
            for i in range (0,slnt):
                 if tab3[i].max()>0:
#                    naf3=pat+'_'+str(i)+'.'+typei
                    naf3=tabscanName[i]+'.'+typei1

                    npdn3=os.path.join(npd,naf3)
#                    np.putmask(tab3[i],tab3[i]>0,1)
                    imgc=colorimage(tab3[i],classifc[pat])
#                    cv2.imwrite(npdn3,normi(tab3[i]))
                    cv2.imwrite(npdn3,imgc)

#                    naf2=pat2+'_'+str(i)+'.'+typei
                    naf2=tabscanName[i]+'.'+typei1
                    npd2=os.path.join(namedirtopcf,pat2)
#                    npd2=os.path.join(npd2,scan_bmp)
                    npdn2=os.path.join(npd2,naf2)
                    imgc=colorimage(tab2[i],classifc[pat2])
#                    cv2.imwrite(npdn2,normi(tab2[i]))
#                    cv2.imwrite(npdn2,normi(tab2[i]))
                    cv2.imwrite(npdn2,imgc)



#                    naf1=pat1+'_'+str(i)+'.'+typei
                    naf1=tabscanName[i]+'.'+typei1
                    npd1=os.path.join(namedirtopcf,pat1)
#                    npd1=os.path.join(npd1,scan_bmp)
                    npdn1=os.path.join(npd1,naf1)
                    imgc=colorimage(tab1[i],classifc[pat1])
#                    cv2.imwrite(npdn1,normi(tab1[i]))
                    cv2.imwrite(npdn1,imgc)
    return tab3


listdirc= [ name for name in os.listdir(namedirtopc) if os.path.isdir(os.path.join(namedirtopc, name)) and \
            name not in alreadyDone]

 
print 'class used :',usedclassif

for f in listdirc:

    print('work on:',f)
    errorfile = open(eferror, 'a')
    tn = datetime.datetime.now()
    todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
    errorfile.write('started ' +namedirHUG+' '+f+' at :'+todayn)
    errorfile.close()

    nbpf=0
    listsliceok=[]
    tabroipat={}
#    namedirtopcf=namedirtopc+'/'+f
    namedirtopcf=os.path.join(namedirtopc,f)

    if os.path.isdir(namedirtopcf):
        sroidir=os.path.join(namedirtopcf,sroi)
        remove_folder(sroidir)
        os.mkdir(sroidir)

    dimtabx,dimtaby,slnt = genepara(namedirtopcf)

    tabsroi=np.zeros((slnt,dimtabx,dimtaby,3),np.uint8)
#    tabbg =np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    tabscan =np.zeros((slnt,dimtabx,dimtaby),np.uint16)
    tabslung =np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    tabscanName={}
    tabscan,tabsroi,tabscanName=genebmp(namedirtopcf, source,tabscanName)
    if  os.path.exists(os.path.join(namedirtopcf, lungmask)):
        tabslung,a,b=genebmp(namedirtopcf, lungmask,tabscanName)
    else:
        tabslung,a,b=genebmp(namedirtopcf, lungmask1,tabscanName)

    for i in usedclassif:
        tabroipat[i]=np.zeros((slnt,dimtabx,dimtaby),np.uint8)

    contenudir = [name for name in os.listdir(namedirtopcf) if name in usedclassif and name not in derivedpat]
    for i in contenudir:
        tabroipat[i],tabsroi,a=genebmp(namedirtopcf, i,tabscanName)

    for i in derivedpat:
        tabroipat[i]=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
        tabroipat[i]=calnewpat(namedirtopcf,i,slnt,dimtabx,dimtaby,tabscanName)

    contenudir = [name for name in os.listdir(namedirtopcf) if name in usedclassif]
    for i in contenudir:
        nbp=pavs(namedirtopcf,i,slnt,dimtabx,dimtaby,tabscanName)
#    pavbg(namedirtopcf,slnt,dimtabx,dimtaby)

        nbpf=nbpf+nbp
    namenbpat=namedirHUG+'_nbpat_'+f+'.txt'
    ofilepw = open(os.path.join(jpegpath,namenbpat), 'w')

    ofilepw.write('number of patches: '+str(nbpf))
    ofilepw.close()
    errorfile = open(eferror, 'a')
    tn = datetime.datetime.now()
    todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
    errorfile.write('completed ' +f+' at :'+todayn)
    errorfile.close()
#

#################################################################
totalpat(jpegpath)
totalnbpat (patchtoppath,picklepathdir)
genelabelloc(patchtoppath,plabelfile,jpegpath)
##########################################################
errorfile = open(eferror, 'a')
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
errorfile.write('completed ' +namedirHUG+' '+subHUG+' at :'+todayn)
errorfile.write('---------------\n')
errorfile.close()
#print listslice
print('completed')