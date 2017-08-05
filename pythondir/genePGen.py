# -*- coding: utf-8 -*-
#Sylvain Kritter 4 august 2017
"""
Top file to generate patches from DICOM database Geneva with HU
this is cross view only
no support of superimposed patterns
first step

version 1.0
"""
from param_pix_t import classif,usedclassif,classifc
from param_pix_t import dimpavx,dimpavy,typei,avgPixelSpacing,thrpatch
from param_pix_t import remove_folder,normi,genelabelloc,totalpat,totalnbpat
from param_pix_t import white
from param_pix_t import patchpicklename,scan_bmp,lungmask,lungmask1,lungmaskbmp,sroi,patchesdirname
from param_pix_t import imagedirname,picklepath,patchfile,source,perrorfile,plabelfile

import cPickle as pickle
import cv2
import datetime
import dicom
import numpy as np
import os

import scipy.misc


#general parameters and file, directory names
#######################################################
#customisation part for datataprep
#global directory for scan file
topdir='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool'
namedirHUG = 'HUG'
#subdir for roi in text
subHUG='ILD_TXT'
#subHUG='ILD3721'
#subHUG='ILD3'
#subHUG='ILDt'

toppatch= 'TOPPATCH'
#extension for output dir
#extendir='essaihug'
extendir='set1'


#labelEnh=('consolidation','reticulation,air_trapping','bronchiectasis','cysts')
labelEnh=()
#imageDepth=255 #number of bits used on dicom images (2 **n)
# average pxixel spacing
print 'class used :',usedclassif
########################################################################
######################  end ############################################
########################################################################
#define the name of directory for patches
patchesdirnametop = 'th'+str(round(thrpatch,1))+'_'+toppatch+'_'+extendir
print patchesdirnametop
#define the name of directory for patches

#full path names
cwdtop=topdir
#path for HUG dicom patient parent directory
path_HUG=os.path.join(cwdtop,namedirHUG)
##directory name with patient databases
namedirtopc =os.path.join(path_HUG,subHUG)
if not os.path.exists(namedirtopc):
    print('directory ',namedirtopc, ' does not exist!')

patchtoppath=os.path.join(cwdtop,patchesdirnametop)

#create patch and jpeg directory
patchpath=os.path.join(patchtoppath,patchesdirname)
#create patch and jpeg directory

#define the name for jpeg files
jpegpath=os.path.join(patchtoppath,imagedirname)

#path for patch pickle
picklepathdir =os.path.join(patchtoppath,picklepath)

#patchpath = final/patches
if not os.path.isdir(patchtoppath):
    os.mkdir(patchtoppath)

#remove_folder(patchpath)
if not os.path.isdir(patchpath):
    os.mkdir(patchpath)

#remove_folder(jpegpath)
if not os.path.isdir(jpegpath):
    os.mkdir(jpegpath)

if not os.path.isdir(picklepathdir):
    os.mkdir(picklepathdir)


eferror=os.path.join(patchtoppath,perrorfile)
errorfile = open(eferror, 'a')
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
errorfile.write('started ' +namedirHUG+' '+subHUG+' at :'+todayn)
errorfile.close()

#filetowrite=os.path.join(namedirtopc,'lislabel.txt')



def genepara(namedirtopcf):
#    dirFileP = os.path.join(namedirtopcf, 'source')
        #list dcm files
    fileList =[name for name in  os.listdir(namedirtopcf) if ".dcm" in name.lower()]
    slnt=0
    for filename in fileList:
        FilesDCM =(os.path.join(namedirtopcf,filename))
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


def genebmp(dirName):
    """generate patches from dicom files and sroi"""
    print ('generate  bmp files from dicom files in :',f)
    (top,tail)=os.path.split(dirName)
#    global constPixelSpacing, dimtabx,dimtaby
    #directory for patches
    bmp_dir = os.path.join(dirName, source)
    remove_folder(bmp_dir)
    os.mkdir(bmp_dir)
    bmp_dir = os.path.join(bmp_dir, scan_bmp)
    remove_folder(bmp_dir)
    os.mkdir(bmp_dir)
#    bgdirf = os.path.join(dirName, bgdir)
#    remove_folder(bgdirf)
#    os.mkdir(bgdirf)
    lung_dir = os.path.join(dirName, lungmask)
    if not os.path.exists(lung_dir):
        lung_dir = os.path.join(dirName, lungmask1)
#            print lung_dir
    lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)

    remove_folder(lung_bmp_dir)
#             if lungmaskbmp not in lunglist:
    os.mkdir(lung_bmp_dir)

    #list dcm files
    fileList = [name for name in os.listdir(dirName) if ".dcm" in name.lower()]
    lunglist = [name for name in os.listdir(lung_dir) if ".dcm" in name.lower()]

    tabscan=np.zeros((slnt,dimtabx,dimtaby),np.int16)
    tabsroi=np.zeros((slnt,dimtabx,dimtaby,3),np.uint8)
#    os.listdir(lung_dir)
    for filename in fileList:
#            print(filename)
#        if ".dcm" in filename.lower():  # check whether the file's DICOM
            FilesDCM =(os.path.join(dirName,filename))
            RefDs = dicom.read_file(FilesDCM)
            dsr= RefDs.pixel_array
            dsr=dsr.astype('int16')
            fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
#                print 'fxs',fxs
            scanNumber=int(RefDs.InstanceNumber)
            endnumslice=filename.find('.dcm')
            imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)+'.'
#            bmpfile=os.path.join(dirFilePbmp,imgcore)
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

            tabscan[scanNumber]=dsr

            imgcored=imgcoredeb+typei
            bmpfiled=os.path.join(bmp_dir,imgcored)
            imgcoresroi='sroi_'+str(scanNumber)+'.'+typei
            bmpfileroi=os.path.join(sroidir,imgcoresroi)
#            print imgcoresroi,bmpfileroi
            textw='n: '+tail+' scan: '+str(scanNumber)

            cv2.imwrite (bmpfiled, dsrforimage)
            dsrforimage=cv2.cvtColor(dsrforimage,cv2.COLOR_GRAY2BGR)
            dsrforimage=tagviews(dsrforimage,textw,0,20)
            cv2.imwrite (bmpfileroi, dsrforimage)
            tabsroi[scanNumber]=dsrforimage


    for lungfile in lunglist:
#             print(lungfile)
#             if ".dcm" in lungfile.lower():  # check whether the file's DICOM
                FilesDCM =(os.path.join(lung_dir,lungfile))
                RefDs = dicom.read_file(FilesDCM)
                dsr= RefDs.pixel_array
                dsr=dsr.astype('int16')
                fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
#                print 'fxs',fxs
                scanNumber=int(RefDs.InstanceNumber)
                endnumslice=filename.find('.dcm')
                imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)+'.'
                imgcore=imgcoredeb+typei
                bmpfile=os.path.join(lung_bmp_dir,imgcore)
                dsr=normi(dsr)
                dsrresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
                cv2.imwrite (bmpfile, dsrresize)

    return tabscan,tabsroi

def reptfulle(tabc,dx,dy,col):
    imgi = np.zeros((dx,dy,3), np.uint8)
    cv2.polylines(imgi,[tabc],True,col)
    cv2.fillPoly(imgi,[tabc],col)
#    tabzi = np.array(imgi)
#    tabz = tabzi[:, :,1]
    return imgi

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


def contour2(im,l):
    col=classifc[l]
#    print l
#    print 'dimtabx' , dimtabx
    vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,0,255,0)
    im2,contours0, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
    cv2.drawContours(vis,contours,-1,col,1,cv2.LINE_AA)
    return vis


def pavs (namedirtopcf,label,loca,slnt,numslice):
    """ generate patches from ROI"""
    (top,tail)=os.path.split(namedirtopcf)
#    print 'pav :',tail,'pattern :',label,'loca :',loca, 'slice :',numslice
    ntotpat=0
    tabf=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    _tabsroi=np.zeros((slnt,dimtabx,dimtaby,3),np.uint8)
#    _tabbg = np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    _tabscan = np.zeros((slnt,dimtabx,dimtaby),np.uint8)
    patpickle=[]

    pxy=float(dimpavx*dimpavy)

    nampadir=os.path.join(patchpath,label)
    nampadirl=os.path.join(nampadir,loca)
    if not os.path.exists(nampadir):
         os.mkdir(nampadir)
         os.mkdir(nampadirl)

    pathpicklepat=os.path.join(picklepathdir,label)
#    print pathpicklepat
    pathpicklepatl=os.path.join(pathpicklepat,loca)
    patchpicklenamepatient=namedirHUG+'_'+subHUG+'_'+tail+'_'+numslice+'_'+patchpicklename

    pathpicklepatfile=os.path.join(pathpicklepatl,patchpicklenamepatient)
    if not os.path.exists(pathpicklepat):
         os.mkdir(pathpicklepat)
    if not os.path.exists(pathpicklepatl):
         os.mkdir(pathpicklepatl)
    if os.path.exists(pathpicklepatfile):
        os.remove(pathpicklepatfile)

#    for scannumb in range (0,slnt):
    tabp = np.zeros((dimtabx, dimtaby), dtype='i')
    tabf=np.copy(tabroipat[label][numslice])
    tabfc=np.copy(tabf)
    tabfc=cv2.cvtColor(tabfc, cv2.COLOR_BGR2GRAY)
    nbp=0
    if tabf.max()>0:
           vis=contour2(tabf,label)
           if vis.sum()>0:

                _tabsroi = np.copy(tabsroi[int(numslice)])
                imn=cv2.add(vis,_tabsroi)
                imn=tagview(imn,label,0,100)
                tabsroi[int(numslice)]=imn
                imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)
                sroifile='sroi_'+str(numslice)+'.'+typei
                filenamesroi=os.path.join(sroidir,sroifile)
#                print 'filenamesroi',filenamesroi
                cv2.imwrite(filenamesroi,imn)

                atabf = np.nonzero(tabf)

                xmin=atabf[1].min()
                xmax=atabf[1].max()
                ymin=atabf[0].min()
                ymax=atabf[0].max()
                tabf=cv2.cvtColor(tabf, cv2.COLOR_BGR2GRAY)
                np.putmask(tabf,tabf>0,1)
                _tabscan=tabscan[int(numslice)]
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
#                                nampa=os.path.join(nampadirl,namedirHUG+'_'+tail+
#                                                   '_'+str(numslice)+'_'+str(i)+'_'+str(j)+'_'+str(nbp)+'.'+typei )                                
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
             tabfc =tabfc+tabp
             ntotpat=ntotpat+nbp
             if numslice not in listsliceok:
                    listsliceok.append(numslice)
             stw=namedirHUG+'_'+tail+'_slice_'+str(numslice)+'_'+label+'_'+loca+'_'+str(nbp)
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

    return ntotpat


def fileext(namefile,curdir,patchpath):
    listlabel=[]

    ofi = open(namefile, 'r')
    t = ofi.read()
    #print( t)
    ofi.close()

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
#            print('slash',loca)
            loca=loca.replace('/','_')
#            print('after',loca)

#        print('label',label)
#        print('localisation',loca)
        if label=='fibrosis':
            label='HC'
        if label not in listlabel:
                    plab=os.path.join(patchpath,label)
                    ploc=os.path.join(plab,loca)

                    listlabel.append(label+'_'+loca)
                    listlabeld=os.listdir(patchpath)
                    if label not in listlabeld:
#                            print label
                            os.mkdir(plab)
#                            os.mkdir(plabNorm)
                    listlocad=os.listdir(plab)
                    if loca not in listlocad:
                            os.mkdir(ploc)
#                            os.mkdir(plocNorm)

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
                nametab1='slice_'+str(slice)+'_'+str(label)+'_'+str(loca)+'_'+str(nbpoint)+'.txt'
                nametab2=os.path.join(curdir,patchfile)
                nametab=os.path.join(nametab2,nametab1)

#                print 'nametab',nametab
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
    return(listlabel,coefi)

listdirc= (os.listdir(namedirtopc))
npat=0

for f in listdirc:
    #f = 35
    print('work on:',f)
    errorfile = open(eferror, 'a')
    tn = datetime.datetime.now()
    todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
    errorfile.write('started ' +f+' at :'+todayn)
    errorfile.close()
    nbpf=0
#    tabroipat[pat][scannumb]

    tabroipat={}
    for label in usedclassif:
        tabroipat[label]={}

    
    listsliceok=[]
    posp=f.find('.',0)
    posu=f.find('_',0)
    namedirtopcf=os.path.join(namedirtopc,f)
    sroidir=os.path.join(namedirtopcf,sroi)
    if os.path.exists(sroidir):
        remove_folder(sroidir)
    os.mkdir(sroidir)

    pathpatchfilecomplet=os.path.join(namedirtopcf,patchfile)
#    pathpatchfilecomplet=unicode(pathpatchfilecomplet)
#    print type(pathpatchfilecomplet)
    if os.path.exists(pathpatchfilecomplet):
        remove_folder(pathpatchfilecomplet)
    os.mkdir(pathpatchfilecomplet)
    #namedirtopcf = final/ILD_DB_txtROIs/35
    if posp==-1 and posu==-1:
        contenudir = os.listdir(namedirtopcf)
#        print(contenudir)
        fif=False
        dimtabx,dimtaby,slnt = genepara(namedirtopcf)
#        print dimtabx,dimtaby,slnt
        tabscan,tabsroi=genebmp(namedirtopcf)

        for f1 in contenudir:

            if f1.find('.txt') >0 and (f1.find('CT')==0 or \
             f1.find('Tho')==0):
#                print f1
                npat+=1
                fif=True
                fileList =f1
                pathf1=os.path.join(namedirtopcf,fileList)
                labell,coefi =fileext(pathf1,namedirtopcf,patchpath)
#                print labell,coefi
                break
        if not fif:
             print('ERROR: no ROI txt content file', f)
             errorfile.write('ERROR: no ROI txt content file in: '+ f+'\n')

        listslice= os.listdir(os.path.join(namedirtopcf,patchfile) )
#        print('listslice',listslice)
        listcore =[]
        for l in listslice:
#                print(pathl)
            il1=l.find('.',0)
            j=0
            while l.find('_',il1-j)!=-1:
                j-=1
            ilcore=l[0:il1-j-1]
            if ilcore not in listcore:
                listcore.append(ilcore)
    #pathl=final/ILD_DB_txtROIs/35/patchfile/slice_2_micronodulesdiffuse_1.txt
#        print('listcore',listcore)
        for c in listcore:
#            print 'debut c',c
            ftab=True
            imgc = np.zeros((dimtabx,dimtaby,3), np.uint8)
            for l in listslice:
#                print('l1',l,'c1:',c)
                if l.find(c,0)==0:
#                    print('debut l',l,'c:',c)


                    pathl=os.path.join(pathpatchfilecomplet,l)
                    tabcff = np.loadtxt(pathl,dtype='f')
                    ofile = open(pathl, 'r')
                    t = ofile.read()
                    #print( t)
                    ofile.close()
                    labpos=t.find('label')
                    labposend=t.find('\n',labpos)
                    labposdeb = t.find(' ',labpos)
                    label=t[labposdeb:labposend].strip()
#                    print label
                    locapos=t.find('local')
                    locaposend=t.find('\n',locapos)
                    locaposdeb = t.find(' ',locapos)
                    loca=t[locaposdeb:locaposend].strip()
                    pos=c.find('_',0)
                    pos1=c.find('_',pos+1)
                    numslice=c[pos+1:pos1]

                    tabccfi=tabcff/avgPixelSpacing

                    tabc=tabccfi.astype(int)

                    col=classifc[label]
                    imgi= reptfulle(tabc,dimtabx,dimtaby,col)

                    imgc=imgc+imgi
#                    print label


#                    if label=='bronchial_wall_thickening':
#                        print 'bronchial_wall_thickening'
#                        cv2.imshow('imgc',imgc)
#                        cv2.waitKey(0)
#                        cv2.destroyAllWindows()

#                    print('end create tables')
                    il=l.find('.',0)
                    iln=l[0:il]
#                    print('fin l',l,'c:',c)
#                    print iln,label,loca
#            print label,usedclassif
            if label in usedclassif:
#
#                print('c :',numslice,label,loca,)
                labeldir=os.path.join(namedirtopcf,label)
                if not os.path.exists(labeldir):
                    os.mkdir(labeldir)
                locadir=os.path.join(labeldir,loca)
                if not os.path.exists(locadir):
                    os.mkdir(locadir)
                namepat=f+'_'+numslice+'.'+typei
                imgcoreScan=os.path.join(locadir,namepat)
                tabtowrite=cv2.cvtColor(imgc,cv2.COLOR_BGR2RGB)
                cv2.imwrite(imgcoreScan,tabtowrite)
                tabroipat[label][numslice]=imgc
                nbp=pavs(namedirtopcf,label,loca,slnt,numslice)
#                print 'nbp',nbp

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



#################################################################
totalpat(jpegpath)
totalnbpat (patchtoppath,picklepathdir)
genelabelloc(patchtoppath,plabelfile,jpegpath)

##########################################################
errorfile = open(eferror, 'a')
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
errorfile.write('completed ' +namedirHUG+' '+subHUG+' at :'+todayn)
errorfile.close()
print('completed')