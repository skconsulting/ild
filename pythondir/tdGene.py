# coding: utf-8
'''create patches from patient on front view GRE, including ROI when overlapping
    sylvain kritter 11 Aug 2017
 '''
 
from param_pix_t import classif,derivedpat,usedclassif,classifc
from param_pix_t import dimpavx,dimpavy,typei,avgPixelSpacing,thrpatch,perrorfile,plabelfile,setdata,pxy
from param_pix_t import remove_folder,normi,genelabelloc,totalpat,totalnbpat,fidclass
from param_pix_t import white
from param_pix_t import patchpicklename,lungmask,patchesdirname
from param_pix_t import imagedirname,picklepath,source,transbmp,sroid ,typei1
 
import os
import cv2
import datetime

import numpy as np

import cPickle as pickle
import dicom

#####################################################################
#define the working directory
#global directory for scan file
topdir='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool'
namedirHUG = 'CHU'
#subdir for roi in text
subHUG='UIP0'
#subHUG='UIP_106530'
#subHUG='UIP0'
#subHUG='UIP_S14740'


#global directory for output patches file
toppatch= 'TOPPATCH'
#extension for output dir
extendir='set0d'
extendir1='3d'
extendir='essaiset0d3d'
############################################################
if len (extendir1)>0:
    extendir1='_'+extendir1

alreadyDone =['S4550','S106530','S107260','S139430','S145210','S14740',
              'S15440','S1830','S28200','S335940','S359750',
              'S72260','S72261']
alreadyDone =[]

#labelEnh=('consolidation','reticulation,air_trapping','bronchiectasis','cysts')
labelEnh=()
locabg='anywhere_CHUG'

isGre=True

labelbg='back_ground'
locabg='td_CHUG'

reserved=['bgdir','sroi','sroi1','bgdir3d','sroi3d']
notclas=['lung','source','B70f']

patchesdirnametop = 'th'+str(round(thrpatch,1))+'_'+toppatch+'_'+extendir+extendir1
print 'name of directory for patches :', patchesdirnametop
#define the name of directory for patches

cwdtop=topdir
#path for HUG dicom patient parent directory
path_HUG=os.path.join(cwdtop,namedirHUG)
##directory name with patient databases
namedirtopc =os.path.join(path_HUG,subHUG)
print('source directory ',namedirtopc)
if not os.path.exists(namedirtopc):
    print('directory ',namedirtopc, ' does not exist!')

#
#dirHUG=os.path.join(cwdtop,HUG)
#patchtoppath=os.path.join(dirHUG,patchesdirnametop)

listHug= [ name for name in os.listdir(namedirtopc) if os.path.isdir(os.path.join(namedirtopc, name)) and \
            name not in alreadyDone]
print 'list of patients :',listHug


patchtoppath=os.path.join(cwdtop,patchesdirnametop)
#create patch and jpeg directory
patchpath=os.path.join(patchtoppath,patchesdirname)
#path for patch pickle
picklepathdir =os.path.join(patchtoppath,picklepath)
#print 'picklepathdir',picklepathdir

#define the name for jpeg files
jpegpath=os.path.join(patchtoppath,imagedirname)


if not os.path.isdir(patchtoppath):
    os.mkdir(patchtoppath)

#remove_folder(patchpath)
if not os.path.isdir(patchpath):
    os.mkdir(patchpath)

#remove_folder(patchNormpath)
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
errorfile.write('--------------------------------\n')

errorfile.write('source directory '+namedirtopc+'\n')
errorfile.write('th : '+ str(thrpatch)+'\n')
errorfile.write('name of directory for patches :'+ patchesdirnametop+'\n')
errorfile.write( 'list of patients :'+str(listHug)+'\n')
errorfile.write('using pattern set: ' +setdata+'\n')
for pat in usedclassif:
    errorfile.write(pat+'\n')
errorfile.write('--------------------------------\n')
errorfile.close()

###############################################################


reserved=reserved+derivedpat


def reshapeScan(tabscan):
    print 'reshape scan'
    tabres=np.moveaxis(tabscan,0,1)
    return tabres

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
       
    FilesDCM =(os.path.join(dirFileP,fileList[0]))
    FilesDCM1 =(os.path.join(dirFileP,fileList[1]))
    RefDs = dicom.read_file(FilesDCM,force=True)
    RefDs1 = dicom.read_file(FilesDCM1,force=True)
    patientPosition=RefDs.PatientPosition
#    SliceThickness=RefDs.SliceThickness
    try:
            slicepitch = np.abs(RefDs.ImagePositionPatient[2] - RefDs1.ImagePositionPatient[2])
    except:
            slicepitch = np.abs(RefDs.SliceLocation - RefDs1.SliceLocation)

    
    SliceThickness=RefDs.SliceThickness
    try:
            SliceSpacingB=RefDs. SpacingBetweenSlices
    except AttributeError:
             print "Oops! No Slice spacing..."
             SliceSpacingB=0

#    slicepitch=float(SliceThickness)+float(SliceSpacingB)
    print 'slice Thickness :',SliceThickness
    print 'Slice spacing',SliceSpacingB
    print 'slice pitch in z :',slicepitch
    print 'patient position :',patientPosition
    errorfile = open(eferror, 'a')
    errorfile.write('---------------\n')

    errorfile.write('number of slices :'+str(slnt)+'\n')
    errorfile.write('slice Thickness :'+str(SliceThickness)+'\n')
    errorfile.write('slice spacing :'+str(SliceSpacingB)+'\n')
    errorfile.write('slice pitch in z :'+str(slicepitch)+'\n')
    errorfile.write('patient position  :'+str(patientPosition)+'\n')

    errorfile.write('--------------------------------\n')
    errorfile.close()
    slnt=slnt+1
    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
    dsr= RefDs.pixel_array
    dsr= dsr-dsr.min()
    dsr=dsr.astype('int16')
    dsrresize=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
    dimtabx=int(dsrresize.shape[0])
    dimtaby=int(dsrresize.shape[1])
    return dimtabx,dimtaby,slnt ,slicepitch

def tagviews(tab,text,x,y):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,text,(x, y), font,0.3,white,1)
    return viseg


def genebmp(dirName, sou,slnt,dx,dy,tabscanName):
    """generate patches from dicom files and sroi"""

    if sou==source:
        tabres=np.zeros((slnt,dx,dy),np.int16)
    else:
        tabres=np.zeros((slnt,dx,dy),np.uint8)


    dirFileP = os.path.join(dirName, sou)

    (top,tail)=os.path.split(dirName)
    print ('generate image in :',tail, 'directory :',sou)
    fileList =[name for name in  os.listdir(dirFileP) if ".dcm" in name.lower()]

    for filename in fileList:
                FilesDCM =(os.path.join(dirFileP,filename))
                RefDs = dicom.read_file(FilesDCM)
                dsr= RefDs.pixel_array
                dsr=dsr.astype('int16')
                fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
                scanNumber=int(RefDs.InstanceNumber)
                endnumslice=filename.find('.dcm')
                imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)
                if dsr.max()>dsr.min():
                    if sou !=source :
                        dsr=normi(dsr)
                        dsr=cv2.resize(dsr,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
                        if sou == lungmask:
                            np.putmask(dsr,dsr>0,100)

                        else:
                            np.putmask(dsr,dsr==1,0)
                            np.putmask(dsr,dsr>0,100)
                    else :
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
                        tabscanName[scanNumber]=imgcoredeb

                    tabres[scanNumber]=  dsr

    return tabres,tabscanName

def contour2(im,l,dx,dy):
    col=classifc[l]
    vis = np.zeros((dx,dy,3), np.uint8)
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
#    print (labnow, text)
    if label == 'back_ground':
        deltay=30
    else:
#        deltay=25*((labnow-1)%5)
        deltay=40+10*(labnow-1)

    viseg=cv2.putText(tab,label,(x, y+deltay), font,0.3,col,1)
    return viseg


def pavs (dirName,pat,dx,dy):
    """ generate patches from ROI"""
    ntotpat=0

    tabf=np.zeros((dx,dy),np.uint8)
    _tabsroi=np.zeros((dx,dy,3),np.uint8)
    _tabscan = np.zeros((dx,dy),np.int16)

    (top,tail)=os.path.split(dirName)
    print 'pav :',tail,'pattern :',pat
    patpickle=[]
    nampadir=os.path.join(patchpath,pat)
    nampadirl=os.path.join(nampadir,locabg)
    if not os.path.exists(nampadir):
         os.mkdir(nampadir)
         os.mkdir(nampadirl)

    pathpicklepat=os.path.join(picklepathdir,pat)
#    print pathpicklepat
    pathpicklepatl=os.path.join(pathpicklepat,locabg)
    patchpicklenamepatient=tail+'_'+patchpicklename

    pathpicklepatfile=os.path.join(pathpicklepatl,patchpicklenamepatient)
    if not os.path.exists(pathpicklepat):
         os.mkdir(pathpicklepat)
    if not os.path.exists(pathpicklepatl):
         os.mkdir(pathpicklepatl)
    if os.path.exists(pathpicklepatfile):
        os.remove(pathpicklepatfile)

    for scannumb in range (0,dy):
       tabp = np.zeros((dx, dy), dtype=np.uint8)
       tabf=np.copy(tabroipat3d[pat][scannumb])

       tabfc=np.copy(tabf)
       nbp=0
       if tabf.max()>0:
           vis=contour2(tabf,pat,dx,dy)
           if vis.sum()>0:
                _tabsroi = np.copy(tabsroi3d[scannumb])
                imn=cv2.add(vis,_tabsroi)
                imn=tagview(imn,pat,0,20)
                tabsroi3d[scannumb]=imn
                imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)

                sroifile='tr_'+str(scannumb)+'.'+typei
                filenamesroi=os.path.join(sroidir,sroifile)
                cv2.imwrite(filenamesroi,imn)

                np.putmask(tabf,tabf>0,1)

                atabf = np.nonzero(tabf)

                xmin=atabf[1].min()
                xmax=atabf[1].max()
                ymin=atabf[0].min()
                ymax=atabf[0].max()


                _tabscan=tabscan3d[scannumb]

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

             stw=namedirHUG+'_'+tail+'_slice_'+str(scannumb)+'_'+pat+'_'+locabg+'_'+str(nbp)
             stww=stw+'.txt'
             flw=os.path.join(jpegpath,stww)
             mfl=open(flw,"w")
             mfl.write('#number of patches: '+str(nbp)+'\n')
             mfl.close()
             stww=stw+'.'+typei
             flw=os.path.join(jpegpath,stww)
#             scipy.misc.imsave(flw, tabfc)
             cv2.imwrite(flw, tabfc)
             pickle.dump(patpickle, open(pathpicklepatfile, "wb"),protocol=-1)

    return ntotpat



def wtebres(dirf,tab,dx,slicepitch,fxs,dxd):

    (top,tail1)=os.path.split(dirf)
    print 'wtebres' ,tail1
    (top1,tail)=os.path.split(top)
    if tail1==source:
        tabres=np.zeros((dx,dxd,dx),np.int16)
    else:
        tabres=np.zeros((dx,dxd,dx),np.uint8)
    tabsroi=np.zeros((dx,dxd,dx,3),np.uint8)

    imgresize=np.zeros((dx,dxd,dx),np.uint8)

    wridir=os.path.join(dirf,transbmp)
    remove_folder(wridir)
    os.mkdir(wridir)

    sroidir=os.path.join(top,sroid)

    for i in range (0,dx):
        if tab[i].max()>1:

            imgresize=cv2.resize(tab[i],None,fx=1,fy=fxs,interpolation=cv2.INTER_LINEAR)
#            print imgresize.shape

            trcore=tail1+'_'+str(i)+'.'
            trscan=trcore+typei
            trscanbmp=trcore+typei
            trscanroi= 'tr_'+str(i)+'.'+typei


            if tail1==lungmask:
                namescan=os.path.join(wridir,trscanbmp)
                cv2.imwrite (namescan, imgresize)
                tabres[i]=imgresize


            if tail1==source:
                trscan=os.path.join(wridir,trscan)
                tabres[i]=imgresize
                imgtowrite=normi(imgresize)
                cv2.imwrite (trscan, imgtowrite)

                namescan=os.path.join(sroidir,trscanroi)
                textw='n: '+tail+' scan: '+str(i)
#                print imgresize.min(),imgresize.max()

#                print imgresize.min(),imgresize.max()
                imgtowrite=cv2.cvtColor(imgtowrite,cv2.COLOR_GRAY2BGR)
                tagviews(imgtowrite,textw,0,20)
                tabsroi[i]=imgtowrite
                cv2.imwrite (namescan, imgtowrite)
            else:
                trscan=os.path.join(wridir,trscanbmp)
                tabres[i]=  imgresize
                cv2.imwrite (trscan, imgresize)

    return tabres,tabsroi

def calnewpat(dirName,pat,slnt,dimtabx,dimtaby):
    print 'new pattern : ',pat

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
            npd=os.path.join(dirName,pat)
            remove_folder(npd)
            os.mkdir(npd)
            npd=os.path.join(npd,transbmp)
            remove_folder(npd)
            os.mkdir(npd)

            for i in range (0,slnt):
                 if tab3[i].max()>0:
                    naf3=pat+'_'+str(i)+'.'+typei
                    npdn3=os.path.join(npd,naf3)

                    cv2.imwrite(npdn3,tab3[i])

                    naf2=pat2+'_'+str(i)+'.'+typei
                    npd2=os.path.join(dirName,pat2)
                    npd2=os.path.join(npd2,transbmp)
                    npdn2=os.path.join(npd2,naf2)

                    cv2.imwrite(npdn2,tab2[i])

                    naf1=pat1+'_'+str(i)+'.'+typei
                    npd1=os.path.join(dirName,pat1)
                    npd1=os.path.join(npd1,transbmp)
                    npdn1=os.path.join(npd1,naf1)
                    cv2.imwrite(npdn1,tab1[i])
    return tab3

def colorimage(image,color):
#    im=image.copy()
    im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    np.putmask(im,im>0,color)

    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    return im


def genebackground(namedir):
    for sln in range(1,slnt):
        tabpbac=np.copy(tabslung[sln])
#        
        patok=False
        for pat in usedclassif:
            if pat !=fidclass(0,classif):
#                print sln,pat
                tabpat=tabroipat[pat][sln]
#                print tabpat.shape
#                if pat =='ground_glass':
#                    print tabpat.shape
#                    cv2.imshow(pat+str(sln),normi(tabpat))
#                    cv2.waitKey(0)
#                    cv2.destroyAllWindows()
                if tabpat.max()>0:
                    patok=True
#                    tabp=cv2.cvtColor(tabpat,cv2.COLOR_BGR2GRAY)
                    np.putmask(tabpat,tabpat>0,255)
                    mask=np.bitwise_not(tabpat)
                    tabpbac=np.bitwise_and(tabpbac,mask)
#                    print tabroipat[fidclass(0,classif)][sln].shape
                    tabroipat[fidclass(0,classif)][sln]=tabpbac
                    
#                    print tabroipat[fidclass(0,classif)][sln].shape
#                    if sln == 13:
#                        cv2.imshow(str(sln)+pat, tabroipat[fidclass(0,classif)][sln])
#                        cv2.waitKey(0)
#                        cv2.destroyAllWindows()
        if patok:
            labeldir=os.path.join(namedir,fidclass(0,classif))
            if not os.path.exists(labeldir):
               os.mkdir(labeldir)
            namepat=tabscanName[sln]+'.'+typei1
            imgcoreScan=os.path.join(labeldir,namepat)
    #                imgcoreScan=os.path.join(locadir,namepat)
            tabtowrite=colorimage(tabroipat[fidclass(0,classif)][sln],classifc[fidclass(0,classif)])
#            tabtowrite=cv2.cvtColor(tabtowrite,cv2.COLOR_BGR2RGB)
            cv2.imwrite(imgcoreScan,tabtowrite)    

#####################################################################################

for f in listHug:
    print f
    errorfile = open(eferror, 'a')
    tn = datetime.datetime.now()
    todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
    errorfile.write('started ' +namedirHUG+' '+f+' at :'+todayn)
    errorfile.close()
    dirf=os.path.join(namedirtopc,f)

    sroidir=os.path.join(dirf,sroid)
    remove_folder(sroidir)
    os.mkdir(sroidir)
    listsliceok=[]
    tabroipat={}
    tabroipat3d={}

    if isGre:
        tabscanName={}
        dimtabx,dimtaby,slnt,slicepitch = genepara(dirf)
        tabscan =np.zeros((slnt,dimtabx,dimtaby),np.int16)
        tabslung =np.zeros((slnt,dimtabx,dimtaby),np.uint8)

        fxs=float(slicepitch/avgPixelSpacing )
        dimtabxd=int(round(fxs*slnt,0))
#        print 'dimtabxd', dimtabxd
        dimtabyd=dimtaby
        print 'dimtabx:',dimtabx,'dimtabxd:',dimtabxd,'dimtabyd:',dimtabyd,'slnt:',slnt

        for i in usedclassif+derivedpat :
            tabroipat[i]=np.zeros((slnt,dimtabx,dimtaby),np.uint8)
            tabroipat3d[i]=np.zeros((dimtabx,dimtabxd,dimtabyd),np.uint8)

        tabscan3d =np.zeros((dimtabx,dimtabxd,dimtabyd),np.int16)
        tabslung3d =np.zeros((dimtabx,dimtabxd,dimtabyd),np.uint8)
        tabsroi3d =np.zeros((dimtabx,dimtabxd,dimtabyd,3),np.uint8)
        tabres=np.zeros((dimtabx,slnt,dimtabyd),np.int16)

        dirg=os.path.join(dirf,source)
        tabscan,tabscanName=genebmp(dirf, source,slnt,dimtabx,dimtaby,tabscanName)

        tabres=reshapeScan(tabscan)

        tabscan3d,tabsroi3d=wtebres(dirg,tabres,dimtabx,slicepitch,fxs,dimtabxd)

        dirg=os.path.join(dirf,lungmask)
        tabslung,a=genebmp(dirf, lungmask,slnt,dimtabx,dimtaby,tabscanName)
        tabres=reshapeScan(tabslung)
        tablung3,a=wtebres(dirg,tabres,dimtabx,slicepitch,fxs,dimtabxd)

        contenudir = [name for name in os.listdir(dirf) if name in usedclassif and name not in derivedpat]

        for g in contenudir:
            tabroipat[g],a=genebmp(dirf, g,slnt,dimtabx,dimtaby,tabscanName)

        for i in derivedpat:
           tabroipat[i]=calnewpat(dirf,i,slnt,dimtabx,dimtaby)
        genebackground(dirf)
        contenudir = [name for name in os.listdir(dirf) if name in usedclassif]
        nbpf=0
        for g in contenudir:
            print g
            if  tabroipat[g].max()>0:
                dirg=os.path.join(dirf,g)
                tabres=reshapeScan(tabroipat[g])
                tabroipat3d[g],a=wtebres(dirg,tabres,dimtabx,slicepitch,fxs,dimtabxd)
                nbp=pavs(dirf,g,dimtabxd,dimtabyd)
                nbpf=nbpf+nbp
        namenbpat=namedirHUG+'_nbpat_'+f+'.txt'
        ofilepw = open(os.path.join(jpegpath,namenbpat), 'w')

        ofilepw.write('number of patches: '+str(nbpf))
        ofilepw.close()
#        pavbg(dirf,dimtabxd,dimtabyd)
    else:
       print 'is not gre'
    errorfile = open(eferror, 'a')
    tn = datetime.datetime.now()
    todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
    errorfile.write('completed ' +f+' at :'+todayn)
    errorfile.write('-------\n')
    errorfile.close()
#################################################################
totalpat(jpegpath)
totalnbpat (patchtoppath,picklepathdir)
genelabelloc(patchtoppath,plabelfile,jpegpath)
errorfile = open(eferror, 'a')
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
errorfile.write('completed  at :'+todayn)
errorfile.close()