# coding: utf-8
#Sylvain Kritter 22 decembre 2016
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
namedirHUG = 'HUG'
#subdir for roi in text
subHUG='SEG'
att =10 #coef for color attenuation
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

segDir='segDir' #directory with roi overlay

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
lowgreen=(0,51,51)

usedclassif = [
        'consolidation',
        'fibrosis',
        'HC',
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
        'HC':1,
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

classifc ={
'back_ground':darkgreen,
#'consolidation':red,
'consolidation':cyan,
'HC':blue,
'fibrosis':blue,
'ground_glass':red,
#'ground_glass':yellow,
'healthy':darkgreen,
#'micronodules':cyan,
'micronodules':green,
#'reticulation':purple,
'reticulation':yellow,
'air_trapping':pink,
'cysts':lightgreen,
 'bronchiectasis':orange,
 'nolung': lowgreen,
 'bronchial_wall_thickening':white,
 'early_fibrosis':white,
 'emphysema':white,
 'increased_attenuation':white,
 'macronodules':white,
 'pcp':white,
 'peripheral_micronodules':white,
 'tuberculosis':white
 }

classifcseg ={
'consolidation':20,
'fibrosis':30,
'HC':30,
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
    print ('generate  bmp files from dicom files in :',f)
    global  dimtabx,dimtaby,fx
    #directory for patches
    bmp_dir = os.path.join(dirName, bmpname)
    seg_dirp = os.path.join(dirName, segDir)

    fileList = [name for name in os.listdir(dirName) if ".dcm" in name.lower()]

    for filename in fileList:
#        print(filename)
            FilesDCM =(os.path.join(dirName,filename))            
            ds = dicom.read_file(FilesDCM)
           
            dsr= ds.pixel_array          
            dsr= dsr-dsr.min()
            c=255.0/dsr.max()
            dsr=dsr*c
            dsr=dsr.astype('uint8')
            if int(ds.Rows) != 512:
                fx= 512.0/int(ds.Rows)
#                print fx
                dsr=cv2.resize(dsr,(512,512),interpolation=cv2.INTER_LINEAR)
            else:
                fx=1
            slicenumber=int(ds.InstanceNumber)                                 
            endnumslice=filename.find('.dcm')
            imgcore=filename[0:endnumslice]+'_'+str(slicenumber)+'.'+typei
            bmpfile=os.path.join(bmp_dir,imgcore)
            segfile=os.path.join(seg_dirp,imgcore)
            scipy.misc.imsave(bmpfile,dsr)
            scipy.misc.imsave(segfile,dsr)
            dimtabx=dsr.shape[0]
            dimtaby=dimtabx
            textw='n: '+f+' scan: '+str(slicenumber)
            
            #special sroiseg
            namescanseg=os.path.join(sroidirseg,imgcore) 
            tabsroiseg = np.zeros((dimtabx,dimtaby,3), np.uint8)  
            scipy.misc.imsave(namescanseg, tabsroiseg) 
            tagviews(namescanseg,textw,0,20)              
         
            lung_dir = os.path.join(dirName, lungmask)
#            print lung_dir
            lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)
            lunglist = [name for name in os.listdir(lung_dir) if ".dcm" in name.lower()]
            remove_folder(lung_bmp_dir)
#             if lungmaskbmp not in lunglist:
            os.mkdir(lung_bmp_dir)
#             print(lung_bmp_dir)
            for lungfile in lunglist:
#                print(lungfile)
           
                 lungDCM =os.path.join(lung_dir,lungfile)  
                 dslung = dicom.read_file(lungDCM)
                 slicelungnumber=int(dslung.InstanceNumber)       
                 dsrlung= dslung.pixel_array          

                 endnumslice=lungfile.find('.dcm')
                 lungcore=lungfile[0:endnumslice]+'_'+str(slicelungnumber)+'.'+typei
                 lungcoref=os.path.join(lung_bmp_dir,lungcore)

                 np.putmask(dsrlung,dsrlung>0,100)
                 scipy.misc.imsave(lungcoref,dsrlung)

def reptfulle(tabc,dx,dy):
    imgi = np.zeros((dx,dy,3), np.uint8)
    cv2.polylines(imgi,[tabc],True,(1,1,1)) 
    cv2.fillPoly(imgi,[tabc],(1,1,1))
#    tabzi = np.array(imgi)
#    tabz = tabzi[:, :,1]   
    return imgi
    

def tagviewb(fig,label,x,y):
    """write text in image according to label and color"""

    imgn=Image.open(fig)
    img=np.array(imgn)
    col=white
    extseg=str(classifcseg[label])
    labnow=classif[label]
#    print (labnow, text)
    if label == 'back_ground':
        x=0
        y=0        
        deltay=60
    else:        
        deltay=25*(labnow-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(img,label+' '+extseg,(x, y+deltay), font, 0.3,col,1)
    cv2.imwrite(fig, viseg)


def tagviewct(tab,label,x,y):
    """write text in image according to label and color"""

    col=classifc[label]
    labnow=classif[label]
#    print (labnow, text)
    if label == 'back_ground':
        x=0
        y=0        
        deltay=60
    else:        
        deltay=25*(labnow-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,label,(x, y+deltay), font, 0.3,col,1)
#    viseg = cv2.cvtColor(viseg,cv2.COLOR_RGB2BGR)
    return viseg


def tagviews(fig,text,x,y):
    """write simple text in image """
    imgn=Image.open(fig)
    col=white
    img=np.array(imgn)
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(img,text,(x, y), font, 0.3,col,1)
    cv2.imwrite(fig, viseg)

def tagviewst(tab,text,x,y):
    """write simple text in image """
    col=white
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,text,(x, y), font, 0.3,col,1)
    return viseg


def contour2(im,l):  
    viseg=np.zeros((dimtabx,dimtaby,3), np.uint8)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
    ret,thresh = cv2.threshold(imgray,0,255,0)
    im2,contours0, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
    colseg=classifcseg[l]
    cv2.drawContours(viseg,contours,-1,(colseg,colseg,colseg),-1,cv2.LINE_4)
    return viseg
   
###
def contour3(im,l):  
    col=classifc[l]
#    print l
#    print 'dimtabx' , dimtabx
    vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
#    print im.shape
#    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(im,0,255,0)
    im2,contours0, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
    cv2.drawContours(vis,contours,-1,col,1,cv2.LINE_AA)
    return vis
      
   
def pavs (imgi,namedirtopcf, iln,label):
    """ generate roi segment """    
    viseg=contour2(imgi,label)
    ret,thresh = cv2.threshold(viseg,0,255,0)
    imgray = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
    masky=cv2.bitwise_not(imgray)
   
    patchpathc=os.path.join(namedirtopcf,sroiseg)
    contenujpg = os.listdir(patchpathc)
#    print patchpathc

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
            imnseg = cv2.bitwise_and(tablscanseg,tablscanseg,mask = masky)
            imnseg=cv2.add(viseg,imnseg)
            cv2.imwrite(namescanseg,imnseg)
            tagviewb (namescanseg,label,0,100)
            break
    return 


    
def geneOverlay (namedirtopcf):
    """ generate overlay from ROI"""
#    print 'pavement'
    (top,tail)=os.path.split(namedirtopcf)
    scanpath=os.path.join(namedirtopcf,bmpname)
    sroipath=os.path.join(namedirtopcf,sroiseg)
    overpath=os.path.join(namedirtopcf,segDir)    
    contenusroi = os.listdir(sroipath)
    contenuscan = os.listdir(scanpath)
#    print contenusroi

    for  n in contenusroi:           
#        print n, typei
        sliceNumberN=rsliceNum(n,'_','.'+typei)
        for s in contenuscan:
            sliceNumberS=rsliceNum(s,'_','.'+typei)
            if sliceNumberN==sliceNumberS:
                namesroi=os.path.join(sroipath,n)
                nameover=os.path.join(overpath,n)
                namescan=os.path.join(scanpath,n)
                imscan = Image.open(namescan)
                imsroi= Image.open(namesroi)
                tabscan = np.array(imscan)
                tabscanc = cv2.cvtColor(tabscan,cv2.COLOR_GRAY2BGR)
                tabsroi=  np.array(imsroi)

                tabover=np.zeros((dimtabx,dimtaby,3), np.uint8)
                for p in  classifcseg:
                    greylevel=classifcseg[p]
                    g3=(greylevel,greylevel,greylevel)
                    masky=cv2.inRange(tabsroi,g3,g3)                    
                    vis=contour3(masky,p)

                    if masky.max()>0:

                        tabover=cv2.add(vis,tabover)                        
                        tabover=tagviewct(tabover,p,0,100) 

                textw='n: '+tail+' scan: '+str(sliceNumberS)
                tabover= tagviewst(tabover,textw,0,20)
                imtowrite=cv2.add(tabscanc,tabover)
                imtowrite = cv2.cvtColor(imtowrite,cv2.COLOR_RGB2BGR)
                cv2.imwrite(nameover,imtowrite)
                        
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

for f in listdirc:
    #f = 35
    print('work on:',f)
    namedirtopcf=os.path.join(namedirtopc,f) 
    
    bmp_dir = os.path.join(namedirtopcf, bmpname)
    remove_folder(bmp_dir)    
    os.mkdir(bmp_dir)    
    
    sroidirseg=os.path.join(namedirtopcf,sroiseg)
    remove_folder(sroidirseg)
    os.mkdir(sroidirseg)

    segDirp=os.path.join(namedirtopcf,segDir)
    remove_folder(segDirp)
    os.mkdir(segDirp)

    remove_folder(os.path.join(namedirtopcf,patchfile))
    os.mkdir(os.path.join(namedirtopcf,patchfile))
    contenudir = [name for name in os.listdir(namedirtopcf) if name.find('.txt')>0]
#        print(contenudir)
    fif=False
    genebmp(namedirtopcf)

    for f1 in contenudir:
        
        if (f1.find('CT')==0 or f1.find('Tho')==0):
#                print f1
            fif=True
            fileList =f1
            ##f1 = CT-INSPIRIUM-1186.txt
            pathf1=namedirtopcf+'/'+fileList
            #pathf1=final/ILD_DB_txtROIs/35/CT-INSPIRIUM-1186.txt         
            coefi =fileext(pathf1,namedirtopcf)
            break         
        
    if not fif:
         print('ERROR: no ROI txt content file', f)

 
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
#            print c
#        tabzc = np.zeros((dimtabx, dimtaby), dtype='i')
        imgc = np.zeros((dimtabx,dimtaby,3), np.uint8)
        for l in listslice:
#                print('l',l,'c:',c)
            if l.find(c,0)==0:
                pathl=namedirtopcf+'/'+patchfile+'/'+l
                tabcff = np.loadtxt(pathl,dtype='f')
                ofile = open(pathl, 'r')
                t = ofile.read()
                #print( t)
                ofile.close()
                labpos=t.find('label')
                labposend=t.find('\n',labpos)
                labposdeb = t.find(' ',labpos)
                label=t[labposdeb:labposend].strip()
                locapos=t.find('local')
                locaposend=t.find('\n',locapos)
                locaposdeb = t.find(' ',locapos)
                loca=t[locaposdeb:locaposend].strip()
#                print 'fx',fx
                tabccfi=fx*tabcff/coefi

                tabc=tabccfi.astype(int)

                print('generate tables from:',l,'in:', f)
                imgi= reptfulle(tabc,dimtabx,dimtaby)                
                imgc=imgc+imgi                    
#                tabzc=tabz+tabzc
                            
#                    print('end create tables')
                il=l.find('.',0)
                iln=l[0:il]
#                    print iln
        if label in usedclassif:
            print('c :',c, label,loca)
            print('creates patches from:',iln, 'in:', f)
            pavs (imgc,namedirtopcf, iln,label)            
    geneOverlay (namedirtopcf)

print('completed')