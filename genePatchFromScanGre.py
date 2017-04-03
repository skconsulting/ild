
# coding: utf-8
#Sylvain Kritter 5 octobre 2016
"""Top file to generate patches from DICOM database internal equalization from CHU Grenoble"""
import os
import sys
#import png
import numpy as np
import shutil
import scipy as sp
import scipy.misc
import dicom
import PIL
from PIL import Image, ImageFont, ImageDraw
import cv2
import matplotlib.pyplot as plt    
#general parameters and file, directory names
#######################################################
#customisation part for datataprep
#global directory for scan file
namedirHUG = 'CHU'
#subdir for roi in text
subHUG='UIPt'


toppatch= 'TOPPATCH'
    
#extension for output dir
extendir='16_set0_16b'
extendir='essai'
#normalization internal procedure or openCV
normiInternal=True
globalHist=True #use histogram equalization on full image
globalHistInternal=False #use internal for global histogram when True otherwise opencv
#patch overlapp tolerance
thrpatch = 0.8
#labelEnh=('consolidation','reticulation,air_trapping','bronchiectasis','cysts')
labelEnh=()
# average pxixel spacing
avgPixelSpacing=0.734
imageDepth=255 #number of bits used on dicom images (2 **n)

pset=0
#########################################################
if pset==2:
    #picklefile    'HC', 'micronodules'
    #patch size in pixels 32 * 32
    dimpavx =16 
    dimpavy = 16

elif pset==0:
    #'consolidation', 'HC','ground_glass', 'micronodules', 'reticulation'
    #patch size in pixels 32 * 32
    dimpavx =16
    dimpavy = 16
    
elif pset==1:
    #    'consolidation', 'ground_glass',
    #patch size in pixels 32 * 32
    dimpavx =28 
    dimpavy = 28
    
elif pset==3:
    #    'air_trapping'
    #patch size in pixels 32 * 32
    dimpavx =82 #or 20
    dimpavy = 82


########################################################################
######################  end ############################################
########################################################################
patchesdirnametop = toppatch+'_'+extendir
#define the name of directory for patches
patchesdirname = 'patches'
#define the name of directory for normalised patches
patchesNormdirname = 'patches_norm'
#define the name for jpeg files
imagedirname='patches_jpeg'
#define name for image directory in patient directory 
bmpname='scan_bmp'
#directory with lung mask dicom
lungmask='lung'
#directory to put  lung mask bmp
lungmaskbmp='scan_bmp'
#directory name with scan with roi
sroi='sroi8'
#directory name with scan with roi
bgdir='bgdir'
bgdirw='bgdirw'
reserved=['bgdir','sroi8','sroi16']
notclas=['lung','source','B70f']
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
#typei='png' #can be jpg

#patch size


font5 = ImageFont.truetype( 'arial.ttf', 5)
font10 = ImageFont.truetype( 'arial.ttf', 10)
font20 = ImageFont.truetype( 'arial.ttf', 20)
labelbg='back_ground'
locabg='anywhere_CHUG'

#end general part
#########################################################
#log files
##error file
errorfile = open(namedirtopc+'_'+extendir+'_genepatcherrortop.txt', 'w')
#filetowrite=os.path.join(namedirtopc,'lislabel.txt')
mflabel=open(namedirtopc+'_'+extendir+'_lislabel.txt',"w")

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



if pset ==0:
    usedclassif = [
        'back_ground',
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        'emphysema'
        ]
    classif ={
        'back_ground':0,
        'consolidation':1,
        'HC':2,
        'ground_glass':3,
        'healthy':4,
        'micronodules':5,
        'reticulation':6,
        'air_trapping':7,
        'cysts':8,
        'bronchiectasis':9,
        
         'bronchial_wall_thickening':10,
         'early_fibrosis':11,
         'emphysema':12,
         'increased_attenuation':13,
         'macronodules':14,
         'pcp':15,
         'peripheral_micronodules':16,
         'tuberculosis':17
        }
elif pset==1:
    usedclassif = [
        'back_ground',
        'consolidation',
        'ground_glass',
        'healthy'
    #    ,'cysts'
        ]
        
    classif ={
    'back_ground':0,
    'consolidation':1,
    'ground_glass':2,
    'healthy':3
    #,'cysts':4
    }
elif pset==2:
        usedclassif = [
        'back_ground',
        'fibrosis',
        'healthy',
        'micronodules',
        'reticulation'
        ]
        
        classif ={
    'back_ground':0,
    'fibrosis':1,
    'healthy':2,
    'micronodules':3,
    'reticulation':4
    }
elif pset==3:
    usedclassif = [
        'back_ground',
        'healthy',
        'air_trapping'
        ]
    classif ={
        'back_ground':0,
        'healthy':1,
        'air_trapping':2
        }
else:
        print 'eRROR :', pset, 'not allowed'


classifc ={
'back_ground':darkgreen,
#'consolidation':red,
'consolidation':cyan,
'HC':blue,
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

#print namedirtopc
 
patchtoppath=os.path.join(path_HUG,patchesdirnametop)
#create patch and jpeg directory
patchpath=os.path.join(patchtoppath,patchesdirname)
#create patch and jpeg directory
patchNormpath=os.path.join(patchtoppath,patchesNormdirname)
#print patchpath
#define the name for jpeg files
jpegpath=os.path.join(patchtoppath,imagedirname)
#print jpegpath

def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)


#patchpath = final/patches
#remove_folder(patchtoppath)
if not os.path.isdir(patchtoppath):
    os.mkdir(patchtoppath)   

#remove_folder(patchpath)
if not os.path.isdir(patchpath):
    os.mkdir(patchpath)   

#remove_folder(patchNormpath)
if not os.path.isdir(patchNormpath):
    os.mkdir(patchNormpath)  

#remove_folder(jpegpath)
if not os.path.isdir(jpegpath):
    os.mkdir(jpegpath)   


#end log files
def fidclass(numero):
    """return class from number"""
    found=False
    for cle, valeur in classif.items():
        
        if valeur == numero:
            found=True
            return cle
      
    if not found:
        return 'unknown'

def rotateImage(image, angle):
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def rsliceNum(s,c,e):
    endnumslice=s.find(e)
    posend=endnumslice
    while s.find(c,posend)==-1:
        posend-=1
    debnumslice=posend+1
    return int((s[debnumslice:endnumslice])) 


def genebmp(dirName):
    """generate patches from dicom files and sroi"""
    print ('generate  bmp files from dicom files in :',dirName)
    
    global constPixelSpacing, dimtabx,dimtaby, source_name
    contenudir = os.listdir(dirName)
    print 'dirName',dirName
    bgdirf = os.path.join(dirName, bgdir)
    remove_folder(bgdirf)    
    os.mkdir(bgdirf)
    dn=False
    for dirFile in contenudir:
        print dirFile
        rotScan=False
        dirFileP = os.path.join(dirName, dirFile)
#        source_name=''
        if dirFile == 'source':
            source_name='source'
        if dirFile == 'B70f':
            source_name='B70f'
            dn=True
            rotScan=True
#        print 'source_name',source_name
        if dirFile not in reserved:        
            bmp_dir = os.path.join(dirFileP, bmpname)
            remove_folder(bmp_dir)    
            os.mkdir(bmp_dir)
        if dirFile not in reserved +usedclassif+notclas :
            print 'not known:',dirFile
            print usedclassif
            errorfile.write(dirFile+' directory is not known\n')
        #list dcm files
        fileList =[name for name in  os.listdir(dirFileP) if ".dcm" in name.lower()]
    
        for filename in fileList:
            
#            if ".dcm" in filename.lower():  # check whether the file's DICOM
                FilesDCM =(os.path.join(dirFileP,filename))  
#                print FilesDCM         
                RefDs = dicom.read_file(FilesDCM) 
                dsr= RefDs.pixel_array
#                print dirName, dsr.min(),dsr.max()
                dsr= dsr-dsr.min()                
                if dsr.max()>0:
                    c=float(imageDepth)/dsr.max()
                    dsr=dsr*c
                    if imageDepth <256:
                        dsr=dsr.astype('uint8')
                    else:
                        dsr=dsr.astype('uint16')
                    #resize the dicom to have always the same pixel/mm
                    fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing       
                    dsrresize1= sp.misc.imresize(dsr,fxs,interp='bicubic',mode=None)
                    if dirFile=='source' or dirFile=='B70f':
                        if globalHist:
                            if globalHistInternal:
                                dsrresize = normi(dsrresize1) 
                            else:
                                dsrresize = cv2.equalizeHist(dsrresize1) 
                        else:
                            dsrresize=dsrresize1 
                    else:
                            dsrresize=dsrresize1                    
   
                    #calculate the  new dimension of scan image
                    dimtabx=int(dsrresize.shape[0])
                    dimtaby=int(dsrresize.shape[1])                     
                    scanNumber=int(RefDs.InstanceNumber)
                    endnumslice=filename.find('.dcm')
                   
                    imgcore=filename[0:endnumslice]+'_'+str(scanNumber)+'.'+typei   
                    bmpfile=os.path.join(bmp_dir,imgcore)
                    if rotScan :
                        dsrresize=rotateImage(dsrresize, 180)
                    scipy.misc.imsave(bmpfile, dsrresize)
    
                    if dirFile == 'lung':
                        bmpfile=os.path.join(bgdirf,imgcore)
                        scipy.misc.imsave(bmpfile, dsrresize)
                        
                    if dirFile == 'B70f' or dirFile =='source':
#                        slicenumber=int(RefDs.InstanceNumber)
                        namescan=os.path.join(sroidir,imgcore)                        
                        textw='n: '+f+' scan: '+str(scanNumber)
                        if rotScan :
                            dsrresize1=rotateImage(dsrresize1, 180)
                        tablscan=cv2.cvtColor(dsrresize1,cv2.COLOR_GRAY2BGR)
                        scipy.misc.imsave(namescan, tablscan)
                        tagviews(namescan,textw,0,20)    
    return dn 

def reptfulle(tabc,dx,dy):
    imgi = np.zeros((dx,dy,3), np.uint8)
    cv2.polylines(imgi,[tabc],True,(1,1,1)) 
    cv2.fillPoly(imgi,[tabc],(1,1,1))
    tabzi = np.array(imgi)
    tabz = tabzi[:, :,1]   
    return tabz, imgi

def normi(img):
     tabi = np.array(img)
#     print(tabi.min(), tabi.max())
     tabi1=tabi-tabi.min()
#     print(tabi1.min(), tabi1.max())
     maxt=float(tabi1.max())
     if maxt==0:
         maxt=1
     tabi2=tabi1*(imageDepth/maxt)
     if imageDepth<256:
         tabi2=tabi2.astype('uint8')
     else:
         tabi2=tabi2.astype('uint16')
#     print(tabi2.min(), tabi2.max())   
     return tabi2


def tagview(fig,label,x,y):
    """write text in image according to label and color"""
#    print ('write label :',label,' at: ', fig)
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    col=classifc[label]
    labnow=classif[label]
#    print (labnow, text)
    if label == 'back_ground':
        x=0
        y=0        
        deltax=0
        deltay=60
    else:        
        deltay=25*((labnow-1)%5)
#        deltax=175*((labnow-1)//5)
        deltax=80*((labnow-1)//5)

#    print (x+deltax,y+deltay)
    draw.text((x, y+deltay),label,col,font=font10)
    imgn.save(fig)

def tagviews(fig,text,x,y):
    """write simple text in image """
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    draw.text((x, y),text,white,font=font10)
    imgn.save(fig)

def pavbg(dirName,dn):
    print('generate back-ground for :',dirName)
    """ generate patches from ROI"""
    global source_name,dimtabx,dimtaby
    (top,tail)=os.path.split(dirName)
    nampadir=os.path.join(patchpath,labelbg)
    if not os.path.exists(nampadir):
                 os.mkdir(nampadir)
                 
    nampadirl=os.path.join(nampadir,locabg)
    if not os.path.exists(nampadirl):
                 os.mkdir(nampadirl)
                 
    nampaNormdir=os.path.join(patchNormpath,labelbg)
    if not os.path.exists(nampaNormdir):
                 os.mkdir(nampaNormdir)
                 
    nampadirNorml=os.path.join(nampaNormdir,locabg)
    if not os.path.exists(nampadirNorml):
                 os.mkdir(nampadirNorml)
                 
    bgdirf = os.path.join(dirName, bgdir)
    
    patchpath1=os.path.join(dirName,source_name)
    patchpathc=os.path.join(patchpath1,bmpname)
   
    lbmp=os.listdir(patchpathc)
    listbg = os.listdir(bgdirf)
#    print patchpathc
    pxy=float(dimpavx*dimpavy) 
    for lm in listbg:
#        print lm
        nbp=0
        tabp = np.zeros((dimtabx, dimtaby), dtype='uint8')
        slicenumber=rsliceNum(lm,'_','.'+typei)
        if dn:
            slicenumber=2*slicenumber
        for ln in  lbmp:
            slicescan=rsliceNum(ln,'_','.'+typei)

            if slicescan==slicenumber  :
                if slicescan not in listsliceok:
                    break
                else:
                  nambmp=os.path.join(patchpathc,ln)
                  namebg=os.path.join(bgdirf,lm)
           
    #        print namebg
    #find the same name in bgdir directory
                  origbg = Image.open(namebg,'r')
                  origbl= origbg.convert('L')
           
#            for l in lbmp:
#    #          print('l de pavb',l)
#    #          if l.find(slns)>0 and slicenumber in listsliceok:
#              if l.find(slns)>0 :
#                  nambmp=os.path.join(patchpathc,l)
                  origbmp = Image.open(nambmp,'r')
                  origbmpl= origbmp.convert('L')
                 
    #              print ('nambmp:',nambmp)
                  tabf=np.array(origbl)
    #              imagemax=origbl.getbbox()
    #                                            
    #              min_val=np.min(origbl)
    #              max_val=np.max(origbl)
    #              print min_val, max_val
    #              cv2.imshow('tabf',tabf) 
    ##
    #              cv2.waitKey(0)    
    #              cv2.destroyAllWindows()   
            #put all to 1 if>0
                  tabfc = np.copy(tabf)
                  nz= np.count_nonzero(tabf)
                  if nz>0:
    
                    np.putmask(tabf,tabf>0,1)
                    atabf = np.nonzero(tabf)
                    #tab[y][x]  convention
                    xmin=atabf[1].min()
                    xmax=atabf[1].max()
                    ymin=atabf[0].min()
                    ymax=atabf[0].max()
                  else:
                    xmin=0
                    xmax=0
                    ymin=0
                    ymax=0        
                  i=xmin
                  while i <= xmax:
                            j=ymin
                            while j<=ymax:
        #                        if i%10==0 and j%10==0:
        #                         print(i,j)                                
                                tabpatch=tabf[j:j+dimpavy,i:i+dimpavx]
                                area= tabpatch.sum()
    #                            print area  
                                if float(area)/pxy >thrpatch:
    #                                 print 'good'
        #                            good patch                   
                                     crorig = origbmpl.crop((i, j, i+dimpavx, j+dimpavy))
                                     #detect black pixels
                                     imagemax=crorig.getbbox()
    #                                 print imagemax
                                     nim= np.count_nonzero(crorig)
                                     if nim>0:
                                         min_val=np.min(crorig)
                                         max_val=np.max(crorig)         
                                     else:
                                         min_val=0
                                         max_val=0  
                                     if imagemax!=None and min_val!=max_val:
                                        nbp+=1
                                        nampa=os.path.join(nampadirl,tail+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typei )   
                                        crorig.save(nampa)                                        
                                        imgray =np.array(crorig)
    #                                    imgray = cv2.cvtColor(imgra,cv2.COLOR_BGR2GRAY)
                                                #normalization internal procedure or openCV
                                        if normiInternal:
                                            tabi2 = normi(imgray) 
                                        else:
                                            tabi2 = cv2.equalizeHist(imgray)  
                                        nampa=os.path.join(nampadirNorml,tail+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typei )                                    
  
                                        scipy.misc.imsave(nampa, tabi2)
                                    
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
                                j+=1
                            i+=1
#                  break  
    #              print tabfc.shape , tabfc.dtype    
    #              cv2.imshow('tabfc',tabfc) 
    ###
    #              cv2.waitKey(0)    
    #              cv2.destroyAllWindows()  
    #              print tabp.shape , tabp.dtype 
                  tabpw =tabfc+tabp
                  stw=tail+'_slice_'+str(slicenumber)+'_'+labelbg+'_'+locabg+'_'+str(nbp)
                  stww=stw+'.jpg'
                  flw=os.path.join(jpegpath,stww)
                  scipy.misc.imsave(flw, tabpw) 
                  stww=stw+'.txt'
                  flw=os.path.join(jpegpath,stww)
                  mfl=open(flw,"w")
    #        mfl=open(jpegpath+'/'+f+'_'+slicenumber+'.txt',"w")
                  mfl.write('#number of patches: '+str(nbp)+'\n')
                  mfl.close()
                  break

def contour2(im,l):  
    col=classifc[l]
#    print l
#    print 'dimtabx' , dimtabx
    vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
#    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(im,0,255,0)
    im2,contours0, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
    cv2.drawContours(vis,contours,-1,col,1,cv2.LINE_AA)
    return vis
   
###
    
def pavs (dirName,dn):
    """ generate patches from ROI"""
    global source_name,dimtabx,dimtaby
    print 'pav :',dirName
    ntotpat=0
    bgdirf = os.path.join(dirName, bgdir)
    (top,tail)=os.path.split(dirName)
   #list directory in top
#    contenudir = os.listdir(dirName)
    pxy=float(dimpavx*dimpavy)
    contenudir=[name for name in os.listdir(dirName) if name  in usedclassif]
#    print contenudir
    for dirFile in contenudir :
        dirFileP = os.path.join(dirName, dirFile)
        #directory for patches
#        print 'class1:',dirFile,usedclassif
        
#            print 'class:',dirFile
            
        nampadir=os.path.join(patchpath,dirFile)
        nampadirl=os.path.join(nampadir,locabg)
        if not os.path.exists(nampadir):
             os.mkdir(nampadir)
             os.mkdir(nampadirl)
        nampaNormdir=os.path.join(patchNormpath,dirFile)
        nampadirNorml=os.path.join(nampaNormdir,locabg)
        if not os.path.exists(nampaNormdir):
             os.mkdir(nampaNormdir)
             
             os.mkdir(nampadirNorml)                     
   
        dirFilePs = os.path.join(dirFileP, bmpname)
        #list file in roi directory
        listmroi = os.listdir(dirFilePs)
        for filename in listmroi:

             slicenumber= rsliceNum(filename,'_','.'+typei)

             if dn:
                 slicenumber=2*slicenumber

             imgsp = os.path.join(dirName, source_name)
             imgsp_dir = os.path.join(imgsp,bmpname)
             #list slice in source
             listslice = os.listdir(imgsp_dir)
             for l in listslice:

                slicescan= rsliceNum(l,'_','.'+typei)


                if slicescan==slicenumber:
                     tabp = np.zeros((dimtabx, dimtaby), dtype='i')
                     filenamec = os.path.join(dirFilePs, filename)
                     
                     
                     origscan = Image.open(filenamec,'r')
 
                     #mask image
                     origscan = origscan.convert('L')
                     tabf=np.array(origscan)
#                         cv2.imshow('tabf',tabf) 
#                         cv2.waitKey(0)    
#                         cv2.destroyAllWindows()
                     # create ROI as overlay
                     vis=contour2(tabf,dirFile)
                     
#                         atvis = np.nonzero(tabf)
                     if vis.sum()>0:
                             
                     #list slice in directory sroi
                         listsroi =os.listdir(sroidir)
                         
                         for sroi in listsroi:
                             sliceroi= rsliceNum(sroi,'_','.'+typei) 
#                             print slicescan,sliceroi
                             if sliceroi==slicenumber:
                                 filenamesroi = os.path.join(sroidir, sroi)
                                 oriroi = Image.open(filenamesroi)
                                 imroi= oriroi.convert('RGB')           
                                 tablroi = np.array(imroi)
#                                 print vis.shape,tablroi.shape
                                 imn=cv2.add(vis,tablroi)
                                 imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)
                                 cv2.imwrite(filenamesroi,imn)
                                 tagview(filenamesroi,dirFile,0,100)
#                                 cv2.imshow('tablscan',tablscan) 
#                                     cv2.imshow('vis',vis) 
#                                     cv2.imshow('imn',imn) 
#                            #            print namescan
#                                     cv2.waitKey(0)    
#                                     cv2.destroyAllWindows()  
#
                                 break
                     #list slice in bgdir
                     lunglist =os.listdir(bgdirf)                         
                     for lm in lunglist:
                         slicelung= rsliceNum(lm,'_','.'+typei) 
#                             print ' list file in bgdir', lm
                         if dn:
                             slicelung=2*slicelung
#                             print slicelung,slicenumber
                         if slicelung==slicenumber:
            #look in lung maask the name of slice

                            namebg=os.path.join(bgdirf,lm)
#                                print ('namebg:',namebg)
#find the same name in bgdir directory
#                    origbg = Image.open(namebg,'r')
#                    tabhc=np.array(origbg)
                            tabhc=cv2.imread(namebg,0)
#                    cv2.imshow('tabhc1',tabhc) 
                            np.putmask(tabhc,tabhc>0,100)

#                    print slicenumber
#                    del origbg
#                                masky=cv2.inRange(tabf,(1,1,1),(10,10,10))
#                    np.putmask(imgi,imgi>1,200)
                            np.putmask(tabf,tabf==1,0)
#                            tabf1 = np.where(tabf> 1,100,0)
                            np.putmask(tabf,tabf>0,100)
                           
#                                np.putmask(tabu,tabu==2,100)
#                                cv2.imshow('tabf',tabf) 
##                                cv2.imshow('tabu',tabu) 
#                                cv2.waitKey(0)    
#                                cv2.destroyAllWindows()
                            mask=cv2.bitwise_not(tabf)
#                    cv2.imshow('masky',masky) 
#                            print np.shape(tabhc), np.shape(mask),np.shape(tabf1)
                            outy=cv2.bitwise_and(tabhc,mask)
                            cv2.imwrite(namebg,outy)
                            break
 
 
 
 
                     atabf = np.nonzero(tabf)
                     imagemax=tabf.sum()
                             
                     if imagemax>0:
                    #tab[y][x]  convention
                         xmin=atabf[1].min()
                         xmax=atabf[1].max()
                         ymin=atabf[0].min()
                         ymax=atabf[0].max()
                     else:
                         xmin=0
                         xmax=2
                         ymin=0
                         ymax=2
#                 imgmask=cv2.imread(filenamec)
#                 tabimgmask=np.array(imgmask)
                 
#                     
#                     np.putmask(tabf,tabf==1,0)  
#                     np.putmask(tabf,tabf>0,1)
                     np.putmask(tabf,tabf==1,0)
#                            tabf1 = np.where(tabf> 1,100,0)
                     np.putmask(tabf,tabf>0,1)
#                     tabf = np.where(tabf> 1,1,0)
#                         tabu=np.copy(tabf)
#                         np.putmask(tabu,tabu>0,150)
#                         cv2.imshow('tabu',tabu) 
#                         cv2.waitKey(0)    
#                         cv2.destroyAllWindows()  
                     imgscanp = os.path.join(imgsp_dir, l)
                     origbmp = Image.open(imgscanp,'r')
                    #scan bmp file
                     origbmpl= origbmp.convert('L')
 
                     
                     i=xmin
                     nbp=0
                     while i <= xmax:
                        j=ymin
                        while j<=ymax:
                            tabpatch=tabf[j:j+dimpavy,i:i+dimpavx]
                            area= tabpatch.sum()  
                            targ=float(area)/pxy
                            
                            if targ >thrpatch:
#                                    print area,pxy, i,j
#             #good patch        
#                                    if i== 153 and j == 222:
#                                        np.putmask(tabf,tabf>0,150)
#                                        print tabf[j:j+dimpavy,i:i+dimpavx]
#                                        scipy.misc.imsave('a.bmp', tabf) 
        #                        print ('slicenumber',slicenumber)
                                                  
                                crorig = origbmpl.crop((i, j, i+dimpavx, j+dimpavy))
                                 #detect black pixels
                                 #imagemax=(crorig.getextrema())
                                imagemax=crorig.getbbox()
                                nim= np.count_nonzero(crorig)
                                if nim>0:
                                     min_val=np.min(crorig)
                                     max_val=np.max(crorig)         
                                else:
                                     min_val=0
                                     max_val=0                              
                                if imagemax!=None and min_val!=max_val:   
                                    nbp+=1                                   
                                    nampa=os.path.join(nampadirl,tail+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typei )
                                    crorig.save(nampa) 
                #normalize patches and put in patches_norm
                #                                 tabi2=normi(crorig)
                                    imgra =np.array(crorig)
#                                    imgray = cv2.cvtColor(imgra,cv2.COLOR_BGR2GRAY)
                                    if normiInternal:
                                                tabi2 = normi(imgra) 
                                    else:
                                                tabi2 = cv2.equalizeHist(imgra)  
        #                            tabi2 = cv2.equalizeHist(imgray)
                                
                                    nampa=os.path.join(nampadirNorml,tail+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typei )                                    
                                    scipy.misc.imsave(nampa, tabi2)
                                    
                                #                print('pavage',i,j)  
#                                    strpac=strpac+str(i)+' '+str(j)+'\n'
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
                                    if dirFile not in labelEnh:
                                        tabf[j:j+dimpavy,i:i+dimpavx]=0
                                    else:
                                         tabf[j:j+dimpavy/2,i:i+dimpavx/2]=0                          
                            j+=1
                        i+=1
                         
                     
                     origscan =origscan+tabp
                     ntotpat=ntotpat+nbp

                     if nbp>0:
                         if slicenumber not in listsliceok:
                                    listsliceok.append(slicenumber) 
                         stw=tail+'_slice_'+str(slicenumber)+'_'+dirFile+'_'+locabg+'_'+str(nbp)
                         stww=stw+'.txt'
                         flw=os.path.join(jpegpath,stww)
                         mfl=open(flw,"w")
                         mfl.write('#number of patches: '+str(nbp)+'\n')
                         mfl.close()
                         stww=stw+'.jpg'
                         flw=os.path.join(jpegpath,stww)
                         scipy.misc.imsave(flw, origscan)
                     else:
                         tw=tail+'_slice_'+str(slicenumber)+'_'+dirFile+'_'+locabg
                         print tw,' :no patch'
                         errorfile.write(tw+' :no patch\n')
                     break
#                             print listsliceok
#    print listsliceok
    return ntotpat




listdirc= (os.listdir(namedirtopc))
npat=0
#listdirc=('S335940',)
for f in listdirc:
#    f = 'S335940'
    print('work on:',f)

    nbpf=0
    listsliceok=[]

#    namedirtopcf=namedirtopc+'/'+f
    namedirtopcf=os.path.join(namedirtopc,f) 
    if os.path.isdir(namedirtopcf):    
        sroidir=os.path.join(namedirtopcf,sroi)
        remove_folder(sroidir)
        os.mkdir(sroidir)
#
#    remove_folder(namedirtopcf+'/patchfile')
#    os.mkdir(namedirtopcf+'/patchfile')
    #namedirtopcf = final/ILD_DB_txtROIs/35
#    if posp==-1 and posu==-1:
#        print(contenudir)
#    fif=False
#    for fi in contenudir:
#        sb=os.path.join(namedirtopcf,fi)
    s=genebmp(namedirtopcf)
    nbp=pavs(namedirtopcf,s)
 
    pavbg(namedirtopcf,s)
    

    

    
#    if label in usedclassif:
#                print('c :',c, label,loca)
#                print('creates patches from:',iln, 'in:', f)
#                nbp,tabz1=pavs (imgc,tabzc,dimtabx,dimtaby,dimpavx,dimpavy,namedirtopcf,\
#                    jpegpath, patchpath,thrpatch,iln,f,label,loca,typei,errorfile)
#                print('end create patches')
    nbpf=nbpf+nbp
#            #create patches for back-ground
#        pavbg(namedirtopcf,dimtabx,dimtaby,dimpavx,dimpavy)
##    print(f,nbpf)
    ofilepw = open(jpegpath+'/nbpat_'+f+'.txt', 'w')
    ofilepw.write('number of patches: '+str(nbpf))
    ofilepw.close()
#    
    
#################################################################    
#   calculate number of patches
contenupatcht = os.listdir(jpegpath) 
#print(contenupatcht)
npatcht=0
for npp in contenupatcht:
#    print('1',npp)
    if npp.find('.txt')>0 and npp.find('nbp')<0:
#        print('2',npp)
        ofilep = open(jpegpath+'/'+npp, 'r')
        tp = ofilep.read()
#        print( tp)
        ofilep.close()
        numpos2=tp.find('number')
        numposend2=len(tp)
        #tp.find('\n',numpos2)
        numposdeb2 = tp.find(':',numpos2)
        nump2=tp[numposdeb2+1:numposend2].strip()
#        print(nump2)
        numpn2=int(nump2)
        npatcht=npatcht+numpn2
#        print(npatch)
ofilepwt = open(jpegpath+'/totalnbpat.txt', 'w')
ofilepwt.write('number of patches: '+str(npatcht))
ofilepwt.close()
#mf.write('================================\n')
#mf.write('number of datasets:'+str(npat)+'\n')
#mf.close()
#################################################################
#data statistics on paches
#nametopc=os.path.join(cwd,namedirtop)
dirlabel=os.walk( patchpath).next()[1]
#file for data pn patches
filepwt = open(namedirtopc+'totalnbpat.txt', 'w')
ntot=0;

labellist=[]
localist=[]

for dirnam in dirlabel:
    dirloca=os.path.join(patchpath,dirnam)
#    print ('dirloca', dirloca)
    listdirloca=os.listdir(dirloca)
    label=dirnam
#    print ('dirname', dirname)

    loca=''
    if dirnam not in labellist:
            labellist.append(dirnam)
#    print('label:',label)
    for dlo in listdirloca:
        loca=dlo
        if dlo not in localist:      
            localist.append(dlo)
#        print('localisation:',loca)
        if label=='' or loca =='':
            print('not found:',dirnam)        
        subdir = os.path.join(dirloca,loca)
#    print(subdir)
        n=0
        listcwd=os.listdir(subdir)
        for ff in listcwd:
            if ff.find(typei) >0 :
                n+=1
                ntot+=1
#        print(label,loca,n) 
        filepwt.write('label: '+label+' localisation: '+loca+\
        ' number of patches: '+str(n)+'\n')
filepwt.close() 
#listslice=[]
#write the log file with label list
mflabel.write('label  _  localisation\n')
mflabel.write('======================\n')
categ=os.listdir(jpegpath)
for f in categ:
#    print f
    if f.find('.txt')>0 and f.find('nb')==0:
        ends=f.find('.txt')
        debs=f.find('_')
        sln=f[debs+1:ends]
#        print sln
        listlabel={}
        
        for f1 in categ:
#                print f1
                if  f1.find(sln+'_')==0 and f1.find('.txt')>0:
#                    print f1
                    debl=f1.find('slice_')
                    debl1=f1.find('_',debl+1)
                    debl2=f1.find('_',debl1+1)
                    endl=f1.find('.txt')
                    posend=endl
                    while f1.find('_',posend)==-1:
                        posend-=1
                    debnumslice=posend+1
                    label=f1[debl2+1:debnumslice-1]
#                    print label
                    ffle1=os.path.join(jpegpath,f1)
                    fr1=open(ffle1,'r')
                    t1=fr1.read()
                    fr1.close()
                    debsp=t1.find(':')
                    endsp=  t1.find('\n')
                    npo=int(t1[debsp+1:endsp])
#                    print label,npo,listlabel
                    if label in listlabel:
                        
                        listlabel[label]=listlabel[label]+npo
                    else:
                        listlabel[label]=npo
#        listslice.append(sln)
        ffle=os.path.join(jpegpath,f)
        fr=open(ffle,'r')
        t=fr.read()
        fr.close()
        debs=t.find(':')
        ends=len(t)
        nump= t[debs+1:ends]
        mflabel.write(sln+' number of patches: '+nump+'\n')
#        print listlabel
        for l in listlabel:
#           print l,l.find(labelbg)
           if l.find(labelbg)<0:
#             print l
             mflabel.write(l+' '+str(listlabel[l])+'\n')
        mflabel.write('---------------------'+'\n')
mflabel.close()

##########################################################
errorfile.write('completed')
errorfile.close()
#print listslice
print('completed')