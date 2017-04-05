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
subHUG='UIPtt'
toppatch= 'TOPPATCHLUNG'
    
#extension for output dir
extendir='15_set0'

pset=0


    #'consolidation', 'HC','ground_glass', 'micronodules', 'reticulation'
    #patch size in pixels 32 * 32
dimpavx =15
dimpavy = 15
    
subsample=1

#normalization internal procedure or openCV
normiInternal=False
#patch overlapp tolerance
thrpatch = 0.8
#labelEnh=('consolidation','reticulation,air_trapping','bronchiectasis','cysts')
labelEnh=()
# average pxixel spacing
avgPixelSpacing=0.734
imageDepth=255 #number of bits used on dicom images (2 **n)
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
sroi='sroi'
#directory name with scan with roi
bgdir='bgdir'
bgdirw='bgdirw'
nolung='nolung'
reserved=['bgdir','sroi']
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
errorfile = open(namedirtopc+'genepatcherrortop.txt', 'w')
#filetowrite=os.path.join(namedirtopc,'lislabel.txt')
mflabel=open(namedirtopc+'lislabel.txt',"w")

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



usedclassif = ['lung']
#        'consolidation',
#        'HC',
#        'ground_glass',
#        'healthy',
#        'micronodules',
#        'reticulation',
#        'air_trapping',
#        'cysts',
#        'bronchiectasis'


classifc ={
    
    'lung':blue,
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
def rsliceNum(s,c,e):
    ''' look for  afile according to slice number'''
    #s: file name, c: delimiter for snumber, e: end of file extension
    endnumslice=s.find(e)
    posend=endnumslice
    while s.find(c,posend)==-1:
        posend-=1
    debnumslice=posend+1
    return int((s[debnumslice:endnumslice])) 

#patchpath = final/patches
remove_folder(patchtoppath)
#if not os.path.isdir(patchpath):
os.mkdir(patchtoppath)   

remove_folder(patchpath)
#if not os.path.isdir(patchpath):
os.mkdir(patchpath)   

remove_folder(patchNormpath)
#if not os.path.isdir(patchNormpath):
os.mkdir(patchNormpath)  

remove_folder(jpegpath)
#if not os.path.isdir(jpegpath):
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
    bgdirf = os.path.join(dirName, bgdir)
#    remove_folder(bgdirf)    
#    os.mkdir(bgdirf)
    dn=False
    for dirFile in contenudir:
        print dirFile
        dirFileP = os.path.join(dirName, dirFile)
        
        if dirFile == 'source':
            source_name='source'
        if dirFile == 'B70f':
            source_name='B70f'
            dn=True

        
#        if dirFile not in reserved:        
#            bmp_dir = os.path.join(dirFileP, bmpname)
#            remove_folder(bmp_dir)    
#            os.mkdir(bmp_dir)
#        if dirFile not in reserved +usedclassif+notclas :
#            print 'not known:',dirFile
#            errorfile.write(dirFile+' directory is not known\n')
#        #list dcm files
        fileList = os.listdir(dirFileP)
#    
        for filename in fileList:
#            
            if ".dcm" in filename.lower():  # check whether the file's DICOM
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
    
                    dsrresize= sp.misc.imresize(dsr,fxs,interp='bicubic',mode=None)
    
                    #calculate the  new dimension of scan image
                    dimtabx=int(dsrresize.shape[0])
                    dimtaby=int(dsrresize.shape[1])       
                    break
#                    endnumslice=filename.find('.dcm')
#                    imgcore=filename[0:endnumslice]+'.'+typei
#                    bmpfile=os.path.join(bmp_dir,imgcore)
#                    if rotScan :
#                        dsrresize=rotateImage(dsrresize, 180)
#                    scipy.misc.imsave(bmpfile, dsrresize)
#    
#                    if dirFile == 'lung':
#                        bmpfile=os.path.join(bgdirf,imgcore)
#                        scipy.misc.imsave(bmpfile, dsrresize)
#                        
#                    if dirFile == 'B70f' or dirFile =='source':
#    #                    print 'azfter tail',tail,f
#    #                    posend=endnumslice
#    #                    while filename.find('-',posend)==-1:
#    #                        posend-=1
#    #                    debnumslice=posend+1
#                        slicenumber=rsliceNum(filename,'-','.dcm')
#                        namescan=os.path.join(sroidir,imgcore)                   
#                        textw='n: '+f+' scan: '+str(slicenumber)
#                        orign = Image.open(bmpfile)
#                        imscanc= orign.convert('RGB')
#                        tablscan = np.array(imscanc)
#                        scipy.misc.imsave(namescan, tablscan)
#                        tagviews(namescan,textw,0,20)   
#    print 'rotscan:',rotScan    
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
     tabi2=tabi1*(imageDepth/float(tabi1.max()-tabi1.min()))
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
#    contenudir = os.listdir(dirName)*
#    print top
    pxy=float(dimpavx*dimpavy)
    contenudir=[name for name in os.listdir(dirName) if name  in usedclassif]
    print contenudir
    for dirFile in contenudir :
         dirFileP = os.path.join(dirName, dirFile)
         print dirFileP
        #directory for patches
#        print 'class1:',dirFile,usedclassif
        
#            print 'class:',dirFile
            
         nampadir=os.path.join(patchpath,dirFile)
         nampadirl=os.path.join(nampadir,locabg)
         nampadirnolung=os.path.join(patchpath,nolung)
         nampadirnolungl=os.path.join(nampadirnolung,locabg)
         if not os.path.exists(nampadir):
             os.mkdir(nampadir)
             os.mkdir(nampadirl)
             os.mkdir(nampadirnolung)
             os.mkdir(nampadirnolungl)
             
         nampaNormdir=os.path.join(patchNormpath,dirFile)
         nampadirNorml=os.path.join(nampaNormdir,locabg)
         nampaNormdirNolung=os.path.join(patchNormpath,nolung)
         nampadirNormNolungl=os.path.join(nampaNormdirNolung,locabg)
         if not os.path.exists(nampaNormdir):
             os.mkdir(nampaNormdir)
             
             os.mkdir(nampadirNorml)     
             os.mkdir(nampaNormdirNolung)
             
             os.mkdir(nampadirNormNolungl)                     
   
         dirFilePs = os.path.join(dirFileP, bmpname)
        #list file in roi directory
         listmroi = os.listdir(dirFilePs)
         listslice=[]
         for filename in listmroi:
#             print 'filename', filename
             slicenumber=rsliceNum(filename,'-','.'+typei)
            
    
              
#                 print 'number of roi:',slicenumber
             if dn:
                 slicenumber=2*slicenumber
             listslice.append(slicenumber)
         print listslice
         listslicesup=[name for name in listslice if listslice.index(name)%subsample ==0 ]
         print listslicesup
         imgsp = os.path.join(dirName, source_name)
         imgsp_dir = os.path.join(imgsp,bmpname)
         #list slice in source
         listslice = os.listdir(imgsp_dir)
    #             print 'listlice',listslice
         for l in listslice:
    #                    print 'list file in source',l
            
            slicescan=rsliceNum(l,'_','.'+typei)
        
    #                    print slicenumber,slicescan
    
            if slicescan in listslicesup:
                 tabp = np.zeros((dimtabx, dimtaby), dtype='i')
                 for f in listmroi:
                     snumroi=rsliceNum(f,'-','.'+typei)
                     if snumroi==slicescan:
                         filenamec = os.path.join(dirFilePs, f)
                         break
                 print 'filenamec',filenamec
             
                 
                 origscan = Image.open(filenamec,'r')
     
                 #mask image
                 origscan = origscan.convert('L')
                 tabf=np.array(origscan)
    #                         cv2.imshow('tabf',tabf) 
    #                         cv2.waitKey(0)    
    #                         cv2.destroyAllWindows()
                 # create ROI as overlay
               
                 
    #                         atvis = np.nonzero(tabf)
                
                     
                   
                 #list slice in bgdir
                 lunglist =os.listdir(bgdirf)                         
                 for lm in lunglist:
    #                             print ' list file in bgdir', lm
                     
                     slicelung=rsliceNum(lm,'-','.'+typei) 
                     if dn:
                         slicelung=2*slicelung
    #                             print slicelung,slicenumber
                     if slicelung==slicescan:
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
                        np.putmask(tabf,tabf>0,255)
                       
    #                                np.putmask(tabu,tabu==2,100)
    #                                cv2.imshow('tabf',tabf) 
    ##                                cv2.imshow('tabu',tabu) 
    #                                cv2.waitKey(0)    
    #                                cv2.destroyAllWindows()
                        mask=cv2.bitwise_not(tabf)
    #                            np.putmask(tabf,tabf>0,100)
    #                            np.putmask(mask,mask>1,100)
    #                            cv2.imshow('tabf',tabf) 
    #                            cv2.imshow('mask',mask) 
    #                            cv2.waitKey(0)    
    #                            cv2.destroyAllWindows()
    #                    cv2.imshow('masky',masky) 
    #                    print np.shape(tabhc), np.shape(masky)
    #                            outy=cv2.bitwise_and(tabhc,mask)
                
    #                            cv2.imwrite(namebg,outy)
                        print 'namebg', namebg
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
             
                 
                 np.putmask(tabf,tabf==1,0)  
                 np.putmask(tabf,tabf>0,1)
    #                         tabu=np.copy(tabf)
    #                         np.putmask(tabu,tabu>0,150)
    #                         cv2.imshow('tabu',tabu) 
    #                         cv2.waitKey(0)    
    #                         cv2.destroyAllWindows()  
                 imgscanp = os.path.join(imgsp_dir, l)
                 origbmp = Image.open(imgscanp,'r')
                #scan bmp file
                 origbmpl= origbmp.convert('L')
     
                 print 'lung'
    #                     origscan = origscan.convert('L')
                 
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
                         
                            if imagemax!=None:
      
                                nbp+=1
    
                               
                                nampa=os.path.join(nampadirl,tail+'_'+str(slicescan)+'_'+str(nbp)+'.'+typei )
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
                            
                                nampa=os.path.join(nampadirNorml,tail+'_'+str(slicescan)+'_'+str(nbp)+'.'+typei )                                    
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
                     if slicescan not in listsliceok:
                                listsliceok.append(slicescan) 
                     stw=tail+'_slice_'+str(slicescan)+'_'+dirFile+'_'+locabg+'_'+str(nbp)
                     stww=stw+'.txt'
                     flw=os.path.join(jpegpath,stww)
                     mfl=open(flw,"w")
                     mfl.write('#number of patches: '+str(nbp)+'\n')
                     mfl.close()
                     stww=stw+'.jpg'
                     flw=os.path.join(jpegpath,stww)
                     scipy.misc.imsave(flw, origscan)
                 else:
                     tw=tail+'_slice_'+str(slicescan)+'_'+dirFile+'_'+locabg
                     print tw,' :no patch'
                     errorfile.write(tw+' :no patch\n')
                     
                 print 'no lung'    
                 tabp = np.zeros((dimtabx, dimtaby), dtype='i')
                 origscan = Image.open(filenamec,'r')
                
                 #mask image
                 origscan = origscan.convert('L')
                 tabu=np.array(origscan)
                 tabf=np.copy(mask)
    #                     np.putmask(tabf,tabf>0,100)
    #                     cv2.imshow('tabu',tabu) 
    #                     cv2.waitKey(0)    
    #                     cv2.destroyAllWindows()  
                 np.putmask(tabf,tabf>0,1)
                 i=0
                 nbp=0
                 while i <= dimtabx:
                    j=0
                    while j<=dimtaby:
                        tabpatch=tabf[j:j+dimpavy,i:i+dimpavx]
                        area= tabpatch.sum()  
                        targ=float(area)/pxy
    #                            if targ <thrpatch:
    #                                print area,pxy, i,j
                        if targ >thrpatch:
    #                                print area,pxy, i,j
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
                         
                            if imagemax!=None:
      
                                nbp+=1
    
                               
                                nampa=os.path.join(nampadirnolungl,tail+'_'+str(slicescan)+'_'+str(nbp)+'.'+typei )
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
                            
                                nampa=os.path.join(nampadirNormNolungl,tail+'_'+str(slicescan)+'_'+str(nbp)+'.'+typei )                                    
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
                     
                 
                 tabu =tabu+tabp
                 ntotpat=ntotpat+nbp
    
                 if nbp>0:
                     if slicescan not in listsliceok:
                                listsliceok.append(slicescan) 
                     stw=tail+'_slice_'+str(slicescan)+'_'+nolung+'_'+locabg+'_'+str(nbp)
                     stww=stw+'.txt'
                     flw=os.path.join(jpegpath,stww)
                     mfl=open(flw,"w")
                     mfl.write('#number of patches: '+str(nbp)+'\n')
                     mfl.close()
                     stww=stw+'.jpg'
                     flw=os.path.join(jpegpath,stww)
                     scipy.misc.imsave(flw, tabu)
                 else:
                     tw=tail+'_slice_'+str(slicescan)+'_'+nolung+'_'+locabg
                     print tw,' :no patch'
                     errorfile.write(tw+' :no patch\n')
    #                             print listsliceok
#    print listsliceok
    return ntotpat




listdirc= (os.listdir(namedirtopc))
npat=0
for f in listdirc:
    #f = 35
    print('work on:',f)

    nbpf=0
    listsliceok=[]

#    namedirtopcf=namedirtopc+'/'+f
    namedirtopcf=os.path.join(namedirtopc,f) 
#    if os.path.isdir(namedirtopcf):    
#        sroidir=os.path.join(namedirtopcf,sroi)
#        remove_folder(sroidir)
#        os.mkdir(sroidir)
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