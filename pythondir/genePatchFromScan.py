# coding: utf-8
#Sylvain Kritter 5 octobre 2016
"""Top file to generate patches from DICOM database equalization from cv2"""
import os
import numpy as np
import shutil
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
namedirHUG = 'HUG'
#subdir for roi in text
subHUG='ILD_TXT'
#subHUG='ILDtt'


toppatch= 'TOPPATCH'  
#extension for output dir
extendir='16_set0_gcicolor'
#extendir='essai'

#normalization internal procedure or openCV
normiInternal=True# when True: use internal normi, otherwise opencv equalhist
globalHist=True #use histogram equalization on full image
globalHistInternal=False #use internal for global histogram when True otherwise opencv
#patch overlapp tolerance
thrpatch = 0.8
#labelEnh=('consolidation','reticulation,air_trapping','bronchiectasis','cysts')
labelEnh=()
imageDepth=255 #number of bits used on dicom images (2 **n)
# average pxixel spacing
avgPixelSpacing=0.734

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
#define the name of directory for patches
patchesdirnametop = toppatch+'_'+extendir
#define the name of directory for patches
patchesdirname = 'patches'
#define the name of directory for normalised patches
patchesNormdirname = 'patches_norm'
imagedirname='patches_jpeg'


bmpname='scan_bmp'
#directory with lung mask dicom
lungmask='lung_mask'
#directory to put  lung mask bmp
lungmaskbmp='bmp'
#directory name with scan with roi
sroi='sroi'
#directory name with scan with roi
bgdir='bgdir'
bgdirw='bgdirw'
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
labelbg='back_ground'
locabg='anywhere'

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



if pset ==0:
    usedclassif = [
        'back_ground',
        'consolidation',
        'HC',
        'fibrosis',
        'ground_glass',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        'emphysema'
 
        'bronchial_wall_thickening',
        'early_fibrosis',
        'increased_attenuation'
        'macronodules'
        'pcp',
        'peripheral_micronodules',
        'tuberculosis'
        ]
    classif ={
        'back_ground':0,
        'consolidation':1,
        'fibrosis':2,
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
        'healthy',
        'cysts'
        ]
        
    classif ={
    'back_ground':0,
    'consolidation':1,
    'ground_glass':2,
    'healthy':3,
    'cysts':4
    }
elif pset==2:
        usedclassif = [
        'back_ground',
        'fibrosis',
        'healthy',
        'micronodules'
        ,'reticulation'
        ]
        
        classif ={
    'back_ground':0,
    'fibrosis':1,
    'healthy':2,
    'micronodules':3,
    'reticulation':4,
    }
elif pset==3:
    usedclassif = [
        'back_ground',
        'healthy',
        'air_trapping',
        ]
    classif ={
        'back_ground':0,
        'healthy':1,
        'air_trapping':2,
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


patchtoppath=os.path.join(path_HUG,patchesdirnametop)

#create patch and jpeg directory
patchpath=os.path.join(patchtoppath,patchesdirname)
#create patch and jpeg directory
patchNormpath=os.path.join(patchtoppath,patchesNormdirname)
#print patchpath
#define the name for jpeg files
jpegpath=os.path.join(patchtoppath,imagedirname)


def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)


#patchpath = final/patches
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

eferror=os.path.join(patchtoppath,'genepatcherrortop.txt')
errorfile = open(eferror, 'w')
#filetowrite=os.path.join(namedirtopc,'lislabel.txt')
eflabel=os.path.join(patchtoppath,'lislabel.txt')
mflabel=open(eflabel,"w")



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

def rsliceNum(s,c,e):
    endnumslice=s.find(e)
    posend=endnumslice
    while s.find(c,posend)==-1:
        posend-=1
    debnumslice=posend+1
    return int((s[debnumslice:endnumslice])) 


def genebmp(dirName):
    """generate patches from dicom files and sroi"""
    print ('generate  bmp files from dicom files in :',f)
    global constPixelSpacing, dimtabx,dimtaby
    #directory for patches
    bmp_dir = os.path.join(dirName, bmpname)
    remove_folder(bmp_dir)    
    os.mkdir(bmp_dir)
    bgdirf = os.path.join(dirName, bgdir)
    remove_folder(bgdirf)    
    os.mkdir(bgdirf)
    lung_dir = os.path.join(dirName, lungmask)
#            print lung_dir
    lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)
    
    remove_folder(lung_bmp_dir)
#             if lungmaskbmp not in lunglist:
    os.mkdir(lung_bmp_dir)
   
    #list dcm files
    fileList = [name for name in os.listdir(dirName) if ".dcm" in name.lower()]
    lunglist = [name for name in os.listdir(lung_dir) if ".dcm" in name.lower()]
#    os.listdir(lung_dir)
    for filename in fileList:
#        print(filename)
#        if ".dcm" in filename.lower():  # check whether the file's DICOM
            FilesDCM =(os.path.join(dirName,filename))  
#           
            ds = dicom.read_file(FilesDCM)
            dsr= ds.pixel_array          
            dsr= dsr-dsr.min()
            c=float(imageDepth)/dsr.max()
            dsr=dsr*c
            if imageDepth <256:
                dsr=dsr.astype('uint8')
            else:
                dsr=dsr.astype('uint16')
    #resize the dicom to have always the same pixel/mm
            fxs=float(ds.PixelSpacing[0])/avgPixelSpacing   
#            print fxs                       
            scanNumber=int(ds.InstanceNumber)
            endnumslice=filename.find('.dcm')                   
            imgcore=filename[0:endnumslice]+'_'+str(scanNumber)+'.'+typei          
            bmpfile=os.path.join(bmp_dir,imgcore)
            dsrresize1= scipy.misc.imresize(dsr,fxs,interp='bicubic',mode=None) 
            namescan=os.path.join(sroidir,imgcore)                   
            textw='n: '+f+' scan: '+str(scanNumber)
#            scipy.misc.imsave(bmpfile,dsrresize1)
#            orign = Image.open(bmpfile)
#            imscanc= orign.convert('RGB')
#            tablscan = np.array(imscanc)
            tablscan=cv2.cvtColor(dsrresize1,cv2.COLOR_GRAY2BGR)
            scipy.misc.imsave(namescan, tablscan)
            tagviews(namescan,textw,0,20)  
            if globalHist:
                if globalHistInternal:
                    dsrresize = normi(dsrresize1) 
                else:
                    dsrresize = cv2.equalizeHist(dsrresize1) 
            else:
                dsrresize=dsrresize1
#            if globalHist:
#                         dsrresize = cv2.equalizeHist(dsrresize1) 
#            else:
#                         dsrresize=dsrresize1 
            
            scipy.misc.imsave(bmpfile,dsrresize)
#            bmpfiler=os.path.join(bmp_dir,imgcore)
#            cv2.imwrite(bmpfile,imgresize)
            dimtabx=dsrresize.shape[0]
            dimtaby=dimtabx
#            namescan=os.path.join(sroidir,imgcore)                   
#            textw='n: '+f+' scan: '+str(scanNumber)
#            orign = Image.open(bmpfile)
#            imscanc= orign.convert('RGB')
#            tablscan = np.array(imscanc)
#            scipy.misc.imsave(namescan, tablscan)
#            tagviews(namescan,textw,0,20)   
                        
#             print(lung_bmp_dir)
    for lungfile in lunglist:
#             print(lungfile)
#             if ".dcm" in lungfile.lower():  # check whether the file's DICOM
                 lungDCM =os.path.join(lung_dir,lungfile)  
                 dslung = dicom.read_file(lungDCM)
                 dsrlung= dslung.pixel_array  
#                 cv2.imshow('dsrlung1',dsrlung) 
#                 cv2.waitKey(0)    
#                 cv2.destroyAllWindows()   
                             
                 dsrlung= dsrlung-dsrlung.min()
#                 print dsrlung.min(),dsrlung.max()
                 if dsrlung.max()>0:
                     c=float(imageDepth)/dsrlung.max()
                 else:
                     c=0
                 dsrlung=dsrlung*c
                 if imageDepth <256:
                     dsrlung=dsrlung.astype('uint8')
                 else:
                     dsrlung=dsrlung.astype('uint16')
#                 cv2.imshow('dsrlung',dsrlung)
                 fxslung=float(dslung.PixelSpacing[0])/avgPixelSpacing 
#                 print fxslung
                 scanNumber=int(dslung.InstanceNumber)
                 endnumslice=lungfile.find('.dcm')                   
                 lungcore=lungfile[0:endnumslice]+'_'+str(scanNumber)+'.'+typei          
                 lungcoref=os.path.join(lung_bmp_dir,lungcore)
                 lungresize= scipy.misc.imresize(dsrlung,fxslung,interp='bicubic',mode=None)            
                 lungresize = cv2.blur(lungresize,(5,5))                 
                 np.putmask(lungresize,lungresize>0,100)
#                 cv2.imshow('lungresize',lungresize) 
#                 cv2.waitKey(0)    
#                 cv2.destroyAllWindows() 
                 scipy.misc.imsave(lungcoref,lungresize)
#                 cv2.imwrite(lungcoref,lungresize)
                 bgdirflm=os.path.join(bgdirf,lungcore)
##                 print lungcoref,bgdirflm                 
                 scipy.misc.imsave(bgdirflm,lungresize)

    
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
        x=2
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

def pavbg(namedirtopcf,dx,dy,px,py):
    print('generate back-ground for :',f)

    bgdirf = os.path.join(namedirtopcf, bgdir)
    patchpathc=os.path.join(namedirtopcf,bmpname)
   
    lbmp=os.listdir(patchpathc)
    listbg = os.listdir(bgdirf)
#    print patchpathc
    pxy=float(px*py) 
    for lm in listbg:
#        print lm
        nbp=0
        tabp = np.zeros((dx, dy), dtype='uint8')
        slicenumber=rsliceNum(lm,'_','.bmp')
        nambmp=os.path.join(patchpathc,lm)
        namebg=os.path.join(bgdirf,lm)
#
#find the same name in bgdir directory
        origbg = Image.open(namebg,'r')
        origbl= origbg.convert('L')
       
        for l in lbmp:
          slicen=rsliceNum(l,'_','.bmp')

          if slicen==slicenumber and slicenumber in listsliceok:
              nambmp=os.path.join(patchpathc,l)
              origbmp = Image.open(nambmp,'r')
              origbmpl= origbmp.convert('L')
              tabf=np.array(origbl)
              imagemax=origbl.getbbox()
#                                            
              min_val=np.min(origbl)
              max_val=np.max(origbl)
#              print min_val, max_val
#              cv2.imshow('tabf',tabf) 
##
#              cv2.waitKey(0)    
#              cv2.destroyAllWindows()   
        #put all to 1 if>0
              tabfc = np.copy(tabf)
              nz= np.count_nonzero(tabf)
              if nz>0 and min_val!=max_val:
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
                            
                            tabpatch=tabf[j:j+py,i:i+px]
                            area= tabpatch.sum()
                           
                            if float(area)/pxy >thrpatch:
#                                 print 'good'
    #                            good patch
        #                   
                                 crorig = origbmpl.crop((i, j, i+px, j+py))
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
#                                 if imagemax!=None:
                                    nbp+=1
                                    nampa='/'+labelbg+'/'+locabg+'/'+f+'_'+str(slicenumber)+'_'+str(nbp)+'.'+typei 
 
                                    crorig.save(patchpath+nampa)
                                    
                                    imgray =np.array(crorig)
#                                    imgray = cv2.cvtColor(imgra,cv2.COLOR_BGR2GRAY)
                                            #normalization internal procedure or openCV
                                    if normiInternal:
                                        tabi2 = normi(imgray) 
                                    else:
                                        tabi2 = cv2.equalizeHist(imgray)  
                                      
                                    scipy.misc.imsave(patchNormpath+nampa, tabi2)
                                
                                    x=0
                                    #we draw the rectange
                                    while x < px:
                                        y=0
                                        while y < py:
                                            tabp[y+j][x+i]=150
                                            if x == 0 or x == px-1 :
                                                y+=1
                                            else:
                                                y+=py-1
                                        x+=1
                                    #we cancel the source
                                    tabf[j:j+py,i:i+px]=0                           
                            j+=1
                        i+=1
                
#              print tabfc.shape , tabfc.dtype    
#              cv2.imshow('tabfc',tabfc) 
###
#              cv2.waitKey(0)    
#              cv2.destroyAllWindows()  
#              print tabp.shape , tabp.dtype 
              tabpw =tabfc+tabp
              scipy.misc.imsave(jpegpath+'/'+f+'_slice_'+str(slicenumber)+\
        '_'+labelbg+'_'+locabg+'.jpg', tabpw) 
              mfl=open(jpegpath+'/'+f+'_slice_'+str(slicenumber)+'_'+labelbg+\
        '_'+locabg+'_1.txt',"w")
#        mfl=open(jpegpath+'/'+f+'_'+slicenumber+'.txt',"w")
              mfl.write('#number of patches: '+str(nbp)+'\n')
              mfl.close()
              break

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
   
###
  
def pavs (imgi,tab,dx,dy,px,py,namedirtopcf,jpegpath,patchpath,thr,\
    iln,f,label,loca,typei,errorfile):
    """ generate patches from ROI"""
#    print 'pavement'
    if label == 'fibrosis':
        label='HC'
    vis=contour2(imgi,label)         
    bgdirf = os.path.join(namedirtopcf, bgdir)    
    patchpathc=os.path.join(namedirtopcf,bmpname)
    # patchpathc path to scan mage bmp   
    contenujpg = os.listdir(patchpathc)
    debnumslice=iln.find('_')+1
    endnumslice=iln.find('_',debnumslice)
    slicenumber=int(iln[debnumslice:endnumslice])

#    print slns
    tabp = np.zeros((dx, dy), dtype='i')

    np.putmask(tab,tab>0,100)
    tabf = np.copy(tab)
    np.putmask(tabf,tabf>0,1)
    pxy=float(px*py)
#    i=max(mintabx-px,0)
    nbp=0
    strpac=''
    errorliststring=[]

    lung_dir1 = os.path.join(namedirtopcf, lungmask)
#    print lung_dir1
    lung_bmp_dir = os.path.join(lung_dir1,lungmaskbmp)
    lunglist = os.listdir(lung_bmp_dir)
    atabf = np.nonzero(tabf)
        #tab[y][x]  convention
    xmin=atabf[1].min()
    xmax=atabf[1].max()
    ymin=atabf[0].min()
    ymax=atabf[0].max()
    found=False
    for  n in contenujpg:   
        
#        print n# list scan images
        slicescan=rsliceNum(n,'_','.'+typei)
        #                    print(slns)
#        print slicescan,slicenumber
        if slicescan==slicenumber:
            found=True
            namebmp=os.path.join(patchpathc,n)
            namescan=os.path.join(sroidir,n)   
#            namebmp=namedirtopcf+'/'+typei+'/'+n
            orig = Image.open(namebmp)
#            print n
            orign = Image.open(namescan)
            imscanc= orign.convert('RGB')
           
            tablscan = np.array(imscanc)
            
            imn=cv2.add(vis,tablscan)
            imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)

            cv2.imwrite(namescan,imn)

            for lm in lunglist: # scan lung mask
#                print lunglist
#                print slin
                slicelung=rsliceNum(n,'_','.'+typei)
                if slicelung==slicenumber:
                    #look in lung maask the name of slice

                    namebg=os.path.join(bgdirf,lm)
#find the same name in bgdir directory
                    tabhc=cv2.imread(namebg,0)
                    np.putmask(tabhc,tabhc>0,100)
                    imgray = cv2.cvtColor(imgi,cv2.COLOR_BGR2GRAY)
                    tabf=np.array(imgray)
#                    print tabf.shape,tabhc.shape
                    np.putmask(tabf,tabf>0,100)
                    mask=cv2.bitwise_not(tabf)
#                    print mask.shape,tabhc.shape
    #                    cv2.imshow('masky',masky) 
    #                    print np.shape(tabhc), np.shape(masky)
                    outy=cv2.bitwise_and(tabhc,mask)                                        

                    cv2.imwrite(namebg,outy)
                    break
                    
            tagview(namescan,label,0,100)
#            print 'tagview1', namescan, label
            if slicenumber not in listsliceok:
                 listsliceok.append(slicenumber )
            i=xmin
            np.putmask(tabf,tabf==1,0)
            np.putmask(tabf,tabf>0,1)
            while i <= xmax:
                j=ymin
                while j<=ymax:
                    tabpatch=tabf[j:j+py,i:i+px]
                    area= tabpatch.sum()  
                    targ=float(area)/pxy

                    if targ >thr:
 #good patch     

                                          
                        crorig = orig.crop((i, j, i+px, j+py))
                         #detect black pixels
                         #imagemax=(crorig.getextrema())
                        imagemax=crorig.getbbox()
                        min_val=np.min(crorig)
                        max_val=np.max(crorig)

                     
                        if imagemax==None or min_val==max_val:

                            errortext='black or mono level pixel  in: '+ f+' '+ iln+'\n'
                            if errortext not in errorliststring:
                                errorliststring.append(errortext)
                                print(errortext)
#                          
                        else:
                            nbp+=1
                            nampa='/'+label+'/'+loca+'/'+f+'_'+iln+'_'+str(nbp)+'.'+typei 

                            crorig.save(patchpath+nampa) 
        #normalize patches and put in patches_norm
                            imgray =np.array(crorig)
                            if normiInternal:
                                        tabi2 = normi(imgray) 
                            else:
                                        tabi2 = cv2.equalizeHist(imgray)  
                            scipy.misc.imsave(patchNormpath+nampa, tabi2)
                            
                        #                print('pavage',i,j)  
                            strpac=strpac+str(i)+' '+str(j)+'\n'
                            x=0
                            #we draw the rectange
                            while x < px:
                                y=0
                                while y < py:
                                    tabp[y+j][x+i]=150
                                    if x == 0 or x == px-1 :
                                        y+=1
                                    else:
                                        y+=py-1
                                x+=1
                            #we cancel the source
                            if label not in labelEnh:
                                tabf[j:j+py,i:i+px]=0
                            else:
                                 tabf[j:j+py/2,i:i+px/2]=0                          
                    j+=1
                i+=1
            break
    
    if not found:
            print('ERROR image not found '+namedirtopcf+'/'+bmpname+'/'+str(slicenumber))            
            
            errorfile.write('ERROR image not found '+namedirtopcf+\
                '/'+bmpname+'/'+str(slicenumber)+'\n')#####
    tabp =tab+tabp
    mfl=open(jpegpath+'/'+f+'_'+iln+'.txt',"w")
    mfl.write('#number of patches: '+str(nbp)+'\n'+strpac)
    mfl.close()
    scipy.misc.imsave(jpegpath+'/'+f+'_'+iln+'.jpg', tabp)
    if len(errorliststring) >0:
        for l in errorliststring:
            errorfile.write(l)
    return nbp,tabp


def fileext(namefile,curdir,patchpath):
    listlabel=[labelbg+'_'+locabg]
    plab=os.path.join(patchpath,labelbg)
    ploc=os.path.join(plab,locabg) 
    plabNorm=os.path.join(patchNormpath,labelbg)
    plocNorm=os.path.join(plabNorm,locabg) 
    if not os.path.exists(plab):
        os.mkdir(plab)
    if not os.path.exists(plabNorm):
        os.mkdir(plabNorm)
    if not os.path.exists(ploc):
        os.mkdir(ploc)
    if not os.path.exists(plocNorm):
        os.mkdir(plocNorm)

    ofi = open(namefile, 'r')
    t = ofi.read()
    #print( t)
    ofi.close()
#
    nslice = t.count('slice')
#    print('number of slice:',nslice)
    numbercon = t.count('contour')
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
                    plabNorm=os.path.join(patchNormpath,label)
#                    print plabNorm
                    plocNorm=os.path.join(plabNorm,loca) 
                    listlabel.append(label+'_'+loca)     
                    listlabeld=os.listdir(patchpath)
                    if label not in listlabeld:
#                            print label
                            os.mkdir(plab)
                            os.mkdir(plabNorm)
                    listlocad=os.listdir(plab)
                    if loca not in listlocad:
                            os.mkdir(ploc)
                            os.mkdir(plocNorm)
                            

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
#                if (int(slice)==19):
#                    print(slapos,slaposnext,nbpposnext,labposnext,\
#                    tabposdeb,tabposend)
        #        print('fin tableau:',tabposend)
                nametab=curdir+'/patchfile/slice_'+str(slice)+'_'+str(label)+\
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
    return(listlabel,coefi)

listdirc= (os.listdir(namedirtopc))
npat=0
for f in listdirc:
    #f = 35
    print('work on:',f)

    nbpf=0
    listsliceok=[]
    posp=f.find('.',0)
    posu=f.find('_',0)
    namedirtopcf=namedirtopc+'/'+f
      
    if os.path.isdir(namedirtopcf):    
        sroidir=os.path.join(namedirtopcf,sroi)
        remove_folder(sroidir)
        os.mkdir(sroidir)

    remove_folder(namedirtopcf+'/patchfile')
    os.mkdir(namedirtopcf+'/patchfile')
    #namedirtopcf = final/ILD_DB_txtROIs/35
    if posp==-1 and posu==-1:
        contenudir = os.listdir(namedirtopcf)
#        print(contenudir)
        fif=False
        genebmp(namedirtopcf)

        for f1 in contenudir:
            
            if f1.find('.txt') >0 and (f1.find('CT')==0 or \
             f1.find('Tho')==0):
#                print f1
                npat+=1
                fif=True
                fileList =f1
                pathf1=namedirtopcf+'/'+fileList            
                labell,coefi =fileext(pathf1,namedirtopcf,patchpath)

                break
        if not fif:
             print('ERROR: no ROI txt content file', f)
             errorfile.write('ERROR: no ROI txt content file in: '+ f+'\n')
        
        listslice= os.listdir(namedirtopcf+'/patchfile') 
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
            ftab=True
            tabzc = np.zeros((dimtabx, dimtaby), dtype='i')
            imgc = np.zeros((dimtabx,dimtaby,3), np.uint8)
            for l in listslice:
#                print('l',l,'c:',c)
                if l.find(c,0)==0:
                    pathl=namedirtopcf+'/patchfile/'+l
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
#                print(label,loca)
                #print(tabcff,coefi)
                    tabccfi=tabcff/avgPixelSpacing
#                    tabccfi=tabcff/coefi

#                print(tabccfi)
                    tabc=tabccfi.astype(int)
                    
#                print(tabc)
                    print('generate tables from:',l,'in:', f)
                    tabz,imgi= reptfulle(tabc,dimtabx,dimtaby)                
                    imgc=imgc+imgi                    
                    tabzc=tabz+tabzc
                                
#                    print('end create tables')
                    il=l.find('.',0)
                    iln=l[0:il]
#                    print iln
            if label in usedclassif:
                print('c :',c, label,loca)
                print('creates patches from:',iln, 'in:', f)
                nbp,tabz1=pavs (imgc,tabzc,dimtabx,dimtaby,dimpavx,dimpavy,namedirtopcf,\
                    jpegpath, patchpath,thrpatch,iln,f,label,loca,typei,errorfile)
                print('end create patches')
                nbpf=nbpf+nbp
            #create patches for back-ground
        pavbg(namedirtopcf,dimtabx,dimtaby,dimpavx,dimpavy)
#    print(f,nbpf)
    ofilepw = open(jpegpath+'/nbpat_'+f+'.txt', 'w')
    ofilepw.write('number of patches: '+str(nbpf))
    ofilepw.close()
    
    
#################################################################    
#   calculate number of patches
contenupatcht = os.listdir(jpegpath) 
#        print(contenupatcht)
npatcht=0
for npp in contenupatcht:
#    print('1',npp)
    if npp.find('.txt')>0 and npp.find('nbp')==0:
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
eftpt=os.path.join(patchtoppath,'totalnbpat.txt')
filepwt = open(eftpt, 'w')
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

#write the log file with label list
mflabel.write('label  _  localisation\n')
mflabel.write('======================\n')
categ=os.listdir(jpegpath)
for f in categ:
    if f.find('.txt')>0 and f.find('nb')==0:
        ends=f.find('.txt')
        debs=f.find('_')
        sln=f[debs+1:ends]
        listlabel={}
        
        for f1 in categ:
                if  f1.find(sln+'_')==0 and f1.find('.txt')>0:
                    debl=f1.find('slice_')
                    debl1=f1.find('_',debl+1)
                    debl2=f1.find('_',debl1+1)
                    endl=f1.find('.txt')
                    j=0
                    while f1.find('_',endl-j)!=-1:
                        j-=1
                    label=f1[debl2+1:endl-j-2]
                    ffle1=os.path.join(jpegpath,f1)
                    fr1=open(ffle1,'r')
                    t1=fr1.read()
                    fr1.close()
                    debsp=t1.find(':')
                    endsp=  t1.find('\n')
                    np=int(t1[debsp+1:endsp])
                    if label in listlabel:
                                listlabel[label]=listlabel[label]+np
                    else:
                        listlabel[label]=np
        listslice.append(sln)
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
           if l !=labelbg+'_'+locabg:
             mflabel.write(l+' '+str(listlabel[l])+'\n')
        mflabel.write('---------------------'+'\n')

mflabel.close()

##########################################################
errorfile.write('completed')
errorfile.close()
print('completed')