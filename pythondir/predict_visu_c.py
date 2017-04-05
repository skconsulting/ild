# coding: utf-8
#Sylvain Kritter 21 septembre 2016
""" generate predict and visu with lung in bmp file already there """
#general parameters and file, directory names"

import os
import cv2
import datetime
import time
import dicom
import scipy
import sys
import shutil
import numpy as np
#import Tkinter as Tk

import PIL
from PIL import Image as ImagePIL
from PIL import  ImageFont, ImageDraw 
import cPickle as pickle
import ild_helpers as H
import cnn_model as CNN4
from keras.models import model_from_json
from Tkinter import *

#########################################################
# for predict
# with or without bg (true if with back_ground)
wbg=True
#to enhance contrast on patch put True
contrast=True
#threshold for patch acceptance
thrpatch = 0.9
#threshold for probability prediction
thrproba = 0
#normalization internal procedure or openCV
normiInternal=True
#global directory for predict file
namedirtop = 'predict_gre'

#directory for storing image out after prediction
predictout='predicted_results'

#directory with lung mask dicom
lungmask='lung_mask'

#directory to put  lung mask bmp
lungmaskbmp='bmp'

#directory name with scan with roi
sroi='sroi'

#subdirectory name to put images
jpegpath = 'patch_jpeg'

#directory with bmp from dicom
scanbmp='scan_bmp'

Xprepkl='X_predict.pkl'
Xrefpkl='X_file_reference.pkl'
subsamplef='subsample.txt'
#subdirectory name to colect pkl files resulting from prediction
picklefile='pickle_sk32'

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
path_patient = os.path.join(cwdtop,namedirtop)
if not os.path.exists(path_patient):
    print 'patient directory does not exists'
    sys.exit()
    
picklein_file = os.path.join(cwdtop,picklefile)
if not os.path.exists(picklein_file):
    print 'model and weight directory does not exists'
    sys.exit()
lpck= os.listdir(picklein_file)
pson=False
pweigh=False
for l in lpck:
    if l.find('.json',0)>0:
        pson=True
    if l.find('ILD_CNN_model_weights',0)==0:

        pweigh=True
if not(pweigh and pson):
    print 'model and/or weight files does not exists'
    sys.exit()     

#print patient_list
# list label not to visualize
excluvisu=['back_ground','healthy']
#excluvisu=[]

#dataset supposed to be healthy
datahealthy=['138']
#end predict part
#########################################################
# general
#image  patch format
typei='bmp' #can be jpg
#dicom file size in pixels
dimtabx = 512
dimtaby = 512
#patch size in pixels 32 * 32
dimpavx =32
dimpavy = 32

mini=dimtabx-dimpavx
minj=dimtaby-dimpavy

pxy=float(dimpavx*dimpavy)

#end general part
#font file imported in top directory
font20 = ImageFont.truetype( 'arial.ttf', 20)
font10 = ImageFont.truetype( 'arial.ttf', 10)
#########################################################
errorfile = open(path_patient+'/predictlog.txt', 'w') 

#color of labels

red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255,255,0)
cyan=(0,255,255)
purple=(255,0,255)
white=(255,255,255)
darkgreen=(11,123,96)
pink =(255,128,150)
lightgreen=(125,237,125)
orange=(255,153,102)



#only label we consider, number will start at 0 anyway
if wbg :
    classif ={
'back_ground':0,
'consolidation':1,
'fibrosis':2,
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
else:
     classif ={
    'consolidation':0,
    'fibrosis':1,
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
'consolidation':red,
'fibrosis':blue,
'ground_glass':yellow,
'healthy':green,
'micronodules':cyan,
'reticulation':purple,
'air_trapping':pink,
'cysts':orange, 
 'bronchiectasis':lightgreen,
  
  
 'bronchial_wall_thickening':white,
 'early_fibrosis':white,
 'emphysema':white,
 'increased_attenuation':white,
 'macronodules':white,
 'pcp':white,
 'peripheral_micronodules':white,
 'tuberculosis':white
 }


def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path,ignore_errors=True)

   
def genebmp(dirName,fn,subs):
    """generate patches from dicom files"""
#    print ('load dicom files in :',dirName, 'scan name:',fn)
    #directory for patches
    
    predictout_f_dir = os.path.join( dirName,picklefile)
    remove_folder(predictout_f_dir)
    os.mkdir(predictout_f_dir)
#    print 'predictoutdir', predictout_f_dir
    subfile=os.path.join(predictout_f_dir,subsamplef)
    subfilec = open(subfile, 'w')
    subfilec.write('subsample '+str(subs)+'\n' )
    subfilec.close()
    bmp_dir = os.path.join(dirName, scanbmp)

    FilesDCM =(os.path.join(dirName,fn))  
#           
    ds = dicom.read_file(FilesDCM)
    endnumslice=fn.find('.dcm')
    imgcore=fn[0:endnumslice]+'.'+typei
   
    posend=endnumslice
    while fn.find('-',posend)==-1:
        posend-=1
    debnumslice=posend+1
    slicenumber=int(fn[debnumslice:endnumslice])

    bmpfile=os.path.join(bmp_dir,imgcore)
    scipy.misc.imsave(bmpfile, ds.pixel_array)

#    lung_dir = os.path.join(dirName, lungmask)
#    
#    lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)
#    lunglist = os.listdir(lung_dir)
#    #             print(lung_bmp_dir)
##    lungFound=False
#    for lungfile in lunglist:
#    #                print(lungfile)
#                    if ".dcm" in lungfile.lower():
#                        endnumslice=lungfile.find('.dcm')
#                        imgcorescan=lungfile[0:endnumslice]+'.'+typei  
#                        posend=endnumslice
#                        while lungfile.find('_',posend)==-1:
#                            posend-=1
#                        debnumslice=posend+1
#                        slicescan=int(lungfile[debnumslice:endnumslice])
##                        print slicescan
#                        if slicescan== slicenumber:
#
#        # check whether the file's DICOM
#                            lungDCM =os.path.join(lung_dir,lungfile)  
#                            dslung = dicom.read_file(lungDCM)
#                           
#                            lungcore=imgcorescan+'.'+typei
#                            lungcoref=os.path.join(lung_bmp_dir,lungcore)
#                            scipy.misc.imsave(lungcoref, dslung.pixel_array)
#                            break
                   

def normi(img):
     """ normalise patches 0 255"""
     tabi = np.array(img)
#     print(tabi.min(), tabi.max())
     tabi1=tabi-tabi.min()
#     print(tabi1.min(), tabi1.max())
     tabi2=tabi1*(255/float(tabi1.max()-tabi1.min()))
#     print(tabi2.min(), tabi2.max())
     return tabi2

def pavgene (namedirtopcf):
        """ generate patches from scan"""
#        print('generate patches on: ',namedirtopcf)
#        n=0
        (dptop,dptail)=os.path.split(namedirtopcf)
#        print 'namedirto',dptail
        namemask1=os.path.join(namedirtopcf,lungmask)
        namemask=os.path.join(namemask1,lungmaskbmp)
#        print namemask
        bmpdir = os.path.join(namedirtopcf,scanbmp)
        jpegpathf=os.path.join(namedirtopcf,jpegpath)
        remove_folder(jpegpathf)    
        os.mkdir(jpegpathf)
        
        listbmp= os.listdir(bmpdir)
#        print(listbmp)
        
        listlungbmp= os.listdir(namemask)    
                
        for img in listbmp:
#             print img
             tflung=False
             endnumslice=img.find('.bmp')
             posend=endnumslice
             while img.find('-',posend)==-1:
                     posend-=1
             debnumslice=posend+1
             slicenumber=int(img[debnumslice:endnumslice])         
             slns='_'+str(slicenumber)+'.'+typei
#             print(slns)
#             print(listlungbmp)
             if listlungbmp!=[]:
#                    print(listlungbmp)
                    for llung in listlungbmp:
                    
    
                        if llung.find(slns) >0:
                            tflung=True
                            lungfile = os.path.join(namemask,llung)
        #                    print(lungfile)
#                            imlung = ImagePIL.open(lungfile)
                            tablung = cv2.imread(lungfile,0)

#                            tablung = np.array(imlung)
                            np.putmask(tablung,tablung>0,1)
                            break
#             print tflung   
             if not tflung:
                    tablung = np.ones((dimtabx, dimtaby), dtype='i')
#             np.putmask(tablung,tablung>0,255)             
#             cv2.imshow('image',tablung) 
#             cv2.waitKey(0)
#             np.putmask(tablung,tablung>0,1)  
#             print tablung 
             bmpfile = os.path.join(bmpdir,img)
             tabf = cv2.imread(bmpfile,1)
#             im = cv2.imread(bmpfile,1)
             im = ImagePIL.open(bmpfile)
#             imc= im.convert('RGB')
#                  
#             tabf = np.array(imc)
#             cv2.imshow('image',tablung) 
#             cv2.waitKey(0)
             nz= np.count_nonzero(tablung)
             if nz>0:
            
                atabf = np.nonzero(tablung)
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
        #        j=maxj
                 while j<=ymax:
#                     print(i,j)
                     tabpatch=tablung[j:j+dimpavy,i:i+dimpavx]
                     area= tabpatch.sum()  
        
#                    check if area above threshold
                     targ=float(area)/pxy
#                     print area, pxy
                     if targ>thrpatch:
#                        print area, pxy
                        crorig = im.crop((i, j, i+dimpavx, j+dimpavy))
                        imagemax=crorig.getbbox()
        #               detect black patch
        #                print (imagemax)
                        if imagemax!=None:
#                            namepatch=patchpathf+'/p_'+str(slicenumber)+'_'+str(i)+'_'+str(j)+'.'+typei
                            if contrast:
                                    if normiInternal:
                                        tabcont=normi(crorig)
                                    else:
                                        npcrorig =np.array(crorig)
                                        tabcont = cv2.equalizeHist(npcrorig)
                                    patch_list.append((dptail,slicenumber,i,j,tabcont))
#                                    n+=1
                            else:
                                patch_list.append((dptail,slicenumber,i,j,crorig))
                            
                            tablung[j:j+dimpavy,i:i+dimpavx]=0
                            x=0
                            while x < dimpavx:
                                y=0
                                while y < dimpavy:
                                    tabf[y+j][x+i]=[255,0,0]
                                    if x == 0 or x == dimpavx-1 :
                                        y+=1
                                    else:
                                        y+=dimpavy-1
                                x+=1                                             
                            
                            
                            
#                            np.putmask(tablung,tablung>0,255)             
#                            cv2.imshow('image',tablung) 
#                            np.putmask(tablung,tablung>0,1)  
#                            cv2.waitKey(0)
                            j+=dimpavy-1
#                     j+=dimpavy
                     j+=1
#                 i+=dimpavx
                 i+=1
#        print namedirtopcf,n
             scipy.misc.imsave(jpegpathf+'/'+'s_'+str(slicenumber)+'.bmp', tabf)
# 
def ILDCNNpredict(patient_dir_s):     
        
#        print ('predict patches on: ',patient_dir_s) 
        (top,tail)=os.path.split(patient_dir_s)
        for fil in patch_list:
            if fil[0]==tail:
               
                dataset_list.append(fil[4])
                nameset_list.append(fil[0:3])
        
        X = np.array(dataset_list)
#        print X[1].shape
        X_predict = np.asarray(np.expand_dims(X,1))/float(255)        
        
        jsonf= os.path.join(picklein_file,'ILD_CNN_model.json')
#        print jsonf
        weigf= os.path.join(picklein_file,'ILD_CNN_model_weights')
#        print weigf
#model and weights fr CNN
        args  = H.parse_args()                          
        train_params = {
     'do' : float(args.do) if args.do else 0.5,        
     'a'  : float(args.a) if args.a else 0.3,          # Conv Layers LeakyReLU alpha param [if alpha set to 0 LeakyReLU is equivalent with ReLU]
     'k'  : int(args.k) if args.k else 4,              # Feature maps k multiplier
     's'  : float(args.s) if args.s else 1,            # Input Image rescale factor
     'pf' : float(args.pf) if args.pf else 1,          # Percentage of the pooling layer: [0,1]
     'pt' : args.pt if args.pt else 'Avg',             # Pooling type: Avg, Max
     'fp' : args.fp if args.fp else 'proportional',    # Feature maps policy: proportional, static
     'cl' : int(args.cl) if args.cl else 5,            # Number of Convolutional Layers
     'opt': args.opt if args.opt else 'Adam',          # Optimizer: SGD, Adagrad, Adam
     'obj': args.obj if args.obj else 'ce',            # Minimization Objective: mse, ce
     'patience' : args.pat if args.pat else 5,         # Patience parameter for early stoping
     'tolerance': args.tol if args.tol else 1.005,     # Tolerance parameter for early stoping [default: 1.005, checks if > 0.5%]
     'res_alias': args.csv if args.csv else 'res'      # csv results filename alias
         }
#        model = H.load_model()

        model = model_from_json(open(jsonf).read())
        model.load_weights(weigf)

        model.compile(optimizer='Adam', loss=CNN4.get_Obj(train_params['obj']))        

        proba = model.predict_proba(X_predict, batch_size=100)
        predictout_f_dir = os.path.join( patient_dir_s,picklefile)
        xfp=os.path.join(predictout_f_dir,Xprepkl)
        pickle.dump(proba, open( xfp, "wb" ))
        xfpr=os.path.join(predictout_f_dir,Xrefpkl)
        pickle.dump(patch_list, open( xfpr, "wb" ))


def fidclass(numero):
    """return class from number"""
    found=False
    for cle, valeur in classif.items():
        
        if valeur == numero:
            found=True
            return cle
      
    if not found:
        return 'unknown'

 
def tagviewn(fig,label,pro,x,y):
    """write text in image according to label and color"""

    col=classifc[label]
#    print col, label
    if wbg :
        labnow=classif[label]-1
    else:
        labnow=classif[label]
#    print (labnow, text)
    if label == 'back_ground':
        x=0
        y=0        
        deltax=0
        deltay=60
    else:        
        deltay=11*((labnow)%5)
        deltax=175*((labnow)//5)

    cv2.putText(fig,label+' '+pro,(x+deltax, y+deltay+10),cv2.FONT_HERSHEY_PLAIN,0.8,col,1)

    
def tagviews(fig,t0,x0,y0,t1,x1,y1,t2,x2,y2,t3,x3,y3,t4,x4,y4):
    """write simple text in image """
    imgn=ImagePIL.open(fig)
    draw = ImageDraw.Draw(imgn)
    draw.rectangle ([x1, y1,x1+100, y1+15],outline='black',fill='black')
    draw.rectangle ([140, 0,512,75],outline='black',fill='black')
    draw.text((x0, y0),t0,white,font=font10)
    draw.text((x1, y1),t1,white,font=font10)
    draw.text((x2, y2),t2,white,font=font10)
#    draw.text((x3, y3),t3,white,font=font10)
    draw.text((x4, y4),t4,white,font=font10)
    imgn.save(fig)

def maxproba(proba):
    """looks for max probability in result"""
    lenp = len(proba)
    m=0
    for i in range(0,lenp):
        if proba[i]>m:
            m=proba[i]
            im=i
    return im,m


def loadpkl(do):
    """crate image directory and load pkl files"""
#    global classdirec
    
    predictout_f_dir = os.path.join( do,picklefile)
    xfp=os.path.join(predictout_f_dir,Xprepkl)
    dd = open(xfp,'rb')
    my_depickler = pickle.Unpickler(dd)
    probaf = my_depickler.load()
    dd.close()  
    
    xfpr=os.path.join(predictout_f_dir,Xrefpkl)
    dd = open(xfpr,'rb')
    my_depickler = pickle.Unpickler(dd)
    patch_listr = my_depickler.load()
    dd.close() 
    
    preprob=[]
    prefile=[]
    
    (top,tail)=os.path.split(do)

    n=0
    for fil in patch_listr:        
        if fil[0]==tail:
#            print n, proba[n]
            preprob.append(probaf[n])
            prefile.append(fil[1:4])
        n=n+1
    return (preprob,prefile)



def  visua(dirpatientdb,cla,wra):

    (dptop,dptail)=os.path.split(dirpatientdb)
    if cla==1:
        topdir=dptail
    else:
        (dptop1,dptail1)=os.path.split(dptop)
        topdir=dptail1
#        print 'topdir visua',topdir
    for i in range (0,len(classif)):
#        print 'visua dptail', topdir
        listelabelfinal[topdir,fidclass(i)]=0
    #directory name with predict out dabasase, will be created in current directory
    predictout_dir = os.path.join(dirpatientdb, predictout)
    predictout_dir_th = os.path.join(predictout_dir,str(thrproba))
#    print predictout_dir_th
    if wra:
        if not os.path.exists(predictout_dir) :
             os.mkdir(predictout_dir)    
        remove_folder(predictout_dir_th)
        os.mkdir(predictout_dir_th)
#    print predictout_dir_th
    (preprob,listnamepatch)=loadpkl(dirpatientdb)
#    print preprob[0], listnamepatch
    dirpatientfdb=os.path.join(dirpatientdb,scanbmp)
    dirpatientfsdb=os.path.join(dirpatientdb,sroi)
    listbmpscan=os.listdir(dirpatientfdb)
#    print dirpatientfdb
    listlabelf={}
#    setname=f
#    tabsim1 = np.zeros((dimtabx, dimtaby), dtype='i')
    for img in listbmpscan:

        listlabel={}
        if os.path.exists(dirpatientfsdb):
#        imgc=os.path.join(dirpatientfdb,img)
            imgc=os.path.join(dirpatientfsdb,img)
        else:
            imgc=os.path.join(dirpatientfdb,img)
 
        endnumslice=img.find('.'+typei)
        imgcore=img[0:endnumslice]
#        print imgcore
        posend=endnumslice
        while img.find('-',posend)==-1:
            posend-=1
        debnumslice=posend+1
        slicenumber=int((img[debnumslice:endnumslice])) 
        imscan = ImagePIL.open(imgc)
        imscanc= imscan.convert('RGB')
        tablscan = np.array(imscanc)
        if imscan.size[0]>512:
            ncr=imscanc.resize((dimtabx,dimtaby),PIL.Image.ANTIALIAS)
            tablscan = np.array(ncr) 
        ill = 0
      
        foundp=False
#        patch_list.append((dptail,slicenumber,i,j,tabcont))
        for ll in listnamepatch:
            slicename=ll[0]    
            proba=preprob[ill]          
            prec, mprobai = maxproba(proba)
            classlabel=fidclass(prec)           
#            print slicenumber, slicename,dptail
            if slicenumber == slicename and\
            (dptail in datahealthy or (classlabel not in excluvisu)):
#                    print slicenumber, slicename,dptail
                    foundp=True
                    if classlabel in listlabel:
                        numl=listlabel[classlabel]
                        listlabel[classlabel]=numl+1
                    else:
                        listlabel[classlabel]=1
                    if classlabel in listlabelf:
                        nlt=listlabelf[classlabel]
                        listlabelf[classlabel]=nlt+1
                    else:
                        listlabelf[classlabel]=1

            ill+=1

        if wra:        
            imgcorefull=imgcore+'.bmp'
            imgname=os.path.join(predictout_dir_th,imgcorefull)
    #        print 'imgname',imgname
    
            cv2.imwrite(imgname,tablscan)
    
           
            if foundp:
                t0='average probability'
            else:
                t0='no recognised label'
            t1='n: '+dptail+' scan: '+str(slicenumber)        
            t2='CONFIDENTIAL - prototype - not for medical use'
            t3='threshold: '+str(thrproba)
            t4=time.asctime()
            tagviews(imgname,t0,0,0,t1,0,20,t2,20,485,t3,0,40,t4,0,50)
        
    if wra:
            errorfile.write('\n'+'number of labels in :'+str(dptail)+'\n' )
#    print listlabelf
    for classlabel in listlabelf:  
          listelabelfinal[topdir,classlabel]=listlabelf[classlabel]
          print 'patient: ',topdir,', label:',classlabel,': ',listlabelf[classlabel]
          string=str(classlabel)+': '+str(listlabelf[classlabel])+'\n' 
#          print string
          errorfile.write(string )

#    
def renomscan(fa):
        num=0
        contenudir = os.listdir(fa)
#        print(contenudir)
        for ff in contenudir:
#            print ff
            if ff.find('.dcm')>0 and ff.find('-')<0:     
                num+=1    
                corfpos=ff.find('.dcm')
                cor=ff[0:corfpos]
                ncff=os.path.join(fa,ff)
#                print ncff
                if num<10:
                    nums='000'+str(num)
                elif num<100:
                    nums='00'+str(num)
                elif num<1000:
                    nums='0'+str(num)
                else:
                    nums=str(num)
                newff=cor+'-'+nums+'.dcm'
    #            print(newff)
                shutil.copyfile(ncff,os.path.join(fa,newff) )
                os.remove(ncff)
def dd(i):
    if (i)<10:
        o='0'+str(i)
    else:
        o=str(i)
    return o


def nothings(x):
    global imgtext
    imgtext = np.zeros((512,512,3), np.uint8)
    pass

def nothing(x):
    pass

def contrast(im,r):   
     tabi = np.array(im)
     r1=0.5+r/100.0
     tabi1=tabi*r1     
     tabi2=np.clip(tabi1,0,255)
     tabi3=tabi2.astype(np.uint8)
     return tabi3

def lumi(im,r):
    tabi = np.array(im)
    r1=r
    tabi1=tabi+r1
    tabi2=np.clip(tabi1,0,255)
    return tabi2

# mouse callback function
def draw_circle(event,x,y,flags,img):
    global ix,iy,quitl,patchi
    patchi=False

    if event == cv2.EVENT_RBUTTONDBLCLK:
        print x, y
    if event == cv2.EVENT_LBUTTONDBLCLK:
 
#        print('identification')
        ix,iy=x,y
       
        patchi=True
#        print 'identification', ix,iy, patchi
        if x>250 and x<270 and y>490 and y<510:
            print 'quit'
            ix,iy=x,y
            quitl=True

def addpatchn(col,lab, xt,yt,imgn):
#    print col,lab
    cv2.rectangle(imgn,(xt,yt),(xt+dimpavx,yt+dimpavy),col,1)
    return imgn
 
def retrievepatch(x,y,top,sln,pr,li):
    tabtext = np.zeros((512,512,3), np.uint8)
    ill=-1
    pfound=False
    for f in li:
        ill+=1 
        slicenumber=f[0]

        if slicenumber == sln:

            xs=f[1]
            ys=f[2]
#            print xs,ys
            if x>xs and x < xs+dimpavx and y>ys and y<ys+dimpavy:
#                     print slicenumber, f, x, xs, y, ys
                     proba=pr[ill]
                     pfound=True

                     n=0
                     cv2.putText(tabtext,'X',(xs-5+dimpavx/2,ys+5+dimpavy/2),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
                     for j in range (0,len(proba)):
                         
#                     for j in range (0,2):
                         if proba[j]>0.01:
                             n=n+1
                             strw=fidclass(j)+ ' {0:.1f}%'.format(100*proba[j])                             
                             cv2.putText(tabtext,strw,(370,460+10*n),cv2.FONT_HERSHEY_PLAIN,0.8,(0,255,0),1)
                             print fidclass(j), ' {0:.2f}%'.format(100*proba[j])
                     print'found'
                     break 
#    cv2.imshow('image',tabtext)                
    if not pfound:
            print'not found'
    return tabtext

def drawpatch(t,lp,preprob,k,top):
    imgn = np.zeros((512,512,3), np.uint8)
    ill = 0
    endnumslice=k.find('.bmp')

#    print imgcore
    posend=endnumslice
    while k.find('-',posend)==-1:
            posend-=1
    debnumslice=posend+1
    slicenumber=int((k[debnumslice:endnumslice])) 
    th=t/100.0
    listlabel={}
    listlabelaverage={}
#    print slicenumber,th
    for ll in lp:

#            print ll
            slicename=ll[0]          
            xpat=ll[1]
            ypat=ll[2]        
        #we find max proba from prediction
            proba=preprob[ill]
           
            prec, mprobai = maxproba(proba)
            mproba=round(mprobai,2)
            classlabel=fidclass(prec)
            classcolor=classifc[classlabel]
       
            
            if mproba >th and slicenumber == slicename and\
            (top in datahealthy or (classlabel not in excluvisu)):
#                    print classlabel
                    if classlabel in listlabel:
#                        print 'found'
                        numl=listlabel[classlabel]
                        listlabel[classlabel]=numl+1
                        cur=listlabelaverage[classlabel]
#                               print (numl,cur)
                        averageproba= round((cur*numl+mproba)/(numl+1),2)
                        listlabelaverage[classlabel]=averageproba
                    else:
                        listlabel[classlabel]=1
                        listlabelaverage[classlabel]=mproba

                    imgn= addpatchn(classcolor,classlabel,xpat,ypat,imgn)


            ill+=1
#            print listlabel        
    for ll1 in listlabel:
#                print ll1,listlabelaverage[ll1]
                tagviewn(imgn,ll1,str(listlabelaverage[ll1]),175,00)
    

    return imgn
    
def opennew(dirk, fl,L):
    pdirk = os.path.join(dirk,L[fl])
    img = cv2.imread(pdirk,1)
    return img,pdirk

def reti(L,c):
    for i in range (0, len(L)):
     if L[i]==c:
         return i
         break
     

def openfichier(k,dirk,top,L):
    nseed=reti(L,k) 
#    print 'openfichier', k, dirk,top,nseed
  
    global ix,iy,quitl,patchi,classdirec
    global imgtext

    patchi=False
    ix=0
    iy=0
    ncf1 = os.path.join(path_patient,top)
    dop =os.path.join(ncf1,picklefile)
    if classdirec==2:
        ll=os.listdir(ncf1)
        for l in ll:
            ncf =os.path.join(ncf1,l)
            dop =os.path.join(ncf,picklefile)
    else:
        ncf=ncf1
            

    subfile=os.path.join(dop,subsamplef)
    filesub=open(subfile,"r")
    fr=filesub.read()
    filesub.close()
    pnsub=fr.find(' ',0)
    nsub=fr[pnsub+1:len(fr)]
    subsample=int(nsub)
    pdirk = os.path.join(dirk,k)
    img = cv2.imread(pdirk,1)
#    print 'openfichier:',k , ncf, pdirk,top
    
    (preprob,listnamepatch)=loadpkl(ncf)      
    
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)

    cv2.createTrackbar( 'Brightness','image',0,100,nothing)
    cv2.createTrackbar( 'Contrast','image',50,100,nothing)
    cv2.createTrackbar( 'Threshold','image',50,100,nothing)
    cv2.createTrackbar( 'Flip','image',nseed,len(L)-1,nothings)
        
    while(1):
        cv2.setMouseCallback('image',draw_circle,img)
        c = cv2.getTrackbarPos('Contrast','image')
        l = cv2.getTrackbarPos('Brightness','image')
        tl = cv2.getTrackbarPos('Threshold','image')
        fl = cv2.getTrackbarPos('Flip','image')

        img,pdirk= opennew(dirk, fl,L)
#        print pdirk
        (topnew,tailnew)=os.path.split(pdirk)
        endnumslice=tailnew.find('.bmp',0)
        posend=endnumslice
        while tailnew.find('-',posend)==-1:
            posend-=1
            debnumslice=posend+1
        slicenumber=int((tailnew[debnumslice:endnumslice])) 
        
        imglumi=lumi(img,l)
        imcontrast=contrast(imglumi,c)        
        
        imgn=drawpatch(tl,listnamepatch,preprob,L[fl],top)

        imgngray = cv2.cvtColor(imgn,cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(imgngray)              
        outy=cv2.bitwise_and(imcontrast,imcontrast,mask=mask_inv)
        imgt=cv2.add(imgn,outy)
 
        cv2.putText(imgt,'quit',(260,500),cv2.FONT_HERSHEY_PLAIN,1,red,1,cv2.LINE_AA)
        cv2.rectangle(imgt,(250,490),(270,510),red,2)
        
        imgvisu=cv2.cvtColor(cv2.add(imgt,imgtext),cv2.COLOR_BGR2RGB)
        cv2.imshow('image',imgvisu)

        if patchi :
            print 'retrieve patch asked'
            imgtext=retrievepatch(ix,iy,top,slicenumber,preprob,listnamepatch)
            patchi=False

        if quitl or cv2.waitKey(20) & 0xFF == 27 :
#            print 'on quitte', quitl
            break
    quitl=False
#    print 'on quitte 2'
    cv2.destroyAllWindows()


def listfichier(dossier):
    Lf=[]
    L= os.listdir(dossier)
#    print L
    for k in L:
        if ".bmp" in k.lower(): 
            Lf.append(k)
    return Lf

def listbtn2(L,dirk,top):
  
#    for widget in cadrestat.winfo_children():
#        widget.destroy()
    for widget in cadreim.winfo_children():
        widget.destroy()
 
#    cadre1.pack(side=RIGHT,fill='x',expand=1)
    canvas = Canvas(cadreim, borderwidth=2, width=200,height=600,background="blue")
    frame = Frame(canvas, background="blue")
    vsb = Scrollbar(cadreim, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)

    vsb.pack(side="right", fill="y")
    canvas.pack(side="right", fill="both", expand=True)
    canvas.create_window((1,1), window=frame, anchor="nw")
#    canvas.create_window((1,1), window=frame)

    frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
       
    for k in L: 
           Button(frame,text=k,command=lambda k = k:\
           openfichier(k,dirk,top,L)).pack(side=TOP,expand=1)



    
def opendir(k):

    global classdirec
    
    for widget in cadrepn.winfo_children():
        widget.destroy()
    for widget in cadrestat.winfo_children():
        widget.destroy()
    Label(cadrepn, bg='lightgreen',text='patient:'+k).pack(side=TOP,fill=X,expand=1)
    tow=''
    fdir=os.path.join(path_patient,k)
    if classdirec==1:   
#        fdir=os.path.join(path_patient,k)
        bmp_dir = os.path.join(fdir, scanbmp)
    else:
        ldir=os.listdir(fdir)
        for ll in ldir:
             fdir = os.path.join(fdir, ll)
             bmp_dir = os.path.join(fdir, scanbmp)
    
    separator = Frame(cadrepn,height=2, bd=10, relief=SUNKEN)
    separator.pack(fill=X)
#    print 'bmp dir', bmp_dir      
    listscanfile =os.listdir(bmp_dir)
   
    ldcm=[]
    for ll in listscanfile:
      if  ll.lower().find('.bmp',0)>0:
         ldcm.append(ll)
    numberFile=len(ldcm)        
    tow='Number of sub sampled scan images: '+ str(numberFile)+\
    '\n\n'+'Predicted patterns: '+'\n' 
    for cle, valeur in listelabelfinal.items():
#             print 'cle valeur', cle,valeur
             for c in classif:
#                 print (k,c)
                 if (k,c) == cle and listelabelfinal[(k,c)]>0:

                     tow=tow+c+' : '+str(listelabelfinal[(k,c)])+'\n'

    Label(cadrestat, text=tow,bg='lightgreen').pack(side=TOP, fill='both',expand=1)
#    print tow
    dirk=os.path.join(fdir,predictout+'/0')
    L=listfichier(dirk)
    listbtn2(L,dirk,k)
    
    
    
def listdossier(dossier): 
    L= os.walk(dossier).next()[1]  
    return L
    
def listbtn(L):   
    cwt = Label(cadrerun,text="Select a patient")
    cwt.pack()
    for widget in cadrerun.winfo_children():       
                widget.destroy()    
    for k in L:
            Button(cadrerun,text=k,command=lambda k = k: opendir(k)).pack(side=LEFT,fill="both",\
            expand=1)

            


def runf():

    listbtn(listdossier( path_patient ))
#    separator = Frame(cadrerun,height=2, bd=10, relief=SUNKEN)
#    separator.pack(fill=X)
    
def onFrameConfigure(canvas):
    '''Reset the scroll region to encompass the inner frame'''
    canvas.configure(scrollregion=canvas.bbox("all"))
    
    
def quit():
    global fenetre
    fenetre.quit()
    fenetre.destroy()   

def runpredict(pp,subs, retou):
    
    global classdirec,path_patient, patch_list, dataset_list, nameset_list, proba
#    cadrestatus.grid(row=1)
    cw = Label(cadrestatus, text="Running",fg='red',bg='blue')
    cw.pack(side=TOP,fill=X)

    path_patient=pp
#    print path_patient
    if os.path.exists(path_patient):
       patient_list= os.walk(path_patient).next()[1]
       for f in patient_list:
            print('work on:',f, 'with subsamples :', subs)
        
            namedirtopcf1 = os.path.join(path_patient,f)
           
            listscanfile1= os.listdir(namedirtopcf1)
            for ll in listscanfile1:
                namedirtopcf=os.path.join(namedirtopcf1,ll)
                if os.path.isdir(namedirtopcf):
#                    print 'it is a dir'
                    listscanfile= os.listdir(namedirtopcf)
                    classdirec=2
        #    for ll in patient_list2:
                elif ll.find('.dcm',0)>0:
        #            print 'it is not a dir'
                    listscanfile=listscanfile1
                    namedirtopcf=namedirtopcf1
#                    print 'write classider'
                    classdirec=1
                    break
                
            ldcm=[]
            for ll in listscanfile:
             if  ll.lower().find('.dcm',0)>0:
                ldcm.append(ll)
            numberFile=len(ldcm)        
            if retou==1:
                patch_list=[]
                dataset_list=[]
                nameset_list=[]
                proba=[]
                bmp_dir = os.path.join(namedirtopcf, scanbmp)
    #    print bmp_dir
                remove_folder(bmp_dir)    
                os.mkdir(bmp_dir) 
#                lung_dir = os.path.join(namedirtopcf, lungmask)
#                if os.path.exists(lung_dir)== False:
#                   os.mkdir(lung_dir)
#                lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)
#                remove_folder(lung_bmp_dir)
#                os.mkdir(lung_bmp_dir)
                for scanumber in range(0,numberFile):
        #            print scanumber
                    if scanumber%subs==0:
                        scanfile=ldcm[scanumber]          
                        genebmp(namedirtopcf,scanfile,subs)
            
                pavgene(namedirtopcf)
                ILDCNNpredict(namedirtopcf)
                visua(namedirtopcf,classdirec,True)
    
            else:
                    
                visua(namedirtopcf,classdirec,False)
            print('completed on: ',f)     
       
       (top, tail)= os.path.split(path_patient)
       for widget in cadrestatus.winfo_children():       
                widget.destroy()
       wcadrewait = Label(cadrestatus, text="completed for "+tail,fg='darkgreen',bg='lightgreen',width=85)
       wcadrewait.pack()

       runf()
    else:
    #            print 'path patient does not exist'
        wer = Label(cadrestatus, text="path for patients does not exist",\
               fg='red',bg='yellow',width=85)
        wer.pack(side=TOP,fill='both')
        bouton1_run = Button(cadrestatus, text="continue", fg='red',\
              bg='yellow',command= lambda: runl1())
        bouton1_run.pack()


def runl1 ():
    for widget in cadrestatus.winfo_children():
                widget.destroy()
    for widget in cadretop.winfo_children():
                widget.destroy()
    for widget in cadrerun.winfo_children():
                widget.destroy()
    for widget in cadrestat.winfo_children():
                widget.destroy()
    for widget in cadreim.winfo_children():
                widget.destroy()
    for widget in cadrepn.winfo_children():
                widget.destroy()
    runl()

def runl ():
    global path_patient
    bouton_quit = Button(cadretop, text="Quit", command= quit,bg='red',fg='yellow')
    bouton_quit.pack(side="top")
    separator = Frame(cadretop,height=2, bd=10, relief=SUNKEN)
    separator.pack(fill=X)
    w = Label(cadretop, text="path for patients:")
    w.pack(side=TOP,fill='both')
    
    clepp = StringVar()

    e = Entry(cadretop, textvariable=clepp,width=98)
    e.delete(0, END)
    e.insert(0, path_patient)
    e.pack(side=TOP,fill='both',expand=1)
##   
    if os.path.exists(path_patient):
        pl=os.listdir(path_patient)
        ll = Label(cadretop, text='list of patient(s):')
        ll.pack()
        tow=''
        for l in pl:
            ld=os.path.join(path_patient,l)
            if os.path.isdir(ld):
                tow =tow+l+' - '
            ll = Label(cadretop, text=tow,fg='blue')
        ll.pack()
    else:     
        ll = Label(cadretop, text='path_patient does not exist:',fg='red',bg='yellow')
        ll.pack()

    separator = Frame(cadretop,height=2, bd=10, relief=SUNKEN)
    separator.pack(fill=X)
    wcadre5 = Label(cadretop, text="subsample:")
    wcadre5.pack(side=LEFT)
    clev = IntVar()
    e = Entry(cadretop, textvariable=clev,width=5)
    e.delete(0, END)
    e.insert(0, "1")
    e.pack(fill='x',side=LEFT)
    
    retour0=IntVar(cadretop)
    bouton0 = Radiobutton(cadretop, text='run predict',variable=retour0,value=1,bd=2)
    bouton1 = Radiobutton(cadretop, text='visu only',variable=retour0,value=0,bd=2)
    bouton0.pack(side=RIGHT)
    bouton1.pack(side=RIGHT)
    bouton_run = Button(cadretop, text="Run", bg='green',fg='blue',\
             command= lambda: runpredict(clepp.get(),clev.get(),retour0.get()))
    bouton_run.pack(side=RIGHT)
#    separator = Frame(cadretop,height=2, bd=10, relief=SUNKEN)
#    separator.pack(fill=X, padx=5, pady=2)


##########################################################
    
t = datetime.datetime.now()
today = str('date: '+dd(t.month)+'-'+dd(t.day)+'-'+str(t.year)+\
'_'+dd(t.hour)+':'+dd(t.minute)+':'+dd(t.second))

print today


quitl=False
patchi=False
listelabelfinal={}
oldc=0
imgtext = np.zeros((512,512,3), np.uint8)

fenetre = Tk()
fenetre.title("predict")
fenetre.geometry("610x800+100+50")



cadretop = LabelFrame(fenetre, width=600, height=100, text='top',borderwidth=5,bg="purple",fg='yellow')
cadretop.grid(row=0)
cadrestatus = LabelFrame(fenetre,width=600, height=20,text="status run",bg='purple',fg='yellow')
cadrestatus.grid(row=1)
cadrerun = LabelFrame(fenetre,text="select a patient",width=600, height=20,fg='yellow',bg='purple')
cadrerun.grid(row=2)
cadrepn = LabelFrame(fenetre,text="patient name list:",width=600, height=20,bg='purple',fg='yellow')
cadrepn.grid(row=3)
cadrestat=LabelFrame(fenetre,text="statistic", width=300,height=20,fg='yellow',bg='purple')
cadrestat.grid(row=4,  sticky=NW )
cadreim=LabelFrame(fenetre,text="images", width=300,height=20,fg='yellow',bg='purple')
cadreim.grid(row=4,  sticky=E)

runl()

patch_list=[]
dataset_list=[]
nameset_list=[]
proba=[]
fenetre.mainloop()

#visuinter()
errorfile.close() 
