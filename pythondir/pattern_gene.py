# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 11:16:58 2017
generator of images and label
@author: sylvain
"""
import os
import cv2
import cPickle as pickle
import time
import shutil
import numpy as np
import random

#nameHug='DUMMY' #name of top directory for patches pivkle from dicom
nameHug='HUG' #name of top directory for patches pivkle from dicom

#subHUG='lu_f'#subdirectory from nameHug input pickle
subHUG='classpatch'#subdirectory from nameHug input pickle

#subHUG='S3'#subdirectory from nameHug input pickle


toppatch= 'TOPPATCH' #name of top directory for image and label generation
subtop_patch='lu_full'#subdirectory for top patch

lpatch=True #to generate images from  lung patch (True) or geometrical
col = False # to generate color patterns (True), NOT TESTED

slnt=5 # number of images to generate
numfig=200 # for geometrical, number of figuress per type 
randomdim=False #True to generate big figures, False for small dimensions
image_rows = 512
image_cols = 512

#image_rows = 96
#image_cols = 96

dimpavx=16
dimpavy=16
#this generates 32² =1024 patches per image
num_bit =1 # 1 for grey images, 3 for rgb, only for geometrical

cwd=os.getcwd()
#
(cwdtop,tail)=os.path.split(cwd)

typei='jpg' #can be jpg
typeibmp='bmp'
sroi='sroi'
bmpname='scan_bmp'
picklepatches='picklepatches' 



black=(0,0,0)
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
parme=(234,136,222)
chatain=(139,108,66)


classif ={
        'back_ground':0,
        'healthy':1,    
        'ground_glass':2,
        'HC':3,
        'reticulation':4,
        'bronchiectasis':5,
        'cysts':6,
         'consolidation':7,
        'micronodules':8,
        'air_trapping':9,
        'GGpret':10
        }

classifc ={
    'back_ground':chatain,
    'consolidation':cyan,
    'HC':blue,
    'ground_glass':red,
    'healthy':darkgreen,
    'micronodules':green,
    'reticulation':yellow,
    'air_trapping':pink,
    'cysts':lightgreen,
    'bronchiectasis':orange,
    'emphysema':chatain,
    'GGpret': parme,

     'nolung': lowgreen,
     'bronchial_wall_thickening':white,
     'early_fibrosis':white,

     'increased_attenuation':white,
     'macronodules':white,
     'pcp':white,
     'peripheral_micronodules':white,
     'tuberculosis':white
 }



path_HUG=os.path.join(cwdtop,nameHug)
namedirtopc =os.path.join(path_HUG,subHUG)
namedirtopcpickle=os.path.join(namedirtopc,picklepatches)

extendir=subtop_patch
patchesdirnametop = toppatch+'_'+extendir
patchtoppath=os.path.join(path_HUG,patchesdirnametop)
patchpicklename='picklepatches.pkl'
picklepath = 'picklepatches'
picklepathdir =os.path.join(patchtoppath,picklepath)
if not os.path.isdir(patchtoppath):
    os.mkdir(patchtoppath)

if not os.path.isdir(picklepathdir):
    os.mkdir(picklepathdir)

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

def normHU(image): #normalize HU images
    image1= (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image1[image1>1] = 1.
    image1[image1<0] = 0.
    image2=image1 - PIXEL_MEAN
    return image2


def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
#         print 'path exist'
         # remove if exists
         shutil.rmtree(path)
         time.sleep(1)
 
def normi(img):
#     tabi2=bytescale(img, low=0, high=255)
     tabi1=img-img.min()
     maxt=float(tabi1.max())
     if maxt==0:
         maxt=1
     tabi2=tabi1*(255/maxt)
     tabi2=tabi2.astype('uint8')
     return tabi2
 
def fidclass(numero,classn):
    """return class from number"""
    found=False
#    print numero
    for cle, valeur in classn.items():

        if valeur == numero:
            found=True
            return cle
    if not found:
        return 'unknown'
       
def tagviews(tab,text,x,y):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,text,(x, y), font,0.3,white,1)
    return viseg
def show_image(im, name='image'):
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def genebmp(dirName,slnt,col):
    """generate patches from dicom files and sroi"""
    print ('generate geometric images and label files in :',dirName)
#    (topb,tailb)=os.path.split(dirName)
#    global constPixelSpacing, dimtabx,dimtaby
    #directory for patches
    bmp_dir = os.path.join(dirName, bmpname)
    remove_folder(bmp_dir)
    os.mkdir(bmp_dir)
    sroidir=os.path.join(dirName,sroi)
    if os.path.exists(sroidir):
        shutil.rmtree(sroidir)
        time.sleep(1)
    os.mkdir(sroidir)
    if num_bit==3:
         img=np.zeros((slnt,image_rows,image_cols,3),np.uint8)
    else:
        img=np.zeros((slnt,image_rows,image_cols),np.uint8)
    mask=np.zeros((slnt,image_rows,image_cols),np.uint8)
    
#    os.listdir(lung_dir)
    for s in range(slnt):
        if col:
#            print(filename)
            dark_color0 = random.randint(0, 100)
            dark_color1 = random.randint(0, 100)
            dark_color2 = random.randint(0, 100)
            img[s,:, :, 0] = dark_color0
            img[s,:, :, 1] = dark_color1
            img[s,:, :, 2] = dark_color2
        
            # Object
            light_color0 = random.randint(dark_color0+1, 255)
            light_color1 = random.randint(dark_color1+1, 255)
            light_color2 = random.randint(dark_color2+1, 255)
            center_0 = random.randint(0, image_rows)
            center_1 = random.randint(0, image_cols)
            r1 = random.randint(10, 56)
            r2 = random.randint(10, 56)
            cv2.ellipse(img[s], (center_0, center_1), (r1, r2), 0, 0, 360, (light_color0, light_color1, light_color2), -1)
            cv2.ellipse(mask[s], (center_0, center_1), (r1, r2), 0, 0, 360, 255, -1)
           
    # White noise
            density = random.uniform(0, 0.1)
            for i in range(image_rows):
                for j in range(image_cols):
                    if random.random() < density:
                        img[s,i, j, 0] = random.randint(0, 255)
                        img[s,i, j, 1] = random.randint(0, 255)
                        img[s,i, j, 2] = random.randint(0, 255)
        else:
            if randomdim:
                nc=0
                while nc<2*numfig:
                    if num_bit==3:
                        color=red
                    else:
                        color=255
                    imgt=np.zeros((image_rows,image_cols,num_bit),np.uint8)
                    maskt=np.zeros((image_rows,image_cols),np.uint8)
                    center_0 = random.randint(0, image_rows)
                    center_1 = random.randint(0, image_cols)
                    
                    if nc%2==0:
                        r1 = random.randint(2, 50)
                        r2 = random.randint(2, 50)
                        cv2.ellipse(imgt, (center_0, center_1), (r1, r2), 0, 0, 360, color, -1)
                        cv2.ellipse(maskt, (center_0, center_1), (r1, r2), 0, 0, 360, 1, -1)
                    else:
                        r1 = random.randint(2, 50)
                        r2 = random.randint(2, 50)
                        cv2.rectangle(imgt, (center_0, center_1), (center_0+r1, center_1+r2), color, -1)
                        cv2.rectangle(maskt, (center_0, center_1),  (center_0+r1, center_1+r2), 2, -1)
    
                    ms=mask[s].copy()
                    mt=np.copy(maskt)
                    np.putmask(ms,ms>0,255) 
                    np.putmask(mt,mt>0,255)
                    xori=cv2.bitwise_and(mt,ms)
    
    #                print xori.max()
                    if xori.max()==0:
                         img[s]=cv2.add(imgt,img[s])
                         mask[s]=cv2.add(maskt,mask[s])
                         nc+=1
            else:
                nc=0
                while nc<2*numfig:
                    if num_bit==3:
                            color=red
                    else:
                            color=255
                    imgt=np.zeros((image_rows,image_cols,num_bit),np.uint8)
                    maskt=np.zeros((image_rows,image_cols),np.uint8)
                    center_0 = random.randint(0, image_rows)
                    center_1 = random.randint(0, image_cols)
                    r1 = random.randint(2,10)
                    r2 =random.randint(2,10)
                    if nc%2==0:
                        
                        cv2.ellipse(imgt, (center_0, center_1), (r1, r2), 0, 0, 360, color, -1)
                        cv2.ellipse(maskt, (center_0, center_1), (r1, r2), 0, 0, 360, 1, -1)
                    else:
                       
                        cv2.rectangle(imgt, (center_0, center_1), (center_0+r1, center_1+r2), color, -1)
                        cv2.rectangle(maskt, (center_0, center_1),  (center_0+r1, center_1+r2), 2, -1)
                    
                    ms=mask[s].copy()
                    mt=np.copy(maskt)
                    np.putmask(ms,ms>0,255) 
                    np.putmask(mt,mt>0,255)
                    xori=cv2.bitwise_and(mt,ms)
        
        #                print xori.max()
                    if xori.max()==0:
                             img[s]=cv2.add(imgt,img[s])
                             mask[s]=cv2.add(maskt,mask[s])
                             nc+=1
#                     print 'overlapp'
                     
                    
#                print nc
#                cv2.imshow('img'+str(nc),img[s])
#                cv2.imshow('xori',normi(xori))
#                cv2.imshow('maskt',normi(maskt))
#                cv2.imshow('mask',normi(mask[s]))
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()
               
#        print 'number of figures:',nc  
#        print img.shape
        dsrforimage=normi(img[s])
        imgcoredeb='img_'+str(s)+'.'+typeibmp             
        bmpfiled=os.path.join(bmp_dir,imgcoredeb)
        if num_bit ==3:
            dsrforimage=cv2.cvtColor(dsrforimage,cv2.COLOR_RGB2BGR)
        cv2.imwrite (bmpfiled, dsrforimage)
    #            print imgcoresroi,bmpfileroi   
        imgcoresroi='mask_'+str(s)+'.'+typeibmp
        bmpfileroi=os.path.join(sroidir,imgcoresroi)
#            print imgcoresroi,bmpfileroi
        dsrforimage=normi(mask[s])
        dsrforimage=cv2.cvtColor(dsrforimage,cv2.COLOR_GRAY2BGR)                     
        cv2.imwrite (bmpfileroi, dsrforimage)

    return img,mask

def geneaug(image,tt):
    if tt==0:
        imout=image
    elif tt==1:
    # 1 90 deg
        imout = np.rot90(image)
    elif tt==2:
    #2 180 deg
        imout = np.rot90(np.rot90(image))
    elif tt==3:
    #3 270 deg
        imout = np.rot90(np.rot90(np.rot90(image)))
    elif tt==4:
    #4 flip fimage left-right
            imout=np.fliplr(image)
    elif tt==5:
    #5 flip fimage left-right +rot 90
        imout = np.rot90(np.fliplr(image))
    elif tt==6:
    #6 flip fimage left-right +rot 180
        imout = np.rot90(np.rot90(np.fliplr(image)))
    elif tt==7:
    #7 flip fimage left-right +rot 270
        imout = np.rot90(np.rot90(np.rot90(np.fliplr(image))))
    elif tt==8:
    # 8 flip fimage up-down
        imout = imout=np.flipud(image)
    elif tt==9:
    #9 flip fimage up-down +rot90
        imout = np.rot90(np.flipud(image))
    elif tt==10:
    #10 flip fimage up-down +rot180
        imout = np.rot90(np.rot90(np.flipud(image)))
    elif tt==11:
    #11 flip fimage up-down +rot270
        imout = np.rot90(np.rot90(np.rot90(np.flipud(image))))

    return imout




def genebmppatch(dirName,slnt,contenupat):

    """generate patches from dicom files and sroi"""
    print ('generate lung patch based images and label files')
    (top,tail)=os.path.split(dirName)
#    global constPixelSpacing, dimtabx,dimtaby
    #directory for patches
    bmp_dir = os.path.join(top, bmpname)
    remove_folder(bmp_dir)
    os.mkdir(bmp_dir)
    sroidir=os.path.join(top,sroi)
    if os.path.exists(sroidir):
        shutil.rmtree(sroidir)
        time.sleep(1)
    os.mkdir(sroidir)

    img=np.zeros((slnt,image_rows,image_cols),np.int16)
    np.putmask(img,img==0,-1000)
    mask=np.zeros((slnt,image_rows,image_cols),np.uint8)
    patdic={}
    print 'contenupat',contenupat
    for pat in contenupat:
        patdic[pat]=[]

    listpat=[name for name in os.listdir(dirName) if name in contenupat]
    for pat in listpat:
        patdir=os.path.join(dirName,pat)
        subd=os.listdir(patdir)
        for loca in subd:
            patsubdir=os.path.join(patdir,loca)
            listpickles=os.listdir(patsubdir)
            for l in listpickles:
                listscan=pickle.load(open(os.path.join(patsubdir,l),"rb"))
                patdic[pat]=patdic[pat]+listscan

    for pat in contenupat:
        print pat,len(patdic[pat])
    listpat.append('back_ground')
    print 'len(listpat)' ,len(listpat) 
    for pat in listpat :
        print pat, classif[pat]
#    os.listdir(lung_dir)
    for s in range(slnt):
        print 'generate image :',s
        i=0
        while i <image_rows:
            j=0
            while j <image_cols: 
#                indexpat=(i+j)%len(listpat)
                indexpat = random.randint(0, len(listpat)-1)
                indexaug = random.randint(0, 11)
#                print indexaug
#                print indexpat
                pat=fidclass(indexpat,classif)
                if indexpat>0:
                    numi = random.randint(0, len(patdic[pat])-1)
                    patch_t_w=patdic[pat][numi]
                    pat_t_w_a=geneaug(patch_t_w,indexaug)
                    img[s][j:j+dimpavy,i:i+dimpavx]=pat_t_w_a
#                    print patch_t_w.shape, patch_t_w.min(),patch_t_w.max()
#                    print pat_t_w_a.shape, pat_t_w_a.min(),pat_t_w_a.max()
#                    print img[s][j:j+dimpavy,i:i+dimpavx].shape,img[s][j:j+dimpavy,i:i+dimpavx].min(),img[s][j:j+dimpavy,i:i+dimpavx].max()
##                    print i,j
#                    ooo
                cv2.rectangle(mask[s], (i, j), (i+dimpavx,j+dimpavy), classif[pat], -1)
                j+=dimpavy
            i+=dimpavx
#        print 'number of classes :',len( np.unique(mask[s]))
        dsrforimage=normi(img[s])
        imgcoredeb='img_'+str(s)+'.'+typeibmp             
        bmpfiled=os.path.join(bmp_dir,imgcoredeb)
        if num_bit ==3:
            dsrforimage=cv2.cvtColor(dsrforimage,cv2.COLOR_RGB2BGR)
        cv2.imwrite (bmpfiled, dsrforimage)
    #            print imgcoresroi,bmpfileroi   
        imgcoresroi='mask_'+str(s)+'.'+typeibmp
        bmpfileroi=os.path.join(sroidir,imgcoresroi)
    #            print imgcoresroi,bmpfileroi
        dsrforimage=normi(mask[s])
        dsrforimage=cv2.cvtColor(dsrforimage,cv2.COLOR_GRAY2BGR)                     
        cv2.imwrite (bmpfileroi, dsrforimage)

    return img,mask




def preparroi(namedirtopcf,tabscan,tabsroi):
    (top,tail)=os.path.split(namedirtopcf)
    pathpicklepat=os.path.join(picklepathdir,tail)
    if not os.path.exists (pathpicklepat):
                os.mkdir(pathpicklepat)
    
    for num in range(slnt):
        patchpicklenamepatient=str(num)+'_'+patchpicklename   
        pathpicklepatfile=os.path.join(pathpicklepat,patchpicklenamepatient)
        scan_list=[]
        mask_list=[]
        scan_list.append(tabscan[num] ) 
#        print tabscan[0].shape,tabscan[0].min(),tabscan[0].max()
#        maski= tabsroi[num].copy()    
#        np.putmask(maski,maski>0,classif['healthy'])

        mask_list.append(tabsroi[num])
#        print tabsroi[num].min(),tabsroi[num].max(),np.unique(tabsroi[num])
#        oooo
#        if num==3:
##            o=normi(maski)
#            n=normi(tabscan[num] )
#            x=normi(tabsroi[num])
##            f=normi(tabroif)
#            cv2.imshow('maski',x)
#            cv2.imshow('datascan[num] ',n)
##            cv2.imshow('tabroix',x)
##            cv2.imshow('tabroif',f)
##            cv2.imwrite('a.bmp',o)
##            cv2.imwrite('b.bmp',x)
##            cv2.imwrite('c.bmp',tabscan[num])
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            
        patpickle=(scan_list,mask_list)
#        print len(scan_list)
        pickle.dump(patpickle, open(pathpicklepatfile, "wb"),protocol=-1)


listpat=[]

if not lpatch:
        sroidir=os.path.join(namedirtopc,sroi)
        if not os.path.exists(namedirtopc):
            os.mkdir(namedirtopc)
        if os.path.exists(sroidir):
            remove_folder(sroidir)
        os.mkdir(sroidir)
        tabscan,tabsroi=genebmp(namedirtopc,slnt,col)    
        preparroi(namedirtopc,tabscan,tabsroi)
        
else:
    print 'based on patches'
    sroidir=os.path.join(namedirtopc,sroi)
    remove_folder(sroidir)
    os.mkdir(sroidir)
#    classifused={
#    'back_ground':0,
#        'healthy':1,    
#        'ground_glass':2}
    classifused=classif
    tabscan,tabsroi=genebmppatch(namedirtopcpickle,slnt,classifused)
    namedirtopcf=namedirtopc
    preparroi(namedirtopc,tabscan,tabsroi)
    