# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:03:33 2017

@author: sylvain
generate data from dicom images for segmentation roi 

-1st step

include generation per pattern
"""
#from __future__ import print_function

from param_pix import cwdtop,bmpname,lungmask,lungmask1,lungmaskbmp,image_rows,image_cols,typei,front_enabled
from param_pix import transbmp,sroi,sourcedcm
from param_pix import white
from param_pix import remove_folder,normi,rsliceNum
from param_pix import classifc,classif

import cPickle as pickle
import cv2
import dicom
import numpy as np
import os



nameHug='HUG'
#nameHug='CHU'
subHUG='ILD0'
#subHUG='ILD_TXT'
#subHUG='UIP2'

#subHUG='UIP'
toppatch= 'TOPROI'

###############################################################

path_HUG=os.path.join(cwdtop,nameHug)
#path_HUG=os.path.join(nameHug,namsubHug)
namedirtopc =os.path.join(path_HUG,subHUG)


#extension for output dir

extendir=subHUG

patchesdirnametop = toppatch+'_'+extendir
patchtoppath=os.path.join(path_HUG,patchesdirnametop)
patchpicklename='picklepatches.pkl'
picklepath = 'picklepatches'
roipicklepath = 'roipicklepatches'
picklepathdir =os.path.join(patchtoppath,picklepath)
roipicklepathdir =os.path.join(patchtoppath,roipicklepath)

if not os.path.isdir(patchtoppath):
    os.mkdir(patchtoppath)

if not os.path.isdir(picklepathdir):
    os.mkdir(picklepathdir)
if not os.path.isdir(roipicklepathdir):
    os.mkdir(roipicklepathdir)

patchtoppath=os.path.join(path_HUG,patchesdirnametop)

def genepara(fileList,namedir):
    print 'gene parametres'
    fileList =[name for name in  os.listdir(namedir) if ".dcm" in name.lower()]
    slnt=0
    for filename in fileList:
        FilesDCM =(os.path.join(namedir,filename))
        RefDs = dicom.read_file(FilesDCM)
        scanNumber=int(RefDs.InstanceNumber)
        if scanNumber>slnt:
            slnt=scanNumber
    print 'number of slices', slnt
    slnt=slnt+1
    return slnt

def tagviews(tab,text,x,y):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,text,(x, y), font,0.3,white,1)
    return viseg
 
def genebmp(dirName,fileList,slnt,hug):
    """generate patches from dicom files and sroi"""
    print ('generate  bmp files from dicom files in :',f)
    (top,tail)=os.path.split(dirName)
#    global constPixelSpacing, dimtabx,dimtaby
    #directory for patches
    bmp_dir = os.path.join(dirName, bmpname)
    remove_folder(bmp_dir)
    os.mkdir(bmp_dir)
    if hug:
        lung_dir = os.path.join(dirName, lungmask)
        if not os.path.exists(lung_dir):
            lung_dir = os.path.join(dirName, lungmask1)
        
    else:
        (top,tail)=os.path.split(dirName)
        lung_dir = os.path.join(top, lungmask)
        if not os.path.exists(lung_dir):
            lung_dir = os.path.join(top, lungmask1)
        
    lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)
    remove_folder(lung_bmp_dir)
    os.mkdir(lung_bmp_dir)
    
    lunglist = [name for name in os.listdir(lung_dir) if ".dcm" in name.lower()]

    tabscan=np.zeros((slnt,image_rows,image_cols),np.int16)
    tabslung=np.zeros((slnt,image_rows,image_cols),np.uint8)
    tabsroi=np.zeros((slnt,image_rows,image_cols,3),np.uint8)
#    os.listdir(lung_dir)
    for filename in fileList:
#            print(filename)
            FilesDCM =(os.path.join(dirName,filename))
            RefDs = dicom.read_file(FilesDCM)
            dsr= RefDs.pixel_array
            dsr=dsr.astype('int16')

            scanNumber=int(RefDs.InstanceNumber)
            endnumslice=filename.find('.dcm')
            imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)+'.'
#            bmpfile=os.path.join(dirFilePbmp,imgcore)
            dsr[dsr == -2000] = 0
            intercept = RefDs.RescaleIntercept
#            print intercept
            slope = RefDs.RescaleSlope
            if slope != 1:
                dsr = slope * dsr.astype(np.float64)
                dsr = dsr.astype(np.int16)

            dsr += np.int16(intercept)
            dsr = dsr.astype('int16')

            dsr=cv2.resize(dsr,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)

            dsrforimage=normi(dsr)

            tabscan[scanNumber]=dsr

            imgcored=imgcoredeb+typei
            bmpfiled=os.path.join(bmp_dir,imgcored)
            imgcoresroi='sroi_'+str(scanNumber)+'.'+typei
            bmpfileroi=os.path.join(sroidir,imgcoresroi)
#            print imgcoresroi,bmpfileroi
            textw='n: '+tail+' scan: '+str(scanNumber)

            cv2.imwrite (bmpfiled, dsrforimage,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
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
#                fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
#                print 'fxs',fxs
                scanNumber=int(RefDs.InstanceNumber)
                endnumslice=filename.find('.dcm')
                imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)+'.'
                imgcore=imgcoredeb+typei
                bmpfile=os.path.join(lung_bmp_dir,imgcore)
                dsr=normi(dsr)
                dsrresize=cv2.resize(dsr,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)

                cv2.imwrite (bmpfile, dsrresize)
#                np.putmask(dsrresize,dsrresize==1,0)
                np.putmask(dsrresize,dsrresize>0,1)
                tabslung[scanNumber]=dsrresize

    return tabscan,tabsroi,tabslung

def contour2(im,l):
    col=classifc[l]
    vis = np.zeros((image_rows,image_cols,3), np.uint8)
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

 
    deltay=40+10*labnow

    viseg=cv2.putText(tab,label,(x, y+deltay), font,0.3,col,1)
    return viseg

def peparescan(numslice,tabs,tabl):
    tablc=tabl.copy().astype(np.int16)
    taba=cv2.bitwise_and(tabs,tabs,mask=tabl)
    np.putmask(tablc,tablc==0,-1000)
    np.putmask(tablc,tablc==1,0)
    tabab=cv2.bitwise_or(taba,tablc)             
    datascan[numslice]=tabab


def preparroi(namedirtopcf):
    (top,tail)=os.path.split(namedirtopcf)

    pathpicklepat=os.path.join(picklepathdir,tail)
    if not os.path.exists (pathpicklepat):
                os.mkdir(pathpicklepat)
    
    for num in numsliceok:
        scan_list=[]
        mask_list=[]
        scan_list.append(datascan[num] )
        patchpicklenamepatient=str(num)+'_'+patchpicklename        
        tabl=tabslung[num].copy()
        np.putmask(tabl,tabl>0,classif['healthy'])
#        np.putmask(tabl,tabl==0,classif['back_ground'])
        
        pathpicklepatfile=os.path.join(pathpicklepat,patchpicklenamepatient)
        
        maskr=tabroi[num].copy()   
#        print 'maskr',maskr.min(),maskr.max()
#        maskrc=maskr.copy()

        np.putmask(maskr,maskr>0,255)
        
        maskr=np.bitwise_not(maskr)          
        
        roi=cv2.bitwise_and(tabl,tabl,mask=maskr)
        
        roif=cv2.add(roi,tabroi[num]) 

        mask_list.append(roif)
#        if num==178:
#            o=normi(roif)
#            n=normi(datascan[num] )
#            x=normi(tabroi[num])
##            f=normi(tabroif)
#            cv2.imshow('roif',o)
#            cv2.imshow('datascan[num] ',n)
#            cv2.imshow('tabroix',x)
##            cv2.imshow('tabroif',f)
#            cv2.imwrite('a.bmp',o)
#            cv2.imwrite('b.bmp',roif)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            
        patpickle=(scan_list,mask_list)
#        print len(scan_list)
        pickle.dump(patpickle, open(pathpicklepatfile, "wb"),protocol=-1)
        for pat in tabroipat[num]:
             listslicef[tail][num][pat]+=1
             roipathpicklepat=os.path.join(roipicklepathdir,pat)   
             if not os.path.exists (roipathpicklepat):
                os.mkdir(roipathpicklepat)

             roipathpicklepatfile=os.path.join(roipathpicklepat,tail+'_'+patchpicklenamepatient)
             pickle.dump(patpickle, open(roipathpicklepatfile, "wb"),protocol=-1)


def create_test_data(namedirtopcf,pat,tabscan,tabsroi,tabslung):
    
    (top,tail)=os.path.split(namedirtopcf)
    print 'create test data for :', tail, 'pattern :',pat
    pathpat=os.path.join(namedirtopcf,pat)
    list_pos=os.listdir(pathpat)
#    print list_pos
    if not front_enabled:
        try:
            list_pos.remove(transbmp)
        except:
                print 'no trans'
#    print list_pos
    list_image=[name for name in os.listdir(pathpat) if name.find('.dcm')>0] 
  
    if len(list_image)>0:
            print 'it is CHU'
            bmp_dir = os.path.join(pathpat, bmpname)
            remove_folder(bmp_dir)
            os.mkdir(bmp_dir)
            for filename in list_image:
                FilesDCM =(os.path.join(pathpat,filename))
                RefDs = dicom.read_file(FilesDCM)
                dsr= RefDs.pixel_array
                dsr=dsr.astype('int16')
                if dsr.max()>0:
                    numslice=int(RefDs.InstanceNumber)
                    if pat not in tabroipat[numslice]:
                        tabroipat[numslice].append(pat) 
                    if numslice not in numsliceok:
                        numsliceok.append(numslice)
                        peparescan(numslice,tabscan[numslice],tabslung[numslice])
                        tabroi[numslice]=np.zeros((tabscan.shape[1],tabscan.shape[2]), np.uint8)
                    endnumslice=filename.find('.dcm')
                    imgcoredeb=filename[0:endnumslice]+'_'+str(numslice)+'.'
        #            bmpfile=os.path.join(dirFilePbmp,imgcore)
                    dsr[dsr == -2000] = 0
                    intercept = RefDs.RescaleIntercept
        #            print intercept
                    slope = RefDs.RescaleSlope
                    if slope != 1:
                        dsr = slope * dsr.astype(np.float64)
                        dsr = dsr.astype(np.int16)      
                    dsr += np.int16(intercept)
                    dsr = dsr.astype('int16')      
                    dsr=cv2.resize(dsr,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)      
                    dsrforimage=normi(dsr)               
                    imgcored=imgcoredeb+typei
                    bmpfiled=os.path.join(bmp_dir,imgcored)
        #            print imgcoresroi,bmpfileroi   
                    cv2.imwrite (bmpfiled, dsrforimage,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
                    
                    np.putmask(dsrforimage,dsrforimage==1,0)
                    newroic=dsrforimage.copy()
                    np.putmask(newroic,newroic>0,1)            

                    oldroi=tabroi[numslice].copy().astype(np.uint8)
                    tabroinum=oldroi.copy()
                    np.putmask(oldroi,oldroi>0,255)
           
                    oldroi=np.bitwise_not(oldroi)          
            
                    tabroix=cv2.bitwise_and(newroic,newroic,mask=oldroi)
            
                    np.putmask(dsrforimage,dsrforimage>0,classif[pat])
            
                    tabroif=cv2.bitwise_and(dsrforimage,dsrforimage,mask=tabroix)
                    tabroif=cv2.add(tabroif,tabroinum)
                    tabroi[numslice]=tabroif
#                    if  numslice==91:
#                        print tabroinum.min(),tabroinum.max()
#                        o=normi(oldroi)
#                        n=normi(tabroinum)
#                        x=normi(tabroix)
#                        f=normi(tabroif)
#                        g=normi(tabroi[numslice])
#                        cv2.imshow('oldroi',o)
#                        cv2.imshow('tabroinum',n)
#                        cv2.imshow('tabroix',x)
#                        cv2.imshow('tabroif',f)
#                        cv2.imshow('tabroi[numslice]',g)
#                        cv2.waitKey(0)
#                        cv2.destroyAllWindows()
#                    o=normi(oldroi)
#                    n=normi(dsrforimage)
#                    x=normi(tabroix)
#                    f=normi(tabroif)
#                    cv2.imshow('oldroi',o)
#                    cv2.imshow('newroi',n)
#                    cv2.imshow('tabroix',x)
#                    cv2.imshow('tabroif',f)
#                    cv2.waitKey(0)
#                    cv2.destroyAllWindows()
##                
#                                       
                    if dsrforimage.max()>0:
                       vis=contour2(dsrforimage,pat)
                       if vis.sum()>0:
                        _tabsroi = np.copy(tabsroi[numslice])
                        imn=cv2.add(vis,_tabsroi)
                        imn=tagview(imn,pat,0,100)
                        tabsroi[numslice]=imn
                        imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)
                        sroifile='sroi_'+str(numslice)+'.'+typei
                        filenamesroi=os.path.join(sroidir,sroifile)
                        cv2.imwrite(filenamesroi,imn)
            
                 
 
    else:
        for d in list_pos:
            
            print 'localisation : ',d
            pathpat2=os.path.join(pathpat,d)
            list_image=os.listdir(pathpat2)
       
            for l in list_image:
                pos=l.find('.')      
                ext=l[pos:len(l)]
                numslice=rsliceNum(l,'_',ext)
#                print numslice,tabroipat[numslice]
                if pat not in tabroipat[numslice]:
                    tabroipat[numslice].append(pat)                    
                if numslice not in numsliceok:
                    numsliceok.append(numslice)
                    
                    peparescan(numslice,tabscan[numslice],tabslung[numslice])
                    tabroi[numslice]=np.zeros((tabscan.shape[1],tabscan.shape[2]), np.uint8)
#                print numslice,tabroipat[numslice]
    #            tabl=tabslung[numslice].copy()
    #            np.putmask(tabl,tabl>0,1)
    
                newroi = cv2.imread(os.path.join(pathpat2, l), 0) 
                newroi=cv2.resize(newroi,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)
                newroic=newroi.copy()
                np.putmask(newroic,newroic>0,1)            
    
                oldroi=tabroi[numslice].copy().astype(np.uint8)
                tabroinum=oldroi.copy()
                np.putmask(oldroi,oldroi>0,255)
               
                oldroi=np.bitwise_not(oldroi)          
                
                tabroix=cv2.bitwise_and(newroic,newroic,mask=oldroi)
                
                np.putmask(newroi,newroi>0,classif[pat])
                
                tabroif=cv2.bitwise_and(newroi,newroi,mask=tabroix)
                tabroif=cv2.add(tabroif,tabroinum)
                tabroi[numslice]=tabroif
#                if pat=='ground_glass' and  numslice==15:
#                    print newroi.min(),newroi.max()
#                    o=normi(oldroi)
#                    n=normi(newroi)
#                    x=normi(tabroix)
#                    f=normi(tabroif)
#                    cv2.imshow('oldroi',o)
#                    cv2.imshow('newroi',n)
#                    cv2.imshow('tabroix',x)
#                    cv2.imshow('tabroif',f)
#                    cv2.waitKey(0)
#                    cv2.destroyAllWindows()
    #                
    #                                       
                if newroi.max()>0:
                   vis=contour2(newroi,pat)
                   if vis.sum()>0:
                    _tabsroi = np.copy(tabsroi[numslice])
                    imn=cv2.add(vis,_tabsroi)
                    imn=tagview(imn,pat,0,100)
                    tabsroi[numslice]=imn
                    imn = cv2.cvtColor(imn, cv2.COLOR_BGR2RGB)
                    sroifile='sroi_'+str(numslice)+'.'+typei
                    filenamesroi=os.path.join(sroidir,sroifile)
                    cv2.imwrite(filenamesroi,imn)
        
    return tabsroi

listdirc= (os.listdir(namedirtopc))
listpat=[]
listslicetot={}
listpatf={}
listslicef={}
totalnumperpat={}
slntdict={}
for pat in classif:
    totalnumperpat[pat]=0
print 'work on: ',namedirtopc
for f in listdirc:
    print '-----------------'
    print('work on:',f)
    print '-----------------'
    listslicef[f]={}
    
    
    numsliceok=[]
    listslicetot[f]=0
    namedirtopcf=os.path.join(namedirtopc,f)
    sroidir=os.path.join(namedirtopcf,sroi)
    if os.path.exists(sroidir):
        remove_folder(sroidir)

    os.mkdir(sroidir)
    contenudir = [name for name in os.listdir(namedirtopcf) if name.find('.dcm')>0]
    if len(contenudir)>0:
        slnt = genepara(contenudir,namedirtopcf)
        tabscan,tabsroi,tabslung=genebmp(namedirtopcf,contenudir,slnt,True)
    else:
          namedirtopcfs=os.path.join(namedirtopcf,sourcedcm)
          contenudir = [name for name in os.listdir(namedirtopcfs) if name.find('.dcm')>0]
          slnt = genepara(contenudir,namedirtopcfs)
          tabscan,tabsroi,tabslung=genebmp(namedirtopcfs,contenudir,slnt,False)
    slntdict[f]=slnt
    for slic in range(slnt):
        listslicef[f][slic]={}
        for pat in classif:
            listslicef[f][slic][pat]=0
    contenupat = [name for name in os.listdir(namedirtopcf) if name in classif]

    datascan={}
    datamask={}
    tabroi={}
    tabroipat={}
    listpatf[f]=contenupat
    for i in range(slnt):
        tabroipat[i]=[]
    for pat in contenupat:
        print 'work on :',pat
        if pat not in listpat:
            listpat.append(pat)
        tabsroi=create_test_data(namedirtopcf,pat,tabscan,tabsroi,tabslung)
    preparroi(namedirtopcf)
    
    listslicetot[f]=len(numsliceok)
    print 'number of different images :',len(numsliceok)
    for i in range(slnt):
#        print i, tabroipat[i]
        for pat in classif:
             if listslicef[f][i][pat] !=0:
                 print  f,i, pat, listslicef[f][i][pat]
                 totalnumperpat[pat]+=1
                 

pathpatfile=os.path.join(patchtoppath,'listpat.txt')
filetw = open(pathpatfile,"w")                       
print '-----------------------------'
print 'list of patterns :',listpat
print '-----------------------------'
filetw.write('list of patterns :'+str(listpat)+'\n')    
filetw.write( '-----------------------------\n')

#for pat in classif:
#    if totalnumperpat[pat] !=0:
#        print 'number of images for ',pat, ' : ', totalnumperpat[pat]
#        filetw.write('number of images for '+pat+ ' : '+ str(totalnumperpat[pat])+'\n')
#print '-----------------------------'  
#filetw.write( '-----------------------------\n')
totsln=0
npattot={}
for pat in classif:
        npattot[pat]=0
      
for f in listdirc:
    npat={}
    for pat in classif:
        npat[pat]=0
    print 'patient :',f
    filetw.write('patient :'+f+'\n')
    print 'number of diferent images :',listslicetot[f]
    filetw.write('number of different  images :'+str(listslicetot[f]) +'\n')
    print 'list of patterns :',listpatf[f]
    
    filetw.write('list of patterns :'+str(listpatf[f]) +'\n')

#    filetw.write( '-----------------------------\n')
    for s in range(slntdict[f]):
        for pat in classif:
            if listslicef[f][s][pat] !=0:
                npat[pat]+=1
                npattot[pat]+=1
    
    
   
    totsln=totsln+listslicetot[f]
    
    
    for pat in classif:
         if npat[pat]!=0:
             print 'number of images for :', pat,' :',  npat[pat] 
             filetw.write('number of images for :'+ pat+' :'+  str(npat[pat])+'\n')
             
    filetw.write('--------------------\n')
    print '-----------------------------'  
    
    
print 'number total of different images',totsln
filetw.write('number total of different images: '+str(totsln) +'\n')
totimages=0
for pat in classif:
         if npattot[pat]!=0:
             print 'number of images for :', pat,' :',  npattot[pat] 
             filetw.write('number of images for :'+ pat+' :'+  str(npattot[pat])+'\n' )
             totimages+=npattot[pat]
             
print ' total number of images :',totimages
filetw.write('total number of images :'+str(totimages)+'\n')
             
#
#pat='back_ground'
#filetw.write(pat+' '+str(classif[pat])+ '\n')
#pat='healthy'
#filetw.write(pat+' '+str(classif[pat])+ '\n')
#filetw.write('--------------------\n')
#for pat in listpat:
#    filetw.write(pat+' '+str(classif[pat])+ '\n')
#for f in listdirc:
#    filetw.write('patient :'+f+'\n')
#    filetw.write('number of images :'+str(listslicetot[f]) +'\n')
#    filetw.write('list of patterns :'+str(listpatf[f]) +'\n')
#    filetw.write('--------------------\n')



filetw.close()

