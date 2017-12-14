# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:48:43 2017
@author: sylvain Kritter 
Version 1.5 

06 September 2017
"""
#from param_pix_r import *
from param_pix_r import path_data,dimtabmenul,dimtabmenur,dimtabnorm,dimtabxdef
from param_pix_r import typei1,typei,typei2
from param_pix_r import source_name,scan_bmp,roi_name,imageDepth,lung_mask_bmp,lung_mask_bmp1,lung_mask,lung_mask1
from param_pix_r import white,black,red,yellow
from param_pix_r import classifc,classif,classifcontour,usedclassif
from param_pix_r import remove_folder,volumeroifile,normi,rsliceNum 
#from appJar import gui
from skimage import measure
#from sklearn.cluster import KMeans
import cv2
import dicom
import os
import cPickle as pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import product

from skimage import morphology


from skimage.segmentation import clear_border
from skimage.measure import label,regionprops
from skimage.morphology import  disk, binary_erosion, binary_closing
from skimage.filters import roberts

from scipy import ndimage as ndi



pattern=''

quitl=False
images={}
tabroi={}

tabroinumber={}

def drawcontours(im,pat):
#    print 'contour',pat
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,1,255,0)
    _,contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2 = np.zeros((dimtabx,dimtabx,3), np.uint8)
    cv2.drawContours(im2,contours,-1,classifc[pat],1)
    im2=cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
    return im2

def fillcontours(im,pat):
    col=classifc[pat]
#    print 'contour',pat
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,1,255,0)
    _,contours0,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]
    im2 = np.zeros((dimtabx,dimtabx,3), np.uint8)

    for cnt in contours:
        cv2.fillPoly(im2, [cnt],col)

    im2=cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
    return im2
              
    
def contour3(im,l):
#    print 'im',im
    col=classifc[l]
    visi = np.zeros((dimtabx,dimtabx,3), np.uint8)
    imagemax= cv2.countNonZero(np.array(im))
    if imagemax>0:
        cv2.fillPoly(visi, [np.array(im)],col)
    return visi

def contour4(vis,im):
    """  creating an hole for im"""
    visi = np.zeros((dimtabx,dimtabx,3), np.uint8)

    imagemax= cv2.countNonZero(np.array(im))
    if imagemax>0:
        cv2.fillPoly(visi, [np.array(im)],white)
        mgray = cv2.cvtColor(visi,cv2.COLOR_BGR2GRAY)
        np.putmask(mgray,mgray>0,255)
        nthresh=cv2.bitwise_not(mgray)
        vis2=cv2.bitwise_and(vis,vis,mask=nthresh)
#        cv2.imshow("vis", vis)
#        cv2.imshow("visi", visi)
#        cv2.imshow("nthresh", nthresh)
#        cv2.imshow("vis2", vis2)
#        cv2.waitKey(0)
    return vis2

def contour5(vis,im,l):
    """  creating an hole for im in vis and fill with im"""
    visi=np.copy(im)
    mgray = cv2.cvtColor(visi,cv2.COLOR_BGR2GRAY)
    np.putmask(mgray,mgray>0,255)
    nthresh=cv2.bitwise_not(mgray)
    vis2=cv2.bitwise_and(vis,vis,mask=nthresh)
    vis3=cv2.bitwise_or(vis2,visi)
    return vis3


def click_and_crop(event, x, y, flags, param):
    global quitl,pattern,dirpath_patient,dirroit,zoneverticalgauche,zoneverticaldroite
    global posxdel,menus,posydel,posyquit,posxquit,posxdellast,posydellast,posxdelall,posydelall
    global posxcomp,posycomp,imagename,posxreset,posyreset,posxvisua,posyvisua,posxeraseroi,posyeraseroi,posxlastp
    global posylastp,scannumber,patternerase,posxgeneh,posygeneh
    global fxs,x0new,y0new,viewasked
#    print 'patern',pattern


    if event == cv2.EVENT_LBUTTONDOWN:

        cv2.rectangle(menus, (150,12), (511,32), black, -1)
#        posrc=0
        
        print 'click location :',x,y
        
        for key1,value1 in classif.items():
            labelfound=False
            xr=5
            yr=25*value1
            xrn=xr+20
            yrn=yr+20
            if x>xr and x<xrn and y>yr and y< yrn:
                print 'this is',key1
                if key1=='erase' :
                    if  pattern!='erase':
                            patternerase=pattern
                    addt='erase :'
                    cv2.rectangle(menus, (200,0), (210,10), classifc[patternerase], -1)
                    cv2.rectangle(menus, (212,0), (511,12), black, -1)
                    cv2.putText(menus,addt+patternerase,(215,10),cv2.FONT_HERSHEY_PLAIN,1.0,classifc[patternerase],1 )
#                    
                    pattern=key1
                else:
                    addt=''
                    patternerase=''
                    pattern=key1
                    viewasked[key1]=True
                    cv2.setTrackbarPos(key1,'SliderRoi1' ,1)
                    cv2.rectangle(menus, (200,0), (210,10), classifc[pattern], -1)
                    cv2.rectangle(menus, (212,0), (511,12), black, -1)
                    cv2.putText(menus,addt+key1,(215,10),cv2.FONT_HERSHEY_PLAIN,1.0,classifc[key1],1 )
                labelfound=True
                break
#            posrc+=1

        if  x> zoneverticalgauche[0][0] and y > zoneverticalgauche[0][1] and x<zoneverticalgauche[1][0] and y<zoneverticalgauche[1][1]:
            print 'this is in menu'
            labelfound=True

        if  x> zoneverticaldroite[0][0] and y > zoneverticaldroite[0][1] and x<zoneverticaldroite[1][0] and y<zoneverticaldroite[1][1]:
            print 'this is in menu'
            labelfound=True

#        if  x> zonehorizontal[0][0] and y > zonehorizontal[0][1] and x<zonehorizontal[1][0] and y<zonehorizontal[1][1]:
#            print 'this is in menu'
#            labelfound=True

        if x>posxdel and x<posxdel+20 and y>posydel and y< posydel+20:
            print 'this is suppress'
            suppress()
            labelfound=True

        if x>posxquit-20 and x<posxquit+20 and y>posyquit-20 and y< posyquit+20:
            print 'this is quit'
            quitl=True
            labelfound=True
#        print posxdellast,posydellast
        if x>posxdellast and x<posxdellast+20 and y>posydellast and y< posydellast+20:
            print 'this is delete last'
            labelfound=True
            dellast()

        if x>posxdelall and x<posxdelall+20 and y>posydelall and y< posydelall+20:
            print 'this is delete all'
            labelfound=True
            delall()

        if x>posxcomp and x<posxcomp+20 and y>posycomp and y< posycomp+20:
            print 'this is completed for all'
            labelfound=True
            completed(imagename,dirpath_patient,dirroit)

        if x>posxreset and x<posxreset+20 and y>posyreset and y< posyreset+20:
            print 'this is reset'
            labelfound=True
            reseted()
        if x>posxvisua and x<posxvisua+20 and y>posyvisua and y< posyvisua+20:
            print 'this is visua'
            labelfound=True
            visua()
        if x>posxeraseroi and x<posxeraseroi+20 and y>posyeraseroi and y< posyeraseroi+20:
            print 'this is erase roi'
            labelfound=True
            eraseroi(imagename,dirpath_patient,dirroit)

        if x>posxlastp and x<posxlastp+20 and y>posylastp and y< posylastp+20:
            print 'this is last point'
            labelfound=True
            closepolygon()
        
        if x>posxgeneh and x<posxgeneh+20 and y>posygeneh and y< posygeneh+20:
            print 'generate healthy'
            labelfound=True
            genehealthy()

        if not labelfound:
            print 'add point',pattern
            if len(pattern)>0 and pattern !='init':
                xnew=int((x-dimtabmenul)/fxs/2)+int(x0new/fxs)
#                ynew=int((y+y0new)/fxs)/2
                ynew=int(y/fxs/2)+int(y0new/fxs)

#                print x,y,xnew,ynew,fxs,x0new
                numeropoly=tabroinumber[pattern][scannumber]
                tabroi[pattern][scannumber][numeropoly].append((xnew, ynew))
#                print pattern,numeropoly
#                print tabroi[pattern][scannumber][numeropoly]
                cv2.rectangle(images[scannumber], (xnew,ynew),
                              (xnew,ynew), classifc[pattern], 1)

                for l in range(0,len(tabroi[pattern][scannumber][numeropoly])-1):
                    cv2.line(images[scannumber], (tabroi[pattern][scannumber][numeropoly][l][0],tabroi[pattern][scannumber][numeropoly][l][1]),
                              (tabroi[pattern][scannumber][numeropoly][l+1][0],tabroi[pattern][scannumber][numeropoly][l+1][1]), classifc[pattern], 1)
                    l+=1
            else:
                cv2.rectangle(menus, (212,0), (511,12), black, -1)
                cv2.putText(menus,'No pattern selected',(215,10),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )

def closepolygon():
#    print 'closepolygon'
    if  len(pattern)>0:
        numeropoly=tabroinumber[pattern][scannumber]
        if len(tabroi[pattern][scannumber][numeropoly])>0:
            numeropoly+=1
#            print 'numeropoly',numeropoly
            tabroinumber[pattern][scannumber]=numeropoly
            tabroi[pattern][scannumber][numeropoly]=[]
        else:
            'wait for new point'
        cv2.rectangle(menus, (150,12), (511,52), black, -1)
        cv2.putText(menus,'polygone closed',(215,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )

def suppress():
    global menus,scannumber
    numeropoly=tabroinumber[pattern][scannumber]
    lastp=len(tabroi[pattern][scannumber][numeropoly])
    if lastp>0:
        cv2.line(images[scannumber], (tabroi[pattern][scannumber][numeropoly][lastp-2][0],tabroi[pattern][scannumber][numeropoly][lastp-2][1]),
                 (tabroi[pattern][scannumber][numeropoly][lastp-1][0],tabroi[pattern][scannumber][numeropoly][lastp-1][1]), black, 1)
        tabroi[pattern][scannumber][numeropoly].pop()
        for l in range(0,len(tabroi[pattern][scannumber][numeropoly])-1):
            cv2.line(images[scannumber], (tabroi[pattern][scannumber][numeropoly][l][0],tabroi[pattern][scannumber][numeropoly][l][1]),
                     (tabroi[pattern][scannumber][numeropoly][l+1][0],tabroi[pattern][scannumber][numeropoly][l+1][1]), classifc[pattern], 1)
            l+=1
    cv2.rectangle(menus, (150,12), (511,52), black, -1)
    cv2.putText(menus,'delete last entry',(215,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )

def genehealthy():
    global pattern,tabroifinal,volumeroi,path_data_writefile,path_data_write
    pattern='healthy'
    cv2.rectangle(menus, (212,0), (511,12), black, -1)
    cv2.rectangle(menus, (200,0), (210,10), classifc[pattern], -1)
    cv2.putText(menus,pattern,(215,10),cv2.FONT_HERSHEY_PLAIN,1.0,classifc[pattern],1 )
    try:
        lungm=tabroifinal['lung'][scannumber]
        mgray=cv2.cvtColor(lungm,cv2.COLOR_BGR2GRAY)
        imagemax= cv2.countNonZero(mgray)
    except:
        imagemax=0
    if imagemax==0:          

        cv2.rectangle(menus, (150,12), (511,52), black, -1)                        
        cv2.putText(menus,'No healthy created since no lung mask',(150,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )

    else:
        clungm=np.copy(lungm)
    #    cv2.imshow("clungm", clungm)
        for key in usedclassif :
            if key not in ['healthy','lung','erase']:
    #            print key
                try:
                    patim=tabroifinal[key][scannumber]
                except:
                    patim=np.zeros((dimtabx,dimtabx,3),np.uint8)
                mgray=cv2.cvtColor(patim,cv2.COLOR_BGR2GRAY)
                imagemax= cv2.countNonZero(mgray)           
                if imagemax>0:  
    #                cv2.imshow(key+"patim", patim)
                    np.putmask(mgray,mgray>0,255)
    #                cv2.imshow(key+"mgray", mgray)
                    nthresh=cv2.bitwise_not(mgray)
                    clungm=cv2.bitwise_and(clungm,clungm,mask=nthresh)
    #    clungm=colorimage(clungm,classifc['healthy'])
        np.putmask(clungm,clungm>0,classifc['healthy'])
    #    cv2.imshow("clungmf", clungm)         
    
        tabroifinal['healthy'][scannumber]=clungm
        pickle.dump(tabroifinal, open(os.path.join(path_data_write,'tabroifinalr'), "wb" ),protocol=-1)
        cv2.rectangle(menus, (150,12), (511,52), black, -1) 
        mgray=cv2.cvtColor(clungm,cv2.COLOR_BGR2GRAY)
        imagemax= cv2.countNonZero(mgray)
        if imagemax>0:                                  
            cv2.putText(menus,'Gene healthy '+' slice:'+str(scannumber),(150,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
            dirroi=os.path.join(dirpath_patient,'healthy')                
            if not os.path.exists(dirroi):
                os.mkdir(dirroi)
            posext=imagename.find('.'+typei1)
            imgcoreScans=imagename[0:posext]+'.'+typei1
            imgcoreScan=os.path.join(dirroi,imgcoreScans)
            imgcoreRoi=os.path.join(dirroit,imgcoreScans)         
            tabtowrite=cv2.cvtColor(clungm,cv2.COLOR_BGR2RGB)
#            print imgcoreScan
            cv2.imwrite(imgcoreScan,tabtowrite)         
            tabgrey=cv2.cvtColor(tabtowrite,cv2.COLOR_BGR2GRAY)
            np.putmask(tabgrey,tabgrey>0,1)
            area= tabgrey.sum()*pixelSpacing*pixelSpacing/100 #in cm2
            if area>0:
                volumeroi[scannumber]['healthy']=area   
                pickle.dump(volumeroi, open(path_data_writefile, "wb" ),protocol=-1)   
                mroi=cv2.imread(imgcoreRoi,1)
#                mroi=cv2.resize(mroi,(dimtabx,dimtabx),interpolation=cv2.INTER_CUBIC)  

                for pat in usedclassif:
#                     print 'pat 1',pat
                     try:
                         tabtowrite=tabroifinal[pat][scannumber]
                         tabgrey=cv2.cvtColor(tabtowrite,cv2.COLOR_BGR2GRAY)
#                         np.putmask(tabgrey,tabgrey>0,1)
                         area= tabgrey.sum()
                         if area>0:
    #                        print 'pat 2',pat
                            ctkey=drawcontours(tabtowrite,pat)
                            mroi=contour5(mroi,ctkey,pat)


#                            mroi=cv2.add(mroi,ctkey)
                     except:
                        continue
                                

                cv2.imwrite(imgcoreRoi,mroi)        
            
        else:
            cv2.putText(menus,'No healthy created since no ROI'+' slice:'+str(scannumber),(150,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )             

def completed(imagename,dirpath_patient,dirroit):
    global scannumber,pixelSpacing,volumeroi,patternerase,tabroifinal,path_data_writefile,path_data_write

#    print 'completed start'
    closepolygon()
    print 'completed for : ',pattern
    if pattern!='erase' :
#    print dirpath_patient
        imgcoreRoi1=os.path.join(dirpath_patient,source_name) 
        imgcoreRoi2=os.path.join(imgcoreRoi1,scan_bmp) 
        imgcoreRoi3=os.path.join(imgcoreRoi2,imagename) 
        mroi=cv2.imread(imgcoreRoi3,1)
#        mroi=cv2.resize(mroi,(dimtabx,dimtabx),interpolation=cv2.INTER_CUBIC)  

#        posext=imagename.find('.'+typei1)
#        imgcoreRois=imagename[0:posext]+'.'+typei
        imgcoreRoi=os.path.join(dirroit,imagename)
    #    mroi=cv2.resize(mroi,None,fx=fxssicom,fy=fxssicom,interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(imgcoreRoi,mroi)
        
        for key in usedclassif :
            if key !='erase':
                numeropoly=tabroinumber[key][scannumber]
#                print tabroinumber['erase'][scannumber]
                for n in range (0,numeropoly+1):
                    if len(tabroi[key][scannumber][n])>0:
                        for l in range(0,len(tabroi[key][scannumber][n])-1):
                                cv2.line(images[scannumber], (tabroi[key][scannumber][n][l][0],tabroi[key][scannumber][n][l][1]),
                                              (tabroi[key][scannumber][n][l+1][0],tabroi[key][scannumber][n][l+1][1]), black, 1)
#                        print '0'
                        ctc=contour3(tabroi[key][scannumber][n],key)
#                        print key,scannumber
                        try:
                            tabroifinal[key][scannumber]=contour5(tabroifinal[key][scannumber],ctc,key)
                        except:
                            tabroifinal[key][scannumber]=np.zeros((dimtabx,dimtabx,3),np.uint8)
                            tabroifinal[key][scannumber]=contour5(tabroifinal[key][scannumber],ctc,key)
                               
                        pickle.dump(tabroifinal, open(os.path.join(path_data_write,'tabroifinalr'), "wb" ),protocol=-1) 

                        images[scannumber]=cv2.addWeighted(images[scannumber],1,tabroifinal[key][scannumber],0.5,0)
                        tabroi[key][scannumber][n]=[]

                tabroinumber[key][scannumber]=0

                try:
                    imgray = cv2.cvtColor(tabroifinal[key][scannumber],cv2.COLOR_BGR2GRAY)
                    imagemax= cv2.countNonZero(imgray)
    
                    if imagemax>0:  
    
                        posext=imagename.find('.'+typei1)
                        imgcoreScans=imagename[0:posext]+'.'+typei1
    #                    imgcoreRois=imagename[0:posext]+'.'+typei
                        dirroi=os.path.join(dirpath_patient,key)                
                        if key in classifcontour:        
                            dirroi=os.path.join(dirpath_patient,lung_maskf)
                            if not os.path.exists(dirroi):
                                os.mkdir(dirroi)
                            dirroi=os.path.join(dirroi,lung_mask_bmpf)
                        if not os.path.exists(dirroi):
                            os.mkdir(dirroi)
    #                    print dirroi
                        imgcoreScan=os.path.join(dirroi,imgcoreScans)
                        imgcoreRoi=os.path.join(dirroit,imgcoreScans)         
                        tabtowrite=cv2.cvtColor(tabroifinal[key][scannumber],cv2.COLOR_BGR2RGB)
            
                        cv2.rectangle(menus, (150,12), (511,52), black, -1) 
                                         
                        if os.path.exists(imgcoreScan):
                            cv2.putText(menus,'ROI '+' slice:'+str(scannumber)+' overwritten',(150,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
                        else:
                            cv2.putText(menus,'New Slice ROI stored'+str(scannumber),(150,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
    
                        ldrroi=os.listdir(dirroi)
    
                        for i in ldrroi:
                            if rsliceNum(i,'_','.'+typei)==scannumber:
                                os.remove(os.path.join(dirroi,i))
                            if rsliceNum(i,'_','.'+typei1)==scannumber:
                                os.remove(os.path.join(dirroi,i))
                            if rsliceNum(i,'_','.'+typei2)==scannumber:
                                os.remove(os.path.join(dirroi,i))
                        cv2.imwrite(imgcoreScan,tabtowrite)  
    
                        tabgrey=cv2.cvtColor(tabtowrite,cv2.COLOR_BGR2GRAY)
                        np.putmask(tabgrey,tabgrey>0,1)
                        area= tabgrey.sum()*pixelSpacing*pixelSpacing/100 #in cm2
    #                    print area, 'pixelSpacing',pixelSpacing
    
                        if area>0:
    
                            volumeroi[scannumber][key]=area      
                            pickle.dump(volumeroi, open(path_data_writefile, "wb" ),protocol=-1)
    
#                        cv2.putText(menus,'Slice ROI stored',(150,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
    
                        mroi=cv2.imread(imgcoreRoi,1)    
                        ctkey=drawcontours(tabtowrite,key)
#                        mroiaroi=cv2.add(mroi,ctkey)
                        mroiaroi=contour5(mroi,ctkey,key)
                        cv2.imwrite(imgcoreRoi,mroiaroi)
                except:
                            continue
        numeropoly=tabroinumber['erase'][scannumber]
#                print tabroinumber['erase'][scannumber]
        for n in range (0,numeropoly+1):
            tabroi['erase'][scannumber][n]=[]

        tabroinumber['erase'][scannumber]=0
            
        images[scannumber]=np.zeros((dimtabx,dimtabx,3), np.uint8)
    else:
        print 'this is erase'        
        key = 'erase'
        numeropoly=tabroinumber[key][scannumber]
#        print key,numeropoly,scannumber
        for n in range (0,numeropoly+1):
            if len(tabroi[key][scannumber][n])>0:
                
                for l in range(0,len(tabroi[key][scannumber][n])-1):
                    cv2.line(images[scannumber], (tabroi[key][scannumber][n][l][0],tabroi[key][scannumber][n][l][1]),
                                (tabroi[key][scannumber][n][l+1][0],tabroi[key][scannumber][n][l+1][1]), black, 1)
                ctc=contour3(tabroi[key][scannumber][n],key)
                tabroi[key][scannumber][n]=[]
#                ctco=np.copy(ctc)
                np.putmask(ctc,ctc>0,255)
                ctcm=np.bitwise_not(ctc)        
                
                try:
                            tabroifinal[patternerase][scannumber]=np.bitwise_and(tabroifinal[patternerase][scannumber],ctcm)
                except:
                            tabroifinal[key][scannumber]=np.zeros((dimtabx,dimtabx,3),np.uint8)
                            tabroifinal[patternerase][scannumber]=np.bitwise_and(tabroifinal[patternerase][scannumber],ctcm)
                
               
                pickle.dump(tabroifinal, open(os.path.join(path_data_write,'tabroifinalr'), "wb" ),protocol=-1)
                                            
        tabroinumber[key][scannumber]=0
                    
        posext=imagename.find('.'+typei1)
        imgcoreScans=imagename[0:posext]+'.'+typei1
#        imgcoreRois=imagename[0:posext]+'.'+typei
        dirroi=os.path.join(dirpath_patient,patternerase)
        
        if patternerase in classifcontour:    
            dirroi=os.path.join(dirpath_patient,lung_maskf)
            dirroi=os.path.join(dirroi,lung_mask_bmpf)                               
  
        imgcoreScan=os.path.join(dirroi,imgcoreScans)
        imgcoreRoi=os.path.join(dirroit,imgcoreScans)         
        tabtowrite=cv2.cvtColor(tabroifinal[patternerase][scannumber],cv2.COLOR_BGR2RGB)    
#            tabtowritec=np.copy(tabtowrite)
        cv2.rectangle(menus, (150,12), (511,52), black, -1) 
                         
        if os.path.exists(imgcoreScan):
            cv2.putText(menus,'ROI '+' slice:'+str(scannumber)+' overwritten',(150,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
        
        if tabtowrite.max()>0:
            cv2.imwrite(imgcoreScan,tabtowrite)  
        else:
            os.remove(imgcoreScan)                

        tabgrey=cv2.cvtColor(tabtowrite,cv2.COLOR_BGR2GRAY)
        np.putmask(tabgrey,tabgrey>0,1)
        area= tabgrey.sum()*pixelSpacing*pixelSpacing/100 #in cm2

        volumeroi[scannumber][patternerase]=area   
        pickle.dump(volumeroi, open(path_data_writefile, "wb" ),protocol=-1)

        cv2.putText(menus,'Slice ROI stored',(215,20),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
        
        mroi=cv2.imread(imgcoreRoi,1)
        mroi=cv2.resize(mroi,(dimtabx,dimtabx),interpolation=cv2.INTER_CUBIC)  

        np.putmask(mroi,mroi==classifc[patternerase],0)
        ctkey=drawcontours(tabtowrite,patternerase)
#        mroiaroi=cv2.add(mroi,ctkey)
#        print mroi.shape,ctkey.shape
        mroiaroi=contour5(mroi,ctkey,key)

        cv2.imwrite(imgcoreRoi,mroiaroi)
    
        images[scannumber]=np.zeros((dimtabx,dimtabx,3), np.uint8)
        
        
def visua():
    images[scannumber] = np.zeros((dimtabx,dimtabx,3), np.uint8)
    for key in usedclassif:
#        print key,viewasked[key]
#        if viewasked[key]==True:
            numeropoly=tabroinumber[key][scannumber]
            for n in range (0,numeropoly+1):
                if len(tabroi[key][scannumber][n])>0:
                        for l in range(0,len(tabroi[key][scannumber][n])-1):
                            cv2.line(images[scannumber], (tabroi[key][scannumber][n][l][0],tabroi[key][scannumber][n][l][1]),
                                          (tabroi[key][scannumber][n][l+1][0],tabroi[key][scannumber][n][l+1][1]), black, 1)
                        ctc=contour3(tabroi[key][scannumber][n],key)
                        ctc=ctc*0.5
                        ctc=ctc.astype(np.uint8)
                        images[scannumber]=contour5(images[scannumber],ctc,key)
#                    images[scannumber]=cv2.addWeighted(images[scannumber],1,ctc,0.5,0)
    cv2.rectangle(menus, (150,12), (511,52), black, -1)
    cv2.putText(menus,' Visualization ROI',(150,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )

def eraseroi(imagename,dirpath_patient,dirroit):
    print 'this is erase roi',pattern
    global tabroifinal,volumeroi,path_data_writefile,path_data_write
    if len(pattern)>0:
        closepolygon()
        delall()
        tabroifinal[pattern][scannumber]=np.zeros((dimtabx,dimtabx,3), np.uint8)
        pickle.dump(tabroifinal, open(os.path.join(path_data_write,'tabroifinalr'), "wb" ),protocol=-1)
        dirroi=os.path.join(dirpath_patient,pattern)
        if pattern in classifcontour:       
            dirroi=os.path.join(dirpath_patient,lung_maskf)
            dirroi=os.path.join(dirroi,lung_mask_bmpf)
        imgcoreScan=os.path.join(dirroi,imagename)
        if os.path.exists(imgcoreScan):
            os.remove(imgcoreScan)
            completed(imagename,dirpath_patient,dirroit)   
            cv2.rectangle(menus, (150,12), (511,52), black, -1)             
            cv2.putText(menus,'ROI '+pattern+' slice:'+str(scannumber)+' erased',(150,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
            volumeroi[scannumber][pattern]=0
            pickle.dump(volumeroi, open(path_data_writefile, "wb" ),protocol=-1)
        else:
            cv2.rectangle(menus, (150,12), (511,52), black, -1)
            cv2.putText(menus,'ROI '+pattern+' slice:'+str(scannumber)+' not exist',(150,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
    else:
        cv2.putText(menus,' no pattern defined',(150,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )

def reseted():
#    global viewasked
    for key in usedclassif:

#        print key,viewasked[key]
#        if viewasked[key]== True:
            numeropoly=tabroinumber[key][scannumber]
            for n in range (0,numeropoly+1):
                if len(tabroi[key][scannumber][n])>0:
                    for l in range(0,len(tabroi[key][scannumber][n])-1):
    
        #                tabroifinal[pattern][tabroi[pattern][scannumber][l][0]][tabroi[pattern][scannumber][l][1]]=classifc[pattern]
                        cv2.line(images[scannumber], (tabroi[key][scannumber][n][l][0],tabroi[key][scannumber][n][l][1]),
                                      (tabroi[key][scannumber][n][l+1][0],tabroi[key][scannumber][n][l+1][1]), black, 1)
                    tabroi[key][scannumber][n]=[]
    images[scannumber]=np.zeros((dimtabx,dimtabx,3), np.uint8)
    cv2.rectangle(menus, (150,12), (511,52), black, -1)
    cv2.putText(menus,' Delete all drawings',(150,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
    
def dellast():
#    global images
    numeropoly=tabroinumber[pattern][scannumber]
    if len(tabroi[pattern][scannumber][numeropoly])>0:
#        print 'len>0'
        for l in range(0,len(tabroi[pattern][scannumber][numeropoly])-1):
            cv2.line(images[scannumber], (tabroi[pattern][scannumber][numeropoly][l][0],tabroi[pattern][scannumber][numeropoly][l][1]),
                     (tabroi[pattern][scannumber][numeropoly][l+1][0],tabroi[pattern][scannumber][numeropoly][l+1][1]), black, 1)

        
        images[scannumber]=contour4(images[scannumber],tabroi[pattern][scannumber][numeropoly])
        tabroi[pattern][scannumber][numeropoly]=[]
        
    elif numeropoly >0 :
#        print 'len=0 and num>0'
        numeropoly=numeropoly-1
        tabroinumber[pattern][scannumber]=numeropoly
        for l in range(0,len(tabroi[pattern][scannumber][numeropoly])-1):
            cv2.line(images[scannumber], (tabroi[pattern][scannumber][numeropoly][l][0],tabroi[pattern][scannumber][numeropoly][l][1]),
                     (tabroi[pattern][scannumber][numeropoly][l+1][0],tabroi[pattern][scannumber][numeropoly][l+1][1]), black, 1)
        images[scannumber]=contour4(images[scannumber],tabroi[pattern][scannumber][numeropoly])
        tabroi[pattern][scannumber][numeropoly]=[]
    else:
        print'length null'
    cv2.rectangle(menus, (150,12), (511,52), black, -1)
    cv2.putText(menus,' Delete last polygon',(150,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )

def delall():
#    global images
#    print 'delall', pattern
    numeropoly=tabroinumber[pattern][scannumber]
    for n in range (0,numeropoly+1):
#        print n
        for l in range(0,len(tabroi[pattern][scannumber][n])-1):
            cv2.line(images[scannumber], (tabroi[pattern][scannumber][n][l][0],tabroi[pattern][scannumber][n][l][1]),
                (tabroi[pattern][scannumber][n][l+1][0],tabroi[pattern][scannumber][n][l+1][1]), black, 1)
            images[scannumber]=contour4(images[scannumber],tabroi[pattern][scannumber][n])
        tabroi[pattern][scannumber][n]=[]
    tabroinumber[pattern][scannumber]=0
    cv2.rectangle(menus, (150,12), (511,52), black, -1)
    cv2.putText(menus,' Delete all polygons',(150,30),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )

def writeslice(num):
#    print 'write',num
    cv2.rectangle(menus, (5,440), (80,450), red, -1)
    cv2.putText(menus,'Slice: '+str(num),(5,450),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )

def contrasti(im,r):
     r1=0.5+r/100.0
     im=im.astype(np.float32)
     tabi1=im*r1
     tabi2=np.clip(tabi1,0,imageDepth)
     tabi3=tabi2.astype(np.uint8)
     return tabi3

def lumi(tabi,r1):
    tabi1=tabi.astype(np.uint16)
    tabi1=tabi1+r1
    tabi2=np.clip(tabi1,0,imageDepth)
    tabi3=tabi2.astype(np.uint8)
    return tabi3


def zoomfunction(im,z,px,py,dx,algo):
    global fxs,x0new,y0new

    fxs=1+(z/50.0)
    if fxs !=1:
            im=im.astype('float32')
            if algo ==0:
                  im=cv2.resize(im,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_NEAREST  )
            elif algo==1:
                  im=cv2.resize(im,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR )
            elif algo==2:
                im=cv2.resize(im,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_AREA  )
            if algo ==3:
                  im=cv2.resize(im,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_CUBIC )
            elif algo==4:
                im=cv2.resize(im,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LANCZOS4 )
            im=normi(im)
#        imgresize=cv2.resize(im,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)

    dimtabxn=im.shape[0]
    dimtabyn=im.shape[1]
    px0=((dimtabxn-dx)/2)*px/50
    py0=((dimtabyn-dx)/2)*py/50

    x0=max(0,px0)

    y0=max(0,py0)
    x1=min(dimtabxn,x0+dx)
    y1=min(dimtabyn,y0+dx)

    crop_img=im[y0:y1,x0:x1]

    x0new=x0
    y0new=y0
    
    return crop_img


def loop(slnt,pdirk,dirpath_patient,dirroi,tabscanRoi,tabscanName,imagetreat):

    global quitl,scannumber,imagename,viewasked,pattern,patternerase
    quitl=False

    pattern='init'
    patternerase='init'
# 
    fl=slnt/2
    cv2.namedWindow('imageRoi',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('imageRoi', (dimtabxdef+2*dimtabmenu),dimtabx)
    cv2.resizeWindow('imageRoi', (dimtabnorm+dimtabmenur+dimtabmenul),dimtabnorm)

    cv2.namedWindow("SliderRoi1",cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('SliderRoi1', 300,1000)

    cv2.createTrackbar( 'Brightness','SliderRoi1',0,100,nothing)
    cv2.createTrackbar( 'Contrast','SliderRoi1',50,100,nothing)
    cv2.createTrackbar( 'Flip','SliderRoi1',slnt/2,slnt-2,nothing)

    cv2.createTrackbar( 'Zoom','SliderRoi1',0,100,nothing)
    
    cv2.createTrackbar( 'Panx','SliderRoi1',50,100,nothing)
    cv2.createTrackbar( 'Pany','SliderRoi1',50,100,nothing)
    cv2.createTrackbar( 'All','SliderRoi1',1,1,nothing)
    cv2.createTrackbar( 'None','SliderRoi1',0,1,nothing)
    cv2.createTrackbar( 'Transp','SliderRoi1',5,10,nothing)
    if imagetreat==True:
        cv2.createTrackbar( 'smoo','SliderRoi1',0,4,nothing)
        cv2.createTrackbar( 'kerneli','SliderRoi1',0,5,nothing)
        cv2.createTrackbar( 'algo','SliderRoi1',3,4,nothing)
#        cv2.createTrackbar( 'floatf','SliderRoi1',0,1,nothing)
    cv2.setMouseCallback("imageRoi", click_and_crop)
    viewasked={}
    for key1 in usedclassif:
#            print key1
            viewasked[key1]=True
            cv2.createTrackbar( key1,'SliderRoi1',0,1,nothing,)

    nbdig=0
    numberentered={}

    while True:    
        
        key = cv2.waitKey(1000)
#        key = cv2.waitKey(1000) & 0xFF
#        if key != -1:
#            print key
        if key >47 and key<58:
            numberfinal=0
            knum=key-48
            numberentered[nbdig]=knum
            nbdig+=1

            for i in range (nbdig):
                numberfinal=numberfinal+numberentered[i]*10**(nbdig-1-i)   
            numberfinal = min(slnt-1,numberfinal)
            if numberfinal>0:
                writeslice(numberfinal)
            
        if key ==8:            
            numberfinal=0
            nbdig=nbdig-1
            for i in range (nbdig):
                numberfinal=numberfinal+numberentered[i]*10**(nbdig-1-i)            
            if numberfinal>0:
                writeslice(numberfinal)
            else:
                cv2.rectangle(menus, (5,440), (80,450), black, -1)
                                   
        if key == ord("c"):
                print 'completed'
                completed(imagename,dirpath_patient,dirroi)
        elif key == ord("d"):
                print 'delete entry'
                suppress()
        elif key == ord("l"):
                print 'delete last polygon'
                dellast()
        elif key == ord("a"):
                print 'delete all'
                delall()
        elif key == ord("r"):
                print 'reset'
                reseted()
        elif key == ord("v"):
                print 'visualize'
                visua()
        elif key == ord("e"):
                print 'erase'
                eraseroi(imagename,dirpath_patient,dirroi)
        elif key == ord("f"):
                print 'close polygon'
                closepolygon()
        elif key == ord("h"):
                print 'gene healthy'
                genehealthy()      
        elif key == ord("q")  or quitl or cv2.waitKey(20) & 0xFF == 27 :
               print 'quit', quitl
               cv2.destroyAllWindows()
               break
        c = cv2.getTrackbarPos('Contrast','SliderRoi1')
        l = cv2.getTrackbarPos('Brightness','SliderRoi1')
        tr = cv2.getTrackbarPos('Transp','SliderRoi1')
        fl = cv2.getTrackbarPos('Flip','SliderRoi1')
        z = cv2.getTrackbarPos('Zoom','SliderRoi1')
        px = cv2.getTrackbarPos('Panx','SliderRoi1')
        py = cv2.getTrackbarPos('Pany','SliderRoi1')
        allview = cv2.getTrackbarPos('All','SliderRoi1')
        noneview = cv2.getTrackbarPos('None','SliderRoi1')
        if imagetreat==True:
            smoo = cv2.getTrackbarPos('smoo','SliderRoi1')
            kerneli = cv2.getTrackbarPos('kerneli','SliderRoi1')
            algo = cv2.getTrackbarPos('algo','SliderRoi1')
#            floatf = cv2.getTrackbarPos('floatf','SliderRoi1')
        else:
            algo=3

        if algo ==0:
                  algot='NEAREST'
        elif algo==1:
                  algot='LINEAR'
        elif algo==2:
                algot='AREA'
        if algo ==3:
                  algot='CUBIC'
        elif algo==4:
                algot='LANCZOS4'
        if allview==1:
            for key2 in usedclassif:
                cv2.setTrackbarPos(key2,'SliderRoi1' ,1)
            cv2.setTrackbarPos('All','SliderRoi1' ,0)
            cv2.setTrackbarPos('lung','SliderRoi1' ,0)
        if noneview==1:
            for key2 in usedclassif:
                cv2.setTrackbarPos(key2,'SliderRoi1' ,0)
            cv2.setTrackbarPos('None','SliderRoi1' ,0)
                
        for key2 in usedclassif:
#            print patternerase
            if pattern==key2 or patternerase==key2:
                cv2.setTrackbarPos(key2,'SliderRoi1' ,1)
            s = cv2.getTrackbarPos(key2,'SliderRoi1')
            
            if s==0:
#            print key
                viewasked[key2]=False
            else:
                 viewasked[key2]=True        
            
        if key ==13:
#                print 'this is return'
                if numberfinal>0:
                    
                    numberfinal=min(numberfinal,slnt-1)
                    writeslice(numberfinal)
                    cv2.setTrackbarPos('Flip','SliderRoi1' ,numberfinal-1)

                    cv2.rectangle(menus, (5,440), (80,450), black, -1)
                    numberfinal=0
                    nbdig=0
                numberentered={}
#        print fl
#        cv2.setTrackbarPos('Flip','Slider2' ,5)
        if key==2424832:
                fl=max(0,fl-1)
                cv2.setTrackbarPos('Flip','SliderRoi1' ,fl)
        if key==2555904:
                fl=min(slnt-2,fl+1)
                cv2.setTrackbarPos('Flip','SliderRoi1' ,fl)
        
        imsstatus=cv2.getWindowProperty('imageRoi', 0)
        imistatus= cv2.getWindowProperty('SliderRoi1', 0)
        if (imsstatus==0) and (imistatus==0)  :
            scannumber=fl+1
            imagename=tabscanName[scannumber]
            image=tabscanRoi[scannumber]            
            imglumi=lumi(image,l)
            image=contrasti(imglumi,c)
            
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)            
            image=zoomfunction(image,z,px,py,dimtabxdef,algo)
            imagesv=images[scannumber]
            imagesview=zoomfunction(imagesv,z,px,py,dimtabxdef,algo)                            
            imageview=cv2.add(image,imagesview)
            imageview=cv2.add(menus,imageview)
            
            for key1 in usedclassif:
                if viewasked[key1]:
                    try:
                        tbroiks=tabroifinal[key1][scannumber]
                        if tbroiks.max()>0:                        
                            tabroifinalview=zoomfunction(tbroiks,z,px,py,dimtabxdef,algo)
                            imageview=cv2.addWeighted(imageview,1,tabroifinalview,tr/10.,0)
                    except:
                      continue
            imageview=imageview.astype('float32')
            if algo ==0:
                  if imagetreat :   print algot
                  imageview=cv2.resize(imageview,(dimtabnorm,dimtabnorm),interpolation=cv2.INTER_NEAREST  )
            elif algo==1:
                  imageview=cv2.resize(imageview,(dimtabnorm,dimtabnorm),interpolation=cv2.INTER_LINEAR )
                  if imagetreat :   print algot
            elif algo==2:
                imageview=cv2.resize(imageview,(dimtabnorm,dimtabnorm),interpolation=cv2.INTER_AREA  )
                if imagetreat :   print algot
            if algo ==3:
                  imageview=cv2.resize(imageview,(dimtabnorm,dimtabnorm),interpolation=cv2.INTER_CUBIC )
                  if imagetreat :   print algot
            elif algo==4:
                imageview=cv2.resize(imageview,(dimtabnorm,dimtabnorm),interpolation=cv2.INTER_LANCZOS4 )
                if imagetreat :   print algot 
                
            if imagetreat==True:
                if kerneli==0:
                    kernel=(1,1)
                elif kerneli==1:
                    kernel=(2,2)
                elif kerneli==2:
                   kernel=(3,3)
                elif kerneli==3:
                   kernel=(4,4)  
                elif kerneli==4:
                   kernel=(5,5) 
                elif kerneli==5:
                   kernel=(6,6) 
                      
                if smoo==1:
                    try:
                        print algot,'blur',kernel
                        imageview=cv2.blur(imageview,kernel)
                    except:
                            print 'blur not ok',kernel
                elif smoo==2:
                    try:
                       print algot,'medianBlur',kernel[0]
                       imageview=cv2.medianBlur(imageview,kernel[0])
                    except:
                            print 'medianBlur not ok',kernel[0]
     
                elif smoo==3:
                    try:
                       print algot,'bilateralFilter',kernel[0]
                       imageview=cv2.bilateralFilter(imageview,kernel[0],75,75)
                    except:
                            print 'bilateralFilterk not ok',kernel[0]
    
                elif smoo==4:
                    try:
                       print algot,'GaussianBlur',kernel
                       imageview=cv2.GaussianBlur(imageview,kernel,0)  
                    except:
                            print 'GaussianBlur not ok',kernel   
            imageview=np.clip(imageview,0,255)    
#            np.putmask(imageview,imageview>255,255)                
            imageview=normi(imageview)
            imageview=np.concatenate((imageview,menuright),axis=1)
            imageview=np.concatenate((menuleft,imageview),axis=1)
            imageview=cv2.cvtColor(imageview,cv2.COLOR_BGR2RGB)
            
            cv2.imshow("imageRoi", imageview)
            cv2.waitKey(25)
        else:
            print 'quit', quitl
            cv2.destroyAllWindows()
            break
            

def tagviews (tab,t0,x0,y0,t1,x1,y1,t2,x2,y2,t3,x3,y3,t4,x4,y4,t5,x5,y5,t6,x6,y6):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_PLAIN
    col=yellow
    sizef=0.8
    sizefs=0.7
    viseg=cv2.putText(tab,t0,(x0, y0), font,sizef,col,1)
    viseg=cv2.putText(viseg,t1,(x1, y1), font,sizef,col,1)
    viseg=cv2.putText(viseg,t2,(x2, y2), font,sizefs,col,1)

    viseg=cv2.putText(viseg,t3,(x3, y3), font,sizef,col,1)
    viseg=cv2.putText(viseg,t4,(x4, y4), font,sizefs,col,1)
    viseg=cv2.putText(viseg,t5,(x5, y5), font,sizef,col,1)
    viseg=cv2.putText(viseg,t6,(x6, y6), font,sizef,col,1)
    return viseg

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, slnt ,srd=False,fill_lung_structures=True,viewi=False):
    print 'start generation'

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    if viewi:
        print 'initial'+str(slnt/2)
        plt.imshow(image[slnt/2])
        plt.show()
        print image.min(), image.max()
        print image.shape
#    airloc= np.argwhere(image==-1024)
#    print airloc.shape
#    if airloc.shape[0]==0:
#        airloc= np.argwhere(image==-1000)
    
    
#    imageo=np.zeros((image.shape[0],image.shape[1],image.shape[2]),np.int16)
#    kernel=(3,3)

    binary_image = np.array(image > -350, dtype=np.int8)+1 # initial 320 350
    
    if viewi:
        print binary_image.min(), binary_image.max()
#        print airloc[0]
#        print len(airloc)
#        print 'loc air',image[airloc[0][0],airloc[0][1],airloc[0][2]]
        print 'after treshold 237'
        plt.imshow(binary_image[slnt/2])
        plt.show()
#    binary_image = clear_border(binary_image)
    labels = measure.label(binary_image)
    if viewi:
        print 'label'
        print labels.min(),labels.max()
 
        plt.imshow(labels[slnt/2])
        plt.show()
        labl1 =np.array(labels==1,dtype=np.int8)
        plt.imshow(labl1[slnt/2])
        plt.show()
        labl2 =np.array(labels==2,dtype=np.int8)
        plt.imshow(labl2[slnt/2])
        plt.show()

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
# 
    
    ls0=labels.shape[0]-1
    ls1=labels.shape[1]-1
    ls2=labels.shape[2]-1

    for i,j,k in product(range (0,5), range (0,5),range(0,5)):

##        print  'i:',i
##        print (i/4)%2, (i/2)%2, i%2
        im=int(i/4.*ls0)
#        for j in range (0,5):#3
        jm=int(j/4.*ls1)
#            for k in range(0,5):
        km=int(k/4.*ls2)
        if im*jm*km==0:
#            print im,jm,km
            background_label=labels[im,jm,km]
            binary_image[background_label == labels] = 2
    
    if viewi:
        print 'after label applied'
        print 'background_label',background_label

        plt.imshow(binary_image[slnt/2])
        plt.show()
    #Fill the air around the person
#    binary_image[background_label == labels] = 2


    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
#            print i,axial_slice
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    if viewi:
        print 'fill_lung_structures'
        plt.imshow(binary_image[slnt/2])
        plt.show()
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
#    imageo=np.zeros((image.shape[0],image.shape[1],image.shape[2]),np.int16)
#   
    if srd: 
        ke=5
        kernele=np.ones((ke,ke),np.uint8)
        kerneld=np.ones((ke,ke),np.uint8)
    
    #    print type(binary_image[0,0,0])
        for i in range (image.shape[0]):
#            edges = roberts(binary_image[i])
#            binary_image[i] = ndi.binary_fill_holes(edges)
            binary_image[i]= cv2.dilate(binary_image[i].astype('uint8'),kerneld,iterations = 5)
            binary_image[i] = cv2.erode(binary_image[i].astype('uint8'),kernele,iterations = 5)
             

#    binary_image=imageo.copy()
    
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    if viewi:
        print 'remove air pocket'
        plt.imshow(binary_image[slnt/2])
        plt.show()
    labels = measure.label(binary_image[slnt/2]) # Different labels are displayed in different colors
#    label_vals = np.unique(labels)
#    print label_vals
    regions = measure.regionprops(labels)
#    print regions
    numlung=0
    for prop in regions:
#        print prop.area
        if prop.area>5000:
            numlung+=1
#    regions = measure.regionprops(labels)
    areas = [r.area for r in regionprops(labels)]
    areassorted=sorted(areas,reverse=True)
    if viewi:
        print 'numlung',numlung
        print 'sorted areas',areassorted
    if len(areassorted)>0:
        if numlung==2 or areassorted[0]>50000:
            ok=True
            print 'successful generation'
        else:
            ok=False
            print 'Need 2nd algortithm'          
    else:
        ok=False
        print 'Need 2nd algortithm'
 
    return binary_image,ok

def get_segmented_lungs(im, plot=False):

    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image.
    '''
#    binary = im < 604-1024
    binary = im < -320
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap='gray')

    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap='gray')
    '''
    Step 3: Label the image.
    '''
    cleared=morph(cleared,5)
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)
    binary = morphology.dilation(binary,np.ones([5,5]))
    return binary



def morph(imgt,k):

    img=imgt.astype('uint8')
    img[img>0]=classifc['lung'][0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
#    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # <- to remove speckles...
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
#    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    return img

def colorimage(image,color):
#    im=image.copy()
    im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    np.putmask(im,im>0,color)
    return im

def genebmplung(fn,tabscanScan,tabscanName,slnt,listsln,tabroifinal,volumeroi):
    """generate lung mask from dicom files"""
    global lung_maskf,lung_mask_bmpf
    (top,tail) =os.path.split(fn)
    print ('load lung segmented dicom files in :',tail)
    
    fmbmp=os.path.join(top,lung_mask1)
    if not os.path.exists(fmbmp):
        fmbmp=os.path.join(top,lung_mask)
        lung_maskf=  lung_mask
        if not os.path.exists(fmbmp):
            os.mkdir(fmbmp)
    else:  
        lung_maskf=  lung_mask1 
                
    fmbmpbmp=os.path.join(fmbmp,lung_mask_bmp1)
    if not os.path.exists(fmbmpbmp):
         fmbmpbmp=os.path.join(fmbmp,lung_mask_bmp)
         lung_mask_bmpf=lung_mask_bmp
    else:
         lung_mask_bmpf=lung_mask_bmp1
    remove_folder(fmbmpbmp)
    os.mkdir(fmbmpbmp)

    listdcm=[name for name in  os.listdir(fmbmp) if name.lower().find('.dcm')>0]
    if len(listdcm)>0:  
        print 'lung scan exists in dcm'
           
        for l in listdcm:
            FilesDCM =(os.path.join(fmbmp,l))
            RefDs = dicom.read_file(FilesDCM,force=True)
    
            dsr= RefDs.pixel_array
            imgresize=normi(dsr)
#            imgresize=cv2.resize(dsr,(dimtabx,dimtabx),interpolation=cv2.INTER_LINEAR)    
            slicenumber=int(RefDs.InstanceNumber)

            imgcoreScan=tabscanName[slicenumber]
            imgresize[imgresize>0]=classifc['lung'][0]
            bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
            cv2.imwrite(bmpfile,imgresize)
    else:
            print 'no lung scan, generation proceeds'
            tabscanlung = np.zeros((slnt,dimtabx,dimtabx), np.uint8)
#            segmented_lungs_fill=np.zeros((slnt,dimtabx,dimtabx), np.uint8)
            srd=False
            debugview=False
            segmented_lungs_fill,ok = segment_lung_mask(tabscanScan,slnt, srd,True,debugview)
            if ok== False:
                srd=True
                segmented_lungs_fill,ok = segment_lung_mask(tabscanScan,slnt, srd,True,debugview)
            if ok== False:
                print 'use 2nd algorihm'
                segmented_lungs_fill=np.zeros((slnt,dimtabx,dimtabx), np.uint8)
                for i in listsln:
                    v=False
                    segmented_lungs_fill[i]=get_segmented_lungs(tabscanScan[i], v)
          
            for i in listsln:
                
                tabscanlung[i]=morph(segmented_lungs_fill[i],13)
                if tabscanlung[i].max()>0:
                    imgcoreScan=tabscanName[i]
#                    print imgcoreScan
                    bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
                    cv2.imwrite(bmpfile,tabscanlung[i])
                    tabroifinal['lung'][i]=colorimage(tabscanlung[i],classifc['lung'])
#                    print tabroifinal['lung'][i].shape
                    tabgrey=np.copy(tabscanlung[i])
                    np.putmask(tabgrey,tabgrey>0,1)
                    area= tabgrey.sum()*pixelSpacing*pixelSpacing/100
                    volumeroi[i]['lung']=area   
    return tabroifinal,volumeroi


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    
    
    
#    print '\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix),  '\033[F'
    print '\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix),  '\033[4A' 

    # Print New Line on Complete
    if iteration == total: 
        print()


def genebmp(fn,nosource,dirroit,centerHU,limitHU):
    """generate patches from dicom files"""
    global lung_maskf,lung_mask_bmpf
    print ('load dicom files in :',fn)
    if not os.path.exists(fn):
        print 'path does not exists'
    (top,tail) =os.path.split(fn)
    (top1,tail1) =os.path.split(top)
   
    fmbmplung=os.path.join(top,lung_mask1)
    if not os.path.exists(fmbmplung):
        fmbmplung=os.path.join(top,lung_mask)
        lung_maskf=  lung_mask 
    else:
        lung_maskf=  lung_mask1
    fmbmplungbmp=os.path.join(fmbmplung,lung_mask_bmp1)
    if not os.path.exists(fmbmplungbmp):
         lung_mask_bmpf=lung_mask_bmp
    else:
         lung_mask_bmpf=lung_mask_bmp1

    fmbmpbmp=os.path.join(fn,scan_bmp)

    remove_folder(fmbmpbmp)
    os.mkdir(fmbmpbmp)
    if nosource:
        fn=top
    
    listdcm=[name for name in  os.listdir(fn) if name.lower().find('.dcm')>0]
    FilesDCM =(os.path.join(fn,listdcm[0]))
    RefDs = dicom.read_file(FilesDCM,force=True)
    dsr= RefDs.pixel_array
    
    dimtabx= dimtabxdef
    slnt=0
    listsln=[]
    for l in listdcm:
        FilesDCM =(os.path.join(fn,l))
        RefDs = dicom.read_file(FilesDCM,force=True)

        slicenumber=int(RefDs.InstanceNumber)
        pixelSpacing=RefDs.PixelSpacing[0]
        listsln.append(slicenumber)
        if slicenumber> slnt:
            slnt=slicenumber
    slnt=slnt+1
    tabscan = np.zeros((slnt,dimtabx,dimtabx),np.int16)
    tabscanRoi=np.zeros((slnt,dimtabx,dimtabx,3),np.uint8)
    tabscanName = {}
    lbHU=centerHU-(1.0*limitHU/2.0)
    lhHU=centerHU+(1.0*limitHU/2.0)
    ll=len(listdcm)
    printProgressBar(0,ll , prefix = 'Progress:', suffix = 'Complete', length = 50)
    i=0
    for l in listdcm:
        i+=1
        if i%10==0:
            printProgressBar(i + 1, ll, prefix = 'Progress:', suffix = 'Complete', length = 50)
        FilesDCM =(os.path.join(fn,l))
        RefDs = RefDs = dicom.read_file(FilesDCM,force=True)
        slicenumber=int(RefDs.InstanceNumber)
        dsr= RefDs.pixel_array
        dsr = dsr.astype('int16')
        dsr[dsr == -2000] = 0
        intercept = RefDs.RescaleIntercept
        slope = RefDs.RescaleSlope
        if slope != 1:
             dsr = slope * dsr.astype(np.float64)
             dsr = dsr.astype(np.int16)
        dsr += np.int16(intercept)       
        if dsr.shape[0]!= dimtabx:
            dsr = dsr.astype('float32')
            dsr=cv2.resize(dsr,(dimtabx,dimtabx),interpolation=cv2.INTER_LINEAR)
            dsr=dsr.astype('int16')

        endnumslice=l.find('.dcm')
        imgcoreScan=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei1
        tabscanName[slicenumber]=imgcoreScan
        tabscan[slicenumber]=dsr.copy()
        
        lbHUt=lbHU
        lhHUt=lhHU        
        np.putmask(dsr,dsr<lbHU,lbHUt)
        np.putmask(dsr,dsr>lhHU,lhHUt)

        dsrnormi=normi(dsr)
        dsrnormi=cv2.cvtColor(dsrnormi,cv2.COLOR_GRAY2RGB)
        bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
        roibmpfile=os.path.join(dirroit,imgcoreScan)

        t2='Prototype '
        t1='Patient: '+tail1
        t0='CONFIDENTIAL'
        t3='Scan: '+str(slicenumber)

        t4=time.asctime()
        t5='CenterHU: '+str(int(centerHU))
        t6='LimitHU: +/-' +str(int(limitHU/2))
    
        anoted_image=tagviews(dsrnormi,
                              t0,dimtabx-200,dimtabx-10,
                              t1,0,dimtabx-21,
                              t2,dimtabx-200,dimtabx-20,
                              t3,0,dimtabx-32,
                              t4,0,dimtabx-10,
                              t5,0,dimtabx-43,
                              t6,0,dimtabx-54)     
        
        
        anoted_image=cv2.cvtColor(anoted_image,cv2.COLOR_BGR2RGB)
        cv2.imwrite(bmpfile,anoted_image)
        tabscanRoi[slicenumber]=anoted_image
        cv2.imwrite(roibmpfile,anoted_image)

    return slnt,tabscan,listsln,pixelSpacing,tabscanName,dimtabx,tabscanRoi

def nothing(x):
    pass

def menudraw(slnt):
    global refPt, cropping,pattern,x0,y0,quitl,tabroi,tabroifinal,menuleft,menuright,images
    global posxdel,posydel,posxquit,posyquit,posxdellast,posydellast,posxdelall,posydelall
    global posxcomp,posycomp,posxreset,posyreset,posxvisua,posyvisua
    global posxeraseroi,posyeraseroi,posxlastp,posylastp,posxgeneh,posygeneh
#    posrc=0
    corectx=dimtabnorm+dimtabmenul
    for key1,value1 in classif.items():
        tabroi[key1]={}
#        tabroifinal[key1]={}
        tabroinumber[key1]={}
        for sln in range(1,slnt):
            tabroinumber[key1][sln]=0
            tabroi[key1][sln]={}
            tabroi[key1][sln][0]=[]
#            tabroifinal[key1][sln]= np.zeros((dimtabx,dimtaby,3), np.uint8)
            if key1 in usedclassif:
                xr=5
                yr=25*value1
                xrn=xr+20
                yrn=yr+20
                cv2.rectangle(menuleft, (xr, yr),(xrn,yrn), classifc[key1], -1)
                cv2.putText(menuleft,key1,(xr+25,yr+15),cv2.FONT_HERSHEY_SIMPLEX,0.8,classifc[key1],2 )
#        posrc+=1
    
    posxinit=dimtabmenur-25
    fontsize=0.8
    fontw=2
    posxdel=posxinit
    posxdelt=dimtabmenur-90
    posydel=15
    cv2.rectangle(menuright, (posxdel,posydel),(posxdel+20,posydel+20), white, -1)
    cv2.putText(menuright,'(d) del',(posxdelt-65, posydel+20),cv2.FONT_HERSHEY_SIMPLEX,fontsize,white,fontw )
    posxdel+=corectx
        
    posxdellast=posxinit
    posydellast=40
    cv2.rectangle(menuright, (posxdellast,posydellast),(posxdellast+20,posydellast+20), white, -1)
    cv2.putText(menuright,'(l) del last',(posxdelt-98, posydellast+20),cv2.FONT_HERSHEY_SIMPLEX,fontsize,white,fontw )
    posxdellast+=corectx

    posxdelall=posxinit
    posydelall=65
    cv2.rectangle(menuright, (posxdelall,posydelall),(posxdelall+20,posydelall+20), white, -1)
    cv2.putText(menuright,'(e) del all',(posxdelt-90, posydelall+20),cv2.FONT_HERSHEY_SIMPLEX,fontsize,white,fontw )
    posxdelall+=corectx
    
    posxlastp=posxinit
    posylastp=90
    cv2.rectangle(menuright, (posxlastp,posylastp),(posxlastp+20,posylastp+20), white, -1)
    cv2.putText(menuright,'(f) last p',(posxdelt-90, posylastp+20),cv2.FONT_HERSHEY_SIMPLEX,fontsize,white,fontw )
    posxlastp+=corectx

    posxcomp=posxinit
    posycomp=115
    cv2.rectangle(menuright, (posxcomp,posycomp),(posxcomp+20,posycomp+20), white, -1)
    cv2.putText(menuright,'(c) completed',(posxdelt-130, posycomp+20),cv2.FONT_HERSHEY_SIMPLEX,fontsize,white,fontw )
    posxcomp+=corectx

    posxreset=posxinit
    posyreset=140
    cv2.rectangle(menuright, (posxreset,posyreset),(posxreset+20,posyreset+20), white, -1)
    cv2.putText(menuright,'(r) reset',(posxdelt-83, posyreset+20),cv2.FONT_HERSHEY_SIMPLEX,fontsize,white,fontw )
    posxreset+=corectx

    posxvisua=posxinit
    posyvisua=165
    cv2.rectangle(menuright, (posxvisua,posyvisua),(posxvisua+20,posyvisua+20), white, -1)
    cv2.putText(menuright,'(v) visua',(posxdelt-80, posyvisua+20),cv2.FONT_HERSHEY_SIMPLEX,fontsize,white,fontw )
    posxvisua+=corectx

    posxeraseroi=posxinit
    posyeraseroi=190
    cv2.rectangle(menuright, (posxeraseroi,posyeraseroi),(posxeraseroi+20,posyeraseroi+20), white, -1)
    cv2.putText(menuright,'(e) eraseroi',(posxdelt-115, posyeraseroi+20),cv2.FONT_HERSHEY_SIMPLEX,fontsize,white,fontw )
    posxeraseroi+=corectx
   
    posxgeneh=posxinit
    posygeneh=215
    cv2.rectangle(menuright, (posxgeneh,posygeneh),(posxgeneh+20,posygeneh+20), white, -1)
    cv2.putText(menuright,'(h) gene Healthy',(posxdelt-155, posygeneh+20),cv2.FONT_HERSHEY_SIMPLEX,fontsize,white,fontw )
    posxgeneh+=corectx
    
    posxquit=posxinit
    posyquit=450
    cv2.rectangle(menuright, (posxquit,posyquit),(posxquit+20,posyquit+20), red, -1)
    cv2.putText(menuright,'(q) quit',(posxdelt-70, posyquit+20),cv2.FONT_HERSHEY_SIMPLEX,fontsize,red,fontw )
    posxquit+=corectx
    
    
def drawcontours2(im,pat,dimtabx,dimtaby):
#    print 'contour',pat
    imgray = np.copy(im)
    ret,thresh = cv2.threshold(imgray,10,255,0)
    _,contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2 = np.zeros((dimtabx,dimtaby,3), np.uint8)
    cv2.drawContours(im2,contours,-1,classifc[pat],2)
    im2=cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
    return im2   

#
def populate(pp,lissln,slnt,pixelSpacing,tabscanName):
    print 'start populate'
    pathroi=os.path.join(pp,roi_name)
    volumeroi={}
    tabroifinal={}
    for sln in lissln:
        volumeroi[sln]={} 
        for pat in usedclassif:
            volumeroi[sln][pat]=0
            
    for key in usedclassif:
        
        dirroi=os.path.join(pp,key)
        tabroifinal[key]={}

        if key in classifcontour:        
#            dirroi=os.path.join(dirroi,lung_mask_bmp) 
            dirroil=os.path.join(dirpath_patient,lung_mask)

            if not os.path.exists(dirroil):
                dirroil=os.path.join(dirpath_patient,lung_mask1)

            if  os.path.exists(dirroil):               
                dirroi=os.path.join(dirroil,lung_mask_bmp)

                if not os.path.exists(dirroi):
                    dirroi=os.path.join(dirroil,lung_mask_bmp1)
            else:
                dirroi=''
                    
        if os.path.exists(dirroi):
#            print 'exist', dirroi
            listroi =[name for name in  os.listdir(dirroi) if name.lower().find('.'+typei1)>0 ]
            if len(listroi)>0:
                for roiimage in listroi:
                     
                    img=os.path.join(dirroi,roiimage)
                    imageroi= cv2.imread(img,1)              
                    imageroi=cv2.resize(imageroi,(dimtabx,dimtabx),interpolation=cv2.INTER_LINEAR)  
    
                    cdelimter='_'
                    extensionimage='.'+typei1
                    slicenumber=rsliceNum(roiimage,cdelimter,extensionimage)
#                    print roiimage,tabscanName[slicenumber]
                    if roiimage!= tabscanName[slicenumber]:
                         imgnew=os.path.join(dirroi,tabscanName[slicenumber])
                         cv2.imwrite(imgnew,imageroi)
                         os.remove(img)
                        
                    if imageroi.max() >0:
    #                    tabroifinal[key]=np.zeros((slnt,dimtabx,dimtaby,3), np.uint8)
                        tabroifinal[key][slicenumber]=imageroi
                                      
                        sroiname=os.path.join(pathroi,tabscanName[slicenumber])
#                        print pathroi
                        imageview=cv2.cvtColor(imageroi,cv2.COLOR_RGB2BGR)  
                          
                        tabgrey=cv2.cvtColor(imageview,cv2.COLOR_BGR2GRAY)
                        np.putmask(tabgrey,tabgrey>0,1)
                        area= tabgrey.sum()*pixelSpacing*pixelSpacing/100
                        volumeroi[slicenumber][key]=area   
    #                    pickle.dump(volumeroi, open(path_data_writefile, "wb" ),protocol=-1)
                        tabroifinal[key][slicenumber]=imageview   
    
                        np.putmask(tabgrey, tabgrey >0, 100)
                        ctkey=drawcontours2(tabgrey,key,dimtabx,dimtabx)
     
                        anoted_image=cv2.imread(sroiname,1) 
                        ctkeym=ctkey.copy()

                        ctkeym=cv2.cvtColor(ctkeym,cv2.COLOR_RGB2GRAY)

                        ctkeym=cv2.cvtColor(ctkeym,cv2.COLOR_GRAY2RGB) 

                        np.putmask(anoted_image, ctkeym >0, 0)

                        anoted_image=cv2.add(anoted_image,ctkey)

                        cv2.imwrite(sroiname,anoted_image)
                              

    return tabroifinal,volumeroi

def initmenus(slnt,dirpath_patient):
    global menuright,menuleft,imageview,zoneverticalgauche,zoneverticaldroite,images,menus

    menuright=np.zeros((dimtabnorm,dimtabmenur,3), np.uint8)
    menuleft=np.zeros((dimtabnorm,dimtabmenul,3), np.uint8)
    menus=np.zeros((dimtabxdef,dimtabxdef,3), np.uint8)
    menuright[:,0:2]=yellow
    menuleft[:,dimtabmenul-2:dimtabmenul]=yellow
 
    for i in range(1,slnt):
        images[i]=np.zeros((dimtabxdef,dimtabxdef,3), np.uint8)
    imageview=np.zeros((dimtabxdef,dimtabxdef,3), np.uint8)

    menudraw(slnt)

    zoneverticalgauche=((0,0),(dimtabmenul,dimtabnorm))
    zoneverticaldroite=((dimtabnorm+dimtabmenur,0),(dimtabnorm+dimtabmenur+dimtabmenul,dimtabnorm))

def openfichierroi(patient,patient_path_complet,centerHU,limitHU,lungask,ForceGenerate,ImageTreatment):
    global dirpath_patient,dirroit,path_data_write,volumeroi,path_data_writefile,pixelSpacing
    global tabroifinal,dimtabx
    print 'load ',patient
    imwait=np.zeros((200,200,3),np.uint8)
    cv2.putText(imwait,'In Progress',(50, 50),cv2.FONT_HERSHEY_PLAIN,1,white,1 )
    cv2.imshow('wait',imwait)
    dirpath_patient=os.path.join(patient_path_complet,patient)
    
    dirsource=os.path.join(dirpath_patient,source_name)
    dirroit=os.path.join(dirpath_patient,roi_name)
    if not os.path.exists(dirsource):
         os.mkdir(dirsource)
    if not os.path.exists(dirroit):
         os.mkdir(dirroit)
         
    listdcm=[name for name in os.listdir(dirsource) if name.find('.dcm')>0]
    nosource=True
    if len(listdcm)>0:
        nosource=False 
    path_data_write=os.path.join(dirpath_patient,path_data)
    if not os.path.exists(path_data_write):
        os.mkdir(path_data_write)
    path_data_writefile=os.path.join(path_data_write,volumeroifile)

    centerHU1=0
    limitHU1=0
    if os.path.exists(os.path.join(path_data_write,'centerHUr')):
        centerHU1=pickle.load( open(os.path.join(path_data_write,'centerHUr'), "rb" ))
    if os.path.exists(os.path.join(path_data_write,'limitHUr')):
        limitHU1=pickle.load( open(os.path.join(path_data_write,'limitHUr'), "rb" ))
        
    if centerHU1==centerHU and limitHU1==limitHU and not ForceGenerate:
        print 'no need to regenerate'
        slnt=pickle.load( open(os.path.join(path_data_write,'slntr'), "rb" ))
        tabscanScan=pickle.load( open(os.path.join(path_data_write,'tabscanScanr'), "rb" ))
        tabscanName=pickle.load( open(os.path.join(path_data_write,'tabscanNamer'), "rb" ))
        listsln=pickle.load( open(os.path.join(path_data_write,'listslnr'), "rb" ))
        tabroifinal=pickle.load( open(os.path.join(path_data_write,'tabroifinalr'), "rb" ))
        pixelSpacing=pickle.load( open(os.path.join(path_data_write,'pixelSpacingr'), "rb" ))
        tabscanRoi=pickle.load( open(os.path.join(path_data_write,'tabscanRoir'), "rb" ))
        volumeroi=pickle.load( open(os.path.join(path_data_write,'volumeroir'), "rb" ))
        dimtabx=pickle.load( open(os.path.join(path_data_write,'dimtabxr'), "rb" ))
        print 'end load'
    else:
        print 'generate'
        slnt,tabscanScan,listsln,pixelSpacing,tabscanName,dimtabx,tabscanRoi=genebmp(dirsource,nosource,dirroit,centerHU,limitHU)  
        tabroifinal,volumeroi=populate(dirpath_patient,listsln,slnt,pixelSpacing,tabscanName)
        pickle.dump(tabscanRoi, open(os.path.join(path_data_write,'tabscanRoir'), "wb" ),protocol=-1) 
        pickle.dump(dimtabx, open(os.path.join(path_data_write,'dimtabxr'), "wb" ),protocol=-1) 
        pickle.dump(centerHU, open(os.path.join(path_data_write,'centerHUr'), "wb" ),protocol=-1) 
        pickle.dump(limitHU, open(os.path.join(path_data_write,'limitHUr'), "wb" ),protocol=-1) 
        pickle.dump(slnt, open(os.path.join(path_data_write,'slntr'), "wb" ),protocol=-1) 
        pickle.dump(tabscanScan, open(os.path.join(path_data_write,'tabscanScanr'), "wb" ),protocol=-1) 
        pickle.dump(tabscanName, open(os.path.join(path_data_write,'tabscanNamer'), "wb" ),protocol=-1) 
        pickle.dump(listsln, open(os.path.join(path_data_write,'listslnr'), "wb" ),protocol=-1) 
        pickle.dump(tabroifinal, open(os.path.join(path_data_write,'tabroifinalr'), "wb" ),protocol=-1) 
        pickle.dump(pixelSpacing, open(os.path.join(path_data_write,'pixelSpacingr'), "wb" ),protocol=-1)
        pickle.dump(volumeroi, open(os.path.join(path_data_write,'volumeroir'), "wb" ),protocol=-1)
        
    dirsourcescan=os.path.join(dirsource,scan_bmp)
    if lungask:
        tabroifinal,volumeroi=genebmplung(dirsource,tabscanScan,tabscanName,slnt,listsln,tabroifinal,volumeroi)
    initmenus(slnt,dirpath_patient)  
    cv2.destroyWindow('wait')  
#    img=Image.fromarray(normi(tabscanScan[260]), 'RGB')
#    plt.imshow(normi(tabscanScan[260]), interpolation='nearest')
#    plt.show()
#    img.show()
#    cv2.waitKey(0)
    loop(slnt,dirsourcescan,dirpath_patient,dirroit,tabscanRoi,tabscanName,ImageTreatment)
    return 'completed'


def checkvolumegeneroi(patient,patient_path_complet):
    global path_data_writefile
    imwait=np.zeros((200,200,3),np.uint8)
    cv2.putText(imwait,'In Progress',(50, 50),cv2.FONT_HERSHEY_PLAIN,1,white,1 )
    cv2.imshow('wait',imwait)
    
    dirpath_patient=os.path.join(patient_path_complet,patient)
#    dirsource=os.path.join(dirpath_patient,source_name)
#    dirroit=os.path.join(dirpath_patient,roi_name)
    path_data_write=os.path.join(dirpath_patient,path_data)
    path_data_writefile=os.path.join(path_data_write,volumeroifile)
    if not os.path.exists(path_data_writefile):
#        cv2.destroyAllWindows()
        txt= 'no volume data generated'
    else:
        volumeroi=pickle.load(open(path_data_writefile, "rb" ))
        
        txt=''
        for value in volumeroi:
            for val2 in volumeroi[value]:
                if volumeroi[value][val2]>0:
    
                    print value,val2,round(volumeroi[value][val2],1)
                    txt=txt+'Slice: '+str(value)+'  '+str(val2)+'  '+str(round(volumeroi[value][val2],1))+'cm2\n'
    cv2.destroyAllWindows()
    return txt