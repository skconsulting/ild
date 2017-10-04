# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:48:43 2017
@author: sylvain Kritter 
Version 1.5 

06 September 2017
"""
#from param_pix_r import *
from param_pix_r import path_data,dimtabmenu,dimtabnorm
from param_pix_r import typei1,typei,typei2
from param_pix_r import source_name,scan_bmp,roi_name,imageDepth,lung_mask_bmp,lung_mask_bmp1,lung_mask,lung_mask1
from param_pix_r import white,black,red,yellow
from param_pix_r import classifc,classif,classifcontour,usedclassif
from param_pix_r import remove_folder,volumeroifile,normi,rsliceNum 

from skimage import measure

import cv2
import dicom
import os
import cPickle as pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import product
import scipy


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
                    cv2.setTrackbarPos(key1,'SliderRoi' ,1)
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
                xnew=int((x+x0new-dimtabmenu)/fxs)
                ynew=int((y+y0new)/fxs)
#                print x,y,xnew,ynew
                numeropoly=tabroinumber[pattern][scannumber]
                tabroi[pattern][scannumber][numeropoly].append((xnew, ynew))
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
        cv2.putText(menus,'polygone closed',(215,20),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )

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
    cv2.putText(menus,'delete last entry',(215,20),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )

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

                numeropoly=tabroinumber[key][scannumber]
#                print '0'
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
    
                        cv2.putText(menus,'Slice ROI stored',(215,20),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
    
                        mroi=cv2.imread(imgcoreRoi,1)    
                        ctkey=drawcontours(tabtowrite,key)
#                        mroiaroi=cv2.add(mroi,ctkey)
                        mroiaroi=contour5(mroi,ctkey,key)
                        cv2.imwrite(imgcoreRoi,mroiaroi)
                except:
                            continue
        
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
#        mroi=cv2.resize(mroi,(dimtabx,dimtabx),interpolation=cv2.INTER_CUBIC)  

        np.putmask(mroi,mroi==classifc[patternerase],0)
        ctkey=drawcontours(tabtowrite,patternerase)
#        mroiaroi=cv2.add(mroi,ctkey)
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
#    print 'this is erase roi',pattern
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
     im=im.astype(np.uint16)
     tabi1=im*r1
     tabi2=np.clip(tabi1,0,imageDepth)
     tabi3=tabi2.astype(np.uint8)
     return tabi3

def lumi(tabi,r):
    r1=r
    tabi1=tabi.astype(np.uint16)
    tabi1=tabi1+r1
    tabi2=np.clip(tabi1,0,imageDepth)
    tabi3=tabi2.astype(np.uint8)
    return tabi3


def zoomfunction(im,z,px,py):
    global fxs,x0new,y0new

    fxs=1+(z/50.0)
    if fxs !=1:
#        print 'resize'
        imgresize=cv2.resize(im,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_CUBIC)
    else:
        imgresize=im
    dimtabxn=imgresize.shape[0]
    dimtabyn=imgresize.shape[1]
    px0=((dimtabxn-dimtabx)/2)*px/50
    py0=((dimtabyn-dimtabx)/2)*py/50

    x0=max(0,px0)

    y0=max(0,py0)
    x1=min(dimtabxn,x0+dimtabx)
    y1=min(dimtabyn,y0+dimtabx)

    crop_img=imgresize[y0:y1,x0:x1]

    x0new=x0
    y0new=y0
    
    return crop_img


def loop(slnt,pdirk,dirpath_patient,dirroi,tabscanRoi,tabscanName):
    global quitl,scannumber,imagename,viewasked,pattern,patternerase
    quitl=False

    pattern='init'
    patternerase='init'
# 
    fl=slnt/2
#     
    image=tabscanRoi[fl+1]

    cv2.namedWindow('imageRoi',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('imageRoi', (dimtabx+2*dimtabmenu),dimtabx)
    cv2.namedWindow("SliderRoi",cv2.WINDOW_NORMAL)

    cv2.createTrackbar( 'Brightness','SliderRoi',0,100,nothing)
    cv2.createTrackbar( 'Contrast','SliderRoi',50,100,nothing)
    cv2.createTrackbar( 'Flip','SliderRoi',slnt/2,slnt-2,nothing)
    cv2.createTrackbar( 'Zoom','SliderRoi',0,100,nothing)
    cv2.createTrackbar( 'imh','SliderRoi',0,4,nothing)
    cv2.createTrackbar( 'Panx','SliderRoi',50,100,nothing)
    cv2.createTrackbar( 'Pany','SliderRoi',50,100,nothing)
    cv2.createTrackbar( 'All','SliderRoi',1,1,nothing)
    cv2.createTrackbar( 'None','SliderRoi',0,1,nothing)
    cv2.setMouseCallback("imageRoi", click_and_crop)
    viewasked={}
    for key1 in usedclassif:
#            print key1
            viewasked[key1]=True
            cv2.createTrackbar( key1,'SliderRoi',0,1,nothing,)

    nbdig=0
    numberentered={}
    while True:         

        key = cv2.waitKey(1000)
#        key = cv2.waitKey(1000) & 0xFF
        if key != -1:
            print key
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
#                cv2.rectangle(menus, (5,470), (80,460), black, -1)                      
                
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
        
        elif key == ord("q")  or quitl or cv2.waitKey(20) & 0xFF == 27 :
               print 'quit', quitl
               cv2.destroyAllWindows()
               break
        c = cv2.getTrackbarPos('Contrast','SliderRoi')
        l = cv2.getTrackbarPos('Brightness','SliderRoi')
        fl = cv2.getTrackbarPos('Flip','SliderRoi')
        z = cv2.getTrackbarPos('Zoom','SliderRoi')
        px = cv2.getTrackbarPos('Panx','SliderRoi')
        py = cv2.getTrackbarPos('Pany','SliderRoi')
        allview = cv2.getTrackbarPos('All','SliderRoi')
        noneview = cv2.getTrackbarPos('None','SliderRoi')
        imh = cv2.getTrackbarPos('imh','SliderRoi')

        
        if allview==1:
            for key2 in usedclassif:
                cv2.setTrackbarPos(key2,'SliderRoi' ,1)
            cv2.setTrackbarPos('All','SliderRoi' ,0)
        if noneview==1:
            for key2 in usedclassif:
                cv2.setTrackbarPos(key2,'SliderRoi' ,0)
            cv2.setTrackbarPos('None','SliderRoi' ,0)
                
        for key2 in usedclassif:
#            print patternerase
            if pattern==key2 or patternerase==key2:
                cv2.setTrackbarPos(key2,'SliderRoi' ,1)
            s = cv2.getTrackbarPos(key2,'SliderRoi')
            
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
                    cv2.setTrackbarPos('Flip','SliderRoi' ,numberfinal-1)

                    cv2.rectangle(menus, (5,440), (80,450), black, -1)
                    numberfinal=0
                    nbdig=0
                numberentered={}
#        print fl
#        cv2.setTrackbarPos('Flip','Slider2' ,5)
        if key==2424832:
                fl=max(0,fl-1)
                cv2.setTrackbarPos('Flip','SliderRoi' ,fl)
        if key==2555904:
                fl=min(slnt-2,fl+1)
                cv2.setTrackbarPos('Flip','SliderRoi' ,fl)
        
        imsstatus=cv2.getWindowProperty('imageRoi', 0)
        imistatus= cv2.getWindowProperty('SliderRoi', 0)
        if (imsstatus==0) and (imistatus==0)  :
            scannumber=fl+1
            imagename=tabscanName[scannumber]
#            imagenamecomplet=os.path.join(pdirk,imagename)
#            print pdirk
#            image = cv2.imread(imagenamecomplet,1)
            imageo=tabscanRoi[scannumber]
            
#            image=image.astype('float32')
            if imh==1:
                kernel=(3,3)
                image=cv2.blur(imageo,kernel)
            elif imh==2:
                kernel=(3,3)
                image=cv2.medianBlur(imageo,kernel[0])
            elif imh==3:
               kernel=(3,3)
               image=cv2.bilateralFilter(imageo,3,75,75)
            elif imh==4:
               kernel=(3,3)
               image=cv2.GaussianBlur(imageo,kernel,0)
            else:
                image=imageo
#                image=np.zeros((512,512,3),np.uint8) 

#            image=image.astype('uint8')
#            image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            
            image=zoomfunction(image,z,px,py)
#            print images[scannumber].shape
            imagesview=zoomfunction(images[scannumber],z,px,py)
    
            
            imglumi=lumi(image,l)
            image=contrasti(imglumi,c)
            
            imageview=cv2.add(image,imagesview)
            imageview=cv2.add(menus,imageview)
            
            for key1 in usedclassif:
                if viewasked[key1]:
                    try:
                        if tabroifinal[key1][scannumber].max()>0:                        
#                            imagetab=cv2.resize(tabroifinal[key1][scannumber],(dimtabx,dimtabx),interpolation=cv2.INTER_CUBIC  )
                            tabroifinalview=zoomfunction(tabroifinal[key1][scannumber],z,px,py)
                            imageview=cv2.addWeighted(imageview,1,tabroifinalview,0.8,0)
                    except:
                      continue
            imageview=np.concatenate((imageview,menuright),axis=1)
            imageview=np.concatenate((menuleft,imageview),axis=1)
            imageview=cv2.cvtColor(imageview,cv2.COLOR_BGR2RGB)
            cv2.imshow("imageRoi", imageview)
        else:
            print 'quit', quitl
            cv2.destroyAllWindows()
            break
            

def tagviews (tab,t0,x0,y0,t1,x1,y1,t2,x2,y2,t3,x3,y3,t4,x4,y4,t5,x5,y5,t6,x6,y6):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    col=yellow
    viseg=cv2.putText(tab,t0,(x0, y0), font,0.35,col,1)
    viseg=cv2.putText(viseg,t1,(x1, y1), font,0.35,col,1)
    viseg=cv2.putText(viseg,t2,(x2, y2), font,0.3,col,1)

    viseg=cv2.putText(viseg,t3,(x3, y3), font,0.35,col,1)
    viseg=cv2.putText(viseg,t4,(x4, y4), font,0.3,col,1)
    viseg=cv2.putText(viseg,t5,(x5, y5), font,0.35,col,1)
    viseg=cv2.putText(viseg,t6,(x6, y6), font,0.35,col,1)
    return viseg

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, slnt ,fill_lung_structures=True):
    print 'start generation'
    viewi=True

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    if viewi:
        print 'initial'+str(slnt/2)
        plt.imshow(image[slnt/2])
        plt.show()
        print image.min(), image.max()
        
    binary_image = np.array(image > -350, dtype=np.int8)+1 # initial 320 350
    
    if viewi:
        print binary_image.min(), binary_image.max()
        print 'after treshold 237'
        plt.imshow(binary_image[slnt/2])
        plt.show()
#    binary_image = clear_border(binary_image)
    labels = measure.label(binary_image)
    if viewi:
        print labels.min(),labels.max()
        print 'label'
        plt.imshow(labels[slnt/2])
        plt.show()

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
#    print labels.shape[0],labels.shape[1],labels.shape[2]

#    background_label = labels[0,0,0]
#    bg={} 
    ls0=labels.shape[0]-1
    ls1=labels.shape[1]-1
    ls2=labels.shape[2]-1

#    for i,j in product(range (0,8), range (1,3)):
#    for i in range (0,8):#8
##        print  'i:',i
##        print (i/4)%2, (i/2)%2, i%2
#        for j in range (1,3):#3
#            print (i/4)%2*ls0,(i/2)%2*ls1,i%2*ls2/j
#            print (i/4)%2*ls0,(i/2)%2*ls1/j,i%2*ls2
#            print (i/4)%2*ls0/j,(i/2)%2*ls1,i%2*ls2
#            background_label=labels[(i/4)%2*ls0,(i/2)%2*ls1,i%2*ls2/j]
#            binary_image[background_label == labels] = 2
#            background_label=labels[(i/4)%2*ls0,(i/2)%2*ls1/j,i%2*ls2]
#            binary_image[background_label == labels] = 2
#            background_label=labels[(i/4)%2*ls0/j,(i/2)%2*ls1,i%2*ls2]
#            binary_image[background_label == labels] = 2  
#    for i in range (0,5):#8
    for i,j,k in product(range (0,4), range (0,4),range(0,4)):

##        print  'i:',i
##        print (i/4)%2, (i/2)%2, i%2
        im=int(i/3.*ls0)
#        for j in range (0,5):#3
        jm=int(j/3.*ls1)
#            for k in range(0,5):
        km=int(k/3.*ls2)
        if im*jm*km==0:
#            print im,jm,km
            background_label=labels[im,jm,km]
            binary_image[background_label == labels] = 2
    if viewi:
        print 'after label applied'
        plt.imshow(binary_image[slnt/2])
        plt.show()
    #Fill the air around the person
#    binary_image[background_label == labels] = 2


    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
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

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    if viewi:
        print 'remove air pocket'
        plt.imshow(binary_image[slnt/2])
        plt.show()
    return binary_image

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
        print 'lung scan exists'
           
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

            segmented_lungs_fill = segment_lung_mask(tabscanScan,slnt, True)
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
    dimtabx= dsr.shape[0]
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
    lbHU=centerHU-1.0*limitHU/2.
    lhHU=centerHU+1.0*limitHU/2.
#    print lbHU,lhHU
    ll=len(listdcm)
    printProgressBar(0,ll , prefix = 'Progress:', suffix = 'Complete', length = 50)
    i=0
    for l in listdcm:
        i+=1
        if i%10==0:
            printProgressBar(i + 1, ll, prefix = 'Progress:', suffix = 'Complete', length = 50)
#        print l
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
#        dsr=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_CUBIC)

        endnumslice=l.find('.dcm')
        imgcoreScan=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei1
#        imgcoreRoi=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei
#        tabscan[slicenumber]=dsr 
        tabscanName[slicenumber]=imgcoreScan
#        print dsr.min(),dsr.max()
        lbHUt=lbHU
        lhHUt=lhHU
        
        np.putmask(dsr,dsr<lbHU,lbHUt)
        np.putmask(dsr,dsr>lhHU,lhHUt)
#        print dsr.min(),dsr.max()
        tabscan[slicenumber]=dsr
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
        anoted_image=tagviews(dsrnormi,t0,dimtabx-300,dimtabx-10,t1,0,dimtabx-20,t2,dimtabx-350,dimtabx-10,
                     t3,0,dimtabx-30,t4,0,dimtabx-10,t5,0,dimtabx-40,t6,0,dimtabx-50)
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
    corectx=dimtabx+dimtabmenu
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
                cv2.putText(menuleft,key1,(xr+25,yr+15),cv2.FONT_HERSHEY_PLAIN,1.0,classifc[key1],1 )
#        posrc+=1
    
    posxinit=dimtabmenu-25
    fontsize=0.9
    posxdel=posxinit
    posydel=15
    cv2.rectangle(menuright, (posxdel,posydel),(posxdel+20,posydel+20), white, -1)
    cv2.putText(menuright,'(d) del',(posxdel-60, posydel+20),cv2.FONT_HERSHEY_PLAIN,fontsize,white,1 )
    posxdel+=corectx
        
    posxdellast=posxinit
    posydellast=40
    cv2.rectangle(menuright, (posxdellast,posydellast),(posxdellast+20,posydellast+20), white, -1)
    cv2.putText(menuright,'(l) del last',(posxdellast-88, posydellast+20),cv2.FONT_HERSHEY_PLAIN,fontsize,white,1 )
    posxdellast+=corectx

    posxdelall=posxinit
    posydelall=65
    cv2.rectangle(menuright, (posxdelall,posydelall),(posxdelall+20,posydelall+20), white, -1)
    cv2.putText(menuright,'(e) del all',(posxdelall-77, posydelall+20),cv2.FONT_HERSHEY_PLAIN,fontsize,white,1 )
    posxdelall+=corectx
    
    posxlastp=posxinit
    posylastp=90
    cv2.rectangle(menuright, (posxlastp,posylastp),(posxlastp+20,posylastp+20), white, -1)
    cv2.putText(menuright,'(f) last p',(posxlastp-95, posylastp+20),cv2.FONT_HERSHEY_PLAIN,fontsize,white,1 )
    posxlastp+=corectx

    posxcomp=posxinit
    posycomp=115
    cv2.rectangle(menuright, (posxcomp,posycomp),(posxcomp+20,posycomp+20), white, -1)
    cv2.putText(menuright,'(c) completed',(posxcomp-115, posycomp+20),cv2.FONT_HERSHEY_PLAIN,fontsize,white,1 )
    posxcomp+=corectx

    posxreset=posxinit
    posyreset=140
    cv2.rectangle(menuright, (posxreset,posyreset),(posxreset+20,posyreset+20), white, -1)
    cv2.putText(menuright,'(r) reset',(posxreset-75, posyreset+20),cv2.FONT_HERSHEY_PLAIN,fontsize,white,1 )
    posxreset+=corectx

    posxvisua=posxinit
    posyvisua=165
    cv2.rectangle(menuright, (posxvisua,posyvisua),(posxvisua+20,posyvisua+20), white, -1)
    cv2.putText(menuright,'(v) visua',(posxvisua-75, posyvisua+20),cv2.FONT_HERSHEY_PLAIN,fontsize,white,1 )
    posxvisua+=corectx

    posxeraseroi=posxinit
    posyeraseroi=190
    cv2.rectangle(menuright, (posxeraseroi,posyeraseroi),(posxeraseroi+20,posyeraseroi+20), white, -1)
    cv2.putText(menuright,'(e) eraseroi',(posxeraseroi-95, posyeraseroi+20),cv2.FONT_HERSHEY_PLAIN,fontsize,white,1 )
    posxeraseroi+=corectx
   
    posxgeneh=posxinit
    posygeneh=215
    cv2.rectangle(menuright, (posxgeneh,posygeneh),(posxgeneh+20,posygeneh+20), white, -1)
    cv2.putText(menuright,'(h) gene Healthy',(posxgeneh-130, posygeneh+20),cv2.FONT_HERSHEY_PLAIN,fontsize,white,1 )
    posxgeneh+=corectx
    
    posxquit=posxinit
    posyquit=450
    cv2.rectangle(menuright, (posxquit,posyquit),(posxquit+20,posyquit+20), red, -1)
    cv2.putText(menuright,'(q) quit',(posxquit-65, posyquit+20),cv2.FONT_HERSHEY_PLAIN,fontsize,red,1 )
    posxquit+=corectx
    
    
def drawcontours2(im,pat,dimtabx,dimtaby):
#    print 'contour',pat
    imgray = np.copy(im)
    ret,thresh = cv2.threshold(imgray,10,255,0)
    _,contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2 = np.zeros((dimtabx,dimtaby,3), np.uint8)
    cv2.drawContours(im2,contours,-1,classifc[pat],1)
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
#                    imageroi=cv2.resize(imageroi,(dimtabx,dimtabx),interpolation=cv2.INTER_CUBIC)  
    
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
#                        anoted_image=cv2.resize(anoted_image,(dimtabx,dimtabx),interpolation=cv2.INTER_CUBIC)  
                        
#                        print sroiname
    
                        anoted_image=cv2.add(anoted_image,ctkey)
                        cv2.imwrite(sroiname,anoted_image)
                              

    return tabroifinal,volumeroi

def initmenus(slnt,dirpath_patient):
    global menuright,menuleft,imageview,zoneverticalgauche,zoneverticaldroite,images,menus

    menuright=np.zeros((dimtabx,dimtabmenu,3), np.uint8)
    menuleft=np.zeros((dimtabx,dimtabmenu,3), np.uint8)
    menus=np.zeros((dimtabx,dimtabx,3), np.uint8)
    menuright[:,0:2]=yellow
    menuleft[:,dimtabmenu-2:dimtabmenu]=yellow
 
    for i in range(1,slnt):
        images[i]=np.zeros((dimtabx,dimtabx,3), np.uint8)
    imageview=np.zeros((dimtabx,dimtabx,3), np.uint8)

    menudraw(slnt)

    zoneverticalgauche=((0,0),(dimtabmenu,dimtabx))
    zoneverticaldroite=((dimtabx+dimtabmenu,0),(dimtabx+(2*dimtabmenu),dimtabx))

def openfichierroi(patient,patient_path_complet,centerHU,limitHU,lungask,ForceGenerate):
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
    loop(slnt,dirsourcescan,dirpath_patient,dirroit,tabscanRoi,tabscanName)
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