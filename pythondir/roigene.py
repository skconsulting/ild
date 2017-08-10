# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:48:43 2017

@author: sylvain
tool for roi generation
version 1.2
10 aout 2017
"""
#from param_pix_r import *
from param_pix_r import path_data,dimtabx,dimtaby
from param_pix_r import typei1,typei
from param_pix_r import source_name,scan_bmp,roi_name,imageDepth,lung_mask_bmp,lung_mask_bmp1,lung_mask,lung_mask1
from param_pix_r import white,black,red
from param_pix_r import classifc,classif,classifcontour,usedclassif
from param_pix_r import remove_folder,volumeroifile,normi

from skimage import measure

import cv2
import dicom
import os
import cPickle as pickle
import numpy as np
import time

pattern=''

quitl=False
images={}
tabroi={}
tabroifinal={}
tabroinumber={}

def contours(im,pat):
#    print 'contour',pat
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,1,255,0)
    _,contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2 = np.zeros((dimtabx,dimtaby,3), np.uint8)
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
    im2 = np.zeros((dimtabx,dimtaby,3), np.uint8)

    for cnt in contours:
        cv2.fillPoly(im2, [cnt],col)

    im2=cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
    return im2
              
    
def contour3(im,l):
#    print 'im',im
    col=classifc[l]
    visi = np.zeros((dimtabx,dimtaby,3), np.uint8)
    imagemax= cv2.countNonZero(np.array(im))
    if imagemax>0:
        cv2.fillPoly(visi, [np.array(im)],col)
    return visi

def contour4(vis,im):
    """  creating an hole for im"""
    visi = np.zeros((dimtabx,dimtaby,3), np.uint8)
#    vis2= np.zeros((dimtabx,dimtaby,3), np.uint8)
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
    global quitl,pattern,dirpath_patient,dirroit,zoneverticalgauche,zoneverticaldroite,zonehorizontal
    global posxdel,menus,posydel,posyquit,posxquit,posxdellast,posydellast,posxdelall,posydelall
    global posxcomp,posycomp,imagename,posxreset,posyreset,posxvisua,posyvisua,posxeraseroi,posyeraseroi,posxlastp
    global posylastp,scannumber,patternerase
    global fxs,x0new,y0new,viewasked
#    print 'patern',pattern


    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(menus, (150,12), (370,32), black, -1)
#        posrc=0
        print 'click location :',x,y

        for key1,value1 in classif.items():
            labelfound=False
            xr=5
            yr=15*value1
            xrn=xr+10
            yrn=yr+10
            if x>xr and x<xrn and y>yr and y< yrn:
                print 'this is',key1
                if key1=='erase' :
                    if  pattern!='erase':
                            patternerase=pattern
                    addt='erase :'
                    cv2.rectangle(menus, (200,0), (210,10), classifc[patternerase], -1)
                    cv2.rectangle(menus, (212,0), (340,12), black, -1)
                    cv2.putText(menus,addt+patternerase,(215,10),cv2.FONT_HERSHEY_PLAIN,0.7,classifc[patternerase],1 )
#                    
                    pattern=key1
                else:
                    addt=''
                    patternerase=''
                    pattern=key1
                    viewasked[key1]=True
                    cv2.setTrackbarPos(key1,'Slider2' ,1)
                    cv2.rectangle(menus, (200,0), (210,10), classifc[pattern], -1)
                    cv2.rectangle(menus, (212,0), (340,12), black, -1)
                    cv2.putText(menus,addt+key1,(215,10),cv2.FONT_HERSHEY_PLAIN,0.7,classifc[key1],1 )
                labelfound=True
                break
#            posrc+=1

        if  x> zoneverticalgauche[0][0] and y > zoneverticalgauche[0][1] and x<zoneverticalgauche[1][0] and y<zoneverticalgauche[1][1]:
            print 'this is in menu'
            labelfound=True

        if  x> zoneverticaldroite[0][0] and y > zoneverticaldroite[0][1] and x<zoneverticaldroite[1][0] and y<zoneverticaldroite[1][1]:
            print 'this is in menu'
            labelfound=True

        if  x> zonehorizontal[0][0] and y > zonehorizontal[0][1] and x<zonehorizontal[1][0] and y<zonehorizontal[1][1]:
            print 'this is in menu'
            labelfound=True

        if x>posxdel and x<posxdel+10 and y>posydel and y< posydel+10:
            print 'this is suppress'
            suppress()
            labelfound=True

        if x>posxquit and x<posxquit+10 and y>posyquit and y< posyquit+10:
            print 'this is quit'
            quitl=True
            labelfound=True
  
        if x>posxdellast and x<posxdellast+10 and y>posydellast and y< posydellast+10:
            print 'this is delete last'
            labelfound=True
            dellast()

        if x>posxdelall and x<posxdelall+10 and y>posydelall and y< posydelall+10:
            print 'this is delete all'
            labelfound=True
            delall()

        if x>posxcomp and x<posxcomp+10 and y>posycomp and y< posycomp+10:
            print 'this is completed for all'
            labelfound=True
            completed(imagename,dirpath_patient,dirroit)

        if x>posxreset and x<posxreset+10 and y>posyreset and y< posyreset+10:
            print 'this is reset'
            labelfound=True
            reseted()
        if x>posxvisua and x<posxvisua+10 and y>posyvisua and y< posyvisua+10:
            print 'this is visua'
            labelfound=True
            visua()
        if x>posxeraseroi and x<posxeraseroi+10 and y>posyeraseroi and y< posyeraseroi+10:
            print 'this is erase roi'
            labelfound=True
            eraseroi(imagename,dirpath_patient,dirroit)

        if x>posxlastp and x<posxlastp+10 and y>posylastp and y< posylastp+10:
            print 'this is last point'
            labelfound=True
            closepolygon()

        if not labelfound:
            print 'add point',pattern
            if len(pattern)>0:
                xnew=int((x+x0new)/fxs)
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
                cv2.rectangle(menus, (212,0), (340,12), black, -1)
                cv2.putText(menus,'No pattern selected',(215,10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

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
        cv2.rectangle(menus, (150,12), (370,52), black, -1)
        cv2.putText(menus,'polygone closed',(215,20),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

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
    cv2.rectangle(menus, (150,12), (370,52), black, -1)
    cv2.putText(menus,'delete last entry',(215,20),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

def completed(imagename,dirpath_patient,dirroit):
    global scannumber,pixelSpacing,volumeroi,patternerase

#    print 'completed start'
    closepolygon()
    print pattern
    if pattern!='erase' :
#    print dirpath_patient
        imgcoreRoi1=os.path.join(dirpath_patient,source_name) 
        imgcoreRoi2=os.path.join(imgcoreRoi1,scan_bmp) 
        imgcoreRoi3=os.path.join(imgcoreRoi2,imagename) 
        mroi=cv2.imread(imgcoreRoi3,1)
        posext=imagename.find('.'+typei1)
        imgcoreRois=imagename[0:posext]+'.'+typei
        imgcoreRoi=os.path.join(dirroit,imgcoreRois)
    #    mroi=cv2.resize(mroi,None,fx=fxssicom,fy=fxssicom,interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(imgcoreRoi,mroi)
    #    cv2.imshow("mroi", mroi)
        for key in usedclassif :
            if viewasked[key]==True:
                print key
                numeropoly=tabroinumber[key][scannumber]
        #        print key,numeropoly,scannumber
                for n in range (0,numeropoly+1):
                    if len(tabroi[key][scannumber][n])>0:
                        for l in range(0,len(tabroi[key][scannumber][n])-1):
                                cv2.line(images[scannumber], (tabroi[key][scannumber][n][l][0],tabroi[key][scannumber][n][l][1]),
                                              (tabroi[key][scannumber][n][l+1][0],tabroi[key][scannumber][n][l+1][1]), black, 1)
                        ctc=contour3(tabroi[key][scannumber][n],key)
                        tabroifinal[key][scannumber]=contour5(tabroifinal[key][scannumber],ctc,key)
                        images[scannumber]=cv2.addWeighted(images[scannumber],1,tabroifinal[key][scannumber],0.5,0)
                        tabroi[key][scannumber][n]=[]
        #                images[scannumber]=cv2.add(images[scannumber],tabroifinal[key][scannumber])
        
                tabroinumber[key][scannumber]=0
                
                imgray = cv2.cvtColor(tabroifinal[key][scannumber],cv2.COLOR_BGR2GRAY)
                imagemax= cv2.countNonZero(imgray)
                
        #        print key,imagemax
                if imagemax>0:  
                    
                    posext=imagename.find('.'+typei1)
                    imgcoreScans=imagename[0:posext]+'.'+typei1
                    imgcoreRois=imagename[0:posext]+'.'+typei
                    dirroi=os.path.join(dirpath_patient,key)                
                    if key in classifcontour:        
                        dirroi=os.path.join(dirpath_patient,lung_maskf)
                        dirroi=os.path.join(dirroi,lung_mask_bmpf)
                    if not os.path.exists(dirroi):
                        os.mkdir(dirroi)
                    imgcoreScan=os.path.join(dirroi,imgcoreScans)
                    imgcoreRoi=os.path.join(dirroit,imgcoreRois)         
                    tabtowrite=cv2.cvtColor(tabroifinal[key][scannumber],cv2.COLOR_BGR2RGB)
        
                    cv2.rectangle(menus, (150,12), (370,52), black, -1) 
                                     
                    if os.path.exists(imgcoreScan):
                        cv2.putText(menus,'ROI '+' slice:'+str(scannumber)+' overwritten',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
        
#                    if key not in classifcontour:
                    cv2.imwrite(imgcoreScan,tabtowrite)  
#                    else:
#                        tabtowrite=fillcontours(tabtowrite,key)
#                        cv2.imwrite(imgcoreScan,tabtowrite)  
        
                    tabgrey=cv2.cvtColor(tabtowrite,cv2.COLOR_BGR2GRAY)
                    np.putmask(tabgrey,tabgrey>0,1)
                    area= tabgrey.sum()*pixelSpacing*pixelSpacing/100 #in cm2
        #            print area, 'pixelSpacing',pixelSpacing
        
                    if area>0:
        
                        volumeroi[scannumber][key]=area   
                        pickle.dump(volumeroi, open(path_data_writefile, "wb" ),protocol=-1)
        #                volumeroi=pickle.load(open(path_data_writefile, "rb" ))
        #                print volumeroi
        
                    cv2.putText(menus,'Slice ROI stored',(215,20),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
                    
                    mroi=cv2.imread(imgcoreRoi,1)
                    ctkey=contours(tabtowrite,key)
                    mroiaroi=cv2.add(mroi,ctkey)
        
                    cv2.imwrite(imgcoreRoi,mroiaroi)
        
            images[scannumber]=np.zeros((dimtabx,dimtaby,3), np.uint8)
    else:
        print 'this is erase'        
        key = 'erase'
#        erasearea()
#        modifp=[]
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
#                for nkey in patternerase:
#                imgray = cv2.cvtColor(tabroifinal[patternerase][scannumber],cv2.COLOR_BGR2GRAY)
#                areab=np.sum(imgray)


                tabroifinal[patternerase][scannumber]=np.bitwise_and(tabroifinal[patternerase][scannumber],ctcm)
#                imgray = cv2.cvtColor(tabroifinal[patternerase][scannumber],cv2.COLOR_BGR2GRAY)
#                areaa=np.sum(imgray)
#
#                if areaa<areab:
#                    modifp.append(nkey)
#                                            
        tabroinumber[key][scannumber]=0
        
#        for key in modifp:    
                    
        posext=imagename.find('.'+typei1)
        imgcoreScans=imagename[0:posext]+'.'+typei1
        imgcoreRois=imagename[0:posext]+'.'+typei
        dirroi=os.path.join(dirpath_patient,patternerase)
        
        if patternerase in classifcontour:    
            dirroi=os.path.join(dirpath_patient,lung_maskf)
            dirroi=os.path.join(dirroi,lung_mask_bmpf)                               
  
        imgcoreScan=os.path.join(dirroi,imgcoreScans)
        imgcoreRoi=os.path.join(dirroit,imgcoreRois)         
        tabtowrite=cv2.cvtColor(tabroifinal[patternerase][scannumber],cv2.COLOR_BGR2RGB)    
#            tabtowritec=np.copy(tabtowrite)
        cv2.rectangle(menus, (150,12), (370,52), black, -1) 
                         
        if os.path.exists(imgcoreScan):
            cv2.putText(menus,'ROI '+' slice:'+str(scannumber)+' overwritten',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

#        if patternerase not in classifcontour:
        cv2.imwrite(imgcoreScan,tabtowrite)  
#        else:
#            tabtowrite=fillcontours(tabtowrite,patternerase)
#                cv2.imshow('tabtowritec',tabtowritec)
#                cv2.imshow('tabtowrite',tabtowrite)
#
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()

        if tabtowrite.max()>0:
            cv2.imwrite(imgcoreScan,tabtowrite)  
        else:

            os.remove(imgcoreScan)                

        tabgrey=cv2.cvtColor(tabtowrite,cv2.COLOR_BGR2GRAY)
        np.putmask(tabgrey,tabgrey>0,1)
        area= tabgrey.sum()*pixelSpacing*pixelSpacing/100 #in cm2

        volumeroi[scannumber][patternerase]=area   
        pickle.dump(volumeroi, open(path_data_writefile, "wb" ),protocol=-1)

        cv2.putText(menus,'Slice ROI stored',(215,20),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
        
        mroi=cv2.imread(imgcoreRoi,1)
        ctkey=contours(tabtowrite,patternerase)
        mroiaroi=cv2.add(mroi,ctkey)

        cv2.imwrite(imgcoreRoi,mroiaroi)
    
        images[scannumber]=np.zeros((dimtabx,dimtaby,3), np.uint8)
        
        
def visua():
    images[scannumber] = np.zeros((dimtabx,dimtaby,3), np.uint8)
    for key in usedclassif:
#        print key,viewasked[key]
        if viewasked[key]==True:
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
    cv2.rectangle(menus, (150,12), (370,52), black, -1)
    cv2.putText(menus,' Visualization ROI',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

def eraseroi(imagename,dirpath_patient,dirroit):
#    print 'this is erase roi',pattern
    if len(pattern)>0:
        closepolygon()
        delall()
        tabroifinal[pattern][scannumber]=np.zeros((dimtabx,dimtaby,3), np.uint8)
        dirroi=os.path.join(dirpath_patient,pattern)
        if pattern in classifcontour:       
            dirroi=os.path.join(dirpath_patient,lung_maskf)
            dirroi=os.path.join(dirroi,lung_mask_bmpf)
        imgcoreScan=os.path.join(dirroi,imagename)
        if os.path.exists(imgcoreScan):
            os.remove(imgcoreScan)
            completed(imagename,dirpath_patient,dirroit)   
            cv2.rectangle(menus, (150,12), (370,52), black, -1)             
            cv2.putText(menus,'ROI '+pattern+' slice:'+str(scannumber)+' erased',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
            volumeroi[scannumber][pattern]=0
            pickle.dump(volumeroi, open(path_data_writefile, "wb" ),protocol=-1)
        else:
            cv2.rectangle(menus, (150,12), (370,52), black, -1)
            cv2.putText(menus,'ROI '+pattern+' slice:'+str(scannumber)+' not exist',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
    else:
        cv2.putText(menus,' no pattern defined',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

#def erasearea():
##    cv2.rectangle(menus, (150,12), (370,52), black, -1)
#    cv2.putText(menus,'erase area',(215,50),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )


def reseted():
#    global viewasked
    for key in usedclassif:

        print key,viewasked[key]
        if viewasked[key]== True:
            numeropoly=tabroinumber[key][scannumber]
            for n in range (0,numeropoly+1):
                if len(tabroi[key][scannumber][n])>0:
                    for l in range(0,len(tabroi[key][scannumber][n])-1):
    
        #                tabroifinal[pattern][tabroi[pattern][scannumber][l][0]][tabroi[pattern][scannumber][l][1]]=classifc[pattern]
                        cv2.line(images[scannumber], (tabroi[key][scannumber][n][l][0],tabroi[key][scannumber][n][l][1]),
                                      (tabroi[key][scannumber][n][l+1][0],tabroi[key][scannumber][n][l+1][1]), black, 1)
                    tabroi[key][scannumber][n]=[]
    images[scannumber]=np.zeros((dimtabx,dimtaby,3), np.uint8)
    cv2.rectangle(menus, (150,12), (370,52), black, -1)
    cv2.putText(menus,' Delete all drawings',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
    
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
    cv2.rectangle(menus, (150,12), (370,52), black, -1)
    cv2.putText(menus,' Delete last polygon',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

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
    cv2.rectangle(menus, (150,12), (370,52), black, -1)
    cv2.putText(menus,' Delete all polygons',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

def writeslice(num):
#    print 'write',num
    cv2.rectangle(menus, (5,470), (80,460), red, -1)
    cv2.putText(menus,'Slice: '+str(num),(5,470),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

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

def rsliceNum(s,c,e):
    ''' look for  afile according to slice number'''
    #s: file name, c: delimiter for snumber, e: end of file extension
    endnumslice=s.find(e)
    posend=endnumslice
    while s.find(c,posend)==-1:
        posend-=1
    debnumslice=posend+1
    return int((s[debnumslice:endnumslice]))


def zoomfunction(im,z,px,py):
    global fxs,x0new,y0new
    fxs=1+(z/50.0)
    imgresize=cv2.resize(im,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
    dimtabxn=imgresize.shape[0]
    dimtabyn=imgresize.shape[1]
    px0=((dimtabxn-dimtabx)/2)*px/50
    py0=((dimtabyn-dimtaby)/2)*py/50

    x0=max(0,px0)

    y0=max(0,py0)
    x1=min(dimtabxn,x0+dimtabx)
    y1=min(dimtabyn,y0+dimtaby)

    crop_img=imgresize[y0:y1,x0:x1]

    x0new=x0
    y0new=y0
    return crop_img


def loop(slnt,pdirk,dirpath_patient,dirroi):
    global quitl,scannumber,imagename,viewasked,pattern
    quitl=False
    pattern='init'
    list_image={}
    cdelimter='_'
    extensionimage='.'+typei1
    limage=[name for name in os.listdir(pdirk) if name.find('.'+typei1,1)>0 ]

    if len(limage)+1==slnt:
#        print 'good'
#
        for iimage in range(0,slnt-1):
    #        print iimage
            s=limage[iimage]
                #s: file name, c: delimiter for snumber, e: end of file extension
            sln=rsliceNum(s,cdelimter,extensionimage)
            list_image[sln]=s
        fl=slnt/2
        imagename=list_image[fl+1]
        imagenamecomplet=os.path.join(pdirk,imagename)
        image = cv2.imread(imagenamecomplet,cv2.IMREAD_ANYDEPTH)
        image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.namedWindow("Slider2",cv2.WINDOW_NORMAL)

        cv2.createTrackbar( 'Brightness','Slider2',0,100,nothing)
        cv2.createTrackbar( 'Contrast','Slider2',50,100,nothing)
        cv2.createTrackbar( 'Flip','Slider2',slnt/2,slnt-2,nothing)
        cv2.createTrackbar( 'Zoom','Slider2',0,100,nothing)
        cv2.createTrackbar( 'Panx','Slider2',50,100,nothing)
        cv2.createTrackbar( 'Pany','Slider2',50,100,nothing)
        cv2.createTrackbar( 'All','Slider2',1,1,nothing)
        cv2.createTrackbar( 'None','Slider2',0,1,nothing)
        cv2.setMouseCallback("image", click_and_crop)
    viewasked={}
    for key1 in usedclassif:
#            print key1
            viewasked[key1]=True
            cv2.createTrackbar( key1,'Slider2',0,1,nothing)
    populate(dirpath_patient,slnt)
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
                cv2.rectangle(menus, (5,470), (80,460), black, -1)                      
                
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
        c = cv2.getTrackbarPos('Contrast','Slider2')
        l = cv2.getTrackbarPos('Brightness','Slider2')
        fl = cv2.getTrackbarPos('Flip','Slider2')
        z = cv2.getTrackbarPos('Zoom','Slider2')
        px = cv2.getTrackbarPos('Panx','Slider2')
        py = cv2.getTrackbarPos('Pany','Slider2')
        allview = cv2.getTrackbarPos('All','Slider2')
        noneview = cv2.getTrackbarPos('None','Slider2')

        for key2 in usedclassif:
                s = cv2.getTrackbarPos(key2,'Slider2')
                if allview==1:
                     viewasked[key2]=True
                     cv2.setTrackbarPos(key2,'Slider2' ,1)
                elif noneview ==1:
                    viewasked[key2]=False
                    cv2.setTrackbarPos(key2,'Slider2' ,0)
                elif s==0:
#            print key
                    viewasked[key2]=False
                else:
                     viewasked[key2]=True        
            
        if key ==13:
#                print 'this is return'
                if numberfinal>0:
                    
                    numberfinal=min(numberfinal,slnt-1)
#                    print numberfinal
                    writeslice(numberfinal)
                    cv2.setTrackbarPos('Flip','Slider2' ,numberfinal-1)
                    cv2.rectangle(menus, (5,470), (80,460), black, -1)
                    numberfinal=0
                    nbdig=0
                numberentered={}
#        print fl
#        cv2.setTrackbarPos('Flip','Slider2' ,5)
        if key==2424832:
                fl=max(0,fl-1)
                cv2.setTrackbarPos('Flip','Slider2' ,fl)
        if key==2555904:
                fl=min(slnt-2,fl+1)
                cv2.setTrackbarPos('Flip','Slider2' ,fl)
        scannumber=fl+1
        imagename=list_image[scannumber]
        imagenamecomplet=os.path.join(pdirk,imagename)
        image = cv2.imread(imagenamecomplet,cv2.IMREAD_ANYDEPTH)
        image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        image=zoomfunction(image,z,px,py)
        imagesview=zoomfunction(images[scannumber],z,px,py)

        
        imglumi=lumi(image,l)
        image=contrasti(imglumi,c)
        imageview=cv2.add(image,imagesview)
        imageview=cv2.add(imageview,menus)
        for key1 in usedclassif:
            if viewasked[key1]:
                tabroifinalview=zoomfunction(tabroifinal[key1][scannumber],z,px,py)
                imageview=cv2.addWeighted(imageview,1,tabroifinalview,0.5,0)
                
        imageview=cv2.cvtColor(imageview,cv2.COLOR_BGR2RGB)
        cv2.imshow("image", imageview)

def tagviews (tab,t0,x0,y0,t1,x1,y1,t2,x2,y2,t3,x3,y3,t4,x4,y4,t5,x5,y5):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    col=white
    viseg=cv2.putText(tab,t0,(x0, y0), font,0.3,col,1)
    viseg=cv2.putText(viseg,t1,(x1, y1), font,0.3,col,1)
    viseg=cv2.putText(viseg,t2,(x2, y2), font,0.3,col,1)

    viseg=cv2.putText(viseg,t3,(x3, y3), font,0.3,col,1)
    viseg=cv2.putText(viseg,t4,(x4, y4), font,0.3,col,1)
    viseg=cv2.putText(viseg,t5,(x5, y5), font,0.3,col,1)
    return viseg

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
def segment_lung_mask(image, fill_lung_structures=True):

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
#    binary_image = clear_border(binary_image)
    labels = measure.label(binary_image)

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
    for i in range (0,8):
#        print  'i:',i
#        print (i/4)%2, (i/2)%2, i%2
        for j in range (1,3):

            background_label=labels[(i/4)%2*ls0,(i/2)%2*ls1,i%2*ls2/j]
            binary_image[background_label == labels] = 2
            background_label=labels[(i/4)%2*ls0,(i/2)%2*ls1/j,i%2*ls2]
            binary_image[background_label == labels] = 2
            background_label=labels[(i/4)%2*ls0/j,(i/2)%2*ls1,i%2*ls2]
            binary_image[background_label == labels] = 2  

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


    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
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

def genebmplung(fn,tabscanScan,slnt,listsln):
    """generate lung mask from dicom files"""
    global lung_maskf,lung_mask_bmpf
    (top,tail) =os.path.split(fn)
    print ('load lung segmented dicom files in :',tail)
    

    fmbmp=os.path.join(top,lung_mask1)
    if not os.path.exists(fmbmp):
        fmbmp=os.path.join(top,lung_mask)
        lung_maskf=  lung_mask  
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
#    print len(listdcm),fmbmp
#    tabscan = np.zeros((slnt,dimtabx,dimtaby), np.uint8)
    if len(listdcm)>0:  
        print 'lung scan exists'
           
        for l in listdcm:
            FilesDCM =(os.path.join(fmbmp,l))
            RefDs = dicom.read_file(FilesDCM,force=True)
    
            dsr= RefDs.pixel_array
            dsr=normi(dsr)
            imgresize=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)    
            slicenumber=int(RefDs.InstanceNumber)

            imgcoreScan=tabscanScan[slicenumber][0]
            bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
            cv2.imwrite(bmpfile,imgresize)
    else:
            print 'no lung scan, generation proceeds'
            tabscan = np.zeros((slnt,dimtabx,dimtaby), np.int16)
            tabscanlung = np.zeros((slnt,dimtabx,dimtaby), np.uint8)
            for i in listsln:
                tabscan[i]=tabscanScan[i][1]
#            print tabscan[i].min(),tabscan[i].max()
            segmented_lungs_fill = segment_lung_mask(tabscan, True)
#            print segmented_lungs_fill.min(),segmented_lungs_fill.max()

            for i in listsln:
#                tabscan[i]=normi(tabscan[i][1])
                
                tabscanlung[i]=morph(segmented_lungs_fill[i],13)
#                tabscanlung[i]=normi(tabscanlung[i])
                imgcoreScan=tabscanScan[i][0]
                bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
                cv2.imwrite(bmpfile,tabscanlung[i])
    return


def genebmp(fn,nosource,dirroit):
    """generate patches from dicom files"""
    global pixelSpacing,lung_maskf,lung_mask_bmpf
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

    slnt=0
    listsln=[]
    for l in listdcm:

        FilesDCM =(os.path.join(fn,l))
        RefDs = RefDs = dicom.read_file(FilesDCM,force=True)
        slicenumber=int(RefDs.InstanceNumber)
        pixelSpacing=RefDs.PixelSpacing[0]
        listsln.append(slicenumber)
        if slicenumber> slnt:
            slnt=slicenumber

#    print 'number of slices', slnt
    slnt=slnt+1
    tabscan = {}
    for i in range(slnt):
        tabscan[i] = []
    for l in listdcm:
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

        dsr=cv2.resize(dsr,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)

        endnumslice=l.find('.dcm')
        imgcoreScan=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei1
        imgcoreRoi=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei
        tt=(imgcoreScan,dsr)
        tabscan[slicenumber]=tt  

        dsr=normi(dsr)

        bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
        roibmpfile=os.path.join(dirroit,imgcoreRoi)

        t2='Prototype Not for medical use'
        t1='Pt: '+tail1
        t0='CONFIDENTIAL'
        t3='Scan: '+str(slicenumber)

        t4=time.asctime()
        t5=''
        anoted_image=tagviews(dsr,t0,dimtabx-80,dimtaby-10,t1,0,dimtaby-20,t2,dimtabx-250,dimtaby-10,
                     t3,0,dimtaby-30,t4,0,dimtaby-10,t5,0,dimtaby-20)
        cv2.imwrite(bmpfile,anoted_image)
        if not os.path.exists(roibmpfile):
            cv2.imwrite(roibmpfile,anoted_image)
#        cv2.imshow('dd',dsr)
    return slnt,tabscan,listsln

def nothing(x):
    pass

def menudraw(slnt):
    global refPt, cropping,pattern,x0,y0,quitl,tabroi,tabroifinal,menus,images
    global posxdel,posydel,posxquit,posyquit,posxdellast,posydellast,posxdelall,posydelall
    global posxcomp,posycomp,posxreset,posyreset,posxvisua,posyvisua
    global posxeraseroi,posyeraseroi,posxlastp,posylastp
#    posrc=0
    for key1,value1 in classif.items():
        tabroi[key1]={}
        tabroifinal[key1]={}
        tabroinumber[key1]={}
        for sln in range(1,slnt):
            tabroinumber[key1][sln]=0
            tabroi[key1][sln]={}
            tabroi[key1][sln][0]=[]
            tabroifinal[key1][sln]= np.zeros((dimtabx,dimtaby,3), np.uint8)

        xr=5
        yr=15*value1
        xrn=xr+10
        yrn=yr+10
        cv2.rectangle(menus, (xr, yr),(xrn,yrn), classifc[key1], -1)
        cv2.putText(menus,key1,(xr+15,yr+10),cv2.FONT_HERSHEY_PLAIN,0.7,classifc[key1],1 )
#        posrc+=1

    posxdel=dimtabx-20
    posydel=15
    cv2.rectangle(menus, (posxdel,posydel),(posxdel+10,posydel+10), white, -1)
    cv2.putText(menus,'(d) del',(posxdel-40, posydel+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

    posxquit=dimtabx-20
    posyquit=dimtaby-50
    cv2.rectangle(menus, (posxquit,posyquit),(posxquit+10,posyquit+10), red, -1)
    cv2.putText(menus,'(q) quit',(posxquit-45, posyquit+10),cv2.FONT_HERSHEY_PLAIN,0.7,red,1 )
    

    posxdellast=dimtabx-20
    posydellast=30
    cv2.rectangle(menus, (posxdellast,posydellast),(posxdellast+10,posydellast+10), white, -1)
    cv2.putText(menus,'(l) del last',(posxdellast-68, posydellast+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

    posxdelall=dimtabx-20
    posydelall=45
    cv2.rectangle(menus, (posxdelall,posydelall),(posxdelall+10,posydelall+10), white, -1)
    cv2.putText(menus,'(e) del all',(posxdelall-57, posydelall+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

    posxcomp=dimtabx-20
    posycomp=75
    cv2.rectangle(menus, (posxcomp,posycomp),(posxcomp+10,posycomp+10), white, -1)
    cv2.putText(menus,'(c) completed',(posxcomp-85, posycomp+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

    posxreset=dimtabx-20
    posyreset=90
    cv2.rectangle(menus, (posxreset,posyreset),(posxreset+10,posyreset+10), white, -1)
    cv2.putText(menus,'(r) reset',(posxreset-55, posyreset+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

    posxvisua=dimtabx-20
    posyvisua=60
    cv2.rectangle(menus, (posxvisua,posyvisua),(posxvisua+10,posyvisua+10), white, -1)
    cv2.putText(menus,'(v) visua',(posxvisua-55, posyvisua+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

    posxeraseroi=dimtabx-20
    posyeraseroi=105
    cv2.rectangle(menus, (posxeraseroi,posyeraseroi),(posxeraseroi+10,posyeraseroi+10), white, -1)
    cv2.putText(menus,'(e) eraseroi',(posxeraseroi-75, posyeraseroi+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

    posxlastp=dimtabx-20
    posylastp=120
    cv2.rectangle(menus, (posxlastp,posylastp),(posxlastp+10,posylastp+10), white, -1)
    cv2.putText(menus,'(f) last p',(posxlastp-75, posylastp+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

#
def populate(pp,sl):
#    print 'populate'
    for key in usedclassif:
        dirroi=os.path.join(pp,key)
        if key in classifcontour:        
#            dirroi=os.path.join(dirroi,lung_mask_bmp) 
            dirroi=os.path.join(dirpath_patient,lung_maskf)
            dirroi=os.path.join(dirroi,lung_mask_bmpf)
    #        print dirroi,sl
        if os.path.exists(dirroi):
                listroi =[name for name in  os.listdir(dirroi) if name.lower().find('.'+typei1)>0]
                for roiimage in listroi:
#                    tabroi[key][sl]=genepoly(os.path.join(dirroi,roiimage))
                    img=os.path.join(dirroi,roiimage)
                    imageroi= cv2.imread(img,1)
                    imageroi=cv2.resize(imageroi,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)  
#                    imageroi=zoomfunction(imageroi,z)
                    cdelimter='_'
                    extensionimage='.'+typei1
                    slicenumber=rsliceNum(roiimage,cdelimter,extensionimage)
                    imageview=cv2.cvtColor(imageroi,cv2.COLOR_RGB2BGR)                    
                    tabgrey=cv2.cvtColor(imageview,cv2.COLOR_BGR2GRAY)
                    np.putmask(tabgrey,tabgrey>0,1)
                    area= tabgrey.sum()*pixelSpacing*pixelSpacing/100
                    volumeroi[slicenumber][key]=area   
#                    if key == 'ground_glass' and slicenumber==4:
##                    print pat, tabroi[143].max()
##                    cv2.imshow('143',normi(tabroi[143]))
#                        cv2.imwrite('a.bmp',normi(tabgrey))
#                        print tabgrey.shape
#                        print pixelSpacing*pixelSpacing
#                        print area
                    pickle.dump(volumeroi, open(path_data_writefile, "wb" ),protocol=-1)
                                       
#                    if key not in classifcontour:
                    tabroifinal[key][slicenumber]=imageview                        
#                    else:
#                        tabroifinal[key][slicenumber]=contours(imageview,key)
     #                    cv2.imshow('imageroi',imageroi)

def initmenus(slnt,dirpath_patient):
    global menus,imageview,zoneverticalgauche,zoneverticaldroite,zonehorizontal

    menus=np.zeros((dimtabx,dimtaby,3), np.uint8)
    for i in range(1,slnt):
        images[i]=np.zeros((dimtabx,dimtaby,3), np.uint8)
    imageview=np.zeros((dimtabx,dimtaby,3), np.uint8)
    menudraw(slnt)

    zoneverticalgauche=((0,0),(25,dimtaby-50))
    zonehorizontal=((0,0),(dimtabx,20))
    zoneverticaldroite=((dimtabx-25,0),(dimtabx,dimtaby))

def openfichierroi(patient,patient_path_complet):
    global dirpath_patient,dirroit,path_data_write,volumeroi,path_data_writefile
    
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
    slnt,tabscanScan,listsln=genebmp(dirsource,nosource,dirroit)
    dirsourcescan=os.path.join(dirsource,scan_bmp)
    initmenus(slnt,dirpath_patient)
    path_data_write=os.path.join(dirpath_patient,path_data)
    path_data_writefile=os.path.join(path_data_write,volumeroifile)
    if not os.path.exists(path_data_write):
         os.mkdir(path_data_write)
         volumeroi={}
         for i in listsln:
             volumeroi[i]={}
             for pat in classif:
                 volumeroi[i][pat]=0

         pickle.dump(volumeroi, open(path_data_writefile, "wb" ),protocol=-1)       
    else:
        if not os.path.exists(path_data_writefile):
            volumeroi={}
            for i in listsln:
                 volumeroi[i]={}
                 for pat in classif:
                     volumeroi[i][pat]=0
        else:
            volumeroi=pickle.load(open(path_data_writefile, "rb" ))
    loop(slnt,dirsourcescan,dirpath_patient,dirroit)
    return 'completed'

def openfichierroilung(patient,patient_path_complet):
    global dirpath_patient,dirroit,volumeroi,path_data_writefile
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
    slnt,tabscanScan,listsln=genebmp(dirsource,nosource,dirroit)
    genebmplung(dirsource,tabscanScan,slnt,listsln)
    path_data_write=os.path.join(dirpath_patient,path_data)
    path_data_writefile=os.path.join(path_data_write,volumeroifile)
    if not os.path.exists(path_data_write):
         os.mkdir(path_data_write)
         volumeroi={}
         for i in listsln:
             volumeroi[i]={}
             for pat in classif:
                 volumeroi[i][pat]=0

         pickle.dump(volumeroi, open(path_data_writefile, "wb" ),protocol=-1)       
    else:
        if not os.path.exists(path_data_writefile):
            volumeroi={}
            for i in listsln:
                 volumeroi[i]={}
                 for pat in classif:
                     volumeroi[i][pat]=0
        else:
            volumeroi=pickle.load(open(path_data_writefile, "rb" ))
    initmenus(slnt,dirpath_patient)
    menudraw(slnt)
    populate(dirpath_patient,slnt)
    
    return 'completed lung'

def checkvolumegeneroi(patient,patient_path_complet):
#    global dirpath_patient
    dirpath_patient=os.path.join(patient_path_complet,patient)
#    dirsource=os.path.join(dirpath_patient,source_name)
#    dirroit=os.path.join(dirpath_patient,roi_name)
    path_data_write=os.path.join(dirpath_patient,path_data)
    path_data_writefile=os.path.join(path_data_write,volumeroifile)
    if not os.path.exists(path_data_writefile):
        return 'no volume data generated'

    volumeroi=pickle.load(open(path_data_writefile, "rb" ))
    """
    print 'len(volumeroi)',len(volumeroi)
    nbpack=20
    nbdict=len(volumeroi)/nbpack
    print 'nbdict',nbdict
    ddict={}
    for i in range (nbdict+1):
        deb=i*nbpack
        fin=min(deb+nbpack,len(volumeroi))
#        print 'deb',deb,'fin',fin
        ddict[i] = dict(volumeroi.items()[deb:fin])
#        print i, 'ln',len(ddict[i])
#    return
    for i in range (nbdict+1):
        print 'pack:',i
        for value in  ddict[i]:
            txt=''
            for val2 in ddict[i][value]:
                    if ddict[i][value][val2]>0:       
                        print value,val2,round(volumeroi[value][val2],1)
                        txt=txt+'Slice number: '+str(value)+' roi: '+str(val2)+' value cm2: '+str(round(volumeroi[value][val2],1))+'\n'
    """
#
    txt=''
    for value in volumeroi:
        for val2 in volumeroi[value]:
            if volumeroi[value][val2]>0:

                print value,val2,round(volumeroi[value][val2],1)
                txt=txt+'Slice: '+str(value)+'  '+str(val2)+'  '+str(round(volumeroi[value][val2],1))+'cm2\n'
#    print len(txt)
    return txt