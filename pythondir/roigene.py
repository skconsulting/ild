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
    global posylastp,scannumber,patternerase,posxgeneh,posygeneh
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
                    cv2.setTrackbarPos(key1,'SliderRoi' ,1)
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

        if x>posxquit-10 and x<posxquit+20 and y>posyquit-10 and y< posyquit+20:
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
        
        if x>posxgeneh and x<posxgeneh+10 and y>posygeneh and y< posygeneh+10:
            print 'generate healthy'
            labelfound=True
            genehealthy()

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

#def colorimage(image,color):
##    im=image.copy()
##    im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
#    np.putmask(im,im>0,color)
#    return im

def genehealthy():
    global pattern,tabroifinal,volumeroi,path_data_writefile,path_data_write
    pattern='healthy'
    cv2.rectangle(menus, (212,0), (340,12), black, -1)
    cv2.rectangle(menus, (200,0), (210,10), classifc[pattern], -1)
    cv2.putText(menus,pattern,(215,10),cv2.FONT_HERSHEY_PLAIN,0.7,classifc[pattern],1 )
    lungm=tabroifinal['lung'][scannumber]
    mgray=cv2.cvtColor(lungm,cv2.COLOR_BGR2GRAY)
    imagemax= cv2.countNonZero(mgray)
    if imagemax==0:          
        cv2.rectangle(menus, (150,12), (370,52), black, -1)                        
        cv2.putText(menus,'No healthy created since no lung mask',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
    else:
        clungm=np.copy(lungm)
    #    cv2.imshow("clungm", clungm)
        for key in usedclassif :
            if key not in ['healthy','lung','erase']:
    #            print key
                patim=tabroifinal[key][scannumber]
                
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
        pickle.dump(tabroifinal, open(os.path.join(path_data_write,'tabroifinal'), "wb" ),protocol=-1)
        cv2.rectangle(menus, (150,12), (370,52), black, -1) 
        mgray=cv2.cvtColor(clungm,cv2.COLOR_BGR2GRAY)
        imagemax= cv2.countNonZero(mgray)
        if imagemax>0:                                  
            cv2.putText(menus,'Gene healthy '+' slice:'+str(scannumber),(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
            dirroi=os.path.join(dirpath_patient,'healthy')                
            if not os.path.exists(dirroi):
                os.mkdir(dirroi)
            posext=imagename.find('.'+typei1)
            imgcoreScans=imagename[0:posext]+'.'+typei1
            imgcoreScan=os.path.join(dirroi,imgcoreScans)
            imgcoreRoi=os.path.join(dirroit,imgcoreScans)         
            tabtowrite=cv2.cvtColor(clungm,cv2.COLOR_BGR2RGB)
            cv2.imwrite(imgcoreScan,tabtowrite)         
            tabgrey=cv2.cvtColor(tabtowrite,cv2.COLOR_BGR2GRAY)
            np.putmask(tabgrey,tabgrey>0,1)
            area= tabgrey.sum()*pixelSpacing*pixelSpacing/100 #in cm2
            if area>0:
                volumeroi[scannumber]['healthy']=area   
                pickle.dump(volumeroi, open(path_data_writefile, "wb" ),protocol=-1)   
                mroi=cv2.imread(imgcoreRoi,1)
                for pat in usedclassif:
#                     print 'pat 1',pat
                     tabtowrite=tabroifinal[pat][scannumber]
                     tabgrey=cv2.cvtColor(tabtowrite,cv2.COLOR_BGR2GRAY)
    #                 np.putmask(tabgrey,tabgrey>0,1)
                     area= tabgrey.sum()
                     if area>0:
#                        print 'pat 2',pat
                        ctkey=drawcontours(tabtowrite,pat)
                        mroi=cv2.add(mroi,ctkey)
                        cv2.imwrite(imgcoreRoi,mroi)        
            
        else:
            cv2.putText(menus,'No healthy created since no ROI'+' slice:'+str(scannumber),(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )             

#def writesroi(img,key,scannumber):
#     tabgrey=cv2.cvtColor(tabgrey,cv2.COLOR_GRAY2BGR)
#    #                np.putmask(tabgrey,tabgrey>0,classifc[key])
#    #                cv2.imwrite(img,tabgrey)
#    sroiname=os.path.join(pathroi,roibmpfile)
##                print sroiname
#    imageview=cv2.cvtColor(imageroi,cv2.COLOR_RGB2BGR)  
#      
#    tabgrey=cv2.cvtColor(imageview,cv2.COLOR_BGR2GRAY)
#    np.putmask(tabgrey,tabgrey>0,1)
#    area= tabgrey.sum()*pixelSpacing*pixelSpacing/100
#    volumeroi[slicenumber][key]=area   
##                    pickle.dump(volumeroi, open(path_data_writefile, "wb" ),protocol=-1)
#    tabroifinal[key][slicenumber]=imageview   
#
#    np.putmask(tabgrey, tabgrey >0, 100)
#    ctkey=drawcontours2(tabgrey,key,dimtabx,dimtaby)
#    anoted_image=cv2.imread(sroiname,1)
##                    plt.imshow(anoted_image)
##                    plt.imshow(ctkey)
##                    plt.show()
##                
#    anoted_image=cv2.add(anoted_image,ctkey)
#    cv2.imwrite(sroiname,anoted_image)
    
    
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
#        posext=imagename.find('.'+typei1)
#        imgcoreRois=imagename[0:posext]+'.'+typei
        imgcoreRoi=os.path.join(dirroit,imagename)
    #    mroi=cv2.resize(mroi,None,fx=fxssicom,fy=fxssicom,interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(imgcoreRoi,mroi)
        
        for key in usedclassif :

                numeropoly=tabroinumber[key][scannumber]

                for n in range (0,numeropoly+1):
                    if len(tabroi[key][scannumber][n])>0:
                        for l in range(0,len(tabroi[key][scannumber][n])-1):
                                cv2.line(images[scannumber], (tabroi[key][scannumber][n][l][0],tabroi[key][scannumber][n][l][1]),
                                              (tabroi[key][scannumber][n][l+1][0],tabroi[key][scannumber][n][l+1][1]), black, 1)

                        ctc=contour3(tabroi[key][scannumber][n],key)
#                        print key,scannumber

                        tabroifinal[key][scannumber]=contour5(tabroifinal[key][scannumber],ctc,key)

                        pickle.dump(tabroifinal, open(os.path.join(path_data_write,'tabroifinal'), "wb" ),protocol=-1) 

#                        writesroi(tabroifinal[key][scannumber],key,scannumber)
                        images[scannumber]=cv2.addWeighted(images[scannumber],1,tabroifinal[key][scannumber],0.5,0)
                        tabroi[key][scannumber][n]=[]

                tabroinumber[key][scannumber]=0

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
        
                    cv2.rectangle(menus, (150,12), (370,52), black, -1) 
                                     
                    if os.path.exists(imgcoreScan):
                        cv2.putText(menus,'ROI '+' slice:'+str(scannumber)+' overwritten',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

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

                    cv2.putText(menus,'Slice ROI stored',(215,20),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

                    mroi=cv2.imread(imgcoreRoi,1)

                    ctkey=drawcontours(tabtowrite,key)
                    mroiaroi=cv2.add(mroi,ctkey)

                    ldrroi=os.listdir(dirroit)
                    for i in ldrroi:
                        if rsliceNum(i,'_','.'+typei)==scannumber:
                            os.remove(os.path.join(dirroit,i))
                        if rsliceNum(i,'_','.'+typei1)==scannumber:
                            os.remove(os.path.join(dirroit,i))
                        if rsliceNum(i,'_','.'+typei2)==scannumber:
                            os.remove(os.path.join(dirroit,i))                   

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
                tabroifinal[patternerase][scannumber]=np.bitwise_and(tabroifinal[patternerase][scannumber],ctcm)
                pickle.dump(tabroifinal, open(os.path.join(path_data_write,'tabroifinal'), "wb" ),protocol=-1)
                                            
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
        cv2.rectangle(menus, (150,12), (370,52), black, -1) 
                         
        if os.path.exists(imgcoreScan):
            cv2.putText(menus,'ROI '+' slice:'+str(scannumber)+' overwritten',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
        ldrroi=os.listdir(dirroi)
        for i in ldrroi:
            if rsliceNum(i,'_','.'+typei)==scannumber:
                os.remove(os.path.join(dirroi,i))
            if rsliceNum(i,'_','.'+typei1)==scannumber:
                os.remove(os.path.join(dirroi,i))
            if rsliceNum(i,'_','.'+typei2)==scannumber:
                os.remove(os.path.join(dirroi,i))           

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
        np.putmask(mroi,mroi==classifc[patternerase],0)
        ctkey=drawcontours(tabtowrite,patternerase)
        mroiaroi=cv2.add(mroi,ctkey)

        cv2.imwrite(imgcoreRoi,mroiaroi)
    
        images[scannumber]=np.zeros((dimtabx,dimtaby,3), np.uint8)
        
        
def visua():
    images[scannumber] = np.zeros((dimtabx,dimtaby,3), np.uint8)
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
    cv2.rectangle(menus, (150,12), (370,52), black, -1)
    cv2.putText(menus,' Visualization ROI',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

def eraseroi(imagename,dirpath_patient,dirroit):
#    print 'this is erase roi',pattern
    global tabroifinal,volumeroi,path_data_writefile,path_data_write
    if len(pattern)>0:
        closepolygon()
        delall()
        tabroifinal[pattern][scannumber]=np.zeros((dimtabx,dimtaby,3), np.uint8)
        pickle.dump(tabroifinal, open(os.path.join(path_data_write,'tabroifinal'), "wb" ),protocol=-1)
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
    cv2.rectangle(menus, (5,440), (80,450), red, -1)
    cv2.putText(menus,'Slice: '+str(num),(5,450),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

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
    global quitl,scannumber,imagename,viewasked,pattern,patternerase
    quitl=False

    pattern='init'
    patternerase='init'
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

        cv2.namedWindow('imageRoi',cv2.WINDOW_NORMAL)
        cv2.namedWindow("SliderRoi",cv2.WINDOW_NORMAL)

        cv2.createTrackbar( 'Brightness','SliderRoi',0,100,nothing)
        cv2.createTrackbar( 'Contrast','SliderRoi',50,100,nothing)
        cv2.createTrackbar( 'Flip','SliderRoi',slnt/2,slnt-2,nothing)
        cv2.createTrackbar( 'Zoom','SliderRoi',0,100,nothing)
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
#                    print numberfinal
                    writeslice(numberfinal)
                    cv2.setTrackbarPos('Flip','SliderRoi' ,numberfinal-1)
#                    cv2.rectangle(menus, (5,470), (80,460), black, -1)
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
            imagename=list_image[scannumber]
            imagenamecomplet=os.path.join(pdirk,imagename)
            image = cv2.imread(imagenamecomplet)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            
            image=zoomfunction(image,z,px,py)

            imagesview=zoomfunction(images[scannumber],z,px,py)
    
            
            imglumi=lumi(image,l)
            image=contrasti(imglumi,c)
            imageview=cv2.add(image,imagesview)
            imageview=cv2.add(imageview,menus)
            for key1 in usedclassif:
                if viewasked[key1]:
                    if tabroifinal[key1][scannumber].max()>0:                        
                        tabroifinalview=zoomfunction(tabroifinal[key1][scannumber],z,px,py)
                        imageview=cv2.addWeighted(imageview,1,tabroifinalview,0.5,0)
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

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    print 'initial 237'
    plt.imshow(image[slnt/2])
    plt.show()
    binary_image = np.array(image > -320, dtype=np.int8)+1
    print 'after treshold 237'
    plt.imshow(binary_image[slnt/2])
    plt.show()
#    binary_image = clear_border(binary_image)
    labels = measure.label(binary_image)
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
    for i in range (0,8):
#        print  'i:',i
#        print (i/4)%2, (i/2)%2, i%2
        for j in range (1,4):

            background_label=labels[(i/4)%2*ls0,(i/2)%2*ls1,i%2*ls2/j]
            binary_image[background_label == labels] = 2
            background_label=labels[(i/4)%2*ls0,(i/2)%2*ls1/j,i%2*ls2]
            binary_image[background_label == labels] = 2
            background_label=labels[(i/4)%2*ls0/j,(i/2)%2*ls1,i%2*ls2]
            binary_image[background_label == labels] = 2  
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

def colorimage(image,color):
#    im=image.copy()
    im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    np.putmask(im,im>0,color)
    return im

def genebmplung(fn,tabscanScan,slnt,listsln,tabroifinal,volumeroi):
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
            imgresize[imgresize>0]=classifc['lung'][0]
            bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
            cv2.imwrite(bmpfile,imgresize)
    else:
            print 'no lung scan, generation proceeds'
            tabscan = np.zeros((slnt,dimtabx,dimtaby), np.int16)
            tabscanlung = np.zeros((slnt,dimtabx,dimtaby), np.uint8)
            for i in listsln:
                tabscan[i]=tabscanScan[i][1]
#            print tabscan[i].min(),tabscan[i].max()
            segmented_lungs_fill = segment_lung_mask(tabscan,slnt, True)
#            print segmented_lungs_fill.min(),segmented_lungs_fill.max()


            for i in listsln:
#                tabscan[i]=normi(tabscan[i][1])
                
                tabscanlung[i]=morph(segmented_lungs_fill[i],13)
                if tabscanlung[i].max()>0:
#                tabscanlung[i]=normi(tabscanlung[i])
                    imgcoreScan=tabscanScan[i][0]
                    bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
                    cv2.imwrite(bmpfile,tabscanlung[i])
                    tabroifinal['lung'][i]=colorimage(tabscanlung[i],classifc['lung'])
#                    print tabroifinal['lung'][i].shape
                    tabgrey=np.copy(tabscanlung[i])
                    np.putmask(tabgrey,tabgrey>0,1)
                    area= tabgrey.sum()*pixelSpacing*pixelSpacing/100
                    volumeroi[i]['lung']=area   
    return tabroifinal,volumeroi


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
    
#    ldiroit=os.listdir(dirroit)
    
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
#    createvolumeroi(dirpath_patient,listsln)
#    populate(top)
#    tabsroi=np.zeros((slnt,dimtabx,dimtaby), np.uint8) 
    slnt=slnt+1
    tabscan = {}
    lbHU=centerHU-limitHU
    lhHU=centerHU+limitHU
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
#        imgcoreRoi=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei
        tt=(imgcoreScan,dsr)
        tabscan[slicenumber]=tt  
        np.putmask(dsr,dsr<lbHU,lbHU)
        np.putmask(dsr,dsr>lhHU,lhHU)
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
        t6='LimitHU: +/-' +str(int(limitHU))
        anoted_image=tagviews(dsrnormi,t0,dimtabx-300,dimtaby-10,t1,0,dimtaby-20,t2,dimtabx-350,dimtaby-10,
                     t3,0,dimtaby-30,t4,0,dimtaby-10,t5,0,dimtaby-40,t6,0,dimtaby-50)
        anoted_image=cv2.cvtColor(anoted_image,cv2.COLOR_BGR2RGB)
        cv2.imwrite(bmpfile,anoted_image)
#        tabsroi[slicenumber]=anoted_image
        
#        
#        for i in ldiroit:
#                if rsliceNum(i,'_','.'+typei)==slicenumber:
#                    img=cv2.imread(os.path.join(dirroit,i))
#                    os.remove(os.path.join(dirroit,i))
#                    cv2.imwrite(roibmpfile,img)
#                if rsliceNum(i,'_','.'+typei1)==slicenumber:
#                    img=cv2.imread(os.path.join(dirroit,i))
#                    os.remove(os.path.join(dirroit,i))
#                    cv2.imwrite(roibmpfile,img)
#                if rsliceNum(i,'_','.'+typei2)==slicenumber:
#                    img=cv2.imread(os.path.join(dirroit,i))
#                    os.remove(os.path.join(dirroit,i))
#                    cv2.imwrite(roibmpfile,img)
            
#        if not os.path.exists(roibmpfile):
        cv2.imwrite(roibmpfile,anoted_image)
#        for pat in usedclassif:
##                     print 'pat 1',pat
#                 try:
##                     print 'try',pat,slicenumber
#                     tabtowrite=tabroifinal[pat][slicenumber]
#                     tabgrey=cv2.cvtColor(tabtowrite,cv2.COLOR_BGR2GRAY)
#    #                 np.putmask(tabgrey,tabgrey>0,1)
#                     area= tabgrey.sum()
#                     if area>0:
#    #                        print 'pat 2',pat
#                        ctkey=drawcontours(tabtowrite,pat)
#                        anoted_image=cv2.add(anoted_image,ctkey)
#                        cv2.imwrite(roibmpfile,anoted_image)      
#                 except:
##                    print 'except',pat,slicenumber
#                    tabroifinal[pat][slicenumber]=np.zeros((dimtabx,dimtaby,3), np.uint8)
#                            
#        cv2.imshow('dd',dsr)
    return slnt,tabscan,listsln,pixelSpacing

def nothing(x):
    pass

def menudraw(slnt):
    global refPt, cropping,pattern,x0,y0,quitl,tabroi,tabroifinal,menus,images
    global posxdel,posydel,posxquit,posyquit,posxdellast,posydellast,posxdelall,posydelall
    global posxcomp,posycomp,posxreset,posyreset,posxvisua,posyvisua
    global posxeraseroi,posyeraseroi,posxlastp,posylastp,posxgeneh,posygeneh
#    posrc=0
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
    
    posxgeneh=dimtabx-20
    posygeneh=135
    cv2.rectangle(menus, (posxgeneh,posygeneh),(posxgeneh+10,posygeneh+10), white, -1)
    cv2.putText(menus,'(h) gene Hea',(posxgeneh-75, posygeneh+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
    
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
def populate(pp,lissln,slnt,pixelSpacing):
#    print 'populate'
#    global pixelSpacing
#    path_data_write=os.path.join(pp,path_data)
    pathroi=os.path.join(pp,roi_name)
#    print 'lung_mask',lung_mask
#    print 'lung_mask1',lung_mask1
#    print 'lung_mask_bmp',lung_mask_bmp
#    print 'lung_mask_bmp1',lung_mask_bmp1
    volumeroi={}
    tabroifinal={}
    for sln in lissln:
        volumeroi[sln]={} 
        for pat in usedclassif:
            volumeroi[sln][pat]=0
            
    for key in usedclassif:
        
        dirroi=os.path.join(pp,key)
        tabroifinal[key]=np.zeros((slnt,dimtabx,dimtaby,3), np.uint8)

        if key in classifcontour:        
#            dirroi=os.path.join(dirroi,lung_mask_bmp) 
            dirroil=os.path.join(dirpath_patient,lung_mask)
#            print '1',dirroil
            if not os.path.exists(dirroil):
                dirroil=os.path.join(dirpath_patient,lung_mask1)
#                print '2',dirroil
            if  os.path.exists(dirroil):               
                dirroi=os.path.join(dirroil,lung_mask_bmp)
#                print '3',dirroi
                if not os.path.exists(dirroi):
                    dirroi=os.path.join(dirroil,lung_mask_bmp1)
#                    print '4',dirroi
            else:
                dirroi=''
                    
#            print 'lung',dirroi
            
#        print key,dirroi
        if os.path.exists(dirroi):
#            print 'exist', dirroi
            listroi =[name for name in  os.listdir(dirroi) if name.lower().find('.'+typei1)>0 ]
            for roiimage in listroi:
#                    tabroi[key][sl]=genepoly(os.path.join(dirroi,roiimage))

                img=os.path.join(dirroi,roiimage)
                imageroi= cv2.imread(img,1)              
                imageroi=cv2.resize(imageroi,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)  

                cdelimter='_'
                extensionimage='.'+typei1
                slicenumber=rsliceNum(roiimage,cdelimter,extensionimage)
                if imageroi.max() >0:
                    tabroifinal[key][slicenumber]=imageroi
                    
                    pose=roiimage.find('.'+typei1)
                    roibmpfile=roiimage[0:pose]+'.'+typei1               
                        
    #                    print slicenumber ,roiimage,key
    #                tabg1=cv2.cvtColor(imageroi,cv2.COLOR_BGR2GRAY)
    #                tabgrey=cv2.cvtColor(tabgrey,cv2.COLOR_GRAY2BGR)
    #                np.putmask(tabgrey,tabgrey>0,classifc[key])
    #                cv2.imwrite(img,tabgrey)
                    sroiname=os.path.join(pathroi,roibmpfile)
    #                print sroiname
                    imageview=cv2.cvtColor(imageroi,cv2.COLOR_RGB2BGR)  
                      
                    tabgrey=cv2.cvtColor(imageview,cv2.COLOR_BGR2GRAY)
                    np.putmask(tabgrey,tabgrey>0,1)
                    area= tabgrey.sum()*pixelSpacing*pixelSpacing/100
                    volumeroi[slicenumber][key]=area   
#                    pickle.dump(volumeroi, open(path_data_writefile, "wb" ),protocol=-1)
                    tabroifinal[key][slicenumber]=imageview   

                    np.putmask(tabgrey, tabgrey >0, 100)
                    ctkey=drawcontours2(tabgrey,key,dimtabx,dimtaby)
                    anoted_image=cv2.imread(sroiname,1)
#                    plt.imshow(anoted_image)
#                    plt.imshow(ctkey)
#                    plt.show()
#                
                    anoted_image=cv2.add(anoted_image,ctkey)
                    cv2.imwrite(sroiname,anoted_image)
       

    return tabroifinal,volumeroi

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



def openfichierroi(patient,patient_path_complet,centerHU,limitHU):
    global dirpath_patient,dirroit,path_data_write,volumeroi,path_data_writefile,pixelSpacing
    global tabroifinal
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
    path_data_writefile=os.path.join(path_data_write,volumeroifile)

#    paramsaveDirf=os.path.join(paramsaveDir,paramname)
#
#    lisdirold=pickle.load(open( paramsaveDirf, "rb" ))
#
#    paramdict=pickle.load(open( paramsaveDirf, "rb" ))
    centerHU1=0
    limitHU1=0
    if os.path.exists(os.path.join(path_data_write,'centerHU')):
        centerHU1=pickle.load( open(os.path.join(path_data_write,'centerHU'), "rb" ))
    if os.path.exists(os.path.join(path_data_write,'limitHU')):
        limitHU1=pickle.load( open(os.path.join(path_data_write,'limitHU'), "rb" ))
        
    if centerHU1==centerHU and limitHU1==limitHU:
        print 'no need to regenerate'
        slnt=pickle.load( open(os.path.join(path_data_write,'slnt'), "rb" ))
        tabscanScan=pickle.load( open(os.path.join(path_data_write,'tabscanScan'), "rb" ))
        listsln=pickle.load( open(os.path.join(path_data_write,'listsln'), "rb" ))
        tabroifinal=pickle.load( open(os.path.join(path_data_write,'tabroifinal'), "rb" ))
        pixelSpacing=pickle.load( open(os.path.join(path_data_write,'pixelSpacing'), "rb" ))
        volumeroi=pickle.load( open(os.path.join(path_data_write,'volumeroi'), "rb" ))
    else:
        slnt,tabscanScan,listsln,pixelSpacing=genebmp(dirsource,nosource,dirroit,centerHU,limitHU)  
        tabroifinal,volumeroi,path_data_writefile=populate(dirpath_patient,listsln,slnt,pixelSpacing)
        pickle.dump(centerHU, open(os.path.join(path_data_write,'centerHU'), "wb" ),protocol=-1) 
        pickle.dump(limitHU, open(os.path.join(path_data_write,'limitHU'), "wb" ),protocol=-1) 
        pickle.dump(slnt, open(os.path.join(path_data_write,'slnt'), "wb" ),protocol=-1) 
        pickle.dump(tabscanScan, open(os.path.join(path_data_write,'tabscanScan'), "wb" ),protocol=-1) 
        pickle.dump(listsln, open(os.path.join(path_data_write,'listsln'), "wb" ),protocol=-1) 
        pickle.dump(tabroifinal, open(os.path.join(path_data_write,'tabroifinal'), "wb" ),protocol=-1) 
        pickle.dump(pixelSpacing, open(os.path.join(path_data_write,'pixelSpacing'), "wb" ),protocol=-1)
        pickle.dump(volumeroi, open(os.path.join(path_data_write,'volumeroi'), "wb" ),protocol=-1)
        
    dirsourcescan=os.path.join(dirsource,scan_bmp)

    
#    tabroifinal=np.zeros((slnt,512,512), np.uint8) 
#    tabroifinal,volumeroi,slnroi=generoi(dirf,tabroi,listsln,slnroi)
    initmenus(slnt,dirpath_patient)  
            
    loop(slnt,dirsourcescan,dirpath_patient,dirroit)
    return 'completed'

def openfichierroilung(patient,patient_path_complet,centerHU,limitHU):
    global dirpath_patient,dirroit,path_data_write,volumeroi,path_data_writefile,pixelSpacing
    global tabroifinal
    
    imwait=np.zeros((200,200,3),np.uint8)
    cv2.putText(imwait,'In Progress',(50, 50),cv2.FONT_HERSHEY_PLAIN,1,white,1 )
    cv2.imshow('wait',imwait)
    dirpath_patient=os.path.join(patient_path_complet,patient)
    dirsource=os.path.join(dirpath_patient,source_name)
    dirroit=os.path.join(dirpath_patient,roi_name)
    path_data_write=os.path.join(dirpath_patient,path_data)
    path_data_writefile=os.path.join(path_data_write,volumeroifile)
    if not os.path.exists(dirsource):
         os.mkdir(dirsource)
    if not os.path.exists(dirroit):
         os.mkdir(dirroit)
    listdcm=[name for name in os.listdir(dirsource) if name.find('.dcm')>0]
    nosource=True
    if len(listdcm)>0:
        nosource=False
    centerHU1=0
    limitHU1=0
    if os.path.exists(os.path.join(path_data_write,'centerHU')):
        centerHU1=pickle.load( open(os.path.join(path_data_write,'centerHU'), "rb" ))
    if os.path.exists(os.path.join(path_data_write,'limitHU')):
        limitHU1=pickle.load( open(os.path.join(path_data_write,'limitHU'), "rb" ))
        
    if centerHU1==centerHU and limitHU1==limitHU:
        print 'no need to regenerate'
        slnt=pickle.load( open(os.path.join(path_data_write,'slnt'), "rb" ))
        tabscanScan=pickle.load( open(os.path.join(path_data_write,'tabscanScan'), "rb" ))
        listsln=pickle.load( open(os.path.join(path_data_write,'listsln'), "rb" ))
        tabroifinal=pickle.load( open(os.path.join(path_data_write,'tabroifinal'), "rb" ))
        pixelSpacing=pickle.load( open(os.path.join(path_data_write,'pixelSpacing'), "rb" ))
        volumeroi=pickle.load( open(os.path.join(path_data_write,'volumeroi'), "rb" ))
        print 'load completed'
    else:
        slnt,tabscanScan,listsln,pixelSpacing=genebmp(dirsource,nosource,dirroit,centerHU,limitHU)
        tabroifinal,volumeroi=populate(dirpath_patient,listsln,slnt,pixelSpacing)


        pickle.dump(centerHU, open(os.path.join(path_data_write,'centerHU'), "wb" ),protocol=-1) 
        pickle.dump(limitHU, open(os.path.join(path_data_write,'limitHU'), "wb" ),protocol=-1) 
        pickle.dump(slnt, open(os.path.join(path_data_write,'slnt'), "wb" ),protocol=-1) 
        pickle.dump(tabscanScan, open(os.path.join(path_data_write,'tabscanScan'), "wb" ),protocol=-1) 
        pickle.dump(listsln, open(os.path.join(path_data_write,'listsln'), "wb" ),protocol=-1) 
        pickle.dump(tabroifinal, open(os.path.join(path_data_write,'tabroifinal'), "wb" ),protocol=-1) 
        pickle.dump(pixelSpacing, open(os.path.join(path_data_write,'pixelSpacing'), "wb" ),protocol=-1)
        pickle.dump(volumeroi, open(os.path.join(path_data_write,'volumeroi'), "wb" ),protocol=-1)
#    slnt,tabscanScan,listsln=genebmp(dirsource,nosource,dirroit,centerHU,limitHU)
    tabroifinal=genebmplung(dirsource,tabscanScan,slnt,listsln,tabroifinal,volumeroi)
#    createvolumeroi(dirpath_patient,listsln)
    
    initmenus(slnt,dirpath_patient)
#    menudraw(slnt)
#    populate(dirpath_patient)
    
    cv2.destroyAllWindows()
    return 'completed lung'

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