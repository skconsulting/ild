# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:21:22 2017

@author: sylvain
version 1.1
28 july 2017
"""
#from param_pix_p import *
from param_pix_p import path_data,datafrontn,datacrossn,dimpavx,dimpavy,surfelem,volelem,volumeroifile,avgPixelSpacing

from param_pix_p import white,red,yellow,grey,black
from param_pix_p import lungimage,source_name,sroi,sroi3d,scan_bmp,transbmp
from param_pix_p import typei,typei1,typei2
from param_pix_p import threeFileMerge,htmldir,threeFile,threeFile3d

from param_pix_p import classifc,classif,usedclassif

from param_pix_p import maxproba,excluvisu,fidclass,rsliceNum

from tdGenePredictGui import predictrun,calculSurface,uipTree

import cPickle as pickle
import os
import cv2
import numpy as np
import webbrowser

def lisdirprocess(d):
#    a=os.listdir(d)
    a= os.walk(d).next()[1]
#    print 'listdirprocess',a
    stsdir={}
    for dd in a:
        stpred={}
        ddd=os.path.join(d,dd)
        datadir=os.path.join(ddd,path_data)
        pathcross=os.path.join(datadir,datacrossn)
        pathfront=os.path.join(datadir,datafrontn)
        if os.path.exists(pathcross):
            stpred['cross']=True
        else:
             stpred['cross']=False
        if os.path.exists(pathfront):
            stpred['front']=True
        else:
            stpred['front']=False
        stsdir[dd]=stpred

    return a,stsdir

def predict(indata,path_patient):
    print 'module predict'
    listdir=[]
    nota=True
    try:
        listdiri= indata['lispatientselect']
    except KeyError:
            print 'No patient selected'
            nota=False
    if nota:
        predictrun(indata,path_patient)
        if type(listdiri)==unicode:
    #        print 'this is unicode'

            listdir.append(str(listdiri))
        else:
                listdir=listdiri
#    print 'lisdir from module after conv',listdir, type(listdir)
    return listdir

def opennew(dirk, fl,L):
    pdirk = os.path.join(dirk,L[fl])
    img = cv2.imread(pdirk,1)
    return img,pdirk

def nothings(x):
    global imgtext
    global dimtabx, dimtaby
    imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
    pass

def nothing(x):
    pass

def draw_circle(event,x,y,flags,img):
    global ix,iy,quitl,patchi
#    patchi=False

    if event == cv2.EVENT_RBUTTONDBLCLK:
        print x, y
    if event == cv2.EVENT_LBUTTONDOWN:

#        print('identification')
        ix,iy=x,y

        patchi=True
        print 'identification', ix,iy, patchi
        dxrect=(dimtaby/2)


        if x>dxrect and x<dxrect+40 and y>dimtabx-30 and y< dimtabx-10:
            print 'quit'
            ix,iy=x,y
            quitl=True


def contrasti(im,r):
#     tabi = np.array(im)
     r1=0.5+r/100.0
     im=im.astype(np.uint16)
     tabi1=im*r1
     tabi2=np.clip(tabi1,0,255)
     tabi3=tabi2.astype(np.uint8)
     return tabi3

def lumi(tabi,r):
    r1=r
    tabi=tabi.astype(np.uint16)
    tabi1=tabi+r1
    tabi2=np.clip(tabi1,0,255)
    tabi3=tabi2.astype(np.uint8)
    return tabi3

def addpatchn(col,lab, xt,yt,imgn):
#    print col,lab
    cv2.rectangle(imgn,(xt,yt),(xt+dimpavx,yt+dimpavy),col,1)
    return imgn

def tagviewn(fig,label,surface,surftot,roi,tl):
    """write text in image according to label and color"""

    col=classifc[label]
    labnow=classif[label]
    
    deltax=110
    deltay=10+(11*labnow)

    gro=0.8
    if surftot>0:
        pc=str(int(round(100.*surface/surftot,0)))
        pcroi=str(int(round(100.*roi/surftot,0)))
    else:
        pc='0'
        pcroi='0'
    if tl:
        cv2.rectangle(fig,(deltax-10, deltay-6),(deltax-5, deltay-1),col,1)
    cv2.putText(fig,label+' pre: '+str(surface)+'cm2 '+pc+'%'+' roi: '+str(roi)+'cm2 '+pcroi+'%',(deltax, deltay),cv2.FONT_HERSHEY_PLAIN,gro,col,1)


def drawpatch(datav,t,dx,dy,slicenumber,va,patch_list_cross_slice,patch_list_ref_slice,volumeroi):

    imgn = np.zeros((dx,dy,3), np.uint8)
   
    th=t/100.0
    numl=0
    listlabel={}
    listlabelaverage={}
    surflabel={}
    for pat in classif:
        surflabel[pat]=0
    surftot=len(patch_list_ref_slice[slicenumber])
    surftotf=surftot*surfelem
    surftot='surface totale :'+str(int(round(surftotf,0)))+'cm2'
    surftotpat=0
    lvexist=False
    if len (volumeroi)>0:
        lv=volumeroi[slicenumber]
        lvexist=True
    
#        print lv
    for ll in patch_list_cross_slice[slicenumber]:
            xpat=ll[0][0]
            ypat=ll[0][1]
        #we find max proba from prediction
            proba=ll[1]

            prec, mprobai = maxproba(proba)

            classlabel=fidclass(prec,classif)
            classcolor=classifc[classlabel]

            if mprobai >th and classlabel not in excluvisu and va[classlabel]==True:
                if classlabel in listlabel:
#                        print 'found'
                    numl=listlabel[classlabel]
                    listlabel[classlabel]=numl+1
                    surflabel[classlabel]= (numl+1)*surfelem
                    cur=listlabelaverage[classlabel]
                    averageproba= round((cur*numl+mprobai)/(numl+1),2)
                    listlabelaverage[classlabel]=averageproba
                else:
                    listlabel[classlabel]=1
                    surflabel[classlabel]=surfelem
                    listlabelaverage[classlabel]=mprobai

                imgn= addpatchn(classcolor,classlabel,xpat,ypat,imgn)
    delx=120
    for ll1 in usedclassif:               
        if ll1 in listlabel:
            tl=True
        else:
            tl=False
        sul=round(surflabel[ll1],1)
        if lvexist:
            suroi=round(lv[ll1],1)
        else:
            suroi=0
        surftotpat=surftotpat+surflabel[ll1]
        tagviewn(datav,ll1,sul,surftotf,suroi,tl)
    ts='Threshold:'+str(t)
    surfunkn=surftotf-surftotpat
    if surftotf>0:
        surfp=str(abs(round(100-(100.0*surftotpat/surftotf),1)))
    else:
        surfp='NA'
        
    sulunk='surface unknow :'+str(abs(round(surfunkn,1)))+'cm2 ='+surfp+'%'
    cv2.putText(datav,ts,(0,140),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
    cv2.putText(datav,surftot,(delx,140),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
    cv2.putText(datav,sulunk,(delx,150),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
    return imgn,datav


def retrievepatch(x,y,sln,dx,dy,patch_list_cross_slice):

    tabtext = np.zeros((dx,dy,3), np.uint8)
#    ill=-1
    pfound=False
    for ll in patch_list_cross_slice[sln]:

#            print ll
#            slicename=ll[0]
            xs=ll[0][0]
            ys=ll[0][1]
        #we find max proba from prediction
            
#            print xs,ys
            if x>xs and x < xs+dimpavx and y>ys and y<ys+dimpavy:
                     print x, y
                     proba=ll[1]
                     pfound=True

                     n=0
                     probatab={}
                     listproba=[]
                     for j in range (0,len(proba)):
                         probatab[fidclass(j,classif)]=proba[j]
#                     print probatab
                     for key, value in  probatab.items():
                         listproba.append( (key,value))
#                     print 'listed',listproba
                     lsorted= sorted(listproba,key=lambda std:std[1],reverse=True)
#                     print 'lsit sorted', lsorted
                     cv2.putText(tabtext,'X',(xs-5+dimpavx/2,ys+5+dimpavy/2),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

                     for j in range (0,3):
                             n=n+1
                             strw=lsorted[j][0]+ ' {0:.1f}%'.format(100*lsorted[j][1])
                             cv2.putText(tabtext,strw,(dy-142,(dx-40)+10*n),cv2.FONT_HERSHEY_PLAIN,0.7,(0,255,0),1)

                             print lsorted[j][0], ' {0:.2f}%'.format(100*lsorted[j][1])
                     print'found'
                     break
#    cv2.imshow('image',tabtext)
    if not pfound:
            print'not found'
    return tabtext



def colorimage(image,color):
#    im=image.copy()
    im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    np.putmask(im,im>0,color)
    return im

def findmaxvolume(dictSurf,poslung):
#    patmax=''
    patmax=''
    surfmax=0
    for pat in usedclassif :
        if pat not in excluvisu:
#            print pat
            if dictSurf[pat][poslung]>surfmax:
                surfmax=dictSurf[pat][poslung]
#                print surfmax,pat
                patmax=pat
    return surfmax,patmax

def initdictP(d, p):
    d[p] = {}
    d[p]['upperset'] = (0, 0)
    d[p]['middleset'] = (0, 0)
    d[p]['lowerset'] = (0, 0)
    d[p]['all'] = (0, 0)
    return d


def openfichiervolume(listHug,path_patient,patch_list_cross_slice,
                      lungSegment,tabMed,thrprobaUIP,patch_list_cross_slice_sub,slicepitch):
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichiervolume start',path_patient
    volelems=volelem*slicepitch # in mml
    print slicepitch
    quitl=False
    dirf=os.path.join(path_patient,listHug)
    dictP = {}  # dictionary with patch in lung segment
    dictPS = {}  # dictionary with total patch area in lung segment
    dictSubP = {}  # dictionary with patch in subpleural
    dictSurf = {}  # dictionary with patch volume in percentage
    dictpostotal={}

    dictPS['upperset'] = (0, 0)
    dictPS['middleset'] = (0, 0)
    dictPS['lowerset'] = (0, 0)
    dictPS['all'] = (0, 0)
    dictPS = calculSurface(dirf,patch_list_cross_slice, tabMed,lungSegment,dictPS)
    voltotal=int(round(((dictPS['all'][0]+dictPS['all'][1])*volelems),0))

    for patt in usedclassif:
        dictP = initdictP(dictP, patt)
        dictSubP = initdictP(dictSubP, patt)
          
    listPosLung=('left_sub_lower',  'left_sub_middle'  ,'left_sub_upper',
                 'right_sub_lower','right_sub_middle','right_sub_upper',
                 'left_lower','left_middle','left_upper',
                  'right_lower','right_middle','right_upper')
    
    cwd=os.getcwd()
    (cwdtop,tail)=os.path.split(cwd)

    path_img=os.path.join(cwdtop,lungimage)
#    print listHug
    lung_left=cv2.imread(os.path.join(path_img,'lung_left.bmp'),1)
    lung_right=cv2.imread(os.path.join(path_img,'lung_right.bmp'),1)

    lung_lower_right=cv2.imread(os.path.join(path_img,'lung_lower_right.bmp'),0)
    lung_middle_right=cv2.imread(os.path.join(path_img,'lung_middle_right.bmp'),0)
    lung_upper_right=cv2.imread(os.path.join(path_img,'lung_upper_right.bmp'),0)

    lung_lower_left=cv2.imread(os.path.join(path_img,'lung_lower_left.bmp'),0)
    lung_middle_left=cv2.imread(os.path.join(path_img,'lung_middle_left.bmp'),0)
    lung_upper_left=cv2.imread(os.path.join(path_img,'lung_upper_left.bmp'),0)

    lung_sub_lower_right=cv2.imread(os.path.join(path_img,'lung_sub_lower_right.bmp'),0)
    lung_sub_middle_right=cv2.imread(os.path.join(path_img,'lung_sub_middle_right.bmp'),0)
    lung_sub_upper_right=cv2.imread(os.path.join(path_img,'lung_sub_upper_right.bmp'),0)

    lung_sub_lower_left=cv2.imread(os.path.join(path_img,'lung_sub_lower_left.bmp'),0)
    lung_sub_middle_left=cv2.imread(os.path.join(path_img,'lung_sub_middle_left.bmp'),0)
    lung_sub_upper_left=cv2.imread(os.path.join(path_img,'lung_sub_upper_left.bmp'),0)


    dictPosImage={}
    dictPosTextImage={}

    dictPosImage['left']=lung_left
    dictPosImage['right']=lung_right
    np.putmask( dictPosImage['left'], dictPosImage['left']>0,255)
    np.putmask( dictPosImage['right'], dictPosImage['right']>0,255)

    dictPosImage['left_lower']=lung_lower_left
    dictPosImage['left_middle']=lung_middle_left
    dictPosImage['left_upper']=lung_upper_left

    dictPosImage['right_lower']=lung_lower_right
    dictPosImage['right_middle']=lung_middle_right
    dictPosImage['right_upper']=lung_upper_right

    dictPosImage['left_sub_lower']=lung_sub_lower_left
    dictPosImage['left_sub_middle']=lung_sub_middle_left
    dictPosImage['left_sub_upper']=lung_sub_upper_left

    dictPosImage['right_sub_lower']=lung_sub_lower_right
    dictPosImage['right_sub_middle']=lung_sub_middle_right
    dictPosImage['right_sub_upper']=lung_sub_upper_right

    dictPosTextImage['left_lower']=(480,570)
    dictPosTextImage['left_middle']=(480,400)
    dictPosTextImage['left_upper']=(450,215)

    dictPosTextImage['right_lower']=(170,570)
    dictPosTextImage['right_middle']=(170,400)
    dictPosTextImage['right_upper']=(200,215)

    dictPosTextImage['left_sub_lower']=(480,660)
    dictPosTextImage['left_sub_middle']=(625,380) #610
    dictPosTextImage['left_sub_upper']=(450,150)

    dictPosTextImage['right_sub_lower']=(170,660)
    dictPosTextImage['right_sub_middle']=(43,380)
    dictPosTextImage['right_sub_upper']=(200,150)

    dimtabx=lung_left.shape[0]
    dimtaby=lung_left.shape[1]
    imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)


    cv2.namedWindow('imageVol',cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("SliderVol",cv2.WINDOW_NORMAL)

    cv2.createTrackbar( 'Threshold','SliderVol',int(thrprobaUIP*100),100,nothing)
    cv2.createTrackbar( 'All','SliderVol',1,1,nothings)
    cv2.createTrackbar( 'Reset','SliderVol',0,1,nothings)
    viewasked={}
    for key in usedclassif:
        viewasked[key]=True
        cv2.createTrackbar( key,'SliderVol',0,1,nothings)
    imgbackg = np.zeros((dimtabx,dimtaby,3), np.uint8)
    posrc=0
    xr=800
    xrn=xr+10
    for key1 in usedclassif:
        yr=15*posrc

        yrn=yr+10
        cv2.rectangle(imgbackg, (xr, yr),(xrn,yrn), classifc[key1], -1)
        cv2.putText(imgbackg,key1,(xr+15,yr+10),cv2.FONT_HERSHEY_PLAIN,1.0,classifc[key1],1 )
        dictpostotal[key1]=(xr-135,yr+10)
        posrc+=1
    #for total volume
    posrc+=1
    yr=15*posrc
    yrn=yr+10

    dictpostotal['total']=(xr-135,yr+10)
    dictpostotal['totalh']=(xr-135,yr+25)
    
    dxrect=(dimtaby/2)
    cv2.rectangle(imgbackg,(dxrect,dimtabx-30),(dxrect+40,dimtabx-10),red,-1)
    #        cv2.rectangle(imgt,(172,358),(192,368),white,-1)
    cv2.putText(imgbackg,'quit',(dxrect+10,dimtabx-10),cv2.FONT_HERSHEY_PLAIN,1,yellow,1,cv2.LINE_AA)
    cv2.putText(imgbackg,'Patient Name:'+listHug,(50,30),cv2.FONT_HERSHEY_PLAIN,1.4,yellow,1,cv2.LINE_AA)
    imgbackg = cv2.add(imgbackg, dictPosImage['right'])
    imgbackg = cv2.add(dictPosImage['left'], imgbackg)
#    cv2.imshow('a',imgbackg)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    tlold=0
    viewaskedold={}
    for keyinit in usedclassif:
        viewaskedold[keyinit]=True
    while(1):
            drawok=False
#            print "corectnumber",corectnumber

            cv2.setMouseCallback('imageVol',draw_circle,imgtext)
            tl = cv2.getTrackbarPos('Threshold','SliderVol')
            allview = cv2.getTrackbarPos('All','SliderVol')
            noneview = cv2.getTrackbarPos('Reset','SliderVol')
            
            if allview==1:
                for key1 in usedclassif:
                    cv2.setTrackbarPos(key1,'SliderVol',0)
                    viewasked[key1]=True

            if noneview==1:
                for key1 in usedclassif:
                    cv2.setTrackbarPos(key1,'SliderVol',0)
                    viewasked[key1]=False
                cv2.setTrackbarPos('Reset','SliderVol',0) 
                cv2.setTrackbarPos('All','SliderVol',0)
                allview=0          
            
            if allview ==0:
                for key1 in usedclassif:
                    s = cv2.getTrackbarPos(key1,'SliderVol')  
                        
                    if s==1 and viewasked[key1]==False:
                        
                        viewasked[key1]=True
                        for key8 in usedclassif:
                             if key8!= key1:
                                cv2.setTrackbarPos(key8,'SliderVol',0)
                                viewasked[key8]=False                      
                  
            
            if tl != tlold:
                tlold=tl
                drawok=True   
            for keyne in usedclassif:
                if viewasked[keyne]!=viewaskedold[keyne]:
                    viewaskedold[keyne]=viewasked[keyne]
                    drawok=True
            if drawok:    
                print 'view'
#                     print key1
                dictP = {}  # dictionary with patch in lung segment    
                dictSubP = {}  # dictionary with patch in subpleural
                dictSurf = {}  # dictionary with patch volume in percentage
                for patt in usedclassif:
                    dictP = initdictP(dictP, patt)
                    dictSubP = initdictP(dictSubP, patt)
                thrprobaUIP=tl/100.0
    
                dictP, dictSubP, dictSurf= uipTree(dirf,patch_list_cross_slice,lungSegment,tabMed,dictPS,
                                                   dictP,dictSubP,dictSurf,thrprobaUIP,patch_list_cross_slice_sub)
                surfmax={}
                patmax={}
                
                imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
                img = np.zeros((dimtabx,dimtaby,3), np.uint8)
                cv2.putText(imgtext,'Threshold : '+str(tl),(50,50),cv2.FONT_HERSHEY_PLAIN,1,yellow,1,cv2.LINE_AA)
                if allview==1:
                    pervoltot=0
                    for patt in usedclassif:
                        vol=int(round(((dictP[patt]['all'][0]+dictP[patt]['all'][1] )*volelems),0))    
                        pervol=int(round(vol*100./voltotal,0))
                        pervoltot=pervoltot+pervol
                        
                        fillblank=''
                        if vol<10:
                            fillblank='   '
                        elif vol <100:
                            fillblank='  '
                        elif vol <1000:
                            fillblank=' '                        
    
    #                    print patt,vol
                        cv2.putText(imgtext,'V: '+str(vol)+'ml '+fillblank+str(pervol)+'%',(dictpostotal[patt][0],dictpostotal[patt][1]),
                                    cv2.FONT_HERSHEY_PLAIN,1.0,classifc[patt],1 )
                    #total volume
                    cv2.putText(imgtext,'Volume Total: '+str(voltotal)+'ml',(dictpostotal['total'][0],dictpostotal['total'][1]),
                                    cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
                    cv2.putText(imgtext,'Total unknown: '+str(100-pervoltot)+'%',(dictpostotal['totalh'][0],dictpostotal['totalh'][1]),
                                    cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
                    for i in listPosLung:
    
        #                dictPosImage[i]=colorimage(dictPosImage[i],black)
                        surfmax[i],patmax[i] =findmaxvolume(dictSurf,i)
    #                    print i,surfmax[i],patmax[i]
    
                        if surfmax[i]>0:
                             colori=classifc[patmax[i]]
                        else:
                             colori=grey
    
        #                print i,surfmax[i],patmax[i],colori
                        lungtw=colorimage(dictPosImage[i],colori)
    
                        img=cv2.add(img,lungtw)
    #                    cv2.imshow('a',imgbackg)
    #                    cv2.waitKey(0)
    #                    cv2.destroyAllWindows()
                        cv2.rectangle(imgtext,(dictPosTextImage[i][0],dictPosTextImage[i][1]-15),(dictPosTextImage[i][0]+55,dictPosTextImage[i][1]),white,-1)
                        cv2.putText(imgtext,str(surfmax[i])+'%',(dictPosTextImage[i][0],dictPosTextImage[i][1]),cv2.FONT_HERSHEY_PLAIN,1.2,grey,1,)
                else:
    
                    for i in listPosLung:
                        olo=0
                        for pat in usedclassif:
    #                        img = np.zeros((dimtabx,dimtaby,3), np.uint8)
                            if viewasked[pat]==True:
                                 olo+=1
                                 vol=int(round(((dictP[pat]['all'][0]+dictP[pat]['all'][1] )*volelems),0)) 
                                 pervol=int(round(vol*100./voltotal,0))
                                 fillblank=''
                                 if vol<10:
                                    fillblank='   '
                                 elif vol <100:
                                    fillblank='  '
                                 elif vol <1000:
                                    fillblank=' '
                                 cv2.putText(imgtext,'V: '+str(vol)+'ml '+fillblank+str(pervol)+'%',(dictpostotal[pat][0],
                                             dictpostotal[pat][1]),
                                             cv2.FONT_HERSHEY_PLAIN,1.0,classifc[pat],1 )
                                 if olo==1:
                                     if dictSurf[pat][i]>0:
                                         colori=classifc[pat]
                                     else:
                                         colori=grey
                                     lungtw=colorimage(dictPosImage[i],colori)
                                     img=cv2.add(img,lungtw)
                                     cv2.rectangle(imgtext,(dictPosTextImage[i][0],dictPosTextImage[i][1]-15),(dictPosTextImage[i][0]+55,dictPosTextImage[i][1]),white,-1)
                                     cv2.putText(imgtext,str(dictSurf[pat][i])+'%',(dictPosTextImage[i][0],dictPosTextImage[i][1]),cv2.FONT_HERSHEY_PLAIN,1.2,grey,1,)
                                 else:
                                     img = np.zeros((dimtabx,dimtaby,3), np.uint8)
                                     imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
                                 
    #                                 lungtw=colorimage(dictPosImage[i],grey)
    #                                 img=cv2.add(img,lungtw)
                    cv2.putText(imgtext,'Volume Total: '+str(voltotal)+'ml',(dictpostotal['total'][0],dictpostotal['total'][1]),
                                    cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
    
                imgtowrite=cv2.add(imgtext,imgbackg)
                imgray = cv2.cvtColor(imgtowrite,cv2.COLOR_BGR2GRAY)
                np.putmask(imgray,imgray>0,255)
                nthresh=cv2.bitwise_not(imgray)
                vis1=cv2.bitwise_and(img,img,mask=nthresh)
                imgtowrite=cv2.add(vis1,imgtowrite)
    #            imgtowrite=cv2.add(imgtext,imgtowrite)
                imgtowrite=cv2.cvtColor(imgtowrite,cv2.COLOR_BGR2RGB)
#            else:
#             print 'no view'
            cv2.imshow('imageVol',imgtowrite)

            if quitl or cv2.waitKey(20) & 0xFF == 27 :
    #            print 'on quitte', quitl
                break
    quitl=False
    #    print 'on quitte 2'
    cv2.destroyWindow("imageVol")
    cv2.destroyWindow("SliderVol")

    return ''


def openfichier(ti,datacross,path_img,thrprobaUIP,patch_list_cross_slice,patch_list_ref_slice):
    def writeslice(num,menus):
        print 'write',num
        cv2.rectangle(menus, (5,40), (150,30), red, -1)
        cv2.putText(menus,'Slice to visualize: '+str(num),(5,40),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
    
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichier start' 
    
    quitl=False
    
    slnt=datacross[0]
    dimtabx=datacross[1]
    dimtaby=datacross[2]

    patchi=False
    ix=0
    iy=0
    (top,tail)=os.path.split(path_img)
    volumeroilocal={}
    pdirk = os.path.join(path_img,source_name)
    pdirkroicross = os.path.join(path_img,sroi)
    pdirkroifront = os.path.join(path_img,sroi3d)

    if ti =="cross view" or ti =="merge view":
        if os.path.exists(pdirkroicross):
            pdirk = pdirkroicross
            path_data_write=os.path.join(path_img,path_data)
            path_data_writefile=os.path.join(path_data_write,volumeroifile)
            if os.path.exists(path_data_writefile):
                volumeroilocal=pickle.load(open(path_data_writefile, "rb" ))                
            sn=''
        else:
            sn=scan_bmp
        corectnumber=1
    else:        
        if os.path.exists(pdirkroifront):
            pdirk = pdirkroifront
            sn=''
        else:
            sn=transbmp
        corectnumber=0
    pdirk = os.path.join(pdirk,sn)

    list_image={}
    cdelimter='_'
    extensionimage='.'+typei
    limage=[name for name in os.listdir(pdirk) if name.find('.'+typei,1)>0 ]
    if len(limage)==0:
         limage=[name for name in os.listdir(pdirk) if name.find('.'+typei1,1)>0 ]
         extensionimage='.'+typei1
    if len(limage)==0:
         limage=[name for name in os.listdir(pdirk) if name.find('.'+typei2,1)>0 ]
         extensionimage='.'+typei2
        
    if ((ti =="cross view" or ti =="merge view") and len(limage)+1==slnt) or ti =="front view":

        for iimage in limage:

            sln=rsliceNum(iimage,cdelimter,extensionimage)
            list_image[sln]=iimage

        image0=os.path.join(pdirk,list_image[slnt/2])
        img = cv2.imread(image0,1)
        img=cv2.resize(img,(dimtaby,dimtabx),interpolation=cv2.INTER_LINEAR)
#        cv2.imshow('cont',img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
        

        cv2.namedWindow('imagecr',cv2.WINDOW_NORMAL)
        cv2.namedWindow("Slidercr",cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("datavisu",cv2.WINDOW_AUTOSIZE)

        cv2.createTrackbar( 'Brightness','Slidercr',0,100,nothing)
        cv2.createTrackbar( 'Contrast','Slidercr',50,100,nothing)
        cv2.createTrackbar( 'Threshold','Slidercr',int(thrprobaUIP*100),100,nothing)
        cv2.createTrackbar( 'Flip','Slidercr',slnt/2,slnt-2,nothings)
        cv2.createTrackbar( 'All','Slidercr',1,1,nothings)
        cv2.createTrackbar( 'None','Slidercr',0,1,nothings)
    
        
        viewasked={}
        for key1 in usedclassif:
#            print key1
            viewasked[key1]=True
            cv2.createTrackbar( key1,'Slidercr',0,1,nothings)
        nbdig=0
        numberentered={}
        initimg = np.zeros((dimtaby,dimtabx,3), np.uint8)
        slicenumberold=0
        tlold=0
        viewaskedold={}
        for keyne in usedclassif:
                viewaskedold[keyne]=False
        while(1):
            datav = np.zeros((200,500,3), np.uint8)           

            cv2.setMouseCallback('imagecr',draw_circle,img)
            c = cv2.getTrackbarPos('Contrast','Slidercr')
            l = cv2.getTrackbarPos('Brightness','Slidercr')
            tl = cv2.getTrackbarPos('Threshold','Slidercr')
            fl = cv2.getTrackbarPos('Flip','Slidercr')
            allview = cv2.getTrackbarPos('All','Slidercr')
            noneview = cv2.getTrackbarPos('None','Slidercr')
            
            key = cv2.waitKey(100) & 0xFF
            if key != 255:
                print key
            if key >47 and key<58:
                numberfinal=0
                knum=key-48
                print 'this is number',knum
                numberentered[nbdig]=knum
                nbdig+=1
    
                for i in range (nbdig):
                    numberfinal=numberfinal+numberentered[i]*10**(nbdig-1-i)            
#                print numberfinal
                numberfinal = min(slnt-2,numberfinal)
                if numberfinal>0:
                    writeslice(numberfinal,initimg)
                        
            if nbdig>0 and key ==8:          
                numberfinal=0
                nbdig=nbdig-1   
                for i in range (nbdig):    
                    numberfinal=numberfinal+numberentered[i]*10**(nbdig-1-i)            
    #            print numberfinal
                if numberfinal>0:
                    writeslice(numberfinal,initimg)
                else:
                    cv2.rectangle(initimg, (5,40), (150,30), black, -1)
            if nbdig>0 and key ==13 and numberfinal>0:
                    print numberfinal
                    numberfinal = min(slnt-2,numberfinal)
#                    writeslice(numberfinal,initimg)
                    cv2.rectangle(initimg, (5,40), (150,30), black, -1)
                    cv2.setTrackbarPos('Flip','Slidercr' ,numberfinal-1)
                    fl=numberfinal
                    numberfinal=0
                    nbdig=0
                    numberentered={}
            
            for key2 in usedclassif:
                s = cv2.getTrackbarPos(key2,'Slidercr')
                if allview==1:
                     viewasked[key2]=True
                elif noneview ==1:
                    viewasked[key2]=False
                elif s==0:
#            print key
                    viewasked[key2]=False
                else:
                     viewasked[key2]=True
            
            slicenumber=fl+corectnumber
            
            imagel=os.path.join(pdirk,list_image[slicenumber])
            img = cv2.imread(imagel,1)               
            img=cv2.resize(img,(dimtaby,dimtabx),interpolation=cv2.INTER_LINEAR)
            img=cv2.add(img,initimg)

            imglumi=lumi(img,l)
            imcontrast=contrasti(imglumi,c)                
            imcontrast=cv2.cvtColor(imcontrast,cv2.COLOR_BGR2RGB)
            drawok=False
            if slicenumber != slicenumberold:
                slicenumberold=slicenumber
                drawok=True
            if tl != tlold:
                tlold=tl
                drawok=True   
            for keyne in usedclassif:
                if viewasked[keyne]!=viewaskedold[keyne]:
                    viewaskedold[keyne]=viewasked[keyne]
#                    print 'change'
                    drawok=True
                    break            
            if drawok:
                imgn,datav= drawpatch(datav,tl,dimtabx,dimtaby,slicenumber,viewasked,patch_list_cross_slice,patch_list_ref_slice,volumeroilocal)
            
            cv2.putText(datav,'slice number :'+str(slicenumber),(10,180),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
            cv2.putText(datav,'patient Name :'+tail,(10,190),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)

            imgngray = cv2.cvtColor(imgn,cv2.COLOR_BGR2GRAY)
            np.putmask(imgngray,imgngray>0,255)
            mask_inv = cv2.bitwise_not(imgngray)
            outy=cv2.bitwise_and(imcontrast,imcontrast,mask=mask_inv)
            imgt=cv2.add(imgn,outy)
            dxrect=(dimtaby/2)
            cv2.rectangle(imgt,(dxrect,dimtabx-30),(dxrect+20,dimtabx-10),red,-1)
            cv2.putText(imgt,'quit',(dxrect+10,dimtabx-10),cv2.FONT_HERSHEY_PLAIN,1,yellow,1,cv2.LINE_AA)
            imgtoshow=cv2.add(imgt,imgtext)
            imgtoshow=cv2.cvtColor(imgtoshow,cv2.COLOR_BGR2RGB)
            datav=cv2.cvtColor(datav,cv2.COLOR_BGR2RGB)
            cv2.imshow('imagecr',imgtoshow)
            cv2.imshow('datavisu',datav)

            if patchi :
                print 'retrieve patch asked'
                imgtext= retrievepatch(ix,iy,slicenumber,dimtabx,dimtaby,patch_list_cross_slice)
                patchi=False

            if quitl or cv2.waitKey(20) & 0xFF == 27 :
                break
        quitl=False
        cv2.destroyWindow("imagecr")
        cv2.destroyWindow("Slidercr")
        cv2.destroyWindow("datavisu")

        return ''
    else:
        print 'error in the number of scan images compared to dicom numbering'
        return 'error in the number of scan images compared to dicom numbering'



def openfichierfrpr(path_patient,tabfromfront,thrprobaUIP):
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichierfrpr start',path_patient
#    dirf=os.path.join(path_patient,listHug)
   
    slnt=len(tabfromfront)   
    print slnt
    quitl=False

    dimtabx=tabfromfront[0].shape[0]
    dimtaby=tabfromfront[0].shape[1]
    imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)


    cv2.namedWindow('imagefrpr',cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("SliderVolumefrpr",cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar( 'Flip','SliderVolumefrpr',slnt/2,slnt-2,nothings)

    imgbackg = np.zeros((dimtabx,dimtaby,3), np.uint8)
    posrc=0
    for key1 in usedclassif:
        xr=dimtabx-100
        yr=15*posrc
        xrn=xr+10
        yrn=yr+10
        cv2.rectangle(imgbackg, (xr, yr),(xrn,yrn), classifc[key1], -1)
        cv2.putText(imgbackg,key1,(xr+15,yr+10),cv2.FONT_HERSHEY_PLAIN,1.0,classifc[key1],1 )
        posrc+=1

    dxrect=(dimtaby/2)
    cv2.rectangle(imgbackg,(dxrect,dimtabx-30),(dxrect+40,dimtabx-10),red,-1)
    #        cv2.rectangle(imgt,(172,358),(192,368),white,-1)
    cv2.putText(imgbackg,'quit',(dxrect+10,dimtabx-10),cv2.FONT_HERSHEY_PLAIN,1,yellow,1,cv2.LINE_AA)


    while(1):
#            print "corectnumber",corectnumber
            imgtext= np.zeros((dimtabx,dimtaby,3), np.uint8)
            cv2.setMouseCallback('imagefrpr',draw_circle,imgtext)
            fl = cv2.getTrackbarPos('Flip','SliderVolumefrpr')
            img=tabfromfront[fl+1]
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            imgtext=cv2.add(imgtext,imgbackg)
            imgtext=cv2.add(imgtext,img)
            imgtowrite=cv2.cvtColor(imgtext,cv2.COLOR_BGR2RGB)
            cv2.imshow('imagefrpr',imgtowrite)
#            cv2.imshow('image',vis1)

            if quitl or cv2.waitKey(20) & 0xFF == 27 :
    #            print 'on quitte', quitl
                break
    quitl=False
    #    print 'on quitte 2'
    cv2.destroyWindow("imagefrpr")
    cv2.destroyWindow("SliderVolumefrpr")
    return ''

def visuarun(indata,path_patient):

    messageout=""
#    print 'path_patient',path_patient
    lpt=indata['lispatientselect']
    pos=lpt.find(' PREDICT!:')
    if pos >0:
            listHug=(lpt[0:pos])
    else:
            pos=str(indata).find(' noPREDICT!')
#            print 'no predict'
            listHug=(lpt[0:pos])
            messageout="no predict!for "+listHug
            return messageout

    patient_path_complet=os.path.join(path_patient,listHug)

    path_data_dir=os.path.join(patient_path_complet,path_data)
    viewstyle=indata['viewstyle']

    pathhtml=os.path.join(patient_path_complet,htmldir)
    if viewstyle=='from cross predict':
            namefilehtml=listHug+'_'+threeFile
            viewfilehtmlcomplet=os.path.join(pathhtml,namefilehtml)
            url = 'file://' + viewfilehtmlcomplet
            webbrowser.open(url, new=2)

    elif viewstyle=='from front predict':
           namefilehtml=listHug+'_'+threeFile3d
           viewfilehtmlcomplet=os.path.join(pathhtml,namefilehtml)
           url = 'file://' + viewfilehtmlcomplet
           webbrowser.open(url, new=2)

    elif viewstyle=='from cross + front merge':
            namefilehtml=listHug+'_'+threeFileMerge
            viewfilehtmlcomplet=os.path.join(pathhtml,namefilehtml)
            url = 'file://' + viewfilehtmlcomplet
            webbrowser.open(url, new=2)

    elif viewstyle=='cross view':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
            patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
            thrprobaUIP=float(indata['thrprobaUIP'])
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,patch_list_cross_slice,patch_list_cross_slice)
            
    elif viewstyle=='front view':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datafront"), "rb" ))
            patch_list_front_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_front_slice"), "rb" ))
            thrprobaUIP=float(indata['thrprobaUIP'])
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,patch_list_front_slice,patch_list_front_slice)
            
    elif viewstyle=='volume view from cross':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
            slicepitch=datarep[3]
            patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
            patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_sub"), "rb" ))
            lungSegment= pickle.load( open( os.path.join(path_data_dir,"lungSegment"), "rb" ))
            tabMed= pickle.load( open( os.path.join(path_data_dir,"tabMed"), "rb" ))
            thrprobaUIP=float(indata['thrprobaUIP'])
            
            messageout = openfichiervolume(listHug,path_patient,patch_list_cross_slice,
                      lungSegment,tabMed,thrprobaUIP,patch_list_cross_slice_sub,slicepitch)
    
    elif viewstyle=='volume view from front':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
            slicepitch=avgPixelSpacing
            patch_list_front_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_front_slice"), "rb" ))
            patch_list_front_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_front_slice_sub"), "rb" ))
            lungSegmentfront= pickle.load( open( os.path.join(path_data_dir,"lungSegmentfront"), "rb" ))
            tabMedfront= pickle.load( open( os.path.join(path_data_dir,"tabMedfront"), "rb" ))
            thrprobaUIP=float(indata['thrprobaUIP'])
            
            messageout = openfichiervolume(listHug,path_patient,patch_list_front_slice,
                      lungSegmentfront,tabMedfront,thrprobaUIP,patch_list_front_slice_sub,slicepitch)

    elif viewstyle=='merge view':

            thrprobaUIP=float(indata['thrprobaUIP'])
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
#            patch_list_merge= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge"), "rb" ))
#            proba_merge= pickle.load( open( os.path.join(path_data_dir,"proba_merge"), "rb" ))
            patch_list_merge_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge_slice"), "rb" ))
            patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,patch_list_merge_slice,patch_list_cross_slice)
    
    elif viewstyle=='front projected view':

            thrprobaUIP=float(indata['thrprobaUIP'])
            tabfromfront= pickle.load( open( os.path.join(path_data_dir,"tabfromfront"), "rb" ))
            messageout=openfichierfrpr(path_patient,tabfromfront,thrprobaUIP)
    
    else:
            messageout='error: unrecognize view style'

    return messageout


#
####
#indata={'lispatient':'36 PREDICT!:  Cross','viewstyle':'volume view','thrpatch':0.8,'thrproba':0.9,'thrprobaUIP':0.9,'thrprobaMerge':0.9,
#        'picklein_file':"pickle_ex80",'picklein_file_front':"pickle_ex81",'23':'on','threedpredictrequest':'Cross Only','subErosion':15
#        }
#path_patient='C:/Users/sylvain/Documents/boulot/startup/radiology/predicttool/patient_directory'
####
#visuarun(indata,path_patient)
##predict(indata,path_patient)
