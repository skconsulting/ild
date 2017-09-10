# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:21:22 2017

@author: sylvain Kritter 

version 1.5
6 September 2017
"""
#from param_pix_p import *
from param_pix_s2 import path_data,reportalldir
from param_pix_s2 import surfelemp,volumeroifilep

from param_pix_s2 import white,red,yellow,black
from param_pix_s2 import source_name,sroi,sroi3d,scan_bmp,transbmp
from param_pix_s2 import typei1,dimtabx,dimtaby
from param_pix_s2 import reportdir,reportfile

from param_pix_s2 import classifc,classifdict,usedclassifdict,oldFormat

from param_pix_s2 import excluvisu,fidclass,rsliceNum,evaluate,evaluatef,evaluatefull,normi

from scorepredict2 import predictrun

import cPickle as pickle
import os
import cv2
import numpy as np
import datetime
#import time


def lisdirprocess(d):
#    a=os.listdir(d)
    a= os.walk(d).next()[1]
#    print 'listdirprocess',a
    if reportalldir in a:
        a.remove(reportalldir)
#    print 'listdirprocess',a
    stsdir={}
    setrefdict={}
    for dd in a:

            stpred={}
            ddd=os.path.join(d,dd)
            datadir=os.path.join(ddd,path_data)
            pathcross=os.path.join(datadir,'datacrosss')
            
            pathfront=os.path.join(datadir,'datafrontns')
            setrefdict[dd]='no'
            if os.path.exists(pathcross):
                datacross= pickle.load( open( os.path.join(datadir,"datacrosss"), "rb" ))
#                print datacross
                try:
                    setrefdict[dd]=datacross[3]
                except:
                    setrefdict[dd]='oldset0'
            if os.path.exists(pathcross):
                stpred['cross']=True
                
            else:
                 stpred['cross']=False
            if os.path.exists(pathfront):
                stpred['front']=True
            else:
                stpred['front']=False
                      
            crosscompletedf=os.path.join(datadir,'crosscompleted')
            frontcompletedf=os.path.join(datadir,'frontcompleted')
            if os.path.exists(crosscompletedf):
                crosscompleted= pickle.load( open(crosscompletedf, "rb") )
                if crosscompleted:
                    stpred['cross']=True
                else:
                    stpred['cross']=False
                    
            if os.path.exists(frontcompletedf):
                frontcompleted= pickle.load( open(frontcompletedf , "rb") )
                if frontcompleted:
                    stpred['front']=True
                else:
                    stpred['front']=False                    

            stsdir[dd]=stpred
#    print setrefdict
    return a,stsdir,setrefdict

def predictmodule(indata,path_patient):
#    print 'module predict'
    message=''
    listdir=[]
    nota=True
    try:
        listdiri= indata['lispatientselect']
    except KeyError:
            print 'No patient selected'
            nota=False
    if nota:
        message=predictrun(indata,path_patient)

        if type(listdiri)==unicode:
    #        print 'this is unicode'

            listdir.append(str(listdiri))
        else:
                listdir=listdiri
#    print 'lisdir from module after conv',listdir, type(listdir)
#    print 'message in predict module',message
    return listdir,message

def opennew(dirk, fl,L):
    pdirk = os.path.join(dirk,L[fl])
    img = cv2.imread(pdirk,1)
    return img,pdirk

def nothings(x):
    global imgtext
    imgtext = np.zeros((dimtaby,dimtabx,3), np.uint8)
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
#    print label,roi
    if roi>0:
        cv2.putText(fig,'%15s'%label+'%10s'%(str(surface)+'cm2')+
                    '%5s'%(pc+'%')+' roi: '+'%10s'%(str(roi)+'cm2')+'%5s'%(pcroi+'%'),
                (deltax, deltay),cv2.FONT_HERSHEY_PLAIN,gro,col,1)
    else:
        cv2.putText(fig,'%15s'%label+'%10s'%(str(surface)+'cm2')+'%5s'%(pc+'%'),
                (deltax, deltay),cv2.FONT_HERSHEY_PLAIN,gro,col,1)
            

def tagvaccuracy(fig,label,precision,recall,fscore):
    """write text in image according to label and color"""

    col=classifc[label]
    labnow=classif[label]
    
    deltax=480
    deltay=10+(11*labnow)
    gro=0.8
    cv2.putText(fig,'precision: '+'%4s'%(str(precision)+'%')+' recall: '+
                '%4s'%(str(recall)+'%')+' Fscore: '+'%4s'%(str(fscore)+'%'),
                (deltax, deltay),cv2.FONT_HERSHEY_PLAIN,gro,col,1)

def tagvcm(fig,cm):
    n= cm.shape[0]

    deltax=10
    deltay=300
    gro=0.7
    deltaxm=110 
    deltaxt=110
    for pat in classif:
        cp=classif[pat]
#        if cp==lcl:
#            dy=0
#        elif cp==0:
#            dy=lcl
#        else:
        dy=cp
        col=classifc[pat]

        cv2.putText(fig,'%15s'%pat,(deltax, deltay+(12*dy)),cv2.FONT_HERSHEY_PLAIN,gro,col,1)
        cv2.putText(fig,pat[0:7],(deltaxt+(60*dy), deltay-15),cv2.FONT_HERSHEY_PLAIN,gro,col,1)

     
    for i in range (0,n):
        for j in range (0,n):
            dx=deltaxm+60*j
            dy=deltay+12*i
            cv2.putText(fig,'%5s'%str(cm[i][j]),(dx, dy),cv2.FONT_HERSHEY_PLAIN,gro,white,1)




def drawpatch(t,dx,dy,slicenumber,va,patch_list_cross_slice,volumeroi,
              slnt,tabroi,num_class,tabscanLung,xtrains):

    imgn = np.zeros((dy,dx,3), np.uint8)
    datav = np.zeros((500,900,3), np.uint8)
#    patchdict=np.zeros((slnt,dx,dy), np.uint8)  
#    np.putmask(patchdict,patchdict==0,classif['lung'])
#    imgpatch=np.zeros((dx,dy), np.uint8)  
    predictpatu=np.zeros((dx,dy), np.uint8) 
    referencepatu=np.zeros((dx,dy), np.uint8) 
   
    th=t/100.0
#    numl=0
    listlabel={}

    surftot=np.count_nonzero(tabscanLung[slicenumber])

    surftotf= surftot*surfelemp/100
    surftot='surface totale :'+str(int(round(surftotf,0)))+'cm2'
    volpat={}
    for pat in usedclassif:
        volpat[pat]=np.zeros((dy,dx), np.uint8)
    
    
    lvexist=False
    if len (volumeroi)>0:
#        print 'volumeroi exists'
        lv=volumeroi[slicenumber]
        for pat,value in lv.items():
#            print pat,value
            if value>0:
                lvexist=True
                break

    imi=  patch_list_cross_slice[xtrains[slicenumber]]
#    print imi.shape
    imclass0=np.argmax(imi, axis=2).astype(np.uint8)
#    cv2.imwrite('a.bmp',normi(imclass0))
#    print imclass0.shape,imclass0.min(),imclass0.max()
    imamax=np.amax(imi, axis=2)

    np.putmask(imamax,imamax>=th,255)
    np.putmask(imamax,imamax<th,0)
    imamax=imamax.astype(np.uint8)
#    print imamax.min(),imamax.max()
#        print type(imamax[0][0])
    imclass=np.bitwise_and(imamax,imclass0)
    imclassc = np.expand_dims(imclass,2)

    imclassc=np.repeat(imclassc,3,axis=2) 
    patlist=[]
    
    for key,value in classif.items():       
        if key not in excluvisu and va[key]:
            imgnp=imclassc.copy()            
            bl=(value,value,value)
            blc=[]
            zz=classifc[key]
#                print zz
            for z in range(3):
                blc.append(int(zz[z]*0.3))

            np.putmask(imgnp,imgnp!=bl,black)

            np.putmask(imgnp,imgnp==bl,blc)
            imgn=cv2.add(imgn,imgnp)
            imgngray = cv2.cvtColor(imgnp,cv2.COLOR_BGR2GRAY)
            np.putmask(imgngray,imgngray>0,1)
#            area= imgngray.sum()* surfelemp /100
#            print key, imgn.min(),imgn.max()
            volpat[key]=imgngray
            if imgn.max()>0:
                patlist.append(key)


    delx=120
    surftotpat=0
    tablung=np.copy(tabscanLung[slicenumber])
    np.putmask(tablung,tablung>0,255)
        
    for ll1 in usedclassif:     
        volpat[ll1]=np.bitwise_and(tablung, volpat[ll1])   
#        if ll1 =='ground_glass':
#            cv2.imshow('ground_glass',normi(volpat[ll1]))
            
        if ll1 in listlabel:            
            tl=True
        else:
            tl=False
        sul=round(np.count_nonzero(volpat[ll1])*surfelemp/100,1)
        if lvexist:
            suroi=round(lv[ll1],1)
        else:
            suroi=0
        surftotpat=surftotpat+sul
        tagviewn(datav,ll1,sul,surftotf,suroi,tl)
    ts='Threshold:'+str(t)
    surfunkn=surftotf-surftotpat
    if surftotf>0:
        surfp=str(abs(round(100-(100.0*surftotpat/surftotf),1)))
    else:
        surfp='NA'
    if surfunkn>0:
        sulunk='surface unknow :'+str(abs(round(surfunkn,1)))+'cm2 ='+surfp+'%'
    else:
        sulunk=''
    cv2.putText(datav,ts,(0,140),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
    cv2.putText(datav,surftot,(delx,140),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
    cv2.putText(datav,sulunk,(delx,150),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
    
    if lvexist:
        tablung=np.copy(tabscanLung[slicenumber])
        np.putmask(tablung,tablung>0,255)
                  
        predictpatu=np.bitwise_and(tablung, imclass0)  
        referencepatu= np.copy(tabroi[slicenumber]) 
        
        referencepat= referencepatu.flatten()            
        predictpat=  predictpatu.flatten()   
#        pc=np.copy(predictpatu)
#        print 'slicenumber',slicenumber
#        print'pred', pc[318][359]
#        print 'ref',tabroi[slicenumber][318][359]
#        print 'predict',pc.min(),pc.max()
#        np.putmask(pc,pc!=2,0)
#        np.putmask(pc,pc==2,200)
#
##        pcc=np.copy(pc)
##        np.putmask(pcc,pcc==0,50)
#        cv2.imshow('predictpat',pc)
#        
#        pc1=np.copy(referencepatu)
#        print 'ref',pc1.min(),pc1.max()
#        np.putmask(pc1,pc1!=2,0)
#        np.putmask(pc1,pc1==2,200)
#
#        print 'refm',pc1.min(),pc1.max()
##        pcc=np.copy(pc)
##        np.putmask(pcc,pcc==0,50)
#        cv2.imshow('refpat',pc1)
       
        precision={}
        recall={}
        fscore={}
        cf=False
        for pat in usedclassif:
            numpat=classif[pat]
            fscore[pat]=0
#            if numpat == 0:
#                numpat=classif['lung']
            precision[pat],recall[pat] = evaluate(referencepat,predictpat,num_class,(numpat,))
            if (precision[pat]+recall[pat])>0:
                fscore[pat]=2*precision[pat]*recall[pat]/(precision[pat]+recall[pat])
                precisioni=int(round(precision[pat]*100,0))
                recalli=int(round(recall[pat]*100,0))
                fscorei=int(round(fscore[pat]*100,0))
                tagvaccuracy(datav,pat,precisioni,recalli,fscorei)
                cf=True       
        
        if cf:
#            print 'num_class',num_class
            cm=evaluatef(referencepat,predictpat,num_class)
            tagvcm(datav,cm)
    datav=cv2.cvtColor(datav,cv2.COLOR_BGR2RGB)
#    cv2.imwrite('b.bmp',imgn)
    return imgn,datav


def retrievepatch(x,y,slicenumber,patch_list_cross_slice,xtrains):

    tabtext = np.zeros((dimtaby,dimtabx,3), np.uint8)
    #    ill=-1

    listproba=[]
    probatab={}
    imi=  patch_list_cross_slice[xtrains[slicenumber]]
    proba= imi[y][x]
      
    for j in range (0,len(proba)):
         probatab[fidclass(j,classif)]=proba[j]
    #                     print probatab
    for key, value in  probatab.items():
         listproba.append( (key,value))
    #                     print 'listed',listproba
    lsorted= sorted(listproba,key=lambda std:std[1],reverse=True)
    #                     print 'lsit sorted', lsorted
    cv2.putText(tabtext,'X',(x-5,y+5),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
    n=0
    for j in range (0,3):
             n=n+1
             strw=lsorted[j][0]+ ' {0:.1f}%'.format(100*lsorted[j][1])
             cv2.putText(tabtext,strw,(dimtaby-142,(dimtabx-40)+10*n),cv2.FONT_HERSHEY_PLAIN,0.7,(0,255,0),1)
    
             print lsorted[j][0], ' {0:.2f}%'.format(100*lsorted[j][1])

    return tabtext


def openfichiervolumetxtall(listHug,path_patient,indata,thrprobaUIP,cnnweigh,f,tp,xtrains):

    num_class=len(classif)

    f.write('Score for list of patients:\n')
    for patient in listHug:
        f.write(str(patient)+' ')
    f.write('\n-----\n')
    pf=True
    for patient in listHug:
        print 'work on :',patient

        patient_path_complet=os.path.join(path_patient,patient)
        path_data_dir=os.path.join(patient_path_complet,path_data)
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss"), "rb" ))
#        slicepitch=datarep[3]
        slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois"), "rb" ))
        print 'slnroi,patient',slnroi,patient
        if tp=='Cross':
            filepatch="proba_crosss"
        if tp=='FrontProjected':
            filepatch="proba_fronts"
        if tp=='Merge':
            filepatch="proba_merges"
            
        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,filepatch), "rb" ))
#        patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_sub"), "rb" ))
#        lungSegment= pickle.load( open( os.path.join(path_data_dir,"lungSegment"), "rb" ))
#        tabMed= pickle.load( open( os.path.join(path_data_dir,"tabMed"), "rb" ))
        tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs"), "rb" ))
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois"), "rb" ))
                  
        ref,pred,messageout = openfichiervolumetxt(patient,path_patient,patch_list_cross_slice,
                      thrprobaUIP,tabroi,datarep,slnroi,tabscanLung,f,cnnweigh,tp,xtrains)

        if pf:

            referencepat=ref
            predictpat=pred
            pf=False
        else:           

            referencepat= np.concatenate((referencepat,ref),axis=0)
            predictpat= np.concatenate((predictpat,pred),axis=0)

        
    cfma(f,referencepat,predictpat,num_class,'all set',thrprobaUIP,cnnweigh,tp)
    f.close()

def cfma(f,referencepat,predictpat,num_class, namep,thrprobaUIP,cnnweigh,tp):
        f.write('confusion matrix for '+namep+'\n')
        f.write(tp+' View , threshold: '+str(thrprobaUIP)+ ' CNN param: '+cnnweigh+'\n\n')
        
        cm=evaluatef(referencepat,predictpat,num_class)
        n= cm.shape[0]

        presip={}
        recallp={}
        fscorep={}
        usedclassifwobg=list(usedclassif)
        usedclassifwobg.remove('back_ground')
        for pat in usedclassif:
            presip[pat]=0
            recallp[pat]=0
            fscorep[pat]=0
            numpat=classif[pat]

            presip[pat], recallp[pat]=evaluate(referencepat,predictpat,num_class,(numpat,))

        for pat in usedclassif:             
         if presip[pat]+recallp[pat]>0:
            fscorep[pat]=2*presip[pat]*recallp[pat]/(presip[pat]+recallp[pat])
        else:
            fscorep[pat]=0    
        f.write(15*' ')
        for pat in usedclassif:
            f.write('%8s'%pat[0:6])
    #    print newclassif
        f.write('  recall \n')
        for i in range (0,n):
            pat=usedclassif[i]
            f.write('%15s'%pat)
            for j in range (0,n):
                f.write('%8s'%str(cm[i][j]))
            f.write( '%8s'%(str(int(round(100*recallp[pat],0)))+'%')+'\n')        
        f.write('----------------------\n')
        f.write('      precision')
        for pat in usedclassif:
            f.write('%8s'%(str(int(round(100*presip[pat],0)))+'%'))
    #    print newclassif
        f.write('\n')
        f.write('         Fscore')
        for pat in usedclassif:
            f.write('%8s'%(str(int(round(100*fscorep[pat],0)))+'%'))
    #    print newclassif
        f.write('\n--------------------\n')
        f.write('score per pattern:\n\n')
        for pat in usedclassifwobg:            
            if  fscorep[pat]>0:
                f.write('%17s'%pat+
                        '  precision:'+'%5s'%(str(int(round(100*presip[pat],0)))+'%')+
                        ' recall: '+'%5s'%(str(int(round(100*recallp[pat],0)))+'%')+
                        ' fscore:'+'%5s'%(str(int(round(100*fscorep[pat],0)))+'%')+'\n')
                  
        precisiont,recallt= evaluatefull(referencepat,predictpat,num_class)
        if precisiont+recallt>0:
            fscore=2*precisiont*recallt/(precisiont+recallt)
        else:
            fscore=0
        precisioni=str(int(round(precisiont*100,0)))+'%'
        recalli=str(int(round(recallt*100,0)))+'%'
        fscorei=str(int(round(fscore*100,0)))+'%'
    #                    print (slicenumber,pat,precisioni,recalli,fscorei)
        f.write('----------------------\n')
        f.write('Global scores for patient '+namep+' (without back_ground):\n')
        f.write('precision: '+precisioni+' recall: '+recalli+' Fscore: '+fscorei+'\n')
        f.write('----------------------\n')

def openfichiervolumetxt(listHug,path_patient,patch_list_cross_slice,
                      thrprobaUIP,tabroi,datacross,slnroi,tabscanLung,f,cnnweigh,tp,xtrains):
    global  quitl,patchi,ix,iy
    print 'openfichiervolume txt start in',path_patient,' for', listHug, 'predict type ',tp
    slnt=datacross[0]
#    dx=datacross[1]
#    dy=datacross[2]

    t = datetime.datetime.now()
 
    f.write('report for patient :'+listHug+
            ' - date : m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+' at '+str(t.hour)+'h '+str(t.minute)+'mn\n')
    f.write(tp +' View , threshold: '+str(thrprobaUIP)+ ' CNN_param: '+cnnweigh +'\n\n')
#    print 'slnroi',slnroi
    slntroi=len(slnroi)
#    print 'slnroi',slnroi,slntroi

    print 'start Fscore'
    patchdict=np.zeros((slnt,dimtaby,dimtabx), np.uint8)
    predictpatu=np.zeros((slntroi,dimtaby,dimtabx), np.uint8)
    referencepatu=np.zeros((slntroi,dimtaby,dimtabx), np.uint8)
    num_class=len(classif)
    th=thrprobaUIP
    for slroi in range (0,slntroi):
#        print 'slroi',slroi
##    for slicenumber in (143,144):
        slicenumber=slnroi[slroi]
#        print slicenumber
#        print 'xtrains[slicenumber]',xtrains[slicenumber]
        if tabroi[slicenumber].max()>0:
            imi=  patch_list_cross_slice[slroi]
#            print imi.shape
            imclass0=np.argmax(imi, axis=2).astype(np.uint8)
            imamax=np.amax(imi, axis=2)

            np.putmask(imamax,imamax>=th,255)
            np.putmask(imamax,imamax<th,0)
            imamax=imamax.astype(np.uint8)

            patchdict[slicenumber]=np.bitwise_and(imamax,imclass0)

            tablung=np.copy(tabscanLung[slicenumber])
            np.putmask(tablung,tablung>0,255)
              
            predictpatu[slroi]=np.bitwise_and(tablung, patchdict[slicenumber])  
            referencepatu[slroi]=np.copy(tabroi[slicenumber])
           
            referencepat= referencepatu[slroi].flatten()
            predictpat=  predictpatu[slroi].flatten()   
            
            precision={}
            recall={}
            fscore={}
            f.write('Slice :'+str(slicenumber)+'\n')
            f.write('    pattern   precision  recall  Fscore\n')

            for pat in usedclassif:
                if pat !='back_ground':
                    numpat=classif[pat]
                    fscore[pat]=0
                    precision[pat],recall[pat] = evaluate(referencepat,predictpat,num_class,(numpat,))
                    if (precision[pat]+recall[pat])>0:
                        fscore[pat]=2*precision[pat]*recall[pat]/(precision[pat]+recall[pat])
                        precisioni=str(round(precision[pat]*100,0))+'%'
                        recalli=str(round(recall[pat]*100,))+'%'
                        fscorei=str(round(fscore[pat]*100,0))+'%'
    #                    print (slicenumber,pat,precisioni,recalli,fscorei)
                        f.write('%14s'%pat+'%7s'%precisioni+'%9s'%recalli+'%9s'%fscorei+'\n')
            f.write('\n')
    referencepat= referencepatu.flatten()
    predictpat=  predictpatu.flatten() 
    cfma(f,referencepat,predictpat,num_class,listHug,thrprobaUIP,cnnweigh,tp)
    f.write('----------------------\n')

    return referencepat,predictpat,''

def writeslice(num,menus):
#        print 'write',num
        cv2.rectangle(menus, (5,60), (150,50), red, -1)
        cv2.putText(menus,'Slice to visualize: '+str(num),(5,60),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
        
def openfichier(ti,datacross,path_img,thrprobaUIP,patch_list_cross_slice,tabroi,
                cnnweigh,tabscanLung,viewstyle,slnroi,xtrains):   
    global  quitl,patchi,ix,iy
    print 'openfichier start' 
    
    quitl=False
    num_class=len(classif)
    slnt=datacross[0]

    print 'list of slices',slnroi
    patchi=False
    ix=0
    iy=0
    (top,tail)=os.path.split(path_img)
    volumeroilocal={}
    pdirk = os.path.join(path_img,source_name)
    pdirkroicross = os.path.join(path_img,sroi)
    pdirkroifront = os.path.join(path_img,sroi3d)

    if ti =="cross view" or ti =="merge view" or ti =='front projected view':
        if os.path.exists(pdirkroicross):
            pdirk = pdirkroicross
            path_data_write=os.path.join(path_img,path_data)
            path_data_writefile=os.path.join(path_data_write,volumeroifilep)
            if os.path.exists(path_data_writefile):
                volumeroilocal=pickle.load(open(path_data_writefile, "rb" ))    
#                print volumeroilocal
            sn=''
        else:
            sn=scan_bmp
    else:        
        if os.path.exists(pdirkroifront):
            pdirk = pdirkroifront
            sn=''
        else:
            sn=transbmp

    pdirk = os.path.join(pdirk,sn)

    list_image={}
    cdelimter='_'
    extensionimage='.'+typei1
    limage=[name for name in os.listdir(pdirk) if name.find('.'+typei1,1)>0 ]
    lenlimage=len(limage)

#    print len(limage), sln

    for iimage in limage:

        sln=rsliceNum(iimage,cdelimter,extensionimage)
        list_image[sln]=iimage
#        print sln

    image0=os.path.join(pdirk,list_image[slnroi[0]])
    img = cv2.imread(image0,1)
    img=cv2.resize(img,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)
#        cv2.imshow('cont',img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    
    
    cv2.namedWindow('imagepredict',cv2.WINDOW_NORMAL)
    cv2.namedWindow("Sliderfi",cv2.WINDOW_NORMAL)
    cv2.namedWindow("datavisu",cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar( 'Brightness','Sliderfi',0,100,nothing)
    cv2.createTrackbar( 'Contrast','Sliderfi',50,100,nothing)
    cv2.createTrackbar( 'Threshold','Sliderfi',int(thrprobaUIP*100),100,nothing)
    cv2.createTrackbar( 'Flip','Sliderfi',0,lenlimage-1,nothings)
    cv2.createTrackbar( 'All','Sliderfi',1,1,nothings)
    cv2.createTrackbar( 'None','Sliderfi',0,1,nothings)
        
    viewasked={}
    for key1 in usedclassif:
#            print key1
        viewasked[key1]=True
        cv2.createTrackbar( key1,'Sliderfi',0,1,nothings)
    nbdig=0
    numberentered={}
    initimg = np.zeros((dimtaby,dimtabx,3), np.uint8)
    slicenumberold=0
    tlold=0
    viewaskedold={}
    
    for keyne in usedclassif:
            viewaskedold[keyne]=False
    datav = np.zeros((500,900,3), np.uint8) 
    imgtext = np.zeros((dimtaby,dimtabx,3), np.uint8)
    while(1):
    
        imgwip = np.zeros((200,200,3), np.uint8)  
                             
        cv2.setMouseCallback('imagepredict',draw_circle,img)
        c = cv2.getTrackbarPos('Contrast','Sliderfi')
        l = cv2.getTrackbarPos('Brightness','Sliderfi')
        tl = cv2.getTrackbarPos('Threshold','Sliderfi')
        fld = cv2.getTrackbarPos('Flip','Sliderfi')
        allview = cv2.getTrackbarPos('All','Sliderfi')
        noneview = cv2.getTrackbarPos('None','Sliderfi')
        fl=slnroi[fld]
        key = cv2.waitKey(1000)
#            if key != -1:
#                print key
            
        if key >47 and key<58:
            numberfinal=0
            knum=key-48
#                print 'this is number',knum
            numberentered[nbdig]=knum
            nbdig+=1
            for i in range (nbdig):
                numberfinal=numberfinal+numberentered[i]*10**(nbdig-1-i)            
#                print numberfinal
            numberfinal = min(slnt-1,numberfinal)
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
                cv2.rectangle(initimg, (5,60), (150,50), black, -1)
                
        if nbdig>0 and key ==13 and numberfinal>0 :
            if numberfinal in slnroi:
#                    print numberfinal
                numberfinal = min(slnt-1,numberfinal)
#                    writeslice(numberfinal,initimg)
                cv2.rectangle(initimg, (5,60), (150,50), black, -1)
  
                fld=slnroi.index(numberfinal)
#                print fl,numberfinal
                cv2.setTrackbarPos('Flip','Sliderfi' ,fld)
            else:
                print 'number not in set'
#                cv2.rectangle(initimg, (5,60), (150,50), black, -1)
                cv2.rectangle(initimg, (5,60), (150,50), red, -1)
                cv2.putText(initimg,'NO ROI slice '+str(numberfinal)+'!',(5,60),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
#                cv2.putText(initimg, 'NO ROI on this slice',(5,60),cv2.FONT_HERSHEY_PLAIN,5,red,2,cv2.LINE_AA)
#                time.sleep(5)
#                fl=numberfinal
                
            numberfinal=0
            nbdig=0
            numberentered={}
        if key==2424832:
            fld=max(0,fld-1)
            cv2.setTrackbarPos('Flip','Sliderfi' ,fld)
        if key==2555904:
            fld=min(lenlimage-1,fld+1)
            cv2.setTrackbarPos('Flip','Sliderfi' ,fld)
            
        if allview==1:
            for key2 in usedclassif:
                cv2.setTrackbarPos(key2,'Sliderfi' ,1)
            cv2.setTrackbarPos('All','Sliderfi' ,0)
        if noneview==1:
            for key2 in usedclassif:
                cv2.setTrackbarPos(key2,'Sliderfi' ,0)
            cv2.setTrackbarPos('None','Sliderfi' ,0)
        for key2 in usedclassif:
            s = cv2.getTrackbarPos(key2,'Sliderfi')
            if s==1:
                 viewasked[key2]=True               
            else:
                 viewasked[key2]=False

        slicenumber=fl
#        print slicenumber
        
        imagel=os.path.join(pdirk,list_image[slicenumber])
        img = cv2.imread(imagel,1)               
        img=cv2.resize(img,(dimtabx,dimtaby),interpolation=cv2.INTER_LINEAR)
#            print img.shape
#            print  initimg.shape
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
        if drawok:
#                print 'view'
            imgtext = np.zeros((dimtaby,dimtabx,3), np.uint8)
            cv2.putText(imgwip,'WIP',(10,10),cv2.FONT_HERSHEY_PLAIN,5,red,2,cv2.LINE_AA)
            cv2.imshow('wip',imgwip)
            
            imgn,datav= drawpatch(tl,dimtabx,dimtaby,slicenumber,viewasked,patch_list_cross_slice,
                                            volumeroilocal,slnt,tabroi,num_class,tabscanLung,xtrains)
                           
            cv2.putText(datav,'slice number :'+str(slicenumber),(10,180),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
            cv2.putText(datav,'patient Name :'+tail,(10,190),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
            cv2.putText(datav,'CNN weight: '+cnnweigh,(10,170),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
            cv2.putText(datav,viewstyle,(10,200),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)

            cv2.destroyWindow("wip")
#        imgngray = cv2.cvtColor(imgn,cv2.COLOR_BGR2GRAY)
#        np.putmask(imgngray,imgngray>0,255)
#        mask_inv = cv2.bitwise_not(imgngray)
#        outy=cv2.bitwise_and(imcontrast,imcontrast,mask=mask_inv)
        imgt=cv2.add(imgn,imcontrast)
        dxrect=(dimtaby/2)
        cv2.rectangle(imgt,(dxrect,dimtabx-30),(dxrect+20,dimtabx-10),red,-1)
        cv2.putText(imgt,'quit',(dxrect+10,dimtabx-10),cv2.FONT_HERSHEY_PLAIN,1,yellow,1,cv2.LINE_AA)
           
        if patchi :
            cv2.putText(imgwip,'WIP',(10,10),cv2.FONT_HERSHEY_PLAIN,5,red,2,cv2.LINE_AA)
            cv2.imshow('wip',imgwip)
#                print 'retrieve patch asked'
            imgtext= retrievepatch(ix,iy,slicenumber,patch_list_cross_slice,xtrains)
            patchi=False
            cv2.destroyWindow("wip")
        imgtoshow=cv2.add(imgt,imgtext)
        imgtoshow=cv2.cvtColor(imgtoshow,cv2.COLOR_BGR2RGB)

        imsstatus=cv2.getWindowProperty('Sliderfi', 0)
        imistatus= cv2.getWindowProperty('imagepredict', 0)
        imdstatus=cv2.getWindowProperty('datavisu', 0)
#            print imsstatus,imistatus,imdstatus
        if (imdstatus==0) and (imsstatus==0) and (imistatus==0)  :
            cv2.imshow('imagepredict',imgtoshow)
            cv2.imshow('datavisu',datav)
        else:
              quitl=True

        if quitl or cv2.waitKey(20) & 0xFF == 27 :
            break
    quitl=False
    cv2.destroyWindow("imagepredict")
    cv2.destroyWindow("Sliderfi")
    cv2.destroyWindow("datavisu")

    return ''
    


def visuarun(indata,path_patient):
    global classif,usedclassif
    print 'visuarun start'

    messageout=""
 
#    print 'path_patient',path_patient
    lpt=indata['lispatientselect']
    pos=lpt.find(' ')
    if pos>0:
        listHug=(lpt[0:pos])
    else:
        listHug=lpt
    patient_path_complet=os.path.join(path_patient,listHug)
    path_data_dir=os.path.join(patient_path_complet,path_data)

    crosscompleted = pickle.load(open( os.path.join(path_data_dir,"crosscompleted"), "rb" ))
  
    if not crosscompleted:
        messageout="no predict!for "+listHug
        return messageout

    if oldFormat:
        setref='set0'
    else:
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss"), "rb" ))
        setref=datarep[3]
#        print setref
    classif=classifdict[setref]
    usedclassif=usedclassifdict[setref]
#    print usedclassif
    viewstyle=indata['viewstyle']

#    pathhtml=os.path.join(patient_path_complet,htmldir)
    
    if viewstyle=='cross view':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss"), "rb" ))
            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois"), "rb" ))
            slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois"), "rb" ))
            xtrains  = pickle.load( open( os.path.join(path_data_dir,"xtrainss"), "rb" ))


            tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs"), "rb" ))
            proba_cross= pickle.load( open( os.path.join(path_data_dir,"proba_crosss"), "rb" ))
            thrproba=float(indata['thrproba'])
            cnnweigh=indata['picklein_file']
#            print cnnweigh
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrproba,
                                   proba_cross,tabroi,cnnweigh,tabscanLung,viewstyle,slnroi,xtrains)
            
    elif viewstyle=='front view':
            return 'not implemented'
            datarep= pickle.load( open( os.path.join(path_data_dir,"datafront"), "rb" ))
#            tabroif= pickle.load( open( os.path.join(path_data_dir,"tabroif"), "rb" ))
            slnt=datarep[0]

            
            tabLung3d= pickle.load( open( os.path.join(path_data_dir,"tabLung3d"), "rb" ))
            tabroi=np.zeros((slnt,dimtaby,dimtabx), np.uint8)
            patch_list_front_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_front_slice"), "rb" ))
            thrprobaUIP=float(indata['thrproba'])
            cnnweigh=indata['picklein_file_front']
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,
                                   patch_list_front_slice,tabroi,cnnweigh,tabLung3d,viewstyle)
            

    elif viewstyle=='merge view':

            thrprobaUIP=float(indata['thrproba'])
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabroi"), "rb" ))
            tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLung"), "rb" ))
            slnroi= pickle.load( open( os.path.join(path_data_dir,"slnroi"), "rb" ))

#            patch_list_merge= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge"), "rb" ))
#            proba_merge= pickle.load( open( os.path.join(path_data_dir,"proba_merge"), "rb" ))
            patch_list_merge_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge_slice"), "rb" ))
#            patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
            cnnweigh=indata['picklein_file']
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,
                                   patch_list_merge_slice,tabroi,cnnweigh,tabscanLung,viewstyle,slnroi)
    
    elif viewstyle=='front projected view':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabroi"), "rb" ))
            slnroi= pickle.load( open( os.path.join(path_data_dir,"slnroi"), "rb" ))

            patch_list_cross_slice_from_front= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_from_front"), "rb" ))
            thrprobaUIP=float(indata['thrproba']) 
            cnnweigh=indata['picklein_file_front']
            tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLung"), "rb" ))
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,
                                   patch_list_cross_slice_from_front,tabroi,cnnweigh,tabscanLung,viewstyle,slnroi)
#            thrprobaUIP=float(indata['thrprobaUIP'])
#            tabfromfront= pickle.load( open( os.path.join(path_data_dir,"tabfromfront"), "rb" ))
#            messageout=openfichierfrpr(path_patient,tabfromfront,thrprobaUIP)
    
    elif viewstyle=='reportCross':
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss"), "rb" ))
#        slicepitch=datarep[3]
        slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois"), "rb" ))
        cnnweigh=indata['picklein_file']

        tp='Cross'
        proba_cross= pickle.load( open( os.path.join(path_data_dir,"proba_crosss"), "rb" ))
#        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,'patch_list_cross_slices'), "rb" ))
#        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
#        patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_sub"), "rb" ))
#        lungSegment= pickle.load( open( os.path.join(path_data_dir,"lungSegment"), "rb" ))
#        tabMed= pickle.load( open( os.path.join(path_data_dir,"tabMed"), "rb" ))
        thrproba=float(indata['thrproba'])
        tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs"), "rb" ))
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois"), "rb" ))
        xtrains  = pickle.load( open( os.path.join(path_data_dir,"xtrainss"), "rb" ))
        dirf=os.path.join(path_patient,listHug)
        dirfreport=os.path.join(dirf,reportdir)   
        if not os.path.exists(dirfreport):
            os.mkdir(dirfreport)
        t = datetime.datetime.now()
        today =tp+'_weight_'+cnnweigh+ '_th'+str(thrproba)+'_tpat'+'_m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'
        repf=os.path.join(dirfreport,reportfile+str(today)+'.txt')
        f=open(repf,'w')
        x,x,messageout = openfichiervolumetxt(listHug,path_patient,proba_cross,thrproba,
                      tabroi,datarep,slnroi,tabscanLung,f,cnnweigh,tp,xtrains)
        f.close()
        os.startfile(repf)
        
    elif viewstyle=='reportFront':
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
#        slicepitch=datarep[3]
        slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois"), "rb" ))
        cnnweigh=indata['picklein_file']

        tp='FrontProjected'
            
        proba_cross= pickle.load( open( os.path.join(path_data_dir,"proba_fronts"), "rb" ))
#        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
#        patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_sub"), "rb" ))
#        lungSegment= pickle.load( open( os.path.join(path_data_dir,"lungSegment"), "rb" ))
#        tabMed= pickle.load( open( os.path.join(path_data_dir,"tabMed"), "rb" ))
        thrproba=float(indata['thrproba'])
        tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs"), "rb" ))
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois"), "rb" ))
        dirf=os.path.join(path_patient,listHug)
        dirfreport=os.path.join(dirf,reportdir)   
        if not os.path.exists(dirfreport):
            os.mkdir(dirfreport)
        t = datetime.datetime.now()
        today =tp+'_weight_'+cnnweigh+ '_th'+str(thrproba)+'_tpat'+'_m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'
        repf=os.path.join(dirfreport,reportfile+str(today)+'.txt')
        f=open(repf,'w')
        x,x,messageout = openfichiervolumetxt(listHug,path_patient,proba_cross,
                      thrproba,
                      tabroi,datarep,slnroi,tabscanLung,f,cnnweigh,tp)
        f.close()
        os.startfile(repf)
    elif viewstyle=='reportMerge':
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
#        slicepitch=datarep[3]
        slnroi= pickle.load( open( os.path.join(path_data_dir,"slnroi"), "rb" ))
        cnnweigh=indata['picklein_file']

        tp='Merge'
            
        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,'patch_list_merge_slice'), "rb" ))
#        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
#        patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_sub"), "rb" ))
#        lungSegment= pickle.load( open( os.path.join(path_data_dir,"lungSegment"), "rb" ))
#        tabMed= pickle.load( open( os.path.join(path_data_dir,"tabMed"), "rb" ))
        thrproba=float(indata['thrproba'])
        thrpatch=pickle.load( open( os.path.join(path_data_dir,"thrpatch"), "rb" ))
        tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLung"), "rb" ))
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabroi"), "rb" ))
        dirf=os.path.join(path_patient,listHug)
        dirfreport=os.path.join(dirf,reportdir)   
        if not os.path.exists(dirfreport):
            os.mkdir(dirfreport)
        t = datetime.datetime.now()
        today =tp+'_weight_'+cnnweigh+ '_th'+str(thrproba)+'_tpat'+str(thrpatch)+'_m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'
        repf=os.path.join(dirfreport,reportfile+str(today)+'.txt')
        f=open(repf,'w')
        x,x,messageout = openfichiervolumetxt(listHug,path_patient,patch_list_cross_slice,
                      thrproba,
                      tabroi,datarep,slnroi,tabscanLung,f,cnnweigh,thrpatch,tp)
        f.close()
        os.startfile(repf)
        
    elif viewstyle=='reportAll':
        tp=indata['viewstylet']
        print 'report All for', tp
        thrproba=float(indata['thrproba'])

        xtrains  = pickle.load( open( os.path.join(path_data_dir,"xtrainss"), "rb" ))

        cnnweigh=indata['picklein_file']
        pathreport=os.path.join(path_patient,reportalldir)
        if not os.path.exists(pathreport):
            os.mkdir(pathreport)
        t = datetime.datetime.now()
        today =tp+'_'+'_weight_'+cnnweigh+ '_th'+str(thrproba)+'_tpat'+'_m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'
        repf=os.path.join(pathreport,reportfile+str(today)+'.txt')
        f=open(repf,'w')
#        remove_folder(pathreport)
        listHug=[]
        a,b,c=lisdirprocess(path_patient)
        for patient in a:
            if b[patient]['cross']==True:
                listHug.append(patient)

        if not os.path.exists(pathreport):
            os.mkdir(pathreport)
        messageout = openfichiervolumetxtall(listHug,path_patient,indata,thrproba,cnnweigh,f,tp,xtrains)
        os.startfile(repf)
        
    else:
            messageout='error: unrecognize view style'

    return messageout