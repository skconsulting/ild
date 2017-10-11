# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:21:22 2017

@author: sylvain Kritter 

version 1.5
6 September 2017
"""
#from param_pix_p import *
from param_pix_s2 import path_data,reportalldir
#from param_pix_s2 import surfelemp,volumeroifilep

from param_pix_s2 import white,red,yellow,black
from param_pix_s2 import source_name,sroi,sroi3d,scan_bmp,transbmp
from param_pix_s2 import typei1,dimtabx,dimtaby
from param_pix_s2 import reportdir,reportfile

from param_pix_s2 import classifc,classifdict,usedclassifdict,oldFormat

from param_pix_s2 import excluvisu,fidclass,rsliceNum,evaluate,evaluatef
#from param_pix_s2 import normi

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
            pathcross=os.path.join(datadir,'datacrosss2')
            
            pathfront=os.path.join(datadir,'datafronts2')
            setrefdict[dd]='no'
            if os.path.exists(pathcross):
                datacross= pickle.load( open( os.path.join(datadir,"datacrosss2"), "rb" ))
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
                      
            crosscompletedf=os.path.join(datadir,'crosscompleteds2')
            frontcompletedf=os.path.join(datadir,'frontcompleteds2')
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


def tagviewn(fig,label,surface,surftot,roi,tl,precision,recall,fscore,spc,npv):
    """write text in image according to label and color"""
    
    col=classifc[label]
    if col==(0,0,0):
        col=(100,100,100)
    labnow=classif[label]
    
    deltax=20
    deltay=30+(11*labnow)

    gro=0.8
    
    
    if surftot>0:
        pc=str(int(round(100.*surface/surftot,0)))
        pcroi=str(int(round(100.*roi/surftot,0)))
    else:
        pc='0'
        pcroi='0'
    if tl:
        cv2.rectangle(fig,(deltax-10, deltay-6),(deltax-5, deltay-1),col,1)

    if roi>0:
        cv2.putText(fig,'%16s'%label+'%8s'%(str(surface))+
                    '%5s'%(pc+'%')+'%8s'%(str(roi))+'%5s'%(pcroi+'%'),
                (deltax, deltay),cv2.FONT_HERSHEY_PLAIN,gro,col,1)
    else:
        cv2.putText(fig,'%16s'%label+'%8s'%(str(surface))+'%5s'%(pc+'%'),
                (deltax, deltay),cv2.FONT_HERSHEY_PLAIN,gro,col,1)
    deltax=350
    if fscore>0:
        
        cv2.putText(fig,'%10s'%(str(precision)+'%')+
                '%10s'%(str(recall)+'%')+'%10s'%(str(fscore)+'%')+
                '%10s'%(str(spc)+'%')+'%10s'%(str(npv)+'%'),
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
    for pat in usedclassif:
        if pat !='back_ground':
            cp=classif[pat]-1
    #        if cp==lcl:
    #            dy=0
    #        elif cp==0:
    #            dy=lcl
    #        else:
            dy=cp
            col=classifc[pat]
            if col == (0,0,0):
                col=(100,100,100)
    
            cv2.putText(fig,'%15s'%pat,(deltax, deltay+(12*dy)),cv2.FONT_HERSHEY_PLAIN,gro,col,1)
            cv2.putText(fig,pat[0:7],(deltaxt+(60*dy), deltay-15),cv2.FONT_HERSHEY_PLAIN,gro,col,1)

     
    for i in range (0,n):
        for j in range (0,n):
            dx=deltaxm+60*j
            dy=deltay+12*i
            cv2.putText(fig,'%5s'%str(cm[i][j]),(dx, dy),cv2.FONT_HERSHEY_PLAIN,gro,white,1)

def cals(cm,pat):
    numpat=classif[pat]-1
    tp=cm[numpat][numpat]
    fp=cm[:,numpat].sum()-tp
    fn=cm[numpat].sum()-tp
    tn=cm.sum()-fp-fn-tp 
    if tp+fp>0:
        prec=1.0*tp/(tp+fp)
    else:
        prec=0
    if tp+fn>0:
            recall=1.0*tp/(tp+fn)
    else:
            recall=0
    if tn+fp>0:
        spc=1.0*tn/(tn+fp)
    else:
        spc=0
    if tn+fn>0:
        npv=1.0*tn/(tn+fn)
    else:
        npv=0
    if prec+recall>0:
        fsc=2*(prec*recall)/(prec+recall)
    else:
        fsc=0
#    return tp,fp,fn,tn,prec,recall,fsc,spc,npv
    return prec,recall,fsc, spc,npv ,tp+fp,tp+fn

def colorimage(image,color):
#    im=image.copy()
    im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    np.putmask(im,im>0,color)
    return im



def drawpatch(t,dx,dy,slicenumber,va,patch_list_cross_slice,volumeroi,
              slnt,tabroi,num_class,tabscanLung,xtrains):

    imgn = np.zeros((dy,dx,3), np.uint8)
    datav = np.zeros((500,900,3), np.uint8)

    predictpatu=np.zeros((dx,dy), np.uint8) 
    referencepatu=np.zeros((dx,dy), np.uint8) 
   
    th=t/100.0
    listlabel=[]

    surftot=np.count_nonzero(tabscanLung[slicenumber])

    surftotf= surftot*surfelem
    surftot='surface totale :'+str(int(round(surftotf,0)))+'cm2'

    imi=  patch_list_cross_slice[xtrains[slicenumber]]
#    imi=np.flip(imi,axis=2)
#    print'1', imi[278][195]
    imclass0=np.argmax(imi, axis=2).astype(np.uint8)
#    print '2',imclass0[278][195]
    imamax=np.amax(imi, axis=2)
 
    tablung1=np.copy(tabscanLung[slicenumber])   
    np.putmask(tablung1,tablung1>0,255)
    imclass0=np.bitwise_and(tablung1, imclass0) 
#    print '3',imclass0[278][195]

    np.putmask(imamax,imamax>=th,255)
    np.putmask(imamax,imamax<th,0)
    imamax=imamax.astype(np.uint8)
    imclass=np.bitwise_and(imamax,imclass0)
#    print '4',imclass[278][195]
#    print '5',tabroi[slicenumber][278][195]
    
    for pat in usedclassif:       
        if pat not in excluvisu :
            if va[pat]:
                imgnp=imclass.copy()               
                np.putmask(imgnp,imgnp!=classif[pat],0)
                if imgnp.max()>0:
                    imgnpc=colorimage(imgnp,classifc[pat])
                    imgn=cv2.addWeighted(imgn,1,imgnpc,0.3,0)
                    if pat not in listlabel:
                        listlabel.append(pat)

    volroi={}
    volpat={}
    precision={}
    recall={}
    fscore={}
    spc={}
    npv={}
    for pat in usedclassif:
        volroi[pat]=0
        volpat[pat]=0
        precision[pat]=0
        recall[pat]=0
        fscore[pat]=0
        spc[pat]=0
        npv[pat]=0       
    

    predictpatu=np.copy(imclass)
    referencepatu= np.copy(tabroi[slicenumber])
    
    referencepat= referencepatu.flatten()             
    predictpat=  predictpatu.flatten() 
#    print num_class 
    cm=evaluatef(referencepat,predictpat,num_class)
    
#    print cm
#    print num_class
#    fer=open('report.txt','w')
#    n= cm.shape[0]
#    for cmi in range (0,n):     
#                        fer.write('  ')
#                        for j in range (0,n):
#                            fer.write(str(cm[cmi][j])+' ')
#                        fer.write('\n')
#    fer.write('----\n')
#    fer.close()
    tagvcm(datav,cm)
    
    cv2.putText(datav,'%16s'%('pattern')+
                    '%8s'%('surface')+ 
                    '%4s'%('%')+
                    '%9s'%('roi')+
                    '%5s'%('%'),
                (20, 10),cv2.FONT_HERSHEY_PLAIN,0.8,yellow,1)
    cv2.putText(datav,'%24s'%('cm2')+ '%13s'%('cm2'),
                (20, 20),cv2.FONT_HERSHEY_PLAIN,0.8,yellow,1)
    if referencepatu.max()>0:
        cv2.putText(datav,'%10s'%('Precision')+
                    '%11s'%('Recall')+ 
                    '%11s'%('Fscore')+
                    '%10s'%('SPC')+
                    '%10s'%('NPV'),
                (350, 10),cv2.FONT_HERSHEY_PLAIN,0.8,yellow,1)
    
    precisionAverage=0
    recallAverage=0
    fscoreAverage=0
    spcAverage=0
    npvAverage=0
    numberp=0
    tpaverage=0
    cpp=0
    cpr=0
    for pat in usedclassif:   
        if pat !='back_ground':
            cpa=classif[pat]-1 

            precision[pat],recall[pat],fscore[pat], spc[pat],npv[pat],volpat[pat],volroi[pat]=cals(cm,pat)    
            if pat in listlabel:            
                tl=True
            else:
                tl=False
            precisioni=int(round(precision[pat]*100,0))
            recalli=int(round(recall[pat]*100,0))
            fscorei=int(round(fscore[pat]*100,0))
            if cm[cpa].sum()>0:
                    numberp+=1
                    tpaverage+=cm[cpa][cpa]
                    cpp+=cm[:,cpa].sum()
                    cpr+=cm[cpa].sum()
                    spcAverage+=spc[pat]
                    npvAverage+=npv[pat]
                     
            spci=int(round(spc[pat]*100,0))
            npvi=int(round(npv[pat]*100,0))
            sul=round((volpat[pat])*surfelem,1)
            suroi=round((volroi[pat])*surfelem,1)            
            tagviewn(datav,pat,sul,surftotf,suroi,tl,precisioni,recalli,fscorei,spci,npvi)
    if cpp>0:
        precisionAverage=1.0*tpaverage/cpp
#        print '2',precisionAverage
    else:
        precisionAverage=0
    if cpr>0:
        recallAverage=1.0*tpaverage/cpr
#        print '2',recallAverage
    else:
        recallAverage=0
#        fscoreAverage/=numberp
    if numberp>0:
#        print spcAverage,numberp
        spcAverage/=numberp
        npvAverage/=numberp
    
    if referencepatu.max()>0:
        cv2.putText(datav,'%10s'%('Precision %')+
                    '%11s'%('Recall%')+ 
                    '%11s'%('Fscores')+
#                    '%10s'%('SPC %')+
#                    '%10s'%('NPV %'),
                    '',
                (10, 450),cv2.FONT_HERSHEY_PLAIN,0.8,yellow,1)
        precisionAverage=int(round(precisionAverage*100,0))
        recallAverage=int(round(recallAverage*100,0))
        if recallAverage+precisionAverage>0:
            fscoreAverage=int(round(2.*precisionAverage*recallAverage/(recallAverage+precisionAverage),0))
        else:
            fscoreAverage=0
#        fscoreAverage=int(round(fscoreAverage*100,0))
        spcAverage=int(round(spcAverage*100,0))
        npvAverage=int(round(npvAverage*100,0))
        cv2.putText(datav,'%10s'%(precisionAverage)+
                    '%11s'%(recallAverage)+ 
                    '%11s'%(fscoreAverage)+
#                    '%10s'%(spcAverage)+
#                    '%10s'%(npvAverage),
                    '',
                    (10, 460),cv2.FONT_HERSHEY_PLAIN,0.8,yellow,1)
    delx=120
    ts='Threshold:'+str(t)
    
    cv2.putText(datav,ts,(10,170),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
    cv2.putText(datav,surftot,(delx,170),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
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

    num_class=len(usedclassif)

    f.write('Score for list of patients:\n')
    for patient in listHug:
        f.write(str(patient)+' ')
    f.write('\n-----\n')
    pf=True
    for patient in listHug:
        print 'work on :',patient

        patient_path_complet=os.path.join(path_patient,patient)
        path_data_dir=os.path.join(patient_path_complet,path_data)
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss2"), "rb" ))
#        slicepitch=datarep[3]
        slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois2"), "rb" ))
#        print 'slnroi,patient',slnroi,patient
        if tp=='Cross':
            filepatch="proba_crosss2"
        if tp=='FrontProjected':
            filepatch="proba_fronts2"
        if tp=='Merge':
            filepatch="proba_merges2"
            
        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,filepatch), "rb" ))
#        patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_sub"), "rb" ))
#        lungSegment= pickle.load( open( os.path.join(path_data_dir,"lungSegment"), "rb" ))
#        tabMed= pickle.load( open( os.path.join(path_data_dir,"tabMed"), "rb" ))
        tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs2"), "rb" ))
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois2"), "rb" ))
                  
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
        for pat in usedclassif:
            presip[pat]=0
            recallp[pat]=0
            fscorep[pat]=0
            numpat=classif[pat]

#            print numpat
            presip[pat], recallp[pat]=evaluate(referencepat,predictpat,num_class,(numpat,))
#            print pat,presip[pat],recallp[pat]

        for pat in usedclassif:             
         if presip[pat]+recallp[pat]>0:
            fscorep[pat]=2*presip[pat]*recallp[pat]/(presip[pat]+recallp[pat])
        else:
            fscorep[pat]=0    
        f.write(15*' ')
        for i in range (0,n):
            pat=fidclass(i+1,classif)
            f.write('%8s'%pat[0:7])
    #    print newclassif
        f.write('  recall \n')
            
        for i in range (0,n):            
            pat=fidclass(i+1,classif)
            f.write('%15s'%pat)
            for j in range (0,n):
                f.write('%8s'%str(cm[i][j]))
            f.write( '%8s'%(str(int(round(100*recallp[pat],0)))+'%')+'\n')        
        f.write('----------------------\n')
        f.write('      precision')
        for i in range (0,n):
            pat=fidclass(i+1,classif)
            f.write('%8s'%(str(int(round(100*presip[pat],0)))+'%'))
    #    print newclassif
        f.write('\n')
        f.write('         Fscore')
        for i in range (0,n):
            pat=fidclass(i+1,classif)
            f.write('%8s'%(str(int(round(100*fscorep[pat],0)))+'%'))
    #    print newclassif
        f.write('\n--------------------\n')

        wrresu(f,cm,'Report for full Patient',1)

        f.write('----------------------\n')

def wrresu(f,cm,obj,refmax):
    volroi={}
    volpat={}
    precision={}
    recall={}
    fscore={}
    spc={}
    npv={}
    for pat in classif:
        volroi[pat]=0
        volpat[pat]=0
        precision[pat]=0
        recall[pat]=0
        fscore[pat]=0
        spc[pat]=0
        npv[pat]=0       
    f.write(str(obj)+'\n')
    f.write('    pattern   precision%  recall%  Fscore%   SPC%     NPV%\n')
   
    precisionAverage=0
    recallAverage=0
    fscoreAverage=0
    spcAverage=0
    npvAverage=0
    numberp=0
    tpaverage=0
    cpp=0
    cpr=0
    for pat in usedclassif:   
        if pat !='back_ground':
            cpa=classif[pat]-1 

            precision[pat],recall[pat],fscore[pat], spc[pat],npv[pat],volpat[pat],volroi[pat]=cals(cm,pat)    
            precisioni=int(round(precision[pat]*100,0))
            recalli=int(round(recall[pat]*100,0))
            fscorei=int(round(fscore[pat]*100,0))
            if cm[cpa].sum()>0:
#                print pat ,cm[cpa]
                numberp+=1
                tpaverage+=cm[cpa][cpa]
                cpp+=cm[:,cpa].sum()
                cpr+=cm[cpa].sum()
                spcAverage+=spc[pat]
                npvAverage+=npv[pat]
                     
                spci=int(round(spc[pat]*100,0))
                npvi=int(round(npv[pat]*100,0)) 
            
                f.write('%14s'%pat+'%7s'%precisioni+
                            '%9s'%recalli+'%9s'%fscorei+
                            '%9s'%spci+'%9s'%npvi+'\n')
    f.write('---\n')
    if cpp>0:
        precisionAverage=1.0*tpaverage/cpp
#        print '2',precisionAverage
    else:
        precisionAverage=0
    if cpr>0:
        recallAverage=1.0*tpaverage/cpr
#        print '2',recallAverage
    else:
        recallAverage=0
#        fscoreAverage/=numberp
    if numberp>0:
#        print spcAverage,numberp
        spcAverage/=numberp
        npvAverage/=numberp
    if refmax>0:

        f.write('%10s'%('Precision %')+
                    '%11s'%('Recall%')+ 
                    '%11s'%('Fscores')+
#                    '%10s'%('SPC%')+
#                    '%10s'%('NPV%')+
                    '\n')
        precisionAverage=int(round(precisionAverage*100,0))
        recallAverage=int(round(recallAverage*100,0))
        if recallAverage+precisionAverage>0:
            fscoreAverage=int(round(2.*precisionAverage*recallAverage/(recallAverage+precisionAverage),0))
        else:
            fscoreAverage=0
        
        spcAverage=int(round(spcAverage*100,0))
        npvAverage=int(round(npvAverage*100,0))
        f.write('%10s'%(precisionAverage)+
                    '%11s'%(recallAverage)+ 
                    '%11s'%(fscoreAverage)+
#                    '%10s'%(spcAverage)+
#                    '%10s'%(npvAverage)+
                    '\n')
    f.write('-----------------------------------\n')

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
#    num_class=len(classif)
    num_class=len(usedclassif)
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

            tablung1=np.copy(tabscanLung[slicenumber])
            np.putmask(tablung1,tablung1>0,255)
#            patchdict[slicenumber]=np.bitwise_and(tablung1, patchdict[slicenumber]) 
            
#            mask=patchdict[slicenumber].copy()
#            tabxorig=patchdict[slicenumber].copy()
#            np.putmask(mask,mask>0,255)
#            tabxn=np.bitwise_not(mask)   
#            tablung=np.copy(tabscanLung[slicenumber])       
#            tablung=np.bitwise_and(tablung, tabxn)
#            np.putmask(tablung,tablung>0,classif['healthy'])
     
            predictpatu[slroi]=np.bitwise_and(tablung1, patchdict[slicenumber]) 
            referencepatu[slroi]=np.copy(tabroi[slicenumber])          
            
            referencepat= referencepatu[slroi].flatten()
            predictpat=  predictpatu[slroi].flatten() 
            cm=evaluatef(referencepat,predictpat,num_class)
            wrresu(f,cm,'Slice: '+str(slicenumber),referencepatu[slroi].max())
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
    num_class=len(usedclassif)
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
            path_data_writefile=os.path.join(path_data_write,'volumeroifiles2')
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
    
    
    cv2.namedWindow('scores2',cv2.WINDOW_NORMAL)
    cv2.namedWindow("Sliderfis2",cv2.WINDOW_NORMAL)
    cv2.namedWindow("datas2",cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar( 'Brightness','Sliderfis2',0,100,nothing)
    cv2.createTrackbar( 'Contrast','Sliderfis2',50,100,nothing)
    cv2.createTrackbar( 'Threshold','Sliderfis2',int(thrprobaUIP*100),100,nothing)
    cv2.createTrackbar( 'Flip','Sliderfis2',0,lenlimage-1,nothings)
    cv2.createTrackbar( 'All','Sliderfis2',1,1,nothings)
    cv2.createTrackbar( 'None','Sliderfis2',0,1,nothings)
        
    viewasked={}
    for key1 in usedclassif:
#            print key1
        viewasked[key1]=True
        cv2.createTrackbar( key1,'Sliderfis2',0,1,nothings)
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
                             
        cv2.setMouseCallback('scores2',draw_circle,img)
        c = cv2.getTrackbarPos('Contrast','Sliderfis2')
        l = cv2.getTrackbarPos('Brightness','Sliderfis2')
        tl = cv2.getTrackbarPos('Threshold','Sliderfis2')
        fld = cv2.getTrackbarPos('Flip','Sliderfis2')
        allview = cv2.getTrackbarPos('All','Sliderfis2')
        noneview = cv2.getTrackbarPos('None','Sliderfis2')
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
                cv2.setTrackbarPos('Flip','Sliderfis2' ,fld)
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
            cv2.setTrackbarPos('Flip','Sliderfis2' ,fld)
        if key==2555904:
            fld=min(lenlimage-1,fld+1)
            cv2.setTrackbarPos('Flip','Sliderfis2' ,fld)
            
        if allview==1:
            for key2 in usedclassif:
                cv2.setTrackbarPos(key2,'Sliderfis2' ,1)
            cv2.setTrackbarPos('All','Sliderfis2' ,0)
        if noneview==1:
            for key2 in usedclassif:
                cv2.setTrackbarPos(key2,'Sliderfis2' ,0)
            cv2.setTrackbarPos('None','Sliderfis2' ,0)
        for key2 in usedclassif:
            s = cv2.getTrackbarPos(key2,'Sliderfis2')
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
                           
            cv2.putText(datav,'slice number :'+str(slicenumber),(10,210),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
            cv2.putText(datav,'patient Name :'+tail,(10,220),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
            cv2.putText(datav,'CNN weight: '+cnnweigh,(10,230),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
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

        imsstatus=cv2.getWindowProperty('Sliderfis2', 0)
        imistatus= cv2.getWindowProperty('scores2', 0)
        imdstatus=cv2.getWindowProperty('datas2', 0)
#            print imsstatus,imistatus,imdstatus
        if (imdstatus==0) and (imsstatus==0) and (imistatus==0)  :
            cv2.imshow('scores2',imgtoshow)
            cv2.imshow('datas2',datav)
        else:
              quitl=True

        if quitl or cv2.waitKey(20) & 0xFF == 27 :
            break
    quitl=False
    cv2.destroyWindow("scores2")
    cv2.destroyWindow("Sliderfis2")
    cv2.destroyWindow("datas2")

    return ''
    


def visuarun(indata,path_patient):
    global classif,usedclassif,surfelem
    print 'visuarun start'

    messageout=""
    viewstyle=indata['viewstyle']
    if viewstyle == 'reportAll':
        lpt=indata['lispatientselect'][0]
    else:
#    print 'path_patient',path_patient
        lpt=indata['lispatientselect']
#    print 'path_patient',path_patient

    pos=lpt.find(' ')
    if pos>0:
        listHug=(lpt[0:pos])
    else:
        listHug=lpt
    patient_path_complet=os.path.join(path_patient,listHug)
    path_data_dir=os.path.join(patient_path_complet,path_data)

    crosscompleted = pickle.load(open( os.path.join(path_data_dir,"crosscompleteds2"), "rb" ))
  
    if not crosscompleted:
        messageout="no predict!for "+listHug
        return messageout
    datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss2"), "rb" ))
    PixelSpacing=datarep[5]
    surfelemp=PixelSpacing*PixelSpacing # for 1 pixel in mm2
    surfelem= surfelemp/100 
    if oldFormat:
        setref='set0'
    else:        
        setref=datarep[3]
    print 'setref',setref
    classif=classifdict[setref]
    usedclassif=usedclassifdict[setref]

#    pathhtml=os.path.join(patient_path_complet,htmldir)
    
    if viewstyle=='cross view':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss2"), "rb" ))
            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois2"), "rb" ))
            slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois2"), "rb" ))
            xtrains  = pickle.load( open( os.path.join(path_data_dir,"xtrainss2"), "rb" ))


            tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs2"), "rb" ))
            proba_cross= pickle.load( open( os.path.join(path_data_dir,"proba_crosss2"), "rb" ))
            thrproba=float(indata['thrproba'])
            cnnweigh=indata['picklein_file']
#            print cnnweigh
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrproba,
                                   proba_cross,tabroi,cnnweigh,tabscanLung,viewstyle,slnroi,xtrains)
            
    elif viewstyle=='front view':
            return 'not implemented'
            datarep= pickle.load( open( os.path.join(path_data_dir,"datafronts2"), "rb" ))

            slnt=datarep[0]

            
            tabLung3d= pickle.load( open( os.path.join(path_data_dir,"tabLung3ds2"), "rb" ))
            tabroi=np.zeros((slnt,dimtaby,dimtabx), np.uint8)
            patch_list_front_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_front_slices2"), "rb" ))
            thrprobaUIP=float(indata['thrproba'])
            cnnweigh=indata['picklein_file_front']
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,
                                   patch_list_front_slice,tabroi,cnnweigh,tabLung3d,viewstyle)
            

    elif viewstyle=='merge view':

            thrprobaUIP=float(indata['thrproba'])
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss2"), "rb" ))
            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois2"), "rb" ))
            tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs2"), "rb" ))
            slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois2"), "rb" ))

            patch_list_merge_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge_slices2"), "rb" ))
            cnnweigh=indata['picklein_file']
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,
                                   patch_list_merge_slice,tabroi,cnnweigh,tabscanLung,viewstyle,slnroi)
    
    elif viewstyle=='front projected view':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss2"), "rb" ))
            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois2"), "rb" ))
            slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois2"), "rb" ))

            patch_list_cross_slice_from_front= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_from_fronts2"), "rb" ))
            thrprobaUIP=float(indata['thrproba']) 
            cnnweigh=indata['picklein_file_front']
            tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs2"), "rb" ))
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,
                                   patch_list_cross_slice_from_front,tabroi,cnnweigh,tabscanLung,viewstyle,slnroi)

    
    elif viewstyle=='reportCross':
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss2"), "rb" ))
#        slicepitch=datarep[3]
        slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois2"), "rb" ))
        cnnweigh=indata['picklein_file']

        tp='Cross'
        proba_cross= pickle.load( open( os.path.join(path_data_dir,"proba_crosss2"), "rb" ))

        thrproba=float(indata['thrproba'])
        tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs2"), "rb" ))
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois2"), "rb" ))
        xtrains  = pickle.load( open( os.path.join(path_data_dir,"xtrainss2"), "rb" ))
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
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacross2"), "rb" ))
#        slicepitch=datarep[3]
        slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois2"), "rb" ))
        cnnweigh=indata['picklein_file']

        tp='FrontProjected'
            
        proba_cross= pickle.load( open( os.path.join(path_data_dir,"proba_fronts2"), "rb" ))

        thrproba=float(indata['thrproba'])
        tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs2"), "rb" ))
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois2"), "rb" ))
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
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacross2"), "rb" ))
#        slicepitch=datarep[3]
        slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois2"), "rb" ))
        cnnweigh=indata['picklein_file']

        tp='Merge'
            
        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,'patch_list_merge_slices2'), "rb" ))

        thrproba=float(indata['thrproba'])
        thrpatch=pickle.load( open( os.path.join(path_data_dir,"thrpatchs2"), "rb" ))
        tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs2"), "rb" ))
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois2"), "rb" ))
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

        xtrains  = pickle.load( open( os.path.join(path_data_dir,"xtrainss2"), "rb" ))

        cnnweigh=indata['picklein_file']
        print cnnweigh
        return ' '
        pathreport=os.path.join(path_patient,reportalldir)
        if not os.path.exists(pathreport):
            os.mkdir(pathreport)
        t = datetime.datetime.now()
        today =tp+'_'+'_weight_'+cnnweigh+ '_th'+str(thrproba)+'_tpat'+'_m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'
        repf=os.path.join(pathreport,reportfile+str(today)+'.txt')
        f=open(repf,'w')
#        remove_folder(pathreport)
        listHug=[]
#        print listHug
        listHugi=indata['lispatientselect']
#        print listHugi
        for lpt in listHugi:
#             print lpt
             pos=lpt.find(' PREDICT!:')
             if pos >0:
                    listHug.append(lpt[0:pos])
             else:
                    pos=lpt.find(' noPREDICT!')
                    if pos >0:
                        listHug.append(lpt[0:pos])
                    else:
                        listHug.append(lpt)
#        print listHug
        print 'report All for', tp, listHug

        if not os.path.exists(pathreport):
            os.mkdir(pathreport)
        messageout = openfichiervolumetxtall(listHug,path_patient,indata,thrproba,cnnweigh,f,tp,xtrains)
        os.startfile(repf)
        
    else:
            messageout='error: unrecognize view style'

    return messageout
