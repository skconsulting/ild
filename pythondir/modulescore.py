# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:21:22 2017

@author: sylvain Kritter 

version 1.5
6 September 2017
"""
#from param_pix_p import *
from param_pix_s import path_data,dimpavx,dimpavy,reportalldir
from param_pix_s import surfelemp

from param_pix_s import white,red,yellow,black
from param_pix_s import source_name,sroi,sroi3d,scan_bmp,transbmp
from param_pix_s import typei1
from param_pix_s import reportdir,reportfile

from param_pix_s import classifc,classifdict,usedclassifdict,oldFormat

from param_pix_s import maxproba,fidclass,rsliceNum,evaluatef

from param_pix_s import normi
from scorepredict import predictrun

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
            
            pathfront=os.path.join(datadir,'datafronts')
            setrefdict[dd]='no'
            if os.path.exists(pathcross):
                datarep= pickle.load( open( os.path.join(datadir,"datacrosss"), "rb" ))
                try:
                    setrefdict[dd]=datarep[5]
#                    print type(setrefdict[dd])
                    if type(setrefdict[dd])==float:
#                        print 'this is float'
                        setrefdict[dd]='oldset0'
                        
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
                      
            crosscompletedf=os.path.join(datadir,'crosscompleteds')
            frontcompletedf=os.path.join(datadir,'frontcompleteds')
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

    return listdir,message

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


def tagviewn(fig,label,surface,surftot,roi,tl,precision,recall,fscore,spc,npv):
    """write text in image according to label and color"""
    
    col=classifc[label]
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
    deltay=30+(11*labnow)
    gro=0.8
    if fscore>0:
        cv2.putText(fig,'Precision: '+'%4s'%(str(precision)+'%')+'  Sensitivity:'+
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
        cp=classif[pat]

        dy=cp
        col=classifc[pat]

        cv2.putText(fig,'%15s'%pat,(deltax, deltay+(12*dy)),cv2.FONT_HERSHEY_PLAIN,gro,col,1)
        cv2.putText(fig,pat[0:7],(deltaxt+(60*dy), deltay-15),cv2.FONT_HERSHEY_PLAIN,gro,col,1)

     
    for i in range (0,n):
        for j in range (0,n):
            dx=deltaxm+60*j
            dy=deltay+12*i
            cv2.putText(fig,'%5s'%str(cm[i][j]),(dx, dy),cv2.FONT_HERSHEY_PLAIN,gro,white,1)
            

def cals(cm,pat):
    numpat=classif[pat]
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
#    print cm
#    print pat,'tp:',tp,'tn:',tn,'fp:',fp,'fn:',fn,'spc:',spc,'npv:',npv
    return prec,recall,fsc, spc,npv ,tp+fp,tp+fn


def drawpatch(t,dx,dy,slicenumber,va,patch_list_cross_slice,volumeroi,slnt,tabroi,num_class,tabscanLung,scaleenh):

    imgn = np.zeros((dx,dy,3), np.uint8)
    datav = np.zeros((500,900,3), np.uint8)
    patchdict=np.zeros((dx,dy), np.uint8)  

    imgpatch=np.zeros((dx,dy), np.uint8)  
#    predictpatu=np.zeros((dx,dy), np.uint8) 
   
    th=t/100.0

    listlabel=[]

    surftot=np.count_nonzero(tabscanLung[slicenumber])

    surftotf= surftot*surfelemp/100
    surftot='surface totale :'+str(int(round(surftotf,0)))+'cm2'
   
    referencepatu= np.copy(tabroi[slicenumber])
    
    if referencepatu.max()>0:
        lvexist=True
    else:
        lvexist=False
                
    dimpavxr=scaleenh*dimpavx
    dimpavyr=scaleenh*dimpavy
    for ll in patch_list_cross_slice[slicenumber]:
            xpat=scaleenh*ll[0][0]
            ypat=scaleenh*ll[0][1]
        #we find max proba from prediction
            proba=ll[1]

            prec, mprobai = maxproba(proba)
           
            if mprobai <th :
                    classlabel='healthy'
            else:
                    classlabel=fidclass(prec,classif)
                
            if classlabel not in listlabel:
                listlabel.append(classlabel)
#                    print listlabel,proba

            cv2.rectangle(imgpatch,(xpat,ypat),(xpat+dimpavxr-1,ypat+dimpavyr-1),classif[classlabel]+1,-1)

            if va[classlabel]==True:
                cv2.rectangle(imgn,(xpat,ypat),(xpat+dimpavxr,ypat+dimpavyr),classifc[classlabel],1)

            if lvexist:
                imgray=np.copy(imgpatch)
                np.putmask(imgray,imgray>0,255)
                mask=np.bitwise_not(imgray)
                patchdict=cv2.bitwise_and(patchdict,patchdict,mask=mask)
                patchdict=cv2.bitwise_or(imgpatch,patchdict)

    volroi={}
    volpat={}
    precision={}
    recall={}
    fscore={}
    spc={}
    npv={}
#    print usedclassif
    for pat in usedclassif:
        volroi[pat]=0
        volpat[pat]=0
        precision[pat]=0
        recall[pat]=0
        fscore[pat]=0
        spc[pat]=0
        npv[pat]=0       
    
    tablung1=np.copy(tabscanLung[slicenumber])
    patchdict=cv2.resize(patchdict,(tablung1.shape[0],tablung1.shape[1]),interpolation=cv2.INTER_LINEAR)
    np.putmask(tablung1,tablung1>0,255)
    predictpatu=np.bitwise_and(tablung1, patchdict) 
    referencepatroi= np.copy(tabroi[slicenumber])
#    referencepatroi=cv2.resize(referencepatroi,(dx,dy),interpolation=cv2.INTER_LINEAR)
    referencepatu=np.bitwise_and(tablung1, referencepatroi) 
#    cv2.imshow('p',normi(predictpatu))
#    cv2.imshow('r',normi(referencepatu))
    
    referencepat= referencepatu.flatten()             
    predictpat=  predictpatu.flatten() 
#    print predictpat.max()

    cm=evaluatef(referencepat,predictpat,num_class)

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
        cv2.putText(datav,'%11s'%('Precision')+
                    '%13s'%('Sensitivity')+ 
                    '%10s'%('Fscore')+
                    '%9s'%('SPC')+
                    '%9s'%('NPV'),
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
        cpa=classif[pat]        
        precision[pat],recall[pat],fscore[pat], spc[pat],npv[pat],volpat[pat],volroi[pat]=cals(cm,pat)
#        if spc[pat]>0:
#                print pat,spc[pat]
        if pat in listlabel:            
            tl=True
        else:
            tl=False
        precisioni=int(round(precision[pat]*100,0))
        recalli=int(round(recall[pat]*100,0))
        fscorei=int(round(fscore[pat]*100,0))
        if cm[classif[pat]].sum()>0:
                numberp+=1
                tpaverage+=cm[cpa][cpa]
                cpp+=cm[:,classif[pat]].sum()
                cpr+=cm[classif[pat]].sum()
                spcAverage+=spc[pat]
                npvAverage+=npv[pat]
           
        spci=int(round(spc[pat]*100,0))
        npvi=int(round(npv[pat]*100,0))
        sul=round((volpat[pat])*surfelemp/100,1)
        suroi=round((volroi[pat])*surfelemp/100,1)
        
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
        cv2.putText(datav,'%12s'%('Precision%')+
                    '%15s'%('Sensitivity%')+ 
                    '%9s'%('Fscore%')+
#                    '%10s'%('SPC %')+
#                    '%10s'%('NPV %')+
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
#                        '%10s'%(spcAverage)+
#                        '%10s'%(npvAverage)+
                        '',
                        (10, 460),cv2.FONT_HERSHEY_PLAIN,0.8,yellow,1)
    delx=120
    ts='Threshold:'+str(t)
    
    cv2.putText(datav,ts,(10,170),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
    cv2.putText(datav,surftot,(delx,170),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
#    cv2.putText(datav,sulunk,(delx,150),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
   
    datav=cv2.cvtColor(datav,cv2.COLOR_BGR2RGB)

    return imgn,datav


def retrievepatch(x,y,sln,dx,dy,patch_list_cross_slice,enhl):

    tabtext = np.zeros((dx,dy,3), np.uint8)
    dimpavxr=enhl*dimpavx
    dimpavyr=enhl*dimpavy
#    ill=-1
    pfound=False
    for ll in patch_list_cross_slice[sln]:

#            print ll
#            slicename=ll[0]
            xs=enhl*ll[0][0]
            ys=enhl*ll[0][1]
        #we find max proba from prediction
            
#            print xs,ys
            if x>xs and x < xs+dimpavxr and y>ys and y<ys+dimpavyr:
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
                     cv2.putText(tabtext,'X',(xs-5+dimpavxr/2,ys+5+dimpavyr/2),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

                     for j in range (0,3):
                             n=n+1
                             strw=lsorted[j][0]+ ' {0:.1f}%'.format(100*lsorted[j][1])
                             cv2.putText(tabtext,strw,(dy-190,(dx-80)+15*n),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

                             print lsorted[j][0], ' {0:.2f}%'.format(100*lsorted[j][1])
                     print'found'
                     break
#    cv2.imshow('image',tabtext)
    if not pfound:
            print'not found'
    return tabtext


def openfichiervolumetxtall(listHug,path_patient,indata,thrprobaUIP,cnnweigh,f,thrpatch,tp):

    num_class=len(classif)
    print 'Score for list of patients:'
    f.write('Score for list of patients:\n')
    for patient in listHug:
        f.write(str(patient)+' ')
    f.write('\n-----\n')
    pf=True
    listroiall=[]
    for patient in listHug:

        patient_path_complet=os.path.join(path_patient,patient)
        path_data_dir=os.path.join(patient_path_complet,path_data)
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss"), "rb" ))
#        slicepitch=datarep[3]
        slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois"), "rb" ))
        if tp=='Cross':
            filepatch="patch_list_cross_slices"
        if tp=='FrontProjected':
            filepatch="patch_list_cross_slice_from_fronts"
        if tp=='Merge':
            filepatch="patch_list_merge_slices"
            
        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,filepatch), "rb" ))
#        patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_sub"), "rb" ))
#        lungSegment= pickle.load( open( os.path.join(path_data_dir,"lungSegment"), "rb" ))
#        tabMed= pickle.load( open( os.path.join(path_data_dir,"tabMed"), "rb" ))
        tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs"), "rb" ))
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois"), "rb" ))
        listroi= pickle.load( open( os.path.join(path_data_dir,"listrois"), "rb" ))
        for key,value in listroi.items():
            for k in value:
                if k not in listroiall:
                    listroiall.append(k)
                          
        ref,pred,messageout = openfichiervolumetxt(patient,path_patient,patch_list_cross_slice,
                      thrprobaUIP,tabroi,datarep,slnroi,tabscanLung,f,cnnweigh,thrpatch,tp,listroi)

        if pf:

            referencepatu=ref
            predictpatu=pred
            pf=False
        else:           
#            print referencepatu.shape
#            print ref.shape
            referencepatu= np.concatenate((referencepatu,ref),axis=0)
            predictpatu= np.concatenate((predictpatu,pred),axis=0)
#    print predictpatu.shape
    referencepat=referencepatu.flatten()
    predictpat=predictpatu.flatten()
#    print 'listroiall',listroiall
    cfma(f,referencepat,predictpat,num_class,'all set',thrprobaUIP,cnnweigh,tp,thrpatch,listroiall)
    f.close()

def cfma(f,referencepat,predictpat,num_class, namep,thrprobaUIP,cnnweigh,tp,thrpatch,listroi):
        f.write('confusion matrix for '+namep+'\n')
        f.write(tp+' View , threshold: '+str(thrprobaUIP)+ ' CNN param: '+cnnweigh+' th_patch'+str(thrpatch)+'\n\n')

        if referencepat.shape[0]==0:
            print 'predict file empty'
            f.write( 'predict file empty\n')
            return
        cm=evaluatef(referencepat,predictpat,num_class)
        n= cm.shape[0]
    
        presip={}
        recallp={}
        fscorep={}
#        print cm
#        fscore={},
        spc={}
        npv={}
        volpat={}
        volroi={}
        for pat in usedclassif:
            presip[pat]=0
            recallp[pat]=0
            fscorep[pat]=0
#            fscore[pat]=0
            spc[pat]=0
            npv[pat]=0
            volpat[pat]=0
            volroi[pat]=0

            presip[pat],recallp[pat],fscorep[pat], spc[pat],npv[pat],volpat[pat],volroi[pat]=cals(cm,pat)
 
        f.write(15*' ')
        for i in range (0,n):
            pat=fidclass(i,classif)
            f.write('%8s'%pat[0:7])
    #    print newclassif
        f.write(' Sensitivity\n')
            
        for i in range (0,n):            
            pat=fidclass(i,classif)
            f.write('%15s'%pat)
            for j in range (0,n):
                f.write('%8s'%str(cm[i][j]))
            f.write( '%8s'%(str(int(round(100*recallp[pat],0)))+'%')+'\n')        
        f.write('----------------------\n')
        f.write('      precision')
        for i in range (0,n):
            pat=fidclass(i,classif)
            f.write('%8s'%(str(int(round(100*presip[pat],0)))+'%'))
    #    print newclassif
        f.write('\n')
        f.write('         Fscore')
        for i in range (0,n):
            pat=fidclass(i,classif)
            f.write('%8s'%(str(int(round(100*fscorep[pat],0)))+'%'))
    #    print newclassif
        f.write('\n--------------------\n')

        wrresu(f,cm,'Report for full Patient',1,listroi)

        f.write('----------------------\n')


def wrresu(f,cm,obj,refmax,listroi):
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
    f.write('pattern   Precision%  Sensitivity%  Fscore%   SPC%     NPV%\n')
    precisionAverage=0
    recallAverage=0
    spcAverage=0
    npvAverage=0
    tpaverage=0
   
    numberp=0
    cpp=0
    cpr=0
    for pat in usedclassif:   
            cpa=classif[pat]        
            precision[pat],recall[pat],fscore[pat], spc[pat],npv[pat],volpat[pat],volroi[pat]=cals(cm,pat)
            if cm[classif[pat]].sum()>0:
                numberp+=1
                tpaverage+=cm[cpa][cpa]
                cpp+=cm[:,classif[pat]].sum()
                cpr+=cm[classif[pat]].sum()
                spcAverage+=spc[pat]
                npvAverage+=npv[pat]

            precisioni=int(round(precision[pat]*100,0))
            recalli=int(round(recall[pat]*100,0))
            fscorei=int(round(fscore[pat]*100,0))
           
            spci=int(round(spc[pat]*100,0))
            npvi=int(round(npv[pat]*100,0))     
#            if precisioni+recalli+fscorei>0:
            f.write('%14s'%pat+'%6s'%precisioni+
                        '%9s'%recalli+'%10s'%fscorei+
                        '%10s'%spci+'%9s'%npvi+'\n')
    f.write('\n')
    if cpp>0:
        precisionAverage=1.0*tpaverage/cpp
    else:
        precisionAverage=0
    if cpr>0:
        recallAverage=1.0*tpaverage/cpr
    else:
        recallAverage=0
    if numberp>0:
        spcAverage/=numberp
        npvAverage/=numberp
    if refmax>0:
        f.write('list of roi: '+str(listroi)+'\n')
        f.write('%10s'%('Precision %')+
                    '%14s'%('Sensitivity%')+ 
                    '%9s'%('Fscore%')+
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
    f.write('---------------------------------\n')


def openfichiervolumetxt(listHug,path_patient,patch_list_cross_slice,
                      thrprobaUIP,tabroi,datacross,slnroi,tabscanLung,f,cnnweigh,thrpatch,tp,listroi):
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichiervolume txt start in',path_patient,' for', listHug, 'predict type ',tp
    slnt=datacross[0]
    dx=datacross[1]
    dy=datacross[2]

    t = datetime.datetime.now()
 
    f.write('report for patient :'+listHug+
            ' - date : m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+' at '+str(t.hour)+'h '+str(t.minute)+'mn\n')
    f.write(tp +' View , threshold: '+str(thrprobaUIP)+ ' CNN_param: '+cnnweigh+' th_patch: '+str(thrpatch) +'\n\n')
 
    slntroi=len(slnroi)

    print 'start Fscore'
    patchdict=np.zeros((slnt,dx,dy), np.uint8)
    predictpatu=np.zeros((slntroi,dx,dy), np.uint8)
    referencepatu=np.zeros((slntroi,dx,dy), np.uint8)
    num_class=len(classif)
    th=thrprobaUIP
    for slroi in range(0,slntroi):
#    for slicenumber in (143,144):
        slicenumber=slnroi[slroi]

        if tabroi[slicenumber].max()>0:
            imgpatch=np.zeros((dx,dy), np.uint8)
            for ll in patch_list_cross_slice[slicenumber]:
                    xpat=ll[0][0]
                    ypat=ll[0][1]
                #we find max proba from prediction
                    proba=ll[1]
        
                    prec, mprobai = maxproba(proba)
                    if mprobai <th:
                        classlabel='healthy'
                    else:
                        classlabel=fidclass(prec,classif)
                        
                    cv2.rectangle(imgpatch,(xpat,ypat),(xpat+dimpavx-1,ypat+dimpavy-1),classif[classlabel]+1,-1)
                    imgray=np.copy(imgpatch)
                    np.putmask(imgray,imgray>0,255)
                    mask=np.bitwise_not(imgray)
#                        print slicenumber, imgpatch.shape
#                        print patchdict[slicenumber].shape
                    patchdict[slicenumber]=cv2.bitwise_and(patchdict[slicenumber],patchdict[slicenumber],mask=mask)
                    patchdict[slicenumber]=cv2.bitwise_or(imgpatch,patchdict[slicenumber])
            tablung1=np.copy(tabscanLung[slicenumber])
            np.putmask(tablung1,tablung1>0,255)
                       
            predictpatu[slroi]=np.bitwise_and(tablung1, patchdict[slicenumber]) 
            referencepatu[slroi]=np.bitwise_and(tablung1, tabroi[slicenumber])  
            
            referencepat= referencepatu[slroi].flatten()
            predictpat=  predictpatu[slroi].flatten() 
            cm=evaluatef(referencepat,predictpat,num_class)
            wrresu(f,cm,'Slice: '+str(slicenumber),referencepatu[slroi].max(),listroi[slicenumber])

#    print 'global matrix'
    listroiall=[]
#    print referencepatu.shape
    referencepat= referencepatu.flatten()
    predictpat=  predictpatu.flatten() 
    for key,value in listroi.items():
        for k in value:
            if k not in listroiall:
                listroiall.append(k)
                          
    cfma(f,referencepat,predictpat,num_class,listHug,thrprobaUIP,cnnweigh,tp,thrpatch,listroiall)
    f.write('----------------------\n')

    return referencepat,predictpat,''

def writeslice(num,menus):
#        print 'write',num
        cv2.rectangle(menus, (5,10), (150,20), red, -1)
        cv2.putText(menus,'Slice to visualize: '+str(num),(5,20),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
        
def openfichier(ti,datacross,path_img,thrprobaUIP,patch_list_cross_slice,tabroi,
                cnnweigh,tabscanLung,viewstyle,slnroi,tabscanroi):   
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichier start' 
    scaleenh=2
    quitl=False
    num_class=len(classif)
    slnt=datacross[0]
    
    dimtabx=scaleenh*datacross[1]
    dimtaby=scaleenh*datacross[2]
#    print slnroi
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
            path_data_writefile=os.path.join(path_data_write,'volumerois')
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
    lenlimage=len(slnroi)

#    print len(limage), sln

    for iimage in limage:

        sln=rsliceNum(iimage,cdelimter,extensionimage)
        list_image[sln]=iimage
#        print sln

#    image0=os.path.join(pdirk,list_image[slnroi[0]])
#    img = cv2.imread(image0,1)
    img=tabscanroi[slnroi[0]]
    img=cv2.resize(img,(dimtaby,dimtabx),interpolation=cv2.INTER_LINEAR)
 
    cv2.namedWindow('score',cv2.WINDOW_NORMAL)
    cv2.namedWindow("Sliderfis",cv2.WINDOW_NORMAL)
    cv2.namedWindow("scoredata",cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar( 'Brightness','Sliderfis',0,100,nothing)
    cv2.createTrackbar( 'Contrast','Sliderfis',50,100,nothing)
    cv2.createTrackbar( 'Threshold','Sliderfis',int(thrprobaUIP*100),100,nothing)
    cv2.createTrackbar( 'Flip','Sliderfis',0,lenlimage-1,nothings)
    cv2.createTrackbar( 'All','Sliderfis',1,1,nothings)
#    cv2.createTrackbar( 'algo','Sliderfis',0,2,nothings)
    cv2.createTrackbar( 'None','Sliderfis',0,1,nothings)
        
    viewasked={}
    for key1 in usedclassif:
#            print key1
        viewasked[key1]=True
        cv2.createTrackbar( key1,'Sliderfis',0,1,nothings)
    nbdig=0
    numberentered={}
    initimg = np.zeros((dimtabx,dimtaby,3), np.uint8)
    slicenumberold=0
    tlold=0
    viewaskedold={}
    
    for keyne in usedclassif:
            viewaskedold[keyne]=False
    datav = np.zeros((500,900,3), np.uint8) 
    imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
    while(1):
    
        imgwip = np.zeros((200,200,3), np.uint8)  
                             
        cv2.setMouseCallback('score',draw_circle,img)
        c = cv2.getTrackbarPos('Contrast','Sliderfis')
        l = cv2.getTrackbarPos('Brightness','Sliderfis')
        tl = cv2.getTrackbarPos('Threshold','Sliderfis')
        fld = cv2.getTrackbarPos('Flip','Sliderfis')
        allview = cv2.getTrackbarPos('All','Sliderfis')
        noneview = cv2.getTrackbarPos('None','Sliderfis')

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
                cv2.rectangle(initimg, (5,10), (150,20), black, -1)
  
                fld=slnroi.index(numberfinal)
#                print fl,numberfinal
                cv2.setTrackbarPos('Flip','Sliderfis' ,fld)
            else:
                print 'number not in set'
#                cv2.rectangle(initimg, (5,60), (150,50), black, -1)
                cv2.rectangle(initimg, (5,10), (150,20), red, -1)
                cv2.putText(initimg,'NO ROI slice '+str(numberfinal)+'!',(5,20),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
#                cv2.putText(initimg, 'NO ROI on this slice',(5,60),cv2.FONT_HERSHEY_PLAIN,5,red,2,cv2.LINE_AA)
#                time.sleep(5)
#                fl=numberfinal
                
            numberfinal=0
            nbdig=0
            numberentered={}
        if key==2424832:
            fld=max(0,fld-1)
            cv2.setTrackbarPos('Flip','Sliderfis' ,fld)
        if key==2555904:
            fld=min(lenlimage-1,fld+1)
            cv2.setTrackbarPos('Flip','Sliderfis' ,fld)
            
        if allview==1:
            for key2 in usedclassif:
                cv2.setTrackbarPos(key2,'Sliderfis' ,1)
            cv2.setTrackbarPos('All','Sliderfis' ,0)
        if noneview==1:
            for key2 in usedclassif:
                cv2.setTrackbarPos(key2,'Sliderfis' ,0)
            cv2.setTrackbarPos('None','Sliderfis' ,0)
        for key2 in usedclassif:
            s = cv2.getTrackbarPos(key2,'Sliderfis')
            if s==1:
                 viewasked[key2]=True               
            else:
                 viewasked[key2]=False

        slicenumber=fl
#        print slicenumber
        
#        imagel=os.path.join(pdirk,list_image[slicenumber])
#        img = cv2.imread(imagel,1) 
        img=tabscanroi[slicenumber] 
        img=lumi(img,l)
        img=contrasti(img,c)    
        img = img.astype('float32')
        img=cv2.resize(img,(dimtaby,dimtabx),interpolation=cv2.INTER_CUBIC)
        img=np.clip(img,0,255)
        img=normi(img)    
#            print img.shape
#            print  initimg.shape
        img=cv2.add(img,initimg)
                
#        imcontrast=cv2.cvtColor(imcontrast,cv2.COLOR_BGR2RGB)
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
            imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
            cv2.putText(imgwip,'WIP',(10,10),cv2.FONT_HERSHEY_PLAIN,5,red,2,cv2.LINE_AA)
            cv2.imshow('wip',imgwip)
            
            imgn,datav= drawpatch(tl,dimtabx,dimtaby,slicenumber,viewasked,patch_list_cross_slice,
                                            volumeroilocal,slnt,tabroi,num_class,tabscanLung,scaleenh)
                           
            cv2.putText(datav,'slice number :'+str(slicenumber),(10,210),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
            cv2.putText(datav,'patient Name :'+tail,(10,220),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
            cv2.putText(datav,'CNN weight: '+cnnweigh,(10,230),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
            cv2.putText(datav,viewstyle,(10,200),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)

            cv2.destroyWindow("wip")
        imgngray = cv2.cvtColor(imgn,cv2.COLOR_BGR2GRAY)
        np.putmask(imgngray,imgngray>0,255)
        mask_inv = cv2.bitwise_not(imgngray)
        outy=cv2.bitwise_and(img,img,mask=mask_inv)
        imgt=cv2.add(imgn,outy)
        dxrect=(dimtaby/2)
        cv2.rectangle(imgt,(dxrect,dimtabx-30),(dxrect+20,dimtabx-10),red,-1)
        cv2.putText(imgt,'quit',(dxrect+10,dimtabx-10),cv2.FONT_HERSHEY_PLAIN,1,yellow,1,cv2.LINE_AA)
           
        if patchi :
            cv2.putText(imgwip,'WIP',(10,10),cv2.FONT_HERSHEY_PLAIN,5,red,2,cv2.LINE_AA)
            cv2.imshow('wip',imgwip)
#                print 'retrieve patch asked'
            imgtext= retrievepatch(ix,iy,slicenumber,dimtabx,dimtaby,patch_list_cross_slice,scaleenh)
            patchi=False
            cv2.destroyWindow("wip")
        imgtoshow=cv2.add(imgt,imgtext)
        imgtoshow=cv2.cvtColor(imgtoshow,cv2.COLOR_BGR2RGB)

        imsstatus=cv2.getWindowProperty('Sliderfis', 0)
        imistatus= cv2.getWindowProperty('score', 0)
        imdstatus=cv2.getWindowProperty('scoredata', 0)
#            print imsstatus,imistatus,imdstatus
        if (imdstatus==0) and (imsstatus==0) and (imistatus==0)  :
            cv2.imshow('score',imgtoshow)
            cv2.imshow('scoredata',datav)
        else:
              quitl=True

        if quitl or cv2.waitKey(20) & 0xFF == 27 :
            break
    quitl=False
    cv2.destroyWindow("score")
    cv2.destroyWindow("Sliderfis")
    cv2.destroyWindow("scoredata")

    return ''
    


def visuarun(indata,path_patient):
    global classif,usedclassif
    print 'visuarun start'

    messageout=""

    viewstyle=indata['viewstyle']
    if viewstyle == 'reportAll':
        lpt=indata['lispatientselect'][0]
    else:
#    print 'path_patient',path_patient
        lpt=indata['lispatientselect']
    pos=lpt.find(' ')
    if pos>0:
        listHug=(lpt[0:pos])
    else:
        listHug=lpt

    patient_path_complet=os.path.join(path_patient,listHug)
    path_data_dir=os.path.join(patient_path_complet,path_data)

    crosscompleted = pickle.load(open( os.path.join(path_data_dir,"crosscompleteds"), "rb" ))
  
    if not crosscompleted:
        messageout="no predict!for "+listHug
        return messageout

    if oldFormat:
        setref='set0'
    else:
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss"), "rb" ))
        setref=datarep[5]
#        print setref
    
        classif=classifdict[setref]
        usedclassif=usedclassifdict[setref]
#    print usedclassif,setref


#    pathhtml=os.path.join(patient_path_complet,htmldir)
    
    if viewstyle=='cross view':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss"), "rb" ))
            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois"), "rb" ))
            slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois"), "rb" ))
            tabscanroi=pickle.load( open( os.path.join(path_data_dir,"tabscanrois"), "rb" ))
            tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs"), "rb" ))
            patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slices"), "rb" ))
            thrproba=float(indata['thrproba'])
            cnnweigh=indata['picklein_file']
#            print cnnweigh
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrproba,
                                   patch_list_cross_slice,tabroi,cnnweigh,tabscanLung,viewstyle,slnroi,tabscanroi)
            
    elif viewstyle=='front view':
            return 'not implemented'
            datarep= pickle.load( open( os.path.join(path_data_dir,"datafronts"), "rb" ))
#            tabroif= pickle.load( open( os.path.join(path_data_dir,"tabroif"), "rb" ))
            slnt=datarep[0]
            dimtabx=datarep[1]
            dimtaby=datarep[2]
            
            tabLung3d= pickle.load( open( os.path.join(path_data_dir,"tabLung3ds"), "rb" ))
            tabroi=np.zeros((slnt,dimtabx,dimtaby), np.uint8)
            patch_list_front_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_front_slices"), "rb" ))
            thrprobaUIP=float(indata['thrproba'])
            cnnweigh=indata['picklein_file_front']
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,
                                   patch_list_front_slice,tabroi,cnnweigh,tabLung3d,viewstyle)
            

    elif viewstyle=='merge view':

            thrprobaUIP=float(indata['thrproba'])
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss"), "rb" ))
            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois"), "rb" ))
            tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs"), "rb" ))
            slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois"), "rb" ))

#            patch_list_merge= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge"), "rb" ))
#            proba_merge= pickle.load( open( os.path.join(path_data_dir,"proba_merge"), "rb" ))
            patch_list_merge_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge_slices"), "rb" ))
#            patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
            cnnweigh=indata['picklein_file']
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,
                                   patch_list_merge_slice,tabroi,cnnweigh,tabscanLung,viewstyle,slnroi)
    
    elif viewstyle=='front projected view':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss"), "rb" ))
            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois"), "rb" ))
            slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois"), "rb" ))

            patch_list_cross_slice_from_front= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_from_fronts"), "rb" ))
            thrprobaUIP=float(indata['thrproba']) 
            cnnweigh=indata['picklein_file_front']
            tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs"), "rb" ))
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
        
        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,'patch_list_cross_slices'), "rb" ))
#        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
#        patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_sub"), "rb" ))
#        lungSegment= pickle.load( open( os.path.join(path_data_dir,"lungSegment"), "rb" ))
#        tabMed= pickle.load( open( os.path.join(path_data_dir,"tabMed"), "rb" ))
        thrproba=float(indata['thrproba'])
        thrpatch=pickle.load( open( os.path.join(path_data_dir,"thrpatchs"), "rb" ))
        tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs"), "rb" ))
        
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois"), "rb" ))
        listroi= pickle.load( open( os.path.join(path_data_dir,"listrois"), "rb" ))
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
                      tabroi,datarep,slnroi,tabscanLung,f,cnnweigh,thrpatch,tp,listroi)
        f.close()
        os.startfile(repf)
        
    elif viewstyle=='reportFront':
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss"), "rb" ))
#        slicepitch=datarep[3]
        slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois"), "rb" ))
        cnnweigh=indata['picklein_file']

        tp='FrontProjected'
            
        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,'patch_list_cross_slice_from_fronts'), "rb" ))
#        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
#        patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_sub"), "rb" ))
#        lungSegment= pickle.load( open( os.path.join(path_data_dir,"lungSegment"), "rb" ))
#        tabMed= pickle.load( open( os.path.join(path_data_dir,"tabMed"), "rb" ))
        thrproba=float(indata['thrproba'])
        thrpatch=pickle.load( open( os.path.join(path_data_dir,"thrpatchs"), "rb" ))
        tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs"), "rb" ))
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois"), "rb" ))
        listroi= pickle.load( open( os.path.join(path_data_dir,"listrois"), "rb" ))
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
                      tabroi,datarep,slnroi,tabscanLung,f,cnnweigh,thrpatch,tp,listroi)
        f.close()
        os.startfile(repf)
    elif viewstyle=='reportMerge':
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss"), "rb" ))
#        slicepitch=datarep[3]
        slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois"), "rb" ))
        cnnweigh=indata['picklein_file']

        tp='Merge'
            
        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,'patch_list_merge_slices'), "rb" ))
#        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
#        patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_sub"), "rb" ))
#        lungSegment= pickle.load( open( os.path.join(path_data_dir,"lungSegment"), "rb" ))
#        tabMed= pickle.load( open( os.path.join(path_data_dir,"tabMed"), "rb" ))
        thrproba=float(indata['thrproba'])
        thrpatch=pickle.load( open( os.path.join(path_data_dir,"thrpatchs"), "rb" ))
        tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs"), "rb" ))
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois"), "rb" ))
        listroi= pickle.load( open( os.path.join(path_data_dir,"listrois"), "rb" ))
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
                      tabroi,datarep,slnroi,tabscanLung,f,cnnweigh,thrpatch,tp,listroi)
        f.close()
        os.startfile(repf)
        
    elif viewstyle=='reportAll':
        tp=indata['viewstylet']
        
        thrproba=float(indata['thrproba'])
        thrpatch=float(indata['thrpatch'])
        cnnweigh=indata['picklein_file']
        pathreport=os.path.join(path_patient,reportalldir)
        if not os.path.exists(pathreport):
            os.mkdir(pathreport)
        t = datetime.datetime.now()
        today =tp+'_'+'_weight_'+cnnweigh+ '_th'+str(thrproba)+'_tpat'+str(thrpatch)+'_m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'
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
        messageout = openfichiervolumetxtall(listHug,path_patient,indata,thrproba,cnnweigh,f,thrpatch,tp)
        os.startfile(repf)
        
    else:
            messageout='error: unrecognize view style'

    return messageout
