# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:21:22 2017

@author: sylvain Kritter 

version 1.5
6 September 2017
"""
#from param_pix_p import *
from param_pix_c import path_data,dimpavx,dimpavy,reportalldir
from param_pix_c import surfelemp

from param_pix_c import white,red,yellow,black
from param_pix_c import source_name,sroi,sroi3d,scan_bmp,transbmp
from param_pix_c import typei1
from param_pix_c import reportdir,reportfile

from param_pix_c import classifc,classifdict,usedclassifdict,oldFormat

from param_pix_c import maxproba,excluvisu,fidclass,rsliceNum,evaluate,evaluatef

from param_pix_c import normi
from comparecalc import predictrun

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
    print 'listdirprocess',a
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

def predictmodule(indata,path_patient_ref,path_patient_comp):
#    print 'module predict'
    message=''
    listdir=[]
    nota=True
    print indata,path_patient_ref,path_patient_comp
    try:
        listdiri= indata['lispatientselectref']
    except KeyError:
            print 'No patient ref selected'
            nota=False
    try:
        listdiri= indata['lispatientselectcomp']
    except KeyError:
            print 'No patient comp selected'
            nota=False
    if nota:
        message=predictrun(indata,path_patient_ref,path_patient_comp)

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

def colorimage(image,color):
#    im=image.copy()
    im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    np.putmask(im,im>0,color)
    return im

def drawpatch(dx,dy,num_class,slicenumber,viewasked,slnroiref,slnroicomp,tabroiref,tabroicomp,tabscanLung,va,volroiref,volroicomp):


    imgn = np.zeros((dx,dy,3), np.uint8)
    datav = np.zeros((500,900,3), np.uint8) 
    listlabel=[]

    surftot=np.count_nonzero(tabscanLung[slicenumber])

    surftotf= surftot*surfelemp/100
    surftot='surface totale :'+str(int(round(surftotf,0)))+'cm2'
   
    referencepatu= np.copy(tabroiref[slicenumber])
    predictpatu=np.copy(tabroicomp[slicenumber])
                  
    
    for pat in usedclassif:
          classcolor=classifc[pat]
          classn=classif[pat]+1
          patchdictvisu=np.copy(tabroicomp[slicenumber])
          if va[pat]==True:
              np.putmask(patchdictvisu,patchdictvisu!=classn,0)
              np.putmask(patchdictvisu,patchdictvisu==classn,255)
              patchdictvisu=colorimage(patchdictvisu,classcolor)
              imgn=cv2.addWeighted(imgn,1,patchdictvisu,0.5,0)
             
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
        sul=round((volroiref[slicenumber][pat]),1)
        suroi=round((volpat[pat])*surfelemp/100,1)
        
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

    cv2.putText(datav,surftot,(delx,170),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
#    cv2.putText(datav,sulunk,(delx,150),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
   
    datav=cv2.cvtColor(datav,cv2.COLOR_BGR2RGB)

    return imgn,datav





def openfichiervolumetxtall( listHug,path_patientref,path_patient_comp,indata,f):
    global classif,usedclassif
    num_class=len(classif)
    print 'Score for list of patients:'
    f.write('Score for list of patients:\n')
    for patient in listHug:
        f.write(str(patient)+' ')
    f.write('\n-----\n')
    pf=True
    listroiall=[]
    
    for patient in listHug:
        posu=patient.find('_')
        namref=patient[posu+1:]
        patient_path_complet_ref=os.path.join(path_patientref,patient)
        path_data_dir_ref=os.path.join(patient_path_complet_ref,path_data)
    
        datacrossref=pickle.load( open( os.path.join(path_data_dir_ref,"datacrossref"), "rb" ))
        setref=datacrossref[3]
        rootcomp=datacrossref[4]  
        classif=classifdict[setref]
        usedclassif=usedclassifdict[setref]
        fcomp=rootcomp+'_'+namref
        patient_path_complet_comp=os.path.join(path_patient_comp,fcomp)
        path_data_dir_comp=os.path.join(patient_path_complet_comp,path_data)
        tabroiref= pickle.load( open( os.path.join(path_data_dir_ref,"tabroiref"), "rb" ))
        slnroiref= pickle.load( open( os.path.join(path_data_dir_ref,"slnroiref"), "rb" ))
        tabroicomp= pickle.load( open( os.path.join(path_data_dir_comp,"tabroicomp"), "rb" ))
        slnroicomp= pickle.load( open( os.path.join(path_data_dir_comp,"slnroicomp"), "rb" ))
        tabscanLungref=pickle.load( open( os.path.join(path_data_dir_ref,"tabscanLungref"), "rb" ))
        volumeroiref=pickle.load( open( os.path.join(path_data_dir_ref,"volumeroiref"), "rb" ))
        volumeroicomp=pickle.load( open( os.path.join(path_data_dir_comp,"volumeroicomp"), "rb" ))
        listroiref=pickle.load( open( os.path.join(path_data_dir_ref,"listroiref"), "rb" ))
        
        for key,value in listroiref.items():
            for k in value:
                if k not in listroiall:
                    listroiall.append(k)
                          
        ref,pred,messageout = openfichiervolumetxt(datacrossref,patient_path_complet_ref,patient_path_complet_comp,
                                   slnroiref,slnroicomp,tabroiref,tabroicomp,
                                   tabscanLungref,volumeroiref,volumeroicomp,f,patient,listroiref)

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
    cfma(f,referencepat,predictpat,num_class,'all set',listroiall)
    f.close()

def cfma(f,referencepat,predictpat,num_class, namep,listroi):
        f.write('confusion matrix for '+str(namep)+'\n')

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
        f.write('  recall \n')
            
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
    f.write('    pattern   precision%  recall%  Fscore%   SPC%     NPV%\n')
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
            if pat in listroi:
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
            if precisioni+recalli+fscorei>0:
                f.write('%14s'%pat+'%7s'%precisioni+
                        '%9s'%recalli+'%9s'%fscorei+
                        '%9s'%spci+'%9s'%npvi+'\n')
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
    f.write('---------------------------------\n')


def  openfichiervolumetxt(datacrossref,patient_path_complet_ref,patient_path_complet_comp,
                                   slnroiref,slnroicomp,tabroiref,tabroicomp,tabscanLungref,volumeroiref,volumeroicomp,f,listHug,listroi):
    
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichiervolume txt start in',patient_path_complet_comp,' for', listHug
#    slnt=datacrossref[0]
    dx=datacrossref[1]
    dy=dx

    t = datetime.datetime.now()
 
    f.write('report for patient :'+str(listHug)+
            ' - date : m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+' at '+str(t.hour)+'h '+str(t.minute)+'mn\n')
 
    slntroi=len(slnroiref)

    print 'start Fscore'
#    patchdict=np.zeros((slnt,dx,dy), np.uint8)
    predictpatu=np.zeros((slntroi,dx,dy), np.uint8)
    referencepatu=np.zeros((slntroi,dx,dy), np.uint8)
    num_class=len(classif)

    for slroi in range(0,slntroi):
#    for slicenumber in (143,144):
        slicenumber=slnroiref[slroi]

        if tabroiref[slicenumber].max()>0:
            referencepatu[slroi]= np.copy(tabroiref[slicenumber])
            predictpatu[slroi]=np.copy(tabroicomp[slicenumber])
        
            
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
                          
    cfma(f,referencepat,predictpat,num_class,listHug,listroiall)
    f.write('----------------------\n')

    return referencepat,predictpat,''

def writeslice(num,menus):
#        print 'write',num
        cv2.rectangle(menus, (5,10), (150,20), red, -1)
        cv2.putText(menus,'Slice to visualize: '+str(num),(5,20),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
        
def openfichier(datacross,patient_path_complet_ref,slnroiref,slnroicomp,tabroiref,tabroicomp,tabscanLung,volroiref,volroicomp):   
    
    
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichier start' 
    
    quitl=False
    num_class=len(classif)
    slnt=datacross[0]
    dimtabx=datacross[1]
    dimtaby=dimtabx
#    print slnroi
    patchi=False
    ix=0
    iy=0
    (top,tail)=os.path.split(patient_path_complet_ref)

#    pdirk = os.path.join(patient_path_complet_ref,source_name)
    pdirk = os.path.join(patient_path_complet_ref,sroi)

    list_image={}
    cdelimter='_'
    extensionimage='.'+typei1
    limage=[name for name in os.listdir(pdirk) if name.find('.'+typei1,1)>0 ]
    lenlimage=len(slnroiref)

#    print len(limage), sln

    for iimage in limage:

        sln=rsliceNum(iimage,cdelimter,extensionimage)
        list_image[sln]=iimage
#        print sln

    image0=os.path.join(pdirk,list_image[slnroiref[0]])
    img = cv2.imread(image0,1)
    img=cv2.resize(img,(dimtaby,dimtabx),interpolation=cv2.INTER_LINEAR)
 
    cv2.namedWindow('compare',cv2.WINDOW_NORMAL)
    cv2.namedWindow("Sliderfic",cv2.WINDOW_NORMAL)
    cv2.namedWindow("comparedata",cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar( 'Brightness','Sliderfic',0,100,nothing)
    cv2.createTrackbar( 'Contrast','Sliderfic',50,100,nothing)
    cv2.createTrackbar( 'Flip','Sliderfic',0,lenlimage-1,nothings)
    cv2.createTrackbar( 'All','Sliderfic',1,1,nothings)
    cv2.createTrackbar( 'None','Sliderfic',0,1,nothings)
        
    viewasked={}
    for key1 in usedclassif:
#            print key1
        viewasked[key1]=True
        cv2.createTrackbar( key1,'Sliderfic',0,1,nothings)
    nbdig=0
    numberentered={}
    initimg = np.zeros((dimtabx,dimtaby,3), np.uint8)
    slicenumberold=0

    viewaskedold={}
    
    for keyne in usedclassif:
            viewaskedold[keyne]=False
    datav = np.zeros((500,900,3), np.uint8) 
    imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
    while(1):
    
        imgwip = np.zeros((200,200,3), np.uint8)  
                             
        cv2.setMouseCallback('compare',draw_circle,img)
        c = cv2.getTrackbarPos('Contrast','Sliderfic')
        l = cv2.getTrackbarPos('Brightness','Sliderfic')
        fld = cv2.getTrackbarPos('Flip','Sliderfic')
        allview = cv2.getTrackbarPos('All','Sliderfic')
        noneview = cv2.getTrackbarPos('None','Sliderfic')
        fl=slnroiref[fld]
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
            if numberfinal in slnroiref:
#                    print numberfinal
                numberfinal = min(slnt-1,numberfinal)
#                    writeslice(numberfinal,initimg)
                cv2.rectangle(initimg, (5,10), (150,20), black, -1)
  
                fld=slnroiref.index(numberfinal)
#                print fl,numberfinal
                cv2.setTrackbarPos('Flip','Sliderfic' ,fld)
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
            cv2.setTrackbarPos('Flip','Sliderfic' ,fld)
        if key==2555904:
            fld=min(lenlimage-1,fld+1)
            cv2.setTrackbarPos('Flip','Sliderfic' ,fld)
            
        if allview==1:
            for key2 in usedclassif:
                cv2.setTrackbarPos(key2,'Sliderfic' ,1)
            cv2.setTrackbarPos('All','Sliderfic' ,0)
        if noneview==1:
            for key2 in usedclassif:
                cv2.setTrackbarPos(key2,'Sliderfic' ,0)
            cv2.setTrackbarPos('None','Sliderfic' ,0)
        for key2 in usedclassif:
            s = cv2.getTrackbarPos(key2,'Sliderfic')
            if s==1:
                 viewasked[key2]=True               
            else:
                 viewasked[key2]=False

        slicenumber=fl
#        print slicenumber
        
        imagel=os.path.join(pdirk,list_image[slicenumber])
        img = cv2.imread(imagel,1)               
        img=cv2.resize(img,(dimtaby,dimtabx),interpolation=cv2.INTER_LINEAR)
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
            
            imgn,datav= drawpatch(dimtabx,dimtaby,num_class,slicenumber,viewasked,
                                  slnroiref,slnroicomp,tabroiref,tabroicomp,tabscanLung,viewasked,volroiref,volroicomp)
                           
            cv2.putText(datav,'slice number :'+str(slicenumber),(10,210),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
            cv2.putText(datav,'patient Name :'+tail,(10,220),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)

            cv2.destroyWindow("wip")
#        imgngray = cv2.cvtColor(imgn,cv2.COLOR_BGR2GRAY)
#        np.putmask(imgngray,imgngray>0,255)
#        mask_inv = cv2.bitwise_not(imgngray)
#        outy=cv2.bitwise_and(imcontrast,imcontrast,mask=mask_inv)
        imgt=cv2.add(imcontrast,imgn)
        dxrect=(dimtaby/2)
        cv2.rectangle(imgt,(dxrect,dimtabx-30),(dxrect+20,dimtabx-10),red,-1)
        cv2.putText(imgt,'quit',(dxrect+10,dimtabx-10),cv2.FONT_HERSHEY_PLAIN,1,yellow,1,cv2.LINE_AA)
           
        imgtoshow=cv2.add(imgt,imgtext)
        imgtoshow=cv2.cvtColor(imgtoshow,cv2.COLOR_BGR2RGB)

        imsstatus=cv2.getWindowProperty('Sliderfic', 0)
        imistatus= cv2.getWindowProperty('compare', 0)
        imdstatus=cv2.getWindowProperty('comparedata', 0)
#            print imsstatus,imistatus,imdstatus
        if (imdstatus==0) and (imsstatus==0) and (imistatus==0)  :
            cv2.imshow('compare',imgtoshow)
            cv2.imshow('comparedata',datav)
        else:
              quitl=True

        if quitl or cv2.waitKey(20) & 0xFF == 27 :
            break
    quitl=False
    cv2.destroyWindow("compare")
    cv2.destroyWindow("Sliderfic")
    cv2.destroyWindow("comparedata")

    return ''
    


def visuarun(indata,path_patient_ref,path_patient_comp):
    global classif,usedclassif
    print 'visuarun start'

    messageout=""
    
#    print usedclassif,setref
    viewstyle=indata['viewstyle']
#    lispatientselectref=indata['lispatientselectref']
#    lispatientselect=indata['lispatientselect']
    lpt=indata['lispatientselect'][0]
    pos=lpt.find(' ')
    if pos>0:
        listHug=(lpt[0:pos])
    else:
        listHug=lpt
   
    patient_path_complet_ref=os.path.join(path_patient_ref,listHug)
    path_data_dir_ref=os.path.join(patient_path_complet_ref,path_data)
    
    posu=listHug.find('_')
    namref=listHug[posu+1:]
    
#    print 'path_data_dir_ref',path_data_dir_ref
#    pathhtml=os.path.join(patient_path_complet,htmldir)
    
    if viewstyle=='view':    
            datacrossref=pickle.load( open( os.path.join(path_data_dir_ref,"datacrossref"), "rb" ))
            setref=datacrossref[3]
            rootcomp=datacrossref[4]  
            fcomp=rootcomp+'_'+namref
            patient_path_complet_comp=os.path.join(path_patient_comp,fcomp)
            path_data_dir_comp=os.path.join(patient_path_complet_comp,path_data)
            classif=classifdict[setref]
            usedclassif=usedclassifdict[setref]
            tabroiref= pickle.load( open( os.path.join(path_data_dir_ref,"tabroiref"), "rb" ))
            slnroiref= pickle.load( open( os.path.join(path_data_dir_ref,"slnroiref"), "rb" ))
            tabroicomp= pickle.load( open( os.path.join(path_data_dir_comp,"tabroicomp"), "rb" ))
            slnroicomp= pickle.load( open( os.path.join(path_data_dir_comp,"slnroicomp"), "rb" ))
            tabscanLungref=pickle.load( open( os.path.join(path_data_dir_ref,"tabscanLungref"), "rb" ))
            volumeroiref=pickle.load( open( os.path.join(path_data_dir_ref,"volumeroiref"), "rb" ))
            volumeroicomp=pickle.load( open( os.path.join(path_data_dir_comp,"volumeroicomp"), "rb" ))
#            print cnnweigh
            messageout=openfichier(datacrossref,patient_path_complet_ref,
                                   slnroiref,slnroicomp,tabroiref,tabroicomp,tabscanLungref,volumeroiref,volumeroicomp)
#            
#    elif viewstyle=='front view':
#            return 'not implemented'
#            datarep= pickle.load( open( os.path.join(path_data_dir,"datafronts"), "rb" ))
##            tabroif= pickle.load( open( os.path.join(path_data_dir,"tabroif"), "rb" ))
#            slnt=datarep[0]
#            dimtabx=datarep[1]
#            dimtaby=datarep[2]
#            
#            tabLung3d= pickle.load( open( os.path.join(path_data_dir,"tabLung3ds"), "rb" ))
#            tabroi=np.zeros((slnt,dimtabx,dimtaby), np.uint8)
#            patch_list_front_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_front_slices"), "rb" ))
#            thrprobaUIP=float(indata['thrproba'])
#            cnnweigh=indata['picklein_file_front']
#            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,
#                                   patch_list_front_slice,tabroi,cnnweigh,tabLung3d,viewstyle)
#            
#
#    elif viewstyle=='merge view':
#
#            thrprobaUIP=float(indata['thrproba'])
#            datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss"), "rb" ))
#            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois"), "rb" ))
#            tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs"), "rb" ))
#            slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois"), "rb" ))
#
##            patch_list_merge= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge"), "rb" ))
##            proba_merge= pickle.load( open( os.path.join(path_data_dir,"proba_merge"), "rb" ))
#            patch_list_merge_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge_slices"), "rb" ))
##            patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
#            cnnweigh=indata['picklein_file']
#            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,
#                                   patch_list_merge_slice,tabroi,cnnweigh,tabscanLung,viewstyle,slnroi)
#    
#    elif viewstyle=='front projected view':
#            datarep= pickle.load( open( os.path.join(path_data_dir,"datacrosss"), "rb" ))
#            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabrois"), "rb" ))
#            slnroi= pickle.load( open( os.path.join(path_data_dir,"slnrois"), "rb" ))
#
#            patch_list_cross_slice_from_front= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_from_fronts"), "rb" ))
#            thrprobaUIP=float(indata['thrproba']) 
#            cnnweigh=indata['picklein_file_front']
#            tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLungs"), "rb" ))
#            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,
#                                   patch_list_cross_slice_from_front,tabroi,cnnweigh,tabscanLung,viewstyle,slnroi)
##            thrprobaUIP=float(indata['thrprobaUIP'])
##            tabfromfront= pickle.load( open( os.path.join(path_data_dir,"tabfromfront"), "rb" ))
##            messageout=openfichierfrpr(path_patient,tabfromfront,thrprobaUIP)
    
    elif viewstyle=='report':
        datacrossref=pickle.load( open( os.path.join(path_data_dir_ref,"datacrossref"), "rb" ))
        setref=datacrossref[3]
        rootcomp=datacrossref[4]  
        fcomp=rootcomp+'_'+namref
        patient_path_complet_comp=os.path.join(path_patient_comp,fcomp)
        path_data_dir_comp=os.path.join(patient_path_complet_comp,path_data)
        tabroiref= pickle.load( open( os.path.join(path_data_dir_ref,"tabroiref"), "rb" ))
        slnroiref= pickle.load( open( os.path.join(path_data_dir_ref,"slnroiref"), "rb" ))
        tabroicomp= pickle.load( open( os.path.join(path_data_dir_comp,"tabroicomp"), "rb" ))
        slnroicomp= pickle.load( open( os.path.join(path_data_dir_comp,"slnroicomp"), "rb" ))
        tabscanLungref=pickle.load( open( os.path.join(path_data_dir_ref,"tabscanLungref"), "rb" ))
        volumeroiref=pickle.load( open( os.path.join(path_data_dir_ref,"volumeroiref"), "rb" ))
        volumeroicomp=pickle.load( open( os.path.join(path_data_dir_comp,"volumeroicomp"), "rb" ))
        listroiref=pickle.load( open( os.path.join(path_data_dir_ref,"listroiref"), "rb" ))
#        dirf=os.path.join(patient_path_complet_comp,listHug)
        dirfreport=os.path.join(patient_path_complet_comp,reportdir)   
        if not os.path.exists(dirfreport):
            os.mkdir(dirfreport)
        t = datetime.datetime.now()
        today ='_weight_'+'_m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'
        repf=os.path.join(dirfreport,reportfile+str(today)+'.txt')
        f=open(repf,'w')
        x,x,messageout = openfichiervolumetxt(datacrossref,patient_path_complet_ref,patient_path_complet_comp,
                                   slnroiref,slnroicomp,tabroiref,tabroicomp,
                                   tabscanLungref,volumeroiref,volumeroicomp,f,listHug,listroiref)
        f.close()
        os.startfile(repf)
        
   
        
    elif viewstyle=='reportAll':
        print 'report All for',

        pathreport=os.path.join(path_patient_comp,reportalldir)
        if not os.path.exists(pathreport):
            os.mkdir(pathreport)
        t = datetime.datetime.now()
        today ='_All_'+'_m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'
        repf=os.path.join(pathreport,reportfile+str(today)+'.txt')
        f=open(repf,'w')
#        remove_folder(pathreport)
        listHug=[]
        a,b,c=lisdirprocess(path_patient_ref)
        for patient in a:
                listHug.append(patient)
        if not os.path.exists(pathreport):
            os.mkdir(pathreport)
        messageout = openfichiervolumetxtall(listHug,path_patient_ref,path_patient_comp,indata,f)
        os.startfile(repf)
        
    else:
            messageout='error: unrecognize view style'

    return messageout
