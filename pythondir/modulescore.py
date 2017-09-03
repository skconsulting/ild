# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:21:22 2017

@author: sylvain
version 1.2
17 august 2017
"""
#from param_pix_p import *
from param_pix_s import path_data,datafrontn,datacrossn,dimpavx,dimpavy,reportalldir
from param_pix_s import surfelemp,volumeroifilep

from param_pix_s import white,red,yellow,black
from param_pix_s import source_name,sroi,sroi3d,scan_bmp,transbmp
from param_pix_s import typei1
from param_pix_s import reportdir,reportfile

from param_pix_s import classifc,classifdict,usedclassifdict,oldFormat,writeFile,volumeweb

from param_pix_s import maxproba,excluvisu,fidclass,rsliceNum,evaluate,evaluatef,evaluatefull

from scorepredict import predictrun

import cPickle as pickle
import os
import cv2
import numpy as np
import datetime


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
            pathcross=os.path.join(datadir,datacrossn)
            
            pathfront=os.path.join(datadir,datafrontn)
            setrefdict[dd]='no'
            if os.path.exists(pathcross):
                datarep= pickle.load( open( os.path.join(datadir,"datacross"), "rb" ))
                try:
                    setrefdict[dd]=datarep[5]
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
            stsdir[dd]=stpred

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
    lcl=len(classif)-1
    for pat in classif:
        cp=classif[pat]
        if cp==lcl:
            dy=0
        elif cp==0:
            dy=lcl
        else:
            dy=cp
        col=classifc[pat]

        cv2.putText(fig,'%15s'%pat,(deltax, deltay+(12*dy)),cv2.FONT_HERSHEY_PLAIN,gro,col,1)
        cv2.putText(fig,pat[0:7],(deltaxt+(60*dy), deltay-15),cv2.FONT_HERSHEY_PLAIN,gro,col,1)

     
    for i in range (0,n):
        for j in range (0,n):
            dx=deltaxm+60*j
            dy=deltay+12*i
            cv2.putText(fig,'%5s'%str(cm[i][j]),(dx, dy),cv2.FONT_HERSHEY_PLAIN,gro,white,1)




def drawpatch(t,dx,dy,slicenumber,va,patch_list_cross_slice,volumeroi,slnt,tabroi,num_class,tabscanLung):

    imgn = np.zeros((dx,dy,3), np.uint8)
    datav = np.zeros((500,900,3), np.uint8)
    patchdict=np.zeros((slnt,dx,dy), np.uint8)  
#    np.putmask(patchdict,patchdict==0,classif['lung'])
    imgpatch=np.zeros((dx,dy), np.uint8)  
    predictpatu=np.zeros((slnt,dx,dy), np.uint8) 
    referencepatu=np.zeros((slnt,dx,dy), np.uint8) 
   
    th=t/100.0
    numl=0
    listlabel={}

    surftot=np.count_nonzero(tabscanLung[slicenumber])

    surftotf= surftot*surfelemp/100
    surftot='surface totale :'+str(int(round(surftotf,0)))+'cm2'
    volpat={}
    for pat in usedclassif:
        volpat[pat]=np.zeros((dx,dy), np.uint8)
    
    
    lvexist=False
    if len (volumeroi)>0:
#        print 'volumeroi exists'
        lv=volumeroi[slicenumber]
        for pat,value in lv.items():
#            print pat,value
            if value>0:
                lvexist=True
                break
    
    for ll in patch_list_cross_slice[slicenumber]:
            xpat=ll[0][0]
            ypat=ll[0][1]
        #we find max proba from prediction
            proba=ll[1]

            prec, mprobai = maxproba(proba)

            classlabel=fidclass(prec,classif)
            classcolor=classifc[classlabel]

            if mprobai >th and classlabel not in excluvisu:
                volpat[classlabel][ypat:ypat+dimpavy,xpat:xpat+dimpavx]=1
                if classlabel in listlabel:
                    numl=listlabel[classlabel]
                    listlabel[classlabel]=numl+1
                else:
                    listlabel[classlabel]=1
                
                if classif[classlabel]>0:
                        cv2.rectangle(imgpatch,(xpat,ypat),(xpat+dimpavx,ypat+dimpavy),classif[classlabel],-1)
                else:
                        cv2.rectangle(imgpatch,(xpat,ypat),(xpat+dimpavx,ypat+dimpavy),classif['lung'],-1)
                if va[classlabel]==True:
                    cv2.rectangle(imgn,(xpat,ypat),(xpat+dimpavx,ypat+dimpavy),classcolor,1)

                if lvexist:
                    imgray=np.copy(imgpatch)
                    np.putmask(imgray,imgray>0,255)
                    mask=np.bitwise_not(imgray)
                    patchdict[slicenumber]=cv2.bitwise_and(patchdict[slicenumber],patchdict[slicenumber],mask=mask)
                    patchdict[slicenumber]=cv2.bitwise_or(imgpatch,patchdict[slicenumber])

    delx=120
    surftotpat=0
    tablung=np.copy(tabscanLung[slicenumber])
    np.putmask(tablung,tablung>0,255)
        
    for ll1 in usedclassif:     
        volpat[ll1]=np.bitwise_and(tablung, volpat[ll1])          
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
                  
        predictpatu[slicenumber]=np.bitwise_and(tablung, patchdict[slicenumber])  
        referencepatu[slicenumber]= np.copy(tabroi[slicenumber]) 
        
        referencepat= referencepatu[slicenumber].flatten()            
        predictpat=  predictpatu[slicenumber].flatten()   
#        pc=np.copy(patchdict[slicenumber])
#        np.putmask(pc,pc==0,200)
##        np.putmask(pc,pc!=4 and pc!=0,100)
##        pcc=np.copy(pc)
##        np.putmask(pcc,pcc==0,50)
#        cv2.imshow('predictpat',pc)
       
        precision={}
        recall={}
        fscore={}
        cf=False
        for pat in usedclassif:
            numpat=classif[pat]
            fscore[pat]=0
            if numpat == 0:
                numpat=classif['lung']
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


def openfichiervolumetxtall(listHug,path_patient,indata,thrprobaUIP,cnnweigh,f):

    num_class=len(classif)
#    t = datetime.datetime.now()
#    today = '_th'+str(thrprobaUIP)+'_m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'
#    repf=os.path.join(pathreport,reportfile+str(today)+'.txt')
#
#    f=open(repf,'w')
    f.write('Score for list of patients:\n')
    for patient in listHug:
        f.write(str(patient)+' ')
    f.write('\n-----\n')
    pf=True
    for patient in listHug:

        patient_path_complet=os.path.join(path_patient,patient)
        path_data_dir=os.path.join(patient_path_complet,path_data)
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
#        slicepitch=datarep[3]
        slnroi= pickle.load( open( os.path.join(path_data_dir,"slnroi"), "rb" ))
        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
#        patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_sub"), "rb" ))
#        lungSegment= pickle.load( open( os.path.join(path_data_dir,"lungSegment"), "rb" ))
#        tabMed= pickle.load( open( os.path.join(path_data_dir,"tabMed"), "rb" ))
        tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLung"), "rb" ))
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabroi"), "rb" ))
                  
        ref,pred,messageout = openfichiervolumetxt(patient,path_patient,patch_list_cross_slice,
                      thrprobaUIP,tabroi,datarep,slnroi,tabscanLung,f,cnnweigh)
#        print ref.shape
        if pf:
#            print 'first'
            referencepat=ref
            predictpat=pred
            pf=False
        else:           
#            print 'after'
            referencepat= np.concatenate((referencepat,ref),axis=0)
            predictpat= np.concatenate((predictpat,pred),axis=0)
#        print referencepat.shape
#    print referencepat.shape

        
    cfma(f,referencepat,predictpat,num_class,'all set',thrprobaUIP,cnnweigh)
    f.close()

def cfma(f,referencepat,predictpat,num_class, namep,thrprobaUIP,cnnweigh):
        f.write('confusion matrix for '+namep+'\n')
        f.write('Cross View , threshold: '+str(thrprobaUIP)+ ' CNN param: '+cnnweigh+'\n\n')

        
        cm=evaluatef(referencepat,predictpat,num_class)
        n= cm.shape[0]
        newclassif=[]
        newclassif.append(fidclass(len(classif)-1,classif))
        for pat in usedclassif:
            cp=classif[pat]
            if cp>0:
                newclassif.append(pat)
        newclassif.append(fidclass(0,classif))
        presip={}
        recallp={}
        fscorep={}
        for pat in newclassif:
            presip[pat]=0
            recallp[pat]=0
            fscorep[pat]=0
            numpat=classif[pat]
            if numpat == 0:
                numpat=classif['lung']
#            print numpat
            presip[pat], recallp[pat]=evaluate(referencepat,predictpat,num_class,(numpat,))
#            print pat,presip[pat],recallp[pat]
        presip['lung'], recallp['lung']=evaluate(referencepat,predictpat,num_class,(0,))
#        print 'lung',presip['lung'],recallp['lung']
        for pat in newclassif:             
         if presip[pat]+recallp[pat]>0:
            fscorep[pat]=2*presip[pat]*recallp[pat]/(presip[pat]+recallp[pat])
        else:
            fscorep[pat]=0    
        f.write(15*' ')
        for pat in newclassif:
            f.write('%8s'%pat[0:6])
    #    print newclassif
        f.write('  recall \n')
            
        for i in range (0,n):
            pat=newclassif[i]
            f.write('%15s'%pat)
            for j in range (0,n):
                f.write('%8s'%str(cm[i][j]))
            f.write( '%8s'%(str(int(round(100*recallp[pat],0)))+'%')+'\n')        
        f.write('----------------------\n')
        f.write('      precision')
        for pat in newclassif:
            f.write('%8s'%(str(int(round(100*presip[pat],0)))+'%'))
    #    print newclassif
        f.write('\n')
        f.write('         Fscore')
        for pat in newclassif:
            f.write('%8s'%(str(int(round(100*fscorep[pat],0)))+'%'))
    #    print newclassif
        f.write('\n--------------------\n')
        f.write('score per pattern:\n\n')
        for pat in newclassif:
            
            if pat !='lung' and fscorep[pat]>0:
                f.write('%14s'%pat+
                        ' recall: '+'%5s'%(str(int(round(100*recallp[pat],0)))+'%')+
                        ' precision:'+'%5s'%(str(int(round(100*presip[pat],0)))+'%')+
                        ' fscore:'+'%5s'%(str(int(round(100*fscorep[pat],0)))+'%')+'\n')
                  
        precisiont,recallt= evaluatefull(referencepat,predictpat,num_class)
        if precisiont+recallt>0:
            fscore=2*precisiont*recallt/(precisiont+recallt)
        else:
            fscore=0
        precisioni=str(round(precisiont*100,0))+'%'
        recalli=str(round(recallt*100,))+'%'
        fscorei=str(round(fscore*100,0))+'%'
    #                    print (slicenumber,pat,precisioni,recalli,fscorei)
        f.write('----------------------\n')
        f.write('Global scores for patient '+namep+' (without lung):\n')
        f.write('precision:'+precisioni+' recall: '+recalli+' Fscore: '+fscorei+'\n')
        f.write('----------------------\n')

def openfichiervolumetxt(listHug,path_patient,patch_list_cross_slice,
                      thrprobaUIP,tabroi,datacross,slnroi,tabscanLung,f,cnnweigh):
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichiervolume txt start in',path_patient,' for', listHug
    slnt=datacross[0]
    dx=datacross[1]
    dy=datacross[2]
#    print 'slicepitch',slicepitch

   
#    dirfreport=pathreport
#    if not os.path.exists(dirfreport):
#        os.mkdir(dirfreport)
    t = datetime.datetime.now()
#    today = '_th'+str(thrprobaUIP)+'_m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'
#    repf=os.path.join(dirfreport,reportfile+str(today)+'.txt')

#    f=open(repf,'w')
    
    f.write('report for patient :'+listHug+
            ' - date : m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+' at '+str(t.hour)+'h '+str(t.minute)+'mn\n')
    f.write('Cross View , threshold: '+str(thrprobaUIP)+ ' CNN param: '+cnnweigh+'\n\n')
 
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
        
                    classlabel=fidclass(prec,classif)
        
                    if mprobai >th and classlabel not in excluvisu:

        
        #                imgn= addpatchn(classcolor,classlabel,xpat,ypat,imgn)
                        if classif[classlabel]>0:
                            cv2.rectangle(imgpatch,(xpat,ypat),(xpat+dimpavx,ypat+dimpavy),classif[classlabel],-1)
                        else:
                            cv2.rectangle(imgpatch,(xpat,ypat),(xpat+dimpavx,ypat+dimpavy),classif['lung'],-1)
        
                        imgray=np.copy(imgpatch)
                        np.putmask(imgray,imgray>0,255)
                        mask=np.bitwise_not(imgray)
                        patchdict[slicenumber]=cv2.bitwise_and(patchdict[slicenumber],patchdict[slicenumber],mask=mask)
                        patchdict[slicenumber]=cv2.bitwise_or(imgpatch,patchdict[slicenumber])



            tablung=np.copy(tabscanLung[slicenumber])
            np.putmask(tablung,tablung>0,255)
              
            predictpatu[slroi]=np.bitwise_and(tablung, patchdict[slicenumber])  
#            referencepatu[slroi]=np.bitwise_and(tablung, tabroi[slicenumber])  
            referencepatu[slroi]=np.copy(tabroi[slicenumber])

#                cv2.imshow('bpred',normi(patchdict[slicenumber]))
#                cv2.imshow('apred',normi(predictpatu[slicenumber]))
#                cv2.imshow('broi',normi(tabroi[slicenumber]))
#                cv2.imshow('aroi',normi(referencepatu[slicenumber]))
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()
            
            
            referencepat= referencepatu[slroi].flatten()
            predictpat=  predictpatu[slroi].flatten()   
            
            precision={}
            recall={}
            fscore={}
            f.write('Slice :'+str(slicenumber)+'\n')
            f.write('    pattern   precision  recall  Fscore\n')

            for pat in usedclassif:
                numpat=classif[pat]
                fscore[pat]=0
                if numpat == 0:
                    numpat=classif['lung']
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
    cfma(f,referencepat,predictpat,num_class,listHug,thrprobaUIP,cnnweigh)
    f.write('----------------------\n')

    return referencepat,predictpat,''

def writeslice(num,menus):
#        print 'write',num
        cv2.rectangle(menus, (5,60), (150,50), red, -1)
        cv2.putText(menus,'Slice to visualize: '+str(num),(5,60),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
        
def openfichier(ti,datacross,path_img,thrprobaUIP,patch_list_cross_slice,tabroi,
                cnnweigh,tabscanLung,viewstyle,slnroi):   
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichier start' 
    
    quitl=False
    num_class=len(classif)
    slnt=datacross[0]
    dimtabx=datacross[1]
    dimtaby=datacross[2]
    print slnroi
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
    img=cv2.resize(img,(dimtaby,dimtabx),interpolation=cv2.INTER_LINEAR)
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
                                            volumeroilocal,slnt,tabroi,num_class,tabscanLung)
                           
            cv2.putText(datav,'slice number :'+str(slicenumber),(10,180),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
            cv2.putText(datav,'patient Name :'+tail,(10,190),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
            cv2.putText(datav,'CNN weight: '+cnnweigh,(10,170),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
            cv2.putText(datav,viewstyle,(10,200),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)

            cv2.destroyWindow("wip")
        imgngray = cv2.cvtColor(imgn,cv2.COLOR_BGR2GRAY)
        np.putmask(imgngray,imgngray>0,255)
        mask_inv = cv2.bitwise_not(imgngray)
        outy=cv2.bitwise_and(imcontrast,imcontrast,mask=mask_inv)
        imgt=cv2.add(imgn,outy)
        dxrect=(dimtaby/2)
        cv2.rectangle(imgt,(dxrect,dimtabx-30),(dxrect+20,dimtabx-10),red,-1)
        cv2.putText(imgt,'quit',(dxrect+10,dimtabx-10),cv2.FONT_HERSHEY_PLAIN,1,yellow,1,cv2.LINE_AA)
           
        if patchi :
            cv2.putText(imgwip,'WIP',(10,10),cv2.FONT_HERSHEY_PLAIN,5,red,2,cv2.LINE_AA)
            cv2.imshow('wip',imgwip)
#                print 'retrieve patch asked'
            imgtext= retrievepatch(ix,iy,slicenumber,dimtabx,dimtaby,patch_list_cross_slice)
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
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
        setref=datarep[5]
#        print setref
    classif=classifdict[setref]
    usedclassif=usedclassifdict[setref]
#    print usedclassif
    viewstyle=indata['viewstyle']

#    pathhtml=os.path.join(patient_path_complet,htmldir)
    
    if viewstyle=='cross view':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabroi"), "rb" ))
            slnroi= pickle.load( open( os.path.join(path_data_dir,"slnroi"), "rb" ))

            tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLung"), "rb" ))
            patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
            thrproba=float(indata['thrproba'])
            cnnweigh=indata['picklein_file']
#            print cnnweigh
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrproba,
                                   patch_list_cross_slice,tabroi,cnnweigh,tabscanLung,viewstyle,slnroi)
            
    elif viewstyle=='front view':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datafront"), "rb" ))
#            tabroif= pickle.load( open( os.path.join(path_data_dir,"tabroif"), "rb" ))
            slnt=datarep[0]
            dimtabx=datarep[1]
            dimtaby=datarep[2]
            
            tabLung3d= pickle.load( open( os.path.join(path_data_dir,"tabLung3d"), "rb" ))
            tabroi=np.zeros((slnt,dimtabx,dimtaby), np.uint8)
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
#            patch_list_merge= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge"), "rb" ))
#            proba_merge= pickle.load( open( os.path.join(path_data_dir,"proba_merge"), "rb" ))
            patch_list_merge_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge_slice"), "rb" ))
#            patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
            cnnweigh=indata['picklein_file']
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,
                                   patch_list_merge_slice,tabroi,cnnweigh,tabscanLung,viewstyle)
    
    elif viewstyle=='front projected view':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabroi"), "rb" ))
            patch_list_cross_slice_from_front= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_from_front"), "rb" ))
            thrprobaUIP=float(indata['thrproba']) 
            cnnweigh=indata['picklein_file_front']
            tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLung"), "rb" ))
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,
                                   patch_list_cross_slice_from_front,tabroi,cnnweigh,tabscanLung,viewstyle)
#            thrprobaUIP=float(indata['thrprobaUIP'])
#            tabfromfront= pickle.load( open( os.path.join(path_data_dir,"tabfromfront"), "rb" ))
#            messageout=openfichierfrpr(path_patient,tabfromfront,thrprobaUIP)
    
    elif viewstyle=='report':
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
#        slicepitch=datarep[3]
        slnroi= pickle.load( open( os.path.join(path_data_dir,"slnroi"), "rb" ))
        cnnweigh=indata['picklein_file']
        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
#        patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_sub"), "rb" ))
#        lungSegment= pickle.load( open( os.path.join(path_data_dir,"lungSegment"), "rb" ))
#        tabMed= pickle.load( open( os.path.join(path_data_dir,"tabMed"), "rb" ))
        thrproba=float(indata['thrproba'])
        tabscanLung= pickle.load( open( os.path.join(path_data_dir,"tabscanLung"), "rb" ))
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabroi"), "rb" ))
        dirf=os.path.join(path_patient,listHug)
        dirfreport=os.path.join(dirf,reportdir)   
        if not os.path.exists(dirfreport):
            os.mkdir(dirfreport)
        t = datetime.datetime.now()
        today = '_weight_'+cnnweigh+ '_th'+str(thrproba)+'_m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'
        repf=os.path.join(dirfreport,reportfile+str(today)+'.txt')
        f=open(repf,'w')
        x,x,messageout = openfichiervolumetxt(listHug,path_patient,patch_list_cross_slice,
                      thrproba,
                      tabroi,datarep,slnroi,tabscanLung,f,cnnweigh)
        f.close()
        os.startfile(repf)
        
    elif viewstyle=='reportAll':
        print 'report All'
        thrproba=float(indata['thrproba'])
        cnnweigh=indata['picklein_file']
        pathreport=os.path.join(path_patient,reportalldir)
        t = datetime.datetime.now()
        today ='_weight_'+cnnweigh+ '_th'+str(thrproba)+'_m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'
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
        messageout = openfichiervolumetxtall(listHug,path_patient,indata,thrproba,cnnweigh,f)
        os.startfile(repf)
        
    else:
            messageout='error: unrecognize view style'

    return messageout
