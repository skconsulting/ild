# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:21:22 2017

@author: sylvain
version 1.2
17 august 2017
"""
#from param_pix_p import *
from param_pix_p import path_data,datafrontn,datacrossn,dimpavx,dimpavy,surfelem,volelem,volumeroifile,avgPixelSpacing

from param_pix_p import white,red,yellow,grey,black
from param_pix_p import lungimage,source_name,sroi,sroi3d,scan_bmp,transbmp
from param_pix_p import typei,typei1,typei2
from param_pix_p import threeFileMerge,htmldir,threeFile,threeFile3d,reportdir,reportfile

from param_pix_p import classifc,classifdict,usedclassifdict,oldFormat,writeFile,volumeweb

from param_pix_p import maxproba,excluvisu,fidclass,rsliceNum,evaluate,evaluatef,evaluatefull

from tdGenePredictGui import predictrun

import cPickle as pickle
import os
import cv2
import numpy as np
import datetime
import webbrowser

def lisdirprocess(d):
#    a=os.listdir(d)
    a= os.walk(d).next()[1]
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
        cv2.putText(fig,'%15s'%label+'%8s'%(str(surface)+'cm2')+
                    '%5s'%(pc+'%')+' roi: '+'%8s'%(str(roi)+'cm2')+'%5s'%(pcroi+'%'),
                (deltax, deltay),cv2.FONT_HERSHEY_PLAIN,gro,col,1)
    else:
        cv2.putText(fig,'%15s'%label+'%8s'%(str(surface)+'cm2')+'%5s'%(pc+'%'),
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




def drawpatch(t,dx,dy,slicenumber,va,patch_list_cross_slice,patch_list_ref_slice,volumeroi,slnt,tabroi,num_class):

    imgn = np.zeros((dx,dy,3), np.uint8)
    datav = np.zeros((500,900,3), np.uint8)
    patchdict=np.zeros((slnt,dx,dy), np.uint8)  
#    np.putmask(patchdict,patchdict==0,classif['lung'])
    imgpatch=np.zeros((dx,dy), np.uint8)  
   
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

#                imgn= addpatchn(classcolor,classlabel,xpat,ypat,imgn)
                if classif[classlabel]>0:
                    cv2.rectangle(imgpatch,(xpat,ypat),(xpat+dimpavx,ypat+dimpavy),classif[classlabel],-1)
                else:
                    cv2.rectangle(imgpatch,(xpat,ypat),(xpat+dimpavx,ypat+dimpavy),classif['lung'],-1)
                if lvexist:
                    imgray=np.copy(imgpatch)
                    np.putmask(imgray,imgray>0,255)
                    mask=np.bitwise_not(imgray)
    #                a=np.copy(patchdict[slicenumber])
    #                print 'a',a.max()
                    patchdict[slicenumber]=cv2.bitwise_and(patchdict[slicenumber],patchdict[slicenumber],mask=mask)
    #                b=np.copy(patchdict[slicenumber])
    #                print 'b',b.max()
                    patchdict[slicenumber]=cv2.bitwise_or(imgpatch,patchdict[slicenumber])
#                c=np.copy(patchdict[slicenumber])
#                print 'c',c.max()
#                cv2.imshow('mask',normi(mask))
##                cv2.imshow('a',normi(a))
##                cv2.imshow('b',normi(b))
#                cv2.imshow('c',normi(c))
#                cv2.waitKey(0)
#                cv2.destroyWindow('c')
#                cv2.imshow
                cv2.rectangle(imgn,(xpat,ypat),(xpat+dimpavx,ypat+dimpavy),classcolor,1)
    
#    for s in range(slnt):
##        print s,patchdict[s].max()
#        if patchdict[s].max()>0:
#            name=str(s)+'.bmp'
#            print patchdict[s].min(),patchdict[s].max()
#            print normi(patchdict[s]).min(),normi(patchdict[s]).max()
#            cv2.imwrite(name,patchdict[s])
#            cv2.imwrite('b'+name,normi(patchdict[s]))
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
    
    if lvexist:
        referencepat= tabroi[slicenumber].flatten()
        
        predictpat=  patchdict[slicenumber].flatten()   
#        cv2.imwrite('roi.bmp',tabroi[slicenumber])
#        cv2.imwrite('patch.bmp',patchdict[slicenumber])
#        cv2.imwrite('nroi.bmp',normi(tabroi[slicenumber]))
#        cv2.imwrite('npatch.bmp',normi(patchdict[slicenumber]))
        
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
            cm=evaluatef(referencepat,predictpat,num_class)
            tagvcm(datav,cm)
    datav=cv2.cvtColor(datav,cv2.COLOR_BGR2RGB)
#    fscore, acc, cm,pres,recall = evaluate(yvf,ysf,num_class,(8,))
#    f=open('a.txt','w')
#    for pat in usedclassif:
#        f.write('precision is : '+ pat+' '+str(precision[pat])+'\n')
#        f.write( 'accuray is : '+ pat+ ' '+str(recall[pat])+'\n')
#        f.write( 'fscore is : '+ pat+ ' '+str(fscore[pat])+'\n')
#        
##    
##    print('Val F-score: '+str(fscore)+'\tVal acc: '+str(acc))  
##    print cm
##    f.write('f-score is : '+ str(fscore)+'\n')
##    f.write( 'accuray is : '+ str(acc)+'\n')
##    f.write('precision is : '+ str(pres)+'\n')
##    f.write( 'recall is : '+ str(recall)+'\n')
##    f.write('confusion matrix\n')
#    n= cm.shape[0]
#    for i in range (0,n):
#        for j in range (0,n):
#           f.write(str(cm[i][j])+' ')
#        f.write('\n')
#    f.close()
    
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

def posP(sln, lungSegment):
    '''define where is the slice number'''
    if sln in lungSegment['upperset']:
        psp = 'upperset'
    elif sln in lungSegment['middleset']:
        psp = 'middleset'
    else:
        psp = 'lowerset'
    return psp


def calcSupNp(dirf, patch_list_cross_slice, pat, tabMed,
              dictSubP,dictP,thrprobaUIP,lungSegment,patch_list_cross_slice_sub):
    '''calculate the number of pat in lungsegment and in subpleural according to thrprobaUIP'''
    (top,tail)=os.path.split(dirf)
#    print 'number of subpleural for :',tail, 'pattern :', pat

    for slicename in lungSegment['allset']:

        psp=posP(slicename, lungSegment)
#        ill = 0
        if len(patch_list_cross_slice[slicename])>0:
            for ll in range (0, len(patch_list_cross_slice[slicename])):
    
                xpat = patch_list_cross_slice[slicename][ll][0][0]
#                ypat = patch_list_cross_slice[slicename][ll][0][1]
                proba = patch_list_cross_slice[slicename][ll][1]
                prec, mprobai = maxproba(proba)
                classlabel = fidclass(prec, classif)
#                if pat == 'bronchiectasis' and slicename==14 and classlabel==pat:
#                    print slicename,pat,xpat,mprobai,classlabel
    
                if xpat >= tabMed[slicename]:
                    pospr = 0
                    pospl = 1
                else:
                    pospr = 1
                    pospl = 0
                if classlabel == pat and mprobai > thrprobaUIP:
                
                    dictP[pat][psp] = (
                    dictP[pat][psp][0] + pospl,
                    dictP[pat][psp][1] + pospr)
                    dictP[pat]['all'] = (dictP[pat]['all'][0] + pospl, dictP[pat]['all'][1] + pospr)
                    
        if len(patch_list_cross_slice_sub[slicename])>0:
            for ll in range (0, len(patch_list_cross_slice_sub[slicename])):
                xpat = patch_list_cross_slice_sub[slicename][ll][0][0]
#                ypat = patch_list_cross_slice_sub[slicename][ll][0][1]
                proba = patch_list_cross_slice_sub[slicename][ll][1]
                prec, mprobai = maxproba(proba)
                classlabel = fidclass(prec, classif)
#                if pat == 'bronchiectasis' and slicename==14 and classlabel==pat:
#                    print slicename,pat,xpat,mprobai,classlabel
                if classlabel == pat and mprobai > thrprobaUIP:
                    if xpat >= tabMed[slicename]:
                        pospr = 0
                        pospl = 1
                    else:
                        pospr = 1
                        pospl = 0
                    dictSubP[pat]['all'] = (
                            dictSubP[pat]['all'][0] + pospl,
                            dictSubP[pat]['all'][1] + pospr)
                    dictSubP[pat][psp] = (
                            dictSubP[pat][psp][0] + pospl,
                            dictSubP[pat][psp][1] + pospr)
#                ill += 1
#        if pat == 'bronchiectasis':
#             print 'tot',slicename, pat, dictP[pat]
#             print 'sub',slicename, pat, dictSubP[pat]
    return dictSubP,dictP

#def writedic(p,v,d):
#    v.write(p+ ' '+d[p]['lower']+' '+d[p]['middle']+' '+d[p]['upper']+' ')
#    v.write(d[p]['left_sub_lower']+' '+d[p]['left_sub_middle']+' '+d[p]['left_sub_upper']+' ')
#    v.write(d[p]['right_sub_lower']+' '+d[p]['right_sub_middle']+' '+d[p]['right_sub_upper']+' ')
#    v.write(d[p]['left_lower']+' '+d[p]['left_middle']+' '+d[p]['left_upper']+' ')
#    v.write(d[p]['right_lower']+' '+d[p]['right_middle']+' '+d[p]['right_upper']+'\n')


def writedict(dirf, dx):
    (top,tail)=os.path.split(dirf)
    print 'write file  for :',tail
    ftw = os.path.join(dirf, str(dx) + '_' + volumeweb)
    volumefile = open(ftw,'w')
    volumefile.write(
        'patient UIP WEB: ' +
        str(tail) +
        ' ' +
        'patch_size: ' +
        str(dx) +
        '\n')
    volumefile.write('pattern   lower  middle  upper')
    volumefile.write('  left_sub_lower  left_sub_middle  left_sub_upper ')
    volumefile.write('  right_sub_lower  right_sub_middle  right_sub_upper ')
    volumefile.write('  left_lower  left_middle  left_upper ')
    volumefile.write(' right_lower  right_middle  right_upper\n')
    return volumefile

def cvsarea(p, f, de, dse, s, dc, wf):
    '''calculate area of patches related to total area
      p: pat
      f: volumefile
      de:  dictP
      dse:  dictSubP,
      s:      dictPS
      dc:      dictSurf,
      wf:      writeFile
     ''' 
    dictint = {}
    d = de[p]
    ds = dse[p]

    llungloc = (('lowerset', 'lower'), ('middleset', 'middle'), ('upperset', 'upper'))
    llunglocsl = (('lowerset', 'left_sub_lower'), ('middleset', 'left_sub_middle'), ('upperset', 'left_sub_upper'))
    llunglocsr = (('lowerset', 'right_sub_lower'), ('middleset','right_sub_middle'), ('upperset', 'right_sub_upper'))
    llunglocl = (('lowerset', 'left_lower'), ('middleset', 'left_middle'), ('upperset', 'left_upper'))
    llunglocr = (('lowerset', 'right_lower'), ('middleset','right_middle'), ('upperset', 'right_upper'))
    if wf:
        f.write(p + ': ')
    for i in llungloc:
        st = s[i[0]][0] + s[i[0]][1]
        if st > 0:
            l = 100 * float(d[i[0]][0] + d[i[0]][1]) / st
            l = round(l, 1)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')
#        if p == 'bronchiectasis':
#             print 'surface',i,st
#             print dictint[i[1]]
    for i in llunglocsl:
        st = s[i[0]][0]
#        print 'cvsarea',s
#        print 'cvsarea i',i[0], i[1]
#        print 'cvsarea p d', p,d
#        oo
        if st > 0:
            l = 100 * float(ds[i[0]][0]) / st
            l = round(l, 1)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')
#        if p == 'bronchiectasis':
#             print 'surface sub left',i,st
#             print dictint[i[1]]

    for i in llunglocsr:
        st = s[i[0]][1]
        if st > 0:
            l = 100 * float(ds[i[0]][1]) / st
            l = round(l, 1)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')
#        if p == 'bronchiectasis':
#             print 'surface sub right',i,st
#             print dictint[i[1]]

    for i in llunglocl:
        st = s[i[0]][0]
        if st > 0:
            l = 100 * float(d[i[0]][0]) / st
            l = round(l, 1)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')
#        if p == 'bronchiectasis':
#             print 'surface left',i,st
#             print dictint[i[1]]
             
    for i in llunglocr:
        st = s[i[0]][1]
        if st > 0:
            l = 100 * float(d[i[0]][1]) / st
            l = round(l, 1)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')
#        if p == 'bronchiectasis':
#             print 'surface right',i,st
#             print dictint[i[1]]

    dc[p] = dictint
    if wf:
        f.write('\n')

    return dc

def uipTree(dirf,patch_list_cross_slice,lungSegment,tabMed,dictPS,
            dictP,dictSubP,dictSurf,thrprobaUIP,patch_list_cross_slice_sub):
    '''calculate the number of reticulation and HC in total and subpleural
    and diffuse micronodules'''
#    print 'calculate volume'
    (top,tail)=os.path.split(dirf)
#    print 'calculate volume in : ',tail

    
#    print '-------------------------------------------'
#    print 'surface total  by segment Left Right :'
#    print dictPS
#    print '-------------------------------------------'
    if writeFile:
         volumefile = writedict(dirf, dimpavx)
    else:
        volumefile = ''
    for pat in usedclassif:
       
        dictSubP,dictP= calcSupNp(dirf, patch_list_cross_slice, pat, tabMed,
              dictSubP,dictP,thrprobaUIP,lungSegment,patch_list_cross_slice_sub)
#        if pat == 'bronchiectasis':
#            print ' volume total for:',pat
#            print dictP[pat]
#            print '-------------------------------------------'
#            print ' volume subpleural for:', pat
#            print dictSubP[pat]
#            print '-------------------------------------------'
#            print ' volume total for:',pat
#            print dictP[pat]
#            print '-------------------------------------------'
        dictSurf = cvsarea(
            pat,
            volumefile,
            dictP,
            dictSubP,
            dictPS,
            dictSurf,
            writeFile)
    if writeFile:
            volumefile.write('---------------------\n')
            volumefile.close()
#    print ' volume for:',tail
#    print dictSurf
#    print '-------------------------------------------'
    return dictP, dictSubP, dictSurf


def initdictP(d, p):
    d[p] = {}
    d[p]['upperset'] = (0, 0)
    d[p]['middleset'] = (0, 0)
    d[p]['lowerset'] = (0, 0)
    d[p]['all'] = (0, 0)
    return d

def calculSurface(dirf,posp, midx,lungSegment,dictPS):
    (top,tail)=os.path.split(dirf)
    '''calculate surface of lung in term of number of rectangles pavement'''
#    print 'calculate surface for :',tail

    for ll in posp:
        sln=ll
        psp=posP(sln, lungSegment)
        midsln=midx[sln]
        for ii in range(len(posp[ll])):
            xpat = posp[ll][ii][0][0]
    
            if xpat >= midsln:
                pospr = 0
                pospl = 1
            else:
                pospr = 1
                pospl = 0
    
            dictPS[psp] = (dictPS[psp][0] + pospl, dictPS[psp][1] + pospr)
            dictPS['all'] = (dictPS['all'][0] + pospl, dictPS['all'][1] + pospr)
    return dictPS


def openfichiervolume(listHug,path_patient,patch_list_cross_slice,
                      lungSegment,tabMed,thrprobaUIP,patch_list_cross_slice_sub,slicepitch):
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichiervolume start',path_patient
    volelems=volelem*slicepitch # in mml
#    print slicepitch
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
    listPosLungSimpleLeft=( 'left_lower','left_middle','left_upper')
    listPosLungSimpleRight=( 'right_lower','right_middle','right_upper')
    listPosLungSimple= listPosLungSimpleLeft+listPosLungSimpleRight
    
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
    dictPosTextImage['left_upper']=(460,215)

    dictPosTextImage['right_lower']=(170,570)
    dictPosTextImage['right_middle']=(170,400)
    dictPosTextImage['right_upper']=(200,215)

    dictPosTextImage['left_sub_lower']=(480,660)
    dictPosTextImage['left_sub_middle']=(625,380) #610
    dictPosTextImage['left_sub_upper']=(450,150)

    dictPosTextImage['right_sub_lower']=(170,660)
    dictPosTextImage['right_sub_middle']=(43,380)
    dictPosTextImage['right_sub_upper']=(200,150)
    
    dictposSubtotal={}
    dictposSubtotal['left']=(620,672)
    dictposSubtotal['tleft']=(620,657)
    dictposSubtotal['right']=(35,672)
    dictposSubtotal['tright']=(35,657)

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
    tlold=-1
    viewaskedold={}
    for keyinit in usedclassif:
        viewaskedold[keyinit]=True
    while(1):
            imgwip = np.zeros((200,200,3), np.uint8)           
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
                cv2.putText(imgwip,'WIP',(10,10),cv2.FONT_HERSHEY_PLAIN,5,red,2,cv2.LINE_AA)
                cv2.imshow('wip',imgwip)
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
#allview 
                if allview==1:
                    pervoltot=0
                    
                    for patt in usedclassif:
                        vol=int(round(((dictP[patt]['all'][0]+dictP[patt]['all'][1] )*volelems),0))    
                        pervol=int(round(vol*100./voltotal,0))
                        pervoltot=pervoltot+pervol
                
    #                    print patt,vol
                        cv2.putText(imgtext,'V: '+'%5s'%(str(vol)+'ml')+'%5s'%(str(pervol)+'%'),(dictpostotal[patt][0],dictpostotal[patt][1]),
                                    cv2.FONT_HERSHEY_PLAIN,1.0,classifc[patt],1 )
                    #total volume
                    cv2.putText(imgtext,'Volume Total: '+str(voltotal)+'ml',(dictpostotal['total'][0],dictpostotal['total'][1]),
                                    cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
                    cv2.putText(imgtext,'Total unknown: '+str(100-pervoltot)+'%',(dictpostotal['totalh'][0],dictpostotal['totalh'][1]),
                                    cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
                    for i in listPosLung:

                        surfmax[i],patmax[i] =findmaxvolume(dictSurf,i)    
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
#one pattern view                
                else:
                    for pat in usedclassif:
                        olo=0
                        if viewasked[pat]==True:
                            #text top
                            olo+=1
                            voltotaltotalpat=(dictP[pat]['all'][0]+dictP[pat]['all'][1])*volelems
                            voltt=int(round(voltotaltotalpat,0))
                            voltotalpatl=dictP[pat]['all'][0]*volelems
                            voltotalpatr=dictP[pat]['all'][1]*volelems
                            if voltotal>0:
                                     pervoltt=int(round(100.*voltotaltotalpat/voltotal,0))
                            else:
                                     pervoltt=0
                            cv2.putText(imgtext,'V: '+'%5s'%(str(voltt)+'ml')+'%5s'%(str(pervoltt)+'%'),(dictpostotal[pat][0],
                                    dictpostotal[pat][1]), cv2.FONT_HERSHEY_PLAIN,1.0,classifc[pat],1 )        
                            #label in figures
                            for i in listPosLungSimple:                                                    
                                 if i in listPosLungSimpleLeft:
                                    indexloca=0
                                    voltotalpat=voltotalpatl
                                 else:
                                    indexloca=1
                                    voltotalpat=voltotalpatr                                                                  
                                 if i == 'left_lower' or i =='right_lower':
                                      posud='lowerset'
                                 elif i =='left_middle' or i == 'right_middle':
                                     posud='middleset'
                                 elif i =='left_upper' or i == 'right_upper':
                                     posud='upperset'
                                 volfloat=dictP[pat][posud][indexloca]*volelems
                                 vol=int(round(volfloat,0))                                                                    
                                 if voltotalpat>0:
                                     pervol=int(round(volfloat*100./voltotalpat,0))
                                 else:
                                     pervol=0
                                                         
                                 if olo==1:
                                     if dictSurf[pat][i]>0:
                                         colori=classifc[pat]
                                     else:
                                         colori=grey
                                     lungtw=colorimage(dictPosImage[i],colori)
                                     img=cv2.add(img,lungtw)
                                     cv2.rectangle(imgtext,(dictPosTextImage[i][0],dictPosTextImage[i][1]-15),(dictPosTextImage[i][0]+55,dictPosTextImage[i][1]),white,-1)
                                     cv2.putText(imgtext,str(pervol)+'%',(dictPosTextImage[i][0],dictPosTextImage[i][1]),cv2.FONT_HERSHEY_PLAIN,1.2,grey,1,)
                                 else:
                                     img = np.zeros((dimtabx,dimtaby,3), np.uint8)
                                     imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)                           
                            #subpleural        
                            volsubleft=dictSubP[pat]['all'][0]*volelems
                            if voltotalpatl>0:
                                 pervolsubleft=int(round(100*volsubleft/voltotalpatl,0))
                            else:
                                 pervolsubleft=0
                            volsubright=dictSubP[pat]['all'][1]*volelems
                            if voltotalpatr>0:
                                 pervolsubright=int(round(100*volsubright/voltotalpatr,0))
                            else:
                                 pervolsubright=0
                            vsubl=int(round(volsubright,0))
                            vsubr=int(round(volsubleft,0))
                            cv2.putText(imgtext,'Subpleural',(dictposSubtotal['tleft'][0],
                                         dictposSubtotal['tleft'][1]), cv2.FONT_HERSHEY_PLAIN,1.0,classifc[pat],1 )
                            cv2.putText(imgtext,str(vsubl)+'ml '+str(pervolsubleft)+'%',(dictposSubtotal['left'][0],
                                         dictposSubtotal['left'][1]), cv2.FONT_HERSHEY_PLAIN,1.0,classifc[pat],1 )
                            cv2.putText(imgtext,'Subpleural',(dictposSubtotal['tright'][0],
                                         dictposSubtotal['tright'][1]), cv2.FONT_HERSHEY_PLAIN,1.0,classifc[pat],1 )
                            cv2.putText(imgtext,str(vsubr)+'ml '+str(pervolsubright)+'%',(dictposSubtotal['right'][0],
                                         dictposSubtotal['right'][1]), cv2.FONT_HERSHEY_PLAIN,1.0,classifc[pat],1 )              
                                 
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
                cv2.destroyWindow("wip")

            cv2.imshow('imageVol',imgtowrite)

            if quitl or cv2.waitKey(20) & 0xFF == 27 :
    #            print 'on quitte', quitl
                break
    quitl=False
    #    print 'on quitte 2'
    cv2.destroyWindow("imageVol")
    cv2.destroyWindow("SliderVol")

    return ''

def openfichiervolumetxt(listHug,path_patient,patch_list_cross_slice,
                      lungSegment,tabMed,thrprobaUIP,patch_list_cross_slice_sub,slicepitch,tabroi,datacross):
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichiervolume start',path_patient
    volelems=volelem*slicepitch # in mml
    slnt=datacross[0]
    dx=datacross[1]
    dy=datacross[2]
    lissln=datacross[4]
    
#    print 'slicepitch',slicepitch

    dirf=os.path.join(path_patient,listHug)
   
    dirfreport=os.path.join(dirf,reportdir)
    if not os.path.exists(dirfreport):
        os.mkdir(dirfreport)
    t = datetime.datetime.now()
    today = '_th'+str(thrprobaUIP)+'_m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'
    repf=os.path.join(dirfreport,reportfile+str(today)+'.txt')

    f=open(repf,'w')
    
    f.write('report for patient :'+listHug+'\n')
    f.write('Cross View\n')
    f.write('date : m'+str(t.month)+'-d'+str(t.day)+'-y'+str(t.year)+' at '+str(t.hour)+'h '+str(t.minute)+'mn\n')
 
    dictP = {}  # dictionary with patch in lung segment
    dictPS = {}  # dictionary with total patch area in lung segment
    dictSubP = {}  # dictionary with patch in subpleural
    dictSurf = {}  # dictionary with patch volume in percentage
    poslr=['left','right','both']
    posud=['upperset','middleset','lowerset','all']

    dictPS['upperset'] = (0, 0)
    dictPS['middleset'] = (0, 0)
    dictPS['lowerset'] = (0, 0)
    dictPS['all'] = (0, 0)
    dictPS = calculSurface(dirf,patch_list_cross_slice, tabMed,lungSegment,dictPS)
    voltotal=int(round(((dictPS['all'][0]+dictPS['all'][1])*volelems),0))

    for patt in usedclassif:
        dictP = initdictP(dictP, patt)
        dictSubP = initdictP(dictSubP, patt)
          
    
#                     print key1
    dictP = {}  # dictionary with patch in lung segment    
    dictSubP = {}  # dictionary with patch in subpleural
    dictSurf = {}  # dictionary with patch volume in percentage
    for patt in usedclassif:
        dictP = initdictP(dictP, patt)
        dictSubP = initdictP(dictSubP, patt)

    dictP, dictSubP, dictSurf= uipTree(dirf,patch_list_cross_slice,lungSegment,tabMed,dictPS,
                                       dictP,dictSubP,dictSurf,thrprobaUIP,patch_list_cross_slice_sub)

    f.write('\n')
    f.write('Threshold : '+str(int(100*thrprobaUIP))+'%\n')
    f.write('-----------------------------------------------\n')
    f.write('Total volume per pattern \n')
#    f.write('%15s'%+'patern'+'%9s'%'volume total'+'%7s'%'volume pattern, right, % volume pattern)\n')
    f.write('%15s'%'pattern'+'%15s'%'volume total'+'%4s'%'%'+
               '%11s'%'Vol. left'+'%4s'%'%'+
               '%12s'%'Vol. right'+'%4s'%('%')+'\n')

    pervoltot=0
    for patt in usedclassif:
#        print dictSubP[patt]
        volf=(dictP[patt]['all'][0]+dictP[patt]['all'][1] )*volelems
        vol=int(round(volf,0))  
        vollf=dictP[patt]['all'][0]*volelems
        voll=int(round(vollf,0))
        if volf>0:
            pervoll=int(round(vollf/volf*100,0))
        else:
            pervoll=0
        volrf=dictP[patt]['all'][1]*volelems
        volr=int(round(volrf,0))   
        if volr>0:
            pervolr=int(round(volrf/volf*100,0))
        else:
            pervolr=0
        pervol=int(round(vol*100./voltotal,0))
        pervoltot=pervoltot+pervol
        
        f.write('%15s'%patt+'%14s'%(str(vol)+'ml')+'%5s'%(str(pervol)+'%')+
               '%10s'%(str(voll)+'ml')+'%5s'%(str(pervoll)+'%')+
               '%11s'%(str(volr)+'ml')+'%5s'%(str(pervolr)+'%')+'\n')

    f.write('\n')
    f.write('Volume Total:     '+str(voltotal)+'ml\n')
    f.write('Total unknown:    '+str(100-pervoltot)+'%\n')
    f.write('-----------------------------------------------\n')

    f.write('Pattern localization (% is in comparison with total pattern area in left or right)\n')
    f.write('\n')


    for patt in usedclassif:       
        f.write(patt+'\n')
        volpatl=dictP[patt]['all'][0]*volelems
        volpatr=dictP[patt]['all'][1]*volelems
        for lr in poslr:
            f.write('  '+lr+'\n')
            if lr=='left':               
                ind=0
                voltotal=volpatl
            elif lr=='right': 
                voltotal=volpatr
                ind=1
            else:
                voltotal=volpatr+volpatl
                ind=2
            for ud in posud:
                    if ind<2:
                        vol=dictP[patt][ud][ind]*volelems
                    else:
                        vol=(dictP[patt][ud][0]+dictP[patt][ud][1])*volelems   
                    if voltotal>0:
                        pervol=int(round(vol*100./voltotal,0))
                    else:
                        pervol=0
                    volrounded=round(vol,1)
                    f.write('    '+'%9s'%ud+': Volume:'+'%8s'%(str(volrounded)+'ml')+'%5s'%(str(pervol)+'%')+'\n')
        
        f.write('  subpleural\n')
        for lr in poslr:
            if lr=='left':               
                ind=0
                voltotal=volpatl
            elif lr=='right': 
                voltotal=volpatr
                ind=1
            else:
                voltotal=volpatr+volpatl
                ind=2
            if ind<2:
                vol=dictSubP[patt]['all'][ind]*volelems
            else:
                vol=(dictSubP[patt]['all'][0]+dictSubP[patt]['all'][1])*volelems
            if voltotal>0:
                pervol=int(round(vol*100./voltotal,0))
            else:
                pervol=0
            volrounded=round(vol,1)
            f.write('    '+'%9s'%lr+': Volume:'+'%8s'%(str(volrounded)+'ml')+'%5s'%(str(pervol)+'%')+'\n')
        
        f.write('------------------------------------\n')
    if tabroi.max()>0:
        print 'start Fscore'
        patchdict=np.zeros((slnt,dx,dy), np.uint8)
        num_class=len(classif)
        th=thrprobaUIP
        for slicenumber in lissln:
    #    for slicenumber in (143,144):
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
    
                referencepat= tabroi[slicenumber].flatten()
                predictpat=  patchdict[slicenumber].flatten()   
                
                precision={}
                recall={}
                fscore={}
                f.write('   pattern   precision  recall  Fscore  for slice :'+str(slicenumber)+'\n')
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
                        
        referencepat= tabroi.flatten()
        predictpat=  patchdict.flatten()       
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
        f.write('scores for patient (without lung):\n')
        f.write('precision:'+precisioni+' recall: '+recalli+' Fscore: '+fscorei+'\n')
        f.write('----------------------\n')

        
        f.write('confusion matrix\n')
        
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
        f.write('\n')
    f.write('----------------------\n')
    f.close()
    return ''

def writeslice(num,menus):
#        print 'write',num
        cv2.rectangle(menus, (5,60), (150,50), red, -1)
        cv2.putText(menus,'Slice to visualize: '+str(num),(5,60),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
        
def openfichier(ti,datacross,path_img,thrprobaUIP,patch_list_cross_slice,patch_list_ref_slice,tabroi,cnnweigh):   
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichier start' 
    
    quitl=False
    num_class=len(classif)
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
#                print volumeroilocal
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
        
        cv2.namedWindow('imagepredict',cv2.WINDOW_NORMAL)
        cv2.namedWindow("Sliderfi",cv2.WINDOW_NORMAL)
        cv2.namedWindow("datavisu",cv2.WINDOW_AUTOSIZE)

        cv2.createTrackbar( 'Brightness','Sliderfi',0,100,nothing)
        cv2.createTrackbar( 'Contrast','Sliderfi',50,100,nothing)
        cv2.createTrackbar( 'Threshold','Sliderfi',int(thrprobaUIP*100),100,nothing)
        cv2.createTrackbar( 'Flip','Sliderfi',slnt/2,slnt-2,nothings)
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
        while(1):
        
            imgwip = np.zeros((200,200,3), np.uint8)  
                                 
            cv2.setMouseCallback('imagepredict',draw_circle,img)
            c = cv2.getTrackbarPos('Contrast','Sliderfi')
            l = cv2.getTrackbarPos('Brightness','Sliderfi')
            tl = cv2.getTrackbarPos('Threshold','Sliderfi')
            fl = cv2.getTrackbarPos('Flip','Sliderfi')
            allview = cv2.getTrackbarPos('All','Sliderfi')
            noneview = cv2.getTrackbarPos('None','Sliderfi')
            
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
                    
            if nbdig>0 and key ==13 and numberfinal>0:
#                    print numberfinal
                    numberfinal = min(slnt-1,numberfinal)
#                    writeslice(numberfinal,initimg)
                    cv2.rectangle(initimg, (5,60), (150,50), black, -1)
                    if corectnumber==0:
                        cv2.setTrackbarPos('Flip','Sliderfi' ,numberfinal)
                        fl=numberfinal
                    else:
                        cv2.setTrackbarPos('Flip','Sliderfi' ,numberfinal-1)
                        fl=numberfinal-1
                    
                    numberfinal=0
                    nbdig=0
                    numberentered={}
            if key==2424832:
                fl=max(0,fl-1)
                cv2.setTrackbarPos('Flip','Sliderfi' ,fl)
            if key==2555904:
                fl=min(slnt-2,fl+1)
                cv2.setTrackbarPos('Flip','Sliderfi' ,fl)
            
            for key2 in usedclassif:
                s = cv2.getTrackbarPos(key2,'Sliderfi')
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
                cv2.putText(imgwip,'WIP',(10,10),cv2.FONT_HERSHEY_PLAIN,5,red,2,cv2.LINE_AA)
                cv2.imshow('wip',imgwip)
                
                imgn,datav= drawpatch(tl,dimtabx,dimtaby,slicenumber,viewasked,patch_list_cross_slice,
                                                patch_list_ref_slice,volumeroilocal,slnt,tabroi,num_class)
                               
                cv2.putText(datav,'slice number :'+str(slicenumber),(10,180),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
                cv2.putText(datav,'patient Name :'+tail,(10,190),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
                cv2.putText(datav,'CNN weight: '+cnnweigh,(10,170),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)

                cv2.destroyWindow("wip")
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
    
            if patchi :
                cv2.putText(imgwip,'WIP',(10,10),cv2.FONT_HERSHEY_PLAIN,5,red,2,cv2.LINE_AA)
                cv2.imshow('wip',imgwip)
#                print 'retrieve patch asked'
                imgtext= retrievepatch(ix,iy,slicenumber,dimtabx,dimtaby,patch_list_cross_slice)
                patchi=False
                cv2.destroyWindow("wip")

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
    else:
        print 'error in the number of scan images compared to dicom numbering'
        return 'error in the number of scan images compared to dicom numbering'



def openfichierfrpr(path_patient,tabfromfront,thrprobaUIP):
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichierfrpr start',path_patient
#    dirf=os.path.join(path_patient,listHug)
   
    slnt=len(tabfromfront)   
#    print slnt
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

    nbdig=0
    numberentered={}

    initimg = np.zeros((dimtabx,dimtaby,3), np.uint8)

    while(1):
#            print "corectnumber",corectnumber
            imgtext= np.zeros((dimtabx,dimtaby,3), np.uint8)
            cv2.setMouseCallback('imagefrpr',draw_circle,imgtext)
            fl = cv2.getTrackbarPos('Flip','SliderVolumefrpr')
            
#            key = cv2.waitKey(1000) & 0xFF
            key = cv2.waitKey(1000)

#            if key != -1:
#                print key
#                
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
                    
            if nbdig>0 and key ==13 and numberfinal>0:
#                    print numberfinal
                    numberfinal = min(slnt-1,numberfinal)
#                    writeslice(numberfinal,initimg)
                    cv2.rectangle(initimg, (5,60), (150,50), black, -1)
                    cv2.setTrackbarPos('Flip','SliderVolumefrpr' ,numberfinal-1)
                    fl=numberfinal-1
                    
                    numberfinal=0
                    nbdig=0
                    numberentered={}
            
            if key==2424832:
                fl=max(0,fl-1)
                cv2.setTrackbarPos('Flip','SliderVolumefrpr' ,fl)
            if key==2555904:
                fl=min(slnt-2,fl+1)
                cv2.setTrackbarPos('Flip','SliderVolumefrpr' ,fl)
                
            img=tabfromfront[fl+1]
            img=cv2.add(initimg,img)
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
    global classif,usedclassif
    print 'visuarun start'

    messageout=""
 
#    print 'path_patient',path_patient
    lpt=indata['lispatientselect']
    pos=lpt.find(' ')
    listHug=(lpt[0:pos])
        
    patient_path_complet=os.path.join(path_patient,listHug)
    path_data_dir=os.path.join(patient_path_complet,path_data)
    
    frontcompleted = pickle.load(open( os.path.join(path_data_dir,"frontcompleted"), "rb" ))
    crosscompleted = pickle.load(open( os.path.join(path_data_dir,"crosscompleted"), "rb" ))
#    centerHU=indata['centerHU']
#    limitHU=indata['limitHU']
    
   
    if not crosscompleted:
        messageout="no predict!for "+listHug
        return messageout

#    if pos >0:
#            listHug=(lpt[0:pos])
#    else:
#            pos=str(indata).find(' noPREDICT!')
##            print 'no predict'
#            listHug=(lpt[0:pos])
            
   
    #↓define classif
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
            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabroi"), "rb" ))
            patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
            thrprobaUIP=float(indata['thrprobaUIP'])
            cnnweigh=indata['picklein_file']
#            print cnnweigh
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,
                                   patch_list_cross_slice,patch_list_cross_slice,tabroi,cnnweigh)
            
    elif viewstyle=='front view':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datafront"), "rb" ))
#            tabroif= pickle.load( open( os.path.join(path_data_dir,"tabroif"), "rb" ))
            slnt=datarep[0]
            tabroi=np.zeros((slnt,dimtabx,dimtabx), np.uint8)
            patch_list_front_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_front_slice"), "rb" ))
            thrprobaUIP=float(indata['thrprobaUIP'])
            cnnweigh=indata['picklein_file_front']
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,patch_list_front_slice,patch_list_front_slice,tabroi,cnnweigh)
            
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
            tabroi= pickle.load( open( os.path.join(path_data_dir,"tabroi"), "rb" ))
#            patch_list_merge= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge"), "rb" ))
#            proba_merge= pickle.load( open( os.path.join(path_data_dir,"proba_merge"), "rb" ))
            patch_list_merge_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge_slice"), "rb" ))
            patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
            cnnweigh=indata['picklein_file']
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,patch_list_merge_slice,patch_list_cross_slice,tabroi,cnnweigh)
    
    elif viewstyle=='front projected view':

            thrprobaUIP=float(indata['thrprobaUIP'])
            tabfromfront= pickle.load( open( os.path.join(path_data_dir,"tabfromfront"), "rb" ))
            messageout=openfichierfrpr(path_patient,tabfromfront,thrprobaUIP)
    
    elif viewstyle=='report':
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
        slicepitch=datarep[3]
        patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
        patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_sub"), "rb" ))
        lungSegment= pickle.load( open( os.path.join(path_data_dir,"lungSegment"), "rb" ))
        tabMed= pickle.load( open( os.path.join(path_data_dir,"tabMed"), "rb" ))
        thrprobaUIP=float(indata['thrprobaUIP'])
        tabroi= pickle.load( open( os.path.join(path_data_dir,"tabroi"), "rb" ))
                  
        messageout = openfichiervolumetxt(listHug,path_patient,patch_list_cross_slice,
                      lungSegment,tabMed,thrprobaUIP,patch_list_cross_slice_sub,slicepitch,tabroi,datarep)
    
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
