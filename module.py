# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:21:22 2017

@author: sylvain
"""
import os
import cv2
import shutil
from tdGenePredictWeb import *
import cPickle as pickle
import numpy as np

dimpavx=16
dimpavy=16
pxy=float(dimpavx*dimpavy)

path_data='data'
source_name='source'
scan_bmp='scan_bmp'
transbmp='trans_bmp'
typei='jpg'
excluvisu=['back_ground','healthy']
datafrontn='datafront'
datacrossn='datacross'
wbg=True
#reservedparm=[ 'thrpatch','thrproba','thrprobaUIP','thrprobaMerge','picklein_file',
#                      'picklein_file_front','tdornot','threedpredictrequest',
#                      'onlyvisuaasked','cross','front','merge']
classif ={
        'back_ground':0,
        'consolidation':1,
        'HC':2,
        'ground_glass':3,
        'healthy':4,
        'micronodules':5,
        'reticulation':6,
        'air_trapping':7,
        'cysts':8,
        'bronchiectasis':9,
        'emphysema':10,
        'GGpret':11
        }


red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255,255,0)
cyan=(0,255,255)
purple=(255,0,255)
white=(255,255,255)
darkgreen=(11,123,96)
pink =(255,128,255)
lightgreen=(125,237,125)
orange=(255,153,102)
lowgreen=(0,51,51)
parme=(234,136,222)
chatain=(139,108,66)

classifc ={
    'back_ground':darkgreen,
    'consolidation':cyan,
    'HC':blue,
    'ground_glass':red,
    'healthy':darkgreen,
    'micronodules':green,
    'reticulation':yellow,
    'air_trapping':pink,
    'cysts':lightgreen,
    'bronchiectasis':orange,
    'emphysema':chatain,
    'GGpret': parme,



     'nolung': lowgreen,
     'bronchial_wall_thickening':white,
     'early_fibrosis':white,

     'increased_attenuation':white,
     'macronodules':white,
     'pcp':white,
     'peripheral_micronodules':white,
     'tuberculosis':white
 }

def lisdirprocess(d):
#    a=os.listdir(d)
    a= os.walk(d).next()[1]
    print 'listdirprocess',a
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

cwd=os.getcwd()

def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path,ignore_errors=True)


    
       
def predict(indata,path_patient):
    print 'module predict'
    listdir=[]
    nota=True
    try:
        listdiri= indata['lispatientselect']       
    except KeyError:
            print 'No patient selected'
            nota=False
#    print 'length de listdiri',len(listdiri)
#    for key, value in indata.items():
#        print key
#        print value
#            if key not in reservedparm:
#                listdir.append(key)
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
    if event == cv2.EVENT_LBUTTONDBLCLK:
 
#        print('identification')
        ix,iy=x,y
       
        patchi=True
        print 'identification', ix,iy, patchi
        dxrect=(dimtaby/2)
        if x>dxrect and x<dxrect+20 and y>dimtabx-30 and y< dimtabx-10:
            print 'quit'
            ix,iy=x,y
            quitl=True


def rsliceNum(s,c,e):
    ''' look for  afile according to slice number'''
    #s: file name, c: delimiter for snumber, e: end of file extension
    endnumslice=s.find(e)
    posend=endnumslice
    while s.find(c,posend)==-1:
        posend-=1
    debnumslice=posend+1
    return int((s[debnumslice:endnumslice])) 

def contrasti(im,r):   
#     tabi = np.array(im)
     r1=0.5+r/100.0
     tabi1=im*r1     
     tabi2=np.clip(tabi1,0,255)
     tabi3=tabi2.astype(np.uint8)
     return tabi3

def lumi(tabi,r):
#    tabi = np.array(im)
    r1=r
    tabi1=tabi+r1
    tabi2=np.clip(tabi1,0,255)
    tabi3=tabi2.astype(np.uint8)
    return tabi3

def maxproba(proba):
    """looks for max probability in result"""
    lenp = len(proba)
    m=0
    for i in range(0,lenp):
        if proba[i]>m:
            m=proba[i]
            im=i
    return im,m

def fidclass(numero,classn):
    """return class from number"""
    found=False
#    print numero
    for cle, valeur in classn.items():
        
        if valeur == numero:
            found=True
            return cle
      
    if not found:
        return 'unknown'    

def addpatchn(col,lab, xt,yt,imgn):
#    print col,lab
    cv2.rectangle(imgn,(xt,yt),(xt+dimpavx,yt+dimpavy),col,1)
    return imgn

def tagviewn(fig,label,pro,nbr,x,y):
    """write text in image according to label and color"""

    col=classifc[label]
#    print col, label
    if wbg :
        labnow=classif[label]-1
    else:
        labnow=classif[label]
#    print (labnow, text)
    if label == 'back_ground':
        x=0
        y=0        
        deltax=0
        deltay=60
    else:                
        deltax=130*((labnow)//5)
        deltay=11*((labnow)%5)
    gro=-x*0.0027+0.9
#    if label=='ground_glass':
#            print label,x,deltax, y,deltay,labnow,gro
    cv2.putText(fig,str(nbr)+' '+label+' '+pro,(x+deltax, y+deltay+10),cv2.FONT_HERSHEY_PLAIN,gro,col,1)

   
def drawpatch(t,lp,preprob,k,dx,dy,slnum,va):
    imgn = np.zeros((dx,dy,3), np.uint8)
    ill = 0
#    endnumslice=k.find('.bmp')
    slicenumber=slnum
#    print imgcore
#    posend=endnumslice
#    while k.find('-',posend)==-1:
#            posend-=1
#    debnumslice=posend+1
#    slicenumber=int((k[debnumslice:endnumslice])) 
    th=t/100.0
    listlabel={}
    listlabelaverage={}
#    print slicenumber,th
    for ll in lp:

#            print ll
            slicename=ll[0]          
            xpat=ll[1]
            ypat=ll[2]        
        #we find max proba from prediction
            proba=preprob[ill]
           
            prec, mprobai = maxproba(proba)

            classlabel=fidclass(prec,classif)
            classcolor=classifc[classlabel]
       
            
            if mprobai >th and slicenumber == slicename and classlabel not in excluvisu and va[classlabel]==True:
#                    print classlabel
                    if classlabel in listlabel:
#                        print 'found'
                        numl=listlabel[classlabel]
                        listlabel[classlabel]=numl+1
                        cur=listlabelaverage[classlabel]
#                               print (numl,cur)
                        averageproba= round((cur*numl+mprobai)/(numl+1),2)
                        listlabelaverage[classlabel]=averageproba
                    else:
                        listlabel[classlabel]=1
                        listlabelaverage[classlabel]=mprobai

                    imgn= addpatchn(classcolor,classlabel,xpat,ypat,imgn)


            ill+=1
#            print listlabel        
    for ll1 in listlabel:
#                print ll1,listlabelaverage[ll1]
                delx=int(dy*0.6-120)
                tagviewn(imgn,ll1,str(round(listlabelaverage[ll1],2)),listlabel[ll1],delx,0)
    ts='Treshold:'+str(t)

    cv2.putText(imgn,ts,(0,50),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
    return imgn

def retrievepatch(x,y,sln,pr,li,dx,dy):
  
    tabtext = np.zeros((dx,dy,3), np.uint8)
    ill=-1
    pfound=False
    for f in li:
        ill+=1 
        slicenumber=f[0]

        if slicenumber == sln:

            xs=f[1]
            ys=f[2]
#            print xs,ys
            if x>xs and x < xs+dimpavx and y>ys and y<ys+dimpavy:
                     print x, y
                     proba=pr[ill]
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
#                     for j in range (0,len(proba)):
#                         
##                     for j in range (0,2):
#                         if proba[j]>0.01:
#                             n=n+1
#                             strw=fidclass(j,classif)+ ' {0:.1f}%'.format(100*proba[j])                             
##                             cv2.putText(tabtext,strw,(dy-142,(dx-60)+10*n),cv2.FONT_HERSHEY_PLAIN,0.7,(0,255,0),1)
#                             
#                             print fidclass(j,classif), ' {0:.2f}%'.format(100*proba[j])
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

def openfichier(ti,datacross,patch_list,proba,path_img):
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    quitl=False
    print 'datacros',datacross
    dimtabx=datacross[1]
    slnt=datacross[0]
    dimtaby= datacross[2]
    print 'openfichier',slnt,dimtabx,ti,dimtaby
    print 'dimltabx, dimtaby',dimtabx,dimtaby
#    slicepitch=datacross[2]
    patchi=False
    ix=0
    iy=0
    if ti =="cross" or ti =="merge":
        sn=scan_bmp
        corectnumber=1
    else:
        sn=transbmp 
        corectnumber=0
    pdirk = os.path.join(path_img,source_name)
    pdirk = os.path.join(pdirk,sn)
    list_image={}
    cdelimter='_'
    extensionimage='.'+typei
    limage=[name for name in os.listdir(pdirk) if name.find('.'+typei,1)>0 ]
#    print 'limag',limage
#    print 'lenght limage',len(limage)
    if len(limage)+1==slnt:
#        print 'good'
#    
        for iimage in range(0,slnt-1):
    #        print iimage
            s=limage[iimage]       
                #s: file name, c: delimiter for snumber, e: end of file extension
            sln=rsliceNum(s,cdelimter,extensionimage)
            list_image[sln]=s
            
        image0=os.path.join(pdirk,list_image[slnt/2])
        img = cv2.imread(image0,1)
    #    cv2.imshow('cont',img)
    #    cv2.waitKey(0)    
    #    cv2.destroyAllWindows()
        
        imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
    #    print 'openfichier:',k , ncf, pdirk,top
        
        (preprob,listnamepatch)=(proba,patch_list)     
        
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.namedWindow("Slider",cv2.WINDOW_NORMAL)
    
        cv2.createTrackbar( 'Brightness','Slider',0,100,nothing)
        cv2.createTrackbar( 'Contrast','Slider',50,100,nothing)
        cv2.createTrackbar( 'Threshold','Slider',50,100,nothing)
        cv2.createTrackbar( 'Flip','Slider',slnt/2,slnt-2,nothings)
        cv2.createTrackbar( 'All','Slider',0,1,nothings)
        cv2.createTrackbar( 'None','Slider',0,1,nothings)
        viewasked={}
        for key,value in classif.items():
#            print key
            viewasked[key]=True
            cv2.createTrackbar( key,'Slider',0,1,nothings)
            
    #    switch = '0 : OFF \n1 : ON'
    #    cv2.createTrackbar(switch, 'image',0,1,nothing)
    
            
        while(1):
             
            cv2.setMouseCallback('image',draw_circle,img)
            c = cv2.getTrackbarPos('Contrast','Slider')
            l = cv2.getTrackbarPos('Brightness','Slider')
            tl = cv2.getTrackbarPos('Threshold','Slider')
            fl = cv2.getTrackbarPos('Flip','Slider')
            allview = cv2.getTrackbarPos('All','Slider')
            noneview = cv2.getTrackbarPos('None','Slider')
                  
            for key,value in classif.items():
                s = cv2.getTrackbarPos(key,'Slider')
                if allview==1:
                     viewasked[key]=True
                elif noneview ==1:
                    viewasked[key]=False
                elif s==0:
#            print key
                    viewasked[key]=False
                else:
                     viewasked[key]=True
                    
    
    #        img,pdirk= opennew(dirk, fl,L)
    #        print pdirk
            imagel=os.path.join(pdirk,list_image[fl])
            img = cv2.imread(imagel,1)
    #        print 'img',img.shape, img.dtype
            
            
    #        (topnew,tailnew)=os.path.split(pdirk)
    #        endnumslice=tailnew.find('.bmp',0)
    #        posend=endnumslice
           
            slicenumber=fl+corectnumber
    #        print slicenumber,fl
    #        ooo
    #        while tailnew.find('_',posend)==-1:
    #            posend-=1
    #            debnumslice=posend+1
    #        slicenumber=int((tailnew[debnumslice:endnumslice])) 
    #        
            imglumi=lumi(img,l)
            imcontrast=contrasti(imglumi,c)        
            imcontrast=cv2.cvtColor(imcontrast,cv2.COLOR_BGR2RGB)
            
    #        print 'imcontrast',imcontrast.shape, imcontrast.dtype
            imgn=drawpatch(tl,listnamepatch,preprob,list_image[slicenumber],dimtabx,dimtaby,slicenumber,viewasked)
    #        imgn=cv2.cvtColor(imgn,cv2.COLOR_BGR2RGB)
            imgngray = cv2.cvtColor(imgn,cv2.COLOR_BGR2GRAY)
    #        print 'imgray',imgngray.shape, imgngray.dtype
            mask_inv = cv2.bitwise_not(imgngray)           
    #        print 'mask_inv',mask_inv.shape, mask_inv.dtype
            outy=cv2.bitwise_and(imcontrast,imcontrast,mask=mask_inv)
            
    #        print 'imgn',imgn.shape, imgn.dtype
    #        print 'outy',outy.shape, outy.dtype
            imgt=cv2.add(imgn,outy)
    #        imgt=np.add(imgn,outy)
    #        imgt=np.clip(imgt,0,255)
    
            dxrect=(dimtaby/2)
    #        print dxrect,dimtaby,dimtabx
    #        dxrect=100
            cv2.rectangle(imgt,(dxrect,dimtabx-30),(dxrect+20,dimtabx-10),red,-1)
    #        cv2.rectangle(imgt,(172,358),(192,368),white,-1)
            cv2.putText(imgt,'quit',(dxrect+10,dimtabx-10),cv2.FONT_HERSHEY_PLAIN,1,yellow,1,cv2.LINE_AA)
            imgtoshow=cv2.add(imgt,imgtext)     
    
            imgtoshow=cv2.cvtColor(imgtoshow,cv2.COLOR_BGR2RGB)
    
            cv2.imshow('image',imgtoshow)
           
            if patchi :
                print 'retrieve patch asked'
                imgtext=retrievepatch(ix,iy,fl+corectnumber,preprob,listnamepatch,dimtabx,dimtaby)
                patchi=False
            
            if quitl or cv2.waitKey(20) & 0xFF == 27 :
    #            print 'on quitte', quitl
                break
        quitl=False
    #    print 'on quitte 2'
        cv2.destroyAllWindows()
        return ''
    else:
        print 'error in the number of scan images compared to dicom numbering'
        return 'error in the number of scan images compared to dicom numbering'
    
def visuarun(indata,path_patient):
    listHug=indata['lispatient']
    viewstyle=indata['typeofview']
    print 'listhug',listHug
    print 'viewstyle',viewstyle       
    print 'indata',indata
    print 'path_patient',path_patient

    patient_path_complet=os.path.join(path_patient,listHug)
    print patient_path_complet
    path_data_dir=os.path.join(patient_path_complet,path_data)
    if viewstyle=='cross':
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "r" ))
        patch_list= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross"), "r" ))
        proba= pickle.load( open( os.path.join(path_data_dir,"proba_cross"), "r" ))
    elif viewstyle=='front':
        datarep= pickle.load( open( os.path.join(path_data_dir,"datafront"), "r" ))
        patch_list= pickle.load( open( os.path.join(path_data_dir,"patch_list_front"), "r" ))
        proba= pickle.load( open( os.path.join(path_data_dir,"proba_front"), "r" ))
    else:
        datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "r" ))
        patch_list= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge"), "r" ))
        proba= pickle.load( open( os.path.join(path_data_dir,"proba_merge"), "r" ))
        
        
        
    messageout=openfichier(viewstyle,datarep,patch_list,proba,patient_path_complet)
    return messageout
#    
#indata={'lispatient':'23','typeofview':'cross','thrpatch':0.8,'thrproba':0.6,'thrprobaUIP':0.6,'thrprobaMerge':0.6,
#        'picklein_file':"pickle_ex74",'picklein_file_front':"pickle_ex711",'23':'on'
#        }
#path_patient='C:/Users/sylvain/Documents/boulot/startup/radiology/predicttool/patient_directory'
##
#visuarun(indata,path_patient)
