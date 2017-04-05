# coding: utf-8
#Sylvain Kritter 21 septembre 2016
""" transform grenoble database for prediction, select a subset """
#general parameters and file, directory names"

import os
import cv2
import datetime
import time
import dicom
import scipy
import sys
import shutil
import numpy as np
#import Tkinter as Tk
import cPickle as pickle
from Tkinter import *
#global environment

picklefileglobal='MHKpredictglobal.pkl'
instdirMHK='MHKpredict'
workdiruser='Documents/boulot/startup/radiology/PREDICT'

tempofile=os.path.join(os.environ['TMP'],picklefileglobal)
workingdir= os.path.join(os.environ['USERPROFILE'],workdiruser)
instdir=os.path.join(os.environ['LOCALAPPDATA'],instdirMHK)

#
print 'instdir', instdir
#print 'workdir', workingfull


#########################################################
#define if the patch set is limited to the only pattrens, or all (set0)
#patchSet='set2'
#picklefile='pickle_sk32'
listset=['set0','set1','set2']
picklefile={}
dimpavxs={}
dimpavys={}

#pattern set definition
#pset==2:  'HC', 'micronodules'
picklefile['set2']='pickle_ex16'
dimpavxs['set2'] =32
dimpavys['set2'] = 32

imageDepth =255
#picklefile['set2']='pickle_ex17'
##patch size in pixels 32 * 32
#dimpavxs['set2'] =28
#dimpavys['set2'] = 28


#pset==0: 'consolidation', 'HC','ground_glass', 'micronodules', 'reticulation'
picklefile['set0']='pickle_ex16'
##patch size in pixels 32 * 32
dimpavxs['set0'] =32 
dimpavys['set0'] = 32
    
#pset==1: 'consolidation', 'ground_glass',
picklefile['set1']='pickle_ex17'
##patch size in pixels 32 * 32
dimpavxs['set1'] =28 
dimpavys['set1'] = 28

#pset==3: 'air-trapping',
#picklefile['set3']='pickle_ex17'
###patch size in pixels 32 * 32
#dimpavxs['set3'] =28 
#dimpavys['set3'] = 28


#subdirectory name to colect pkl filesfor lung  prediction
picklefilelung='pickle_sk8_lung'
#patch size in pixels for lung 32 * 32
lungdimpavx =15
lungdimpavy = 15
#to enhance contrast on patch put True
contrastScanLung=False
#normalization internal procedure or openCV
normiInternalLung=False

sepIsDash=True
#erosion factor for subpleura in mm
subErosion= 15

# with or without bg (true if with back_ground)
wbg=True
#to enhance contrast on patch put True
contrast=True
#path for visua back-ground
vbg='A'
#threshold for patch acceptance overlapp
thrpatch = 0.9
#threshold for probability prediction
thrproba = 0.5
#threshold for probability prediction specific UIP
thrprobaUIP=0.5
#probability for lung acceptance
thrlung=0.7

#subsample by default
subsdef=10
#normalization internal procedure or openCV
normiInternal=True

# average pxixel spacing
avgPixelSpacing=0.734
#workingdirectory='C:\Users\sylvain\Documents\boulot\startup\radiology\PREDICT'
#installdirectory='C:\Users\sylvain\Documents\boulot\startup\radiology\UIP\python'
#global directory for predict file
namedirtop = 'predict_essai'

#directory for storing image out after prediction
predictout='predicted_results'

#directory with lung mask dicom
lungmask='lung_mask'

#directory to put  lung mask bmp
lungmaskbmp='scan_bmp'


#directory name with scan with roi
sroi='sroi'

#subdirectory name to put images
jpegpath = 'patch_jpeg'

#directory with bmp from dicom
scanbmp='scan_bmp'

Xprepkl='X_predict.pkl'
Xrefpkl='X_file_reference.pkl'

lungXprepkl='lung_X_predict.pkl'
lungXrefpkl='lung_X_file_reference.pkl'

#file to store different parameters
subsamplef='subsample.pkl'

#subdirectory name to colect pkl files for prediction





# list label not to visualize
excluvisu=['back_ground','healthy']
#excluvisu=[]

#dataset supposed to be healthy
datahealthy=['138']

#image  patch format
typei='bmp' 

#dicom file size in pixels
#dimtabx = 512
#dimtaby = 512


#########################################################################
cwd=os.getcwd()
glovalf=tempofile
path_patient = os.path.join(workingdir,namedirtop)
varglobal=(thrpatch,thrproba,path_patient,subsdef)

def setva():
        global varglobal,thrproba,thrpatch,subsdef,path_patient
        if not os.path.exists(glovalf) :
            pickle.dump(varglobal, open( glovalf, "wb" ))
        else:
            dd = open(glovalf,'rb')
            my_depickler = pickle.Unpickler(dd)
            varglobal = my_depickler.load()
            dd.close() 
#            print varglobal
            path_patient=varglobal[2]
#            print path_patient
            thrproba=varglobal[1]
            thrpatch=varglobal[0]
            subsdef=varglobal[3]



def newva():
 pickle.dump(varglobal, open( glovalf, "wb" ))


(cwdtop,tail)=os.path.split(cwd)

#if not os.path.exists(path_patient):
#    print 'patient directory does not exists'
#    sys.exit()

setva()   
picklein_file={}

for setn in listset:
    picklein_file[setn] = os.path.join(instdir,picklefile[setn])
#    print picklein_file[setn]
    if not os.path.exists(picklein_file[setn]):
        
        print 'model and weight directory does not exists for: ',setn
        sys.exit()
    lpck= os.listdir(picklein_file[setn])
    pson=False
    pweigh=False
    for l in lpck:
        if l.find('.json',0)>0:
            pson=True
        if l.find('ILD_CNN_model_weights',0)==0:
    
            pweigh=True
    if not(pweigh and pson):
        print 'model and/or weight files does not exists for : ',setn
        sys.exit()     

picklein_lung_file = os.path.join(instdir,picklefilelung)
if not os.path.exists(picklein_lung_file):
    print 'model and weight directory for lung does not exists'
    sys.exit()
lpck= os.listdir(picklein_lung_file)
pson=False
pweigh=False
for l in lpck:
    if l.find('.json',0)>0:
        pson=True
    if l.find('ILD_CNN_model_weights',0)==0:

        pweigh=True
if not(pweigh and pson):
    print 'model and/or weight files for lung does not exists'
    sys.exit()     

pxys={}
for nset in listset:
    pxys[nset]=float(dimpavxs[nset]*dimpavys[nset])

#end general part


#color of labels
black=(0,0,0)
red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255,255,0)
cyan=(0,255,255)
purple=(255,0,255)
white=(255,255,255)
darkgreen=(11,123,96)
pink =(255,128,150)
lightgreen=(125,237,125)
orange=(255,153,102)
lowgreen=(0,51,51)


def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path,ignore_errors=True)


#    

def dd(i):
    if (i)<10:
        o='0'+str(i)
    else:
        o=str(i)
    return o

def rsliceNum(s,c,e):
    ''' look for  afile according to slice number'''
    #s: file name, c: delimiter for snumber, e: end of file extension
    endnumslice=s.find(e)
    posend=endnumslice
    while s.find(c,posend)==-1:
        posend-=1
    debnumslice=posend+1
    return int((s[debnumslice:endnumslice])) 
    
def nothings(x):
    global imgtext
    imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
    pass

def nothing(x):
    pass

def contrast(im,r):   
     tabi = np.array(im)
     r1=0.5+r/100.0
     tabi1=tabi*r1     
     tabi2=np.clip(tabi1,0,255)
     tabi3=tabi2.astype(np.uint8)
     return tabi3

def lumi(im,r):
    tabi = np.array(im)
    r1=r
    tabi1=tabi+r1
    tabi2=np.clip(tabi1,0,255)
    return tabi2

# mouse callback function
def draw_circle(event,x,y,flags,img):
    global ix,iy,quitl,patchi
    patchi=False

    if event == cv2.EVENT_RBUTTONDBLCLK:
        print x, y
    if event == cv2.EVENT_LBUTTONDBLCLK:
 
#        print('identification')
        ix,iy=x,y
       
        patchi=True
#        print 'identification', ix,iy, patchi
        if x>250 and x<270 and y>dimtaby-30 and y<dimtaby-10:
            print 'quit'
            ix,iy=x,y
            quitl=True

#def addpatchn(col,lab, xt,yt,imgn,nset):
##    print col,lab
#    dimpavx=dimpavxs[nset]
#    dimpavy=dimpavys[nset]
#    cv2.rectangle(imgn,(xt,yt),(xt+dimpavx,yt+dimpavy),col,1)
#    return imgn
# 
#def retrievepatch(x,y,top,sln,pr,li,nset):
#    tabtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
#    dimpavx=dimpavxs[nset]
#    dimpavy=dimpavys[nset]
#    ill=-1
#    pfound=False
#    for f in li:
#        ill+=1 
#        slicenumber=f[0]
#
#        if slicenumber == sln:
#
#            xs=f[1]
#            ys=f[2]
##            print xs,ys
#            if x>xs and x < xs+dimpavx and y>ys and y<ys+dimpavy:
#                     print x, y
#                     proba=pr[ill]
#                     pfound=True
#
#                     n=0
#                     cv2.putText(tabtext,'X',(xs-5+dimpavx/2,ys+5+dimpavy/2),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
#                     for j in range (0,len(proba)):
#                         
##                     for j in range (0,2):
#                         if proba[j]>0.01:
#                             n=n+1
#                             strw=fidclass(j,classif[nset])+ ' {0:.1f}%'.format(100*proba[j])                             
#                             cv2.putText(tabtext,strw,(dimtabx-142,(dimtaby-60)+10*n),cv2.FONT_HERSHEY_PLAIN,0.8,(0,255,0),1)
#                             
#                             print fidclass(j,classif[nset]), ' {0:.2f}%'.format(100*proba[j])
#                     print'found'
#                     break 
##    cv2.imshow('image',tabtext)                
#    if not pfound:
#            print'not found'
#    return tabtext
#
#def drawpatch(t,lp,preprob,k,top,nset):
#    imgn = np.zeros((dimtabx,dimtaby,3), np.uint8)
#    ill = 0
#    endnumslice=k.find('.bmp')
#
##    print imgcore
#    posend=endnumslice
#    while k.find('-',posend)==-1:
#            posend-=1
#    debnumslice=posend+1
#    slicenumber=int((k[debnumslice:endnumslice])) 
#    th=t/100.0
#    listlabel={}
#    listlabelaverage={}
##    print slicenumber,th
#    for ll in lp:
#
##            print ll
#            slicename=ll[0]          
#            xpat=ll[1]
#            ypat=ll[2]        
#        #we find max proba from prediction
#            proba=preprob[ill]
#           
#            prec, mprobai = maxproba(proba)
#
#            classlabel=fidclass(prec,classif[nset])
#            classcolor=classifc[classlabel]
#       
#            
#            if mprobai >th and slicenumber == slicename and\
#            (top in datahealthy or (classlabel not in excluvisu)):
##                    print classlabel
#                    if classlabel in listlabel:
##                        print 'found'
#                        numl=listlabel[classlabel]
#                        listlabel[classlabel]=numl+1
#                        cur=listlabelaverage[classlabel]
##                               print (numl,cur)
#                        averageproba= round((cur*numl+mprobai)/(numl+1),2)
#                        listlabelaverage[classlabel]=averageproba
#                    else:
#                        listlabel[classlabel]=1
#                        listlabelaverage[classlabel]=mprobai
#
#                    imgn= addpatchn(classcolor,classlabel,xpat,ypat,imgn,nset)
#
#
#            ill+=1
##            print listlabel        
#    for ll1 in listlabel:
##                print ll1,listlabelaverage[ll1]
#                tagviewn(imgn,ll1,str(round(listlabelaverage[ll1],2)),listlabel[ll1],175,00,nset)
#    ts='Treshold:'+str(t)
##    cv2.putText(imgn,ts,(0,42),cv2.FONT_HERSHEY_PLAIN,1,white,0.8,cv2.LINE_AA)
#    cv2.putText(imgn,ts,(0,42),cv2.FONT_HERSHEY_PLAIN,0.8,white,1,cv2.LINE_AA)
#    return imgn
    
def opennew(dirk, fl,L):
    pdirk = os.path.join(dirk,L[fl])
    img = cv2.imread(pdirk,1)
    return img,pdirk

def reti(L,c):
    for i in range (0, len(L)):
     if L[i]==c:
         return i
         break
     

def openfichier(k,dirk,top,L,nset):
    nseed=reti(L,k) 
#    print 'openfichier', k, dirk,top,nseed
  
    global ix,iy,quitl,patchi,classdirec
    global imgtext, dimtabx,dimtaby
   
    patchi=False
    ix=0
    iy=0
    ncf1 = os.path.join(path_patient,top)
    dop =os.path.join(ncf1,picklefile[nset])
    if classdirec==2:
        ll=os.listdir(ncf1)
        for l in ll:
            ncf =os.path.join(ncf1,l)
            dop =os.path.join(ncf,picklefile[nset])
    else:
        ncf=ncf1
            
    subsample=varglobal[3]
    pdirk = os.path.join(dirk,k)
    img = cv2.imread(pdirk,1)
    dimtabx= img.shape[0]
    dimtaby= dimtabx
    imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
#    print 'openfichier:',k , ncf, pdirk,top
    
    (preprob,listnamepatch)=loadpkl(ncf,nset)      
    
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)

    cv2.createTrackbar( 'Brightness','image',0,100,nothing)
    cv2.createTrackbar( 'Contrast','image',50,100,nothing)
    cv2.createTrackbar( 'Threshold','image',50,100,nothing)
    cv2.createTrackbar( 'Flip','image',nseed,len(L)-1,nothings)
        
    while(1):
        cv2.setMouseCallback('image',draw_circle,img)
        c = cv2.getTrackbarPos('Contrast','image')
        l = cv2.getTrackbarPos('Brightness','image')
        tl = cv2.getTrackbarPos('Threshold','image')
        fl = cv2.getTrackbarPos('Flip','image')

        img,pdirk= opennew(dirk, fl,L)
#        print pdirk
        
        
        (topnew,tailnew)=os.path.split(pdirk)
        endnumslice=tailnew.find('.bmp',0)
        posend=endnumslice
        while tailnew.find('-',posend)==-1:
            posend-=1
            debnumslice=posend+1
        slicenumber=int((tailnew[debnumslice:endnumslice])) 
        
        imglumi=lumi(img,l)
        imcontrast=contrast(imglumi,c)        
        imcontrast=cv2.cvtColor(imcontrast,cv2.COLOR_BGR2RGB)
#        print imcontrast.shape, imcontrast.dtype
        imgn=drawpatch(tl,listnamepatch,preprob,L[fl],top,nset)
#        imgn=cv2.cvtColor(imgn,cv2.COLOR_BGR2RGB)
        imgngray = cv2.cvtColor(imgn,cv2.COLOR_BGR2GRAY)
#        print imgngray.shape, imgngray.dtype
        mask_inv = cv2.bitwise_not(imgngray)              
        outy=cv2.bitwise_and(imcontrast,imcontrast,mask=mask_inv)
        imgt=cv2.add(imgn,outy)
 
       
        cv2.rectangle(imgt,(250,dimtaby-10),(270,dimtaby-30),red,-1)
        cv2.putText(imgt,'quit',(260,dimtaby-10),cv2.FONT_HERSHEY_PLAIN,1,yellow,1,cv2.LINE_AA)
        imgtoshow=cv2.add(imgt,imgtext)        
        imgtoshow=cv2.cvtColor(imgtoshow,cv2.COLOR_BGR2RGB)

        
        cv2.imshow('image',imgtoshow)

        if patchi :
            print 'retrieve patch asked'
            imgtext=retrievepatch(ix,iy,top,slicenumber,preprob,listnamepatch,nset)
            patchi=False

        if quitl or cv2.waitKey(20) & 0xFF == 27 :
#            print 'on quitte', quitl
            break
    quitl=False
#    print 'on quitte 2'
    cv2.destroyAllWindows()


def listfichier(dossier):
    Lf=[]
    L= os.listdir(dossier)
#    print L
    for k in L:
        if ".bmp" in k.lower(): 
            Lf.append(k)
    return Lf

def listbtn2(L,dirk,top,nset):
    for widget in cadreim.winfo_children():
        widget.destroy()
    canvas = Canvas(cadreim, borderwidth=2, width=200,height=600,background="blue")
    frame = Frame(canvas, background="blue")
    vsb = Scrollbar(cadreim, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)

    vsb.pack(side="right", fill="y")
    canvas.pack(side="right", fill="both", expand=True)
    canvas.create_window((1,1), window=frame, anchor="nw")
#    canvas.create_window((1,1), window=frame)

    frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
       
    for k in L: 
           Button(frame,text=k,command=lambda k = k:\
           openfichier(k,dirk,top,L,nset)).pack(side=TOP,expand=1)

    
def opendir(k,nset):
    global classdirec
#    
#    for widget in cadrepn.winfo_children():
#        widget.destroy()
    for widget in cadrestat.winfo_children():
        widget.destroy()
    Label(cadrestat, bg='lightgreen',text='patient:'+k).pack(side=TOP,fill=X,expand=1)
    tow=''
    fdir=os.path.join(path_patient,k)
    if classdirec==1:   
#        fdir=os.path.join(path_patient,k)
        bmp_dir = os.path.join(fdir, scanbmp)
    else:
        ldir=os.listdir(fdir)
        for ll in ldir:
             fdir = os.path.join(fdir, ll)
             bmp_dir = os.path.join(fdir, scanbmp)
    
    separator = Frame(cadrestat,height=2, bd=10, relief=SUNKEN)
    separator.pack(fill=X)
#    print 'bmp dir', bmp_dir      
    listscanfile =os.listdir(bmp_dir)
   
    ldcm=[]
    for ll in listscanfile:
      if  ll.lower().find('.bmp',0)>0:
         ldcm.append(ll)
    numberFile=len(ldcm)        
    tow='Number of sub sampled scan images: '+ str(numberFile)+\
    '\n\n'+'Predicted patterns: '+'\n' 
    for cle, valeur in listelabelfinal.items():
#             print 'cle valeur', cle,valeur
             for c in classif[nset]:
#                 print (k,c)
                 if (k,c) == cle and listelabelfinal[(k,c)]>0:

                     tow=tow+c+' : '+str(listelabelfinal[(k,c)])+'\n'

    Label(cadrestat, text=tow,bg='lightgreen').pack(side=TOP, fill='both',expand=1)
#    print tow
    dirkinter=os.path.join(fdir,predictout)
    dirk=os.path.join(dirkinter,vbg)
    L=listfichier(dirk)
    listbtn2(L,dirk,k,nset)
    
       
def listdossier(dossier): 
    L= os.walk(dossier).next()[1]  
    return L
    
def listbtn(L,nset):   
    cwt = Label(cadrerun,text="Select a patient")
    cwt.pack()
    for widget in cadrerun.winfo_children():       
                widget.destroy()    
    for k in L:
            Button(cadrerun,text=k,command=lambda k = k: opendir(k,nset)).pack(side=LEFT,fill="both",\
            expand=1)

def runf(nset):

    listbtn(listdossier( path_patient ),nset)

    
def onFrameConfigure(canvas):
    '''Reset the scroll region to encompass the inner frame'''
    canvas.configure(scrollregion=canvas.bbox("all"))
    
    
def quit():
    global fenetre
    fenetre.quit()
    fenetre.destroy()   




def runpredict(pp,subs,thrp, thpro,retou):
    
    global classdirec,path_patient, patch_list, \
             proba,subsdef,varglobal,thrproba
    global dimtabx, dimtaby
    global lungSegment
    for widget in cadretop.winfo_children():       
                widget.destroy()    
    for widget in cadrelistpatient.winfo_children():
               widget.destroy()
    for widget in cadreparam.winfo_children():
               widget.destroy()
    for widget in cadrerun.winfo_children():
               widget.destroy()
    for widget in cadrestat.winfo_children():
               widget.destroy()
    for widget in cadreim.winfo_children():
               widget.destroy()
                  
    cw = Label(cadrestatus, text="Running",fg='red',bg='blue')
    cw.pack(side=TOP,fill=X)
    thrpatch=thrp
    thrproba=thpro
    subsdef =subs
    path_patient=pp
    varglobal=(thrpatch,thrproba,path_patient,subsdef)  
    newva()
    runl()
   
    (top,tail)=os.path.split(path_patient)
   
    spdir = os.path.join(top, tail+'_s')
    remove_folder(spdir)    
    os.mkdir(spdir) 
#    print path_patient
    if os.path.exists(path_patient):
       
       patient_list= os.walk(path_patient).next()[1]
       for f in patient_list:
            
            lungSegment={}
            print('================================================') 
            print('work on:',f, 'with subsamples :', subs)        
            namedirtopcf1 = os.path.join(path_patient,f)           
            listscanfile1= os.listdir(namedirtopcf1)
            for ll in listscanfile1:
                namedirtopcf=os.path.join(namedirtopcf1,ll)
                if os.path.isdir(namedirtopcf):
#                    print 'it is a dir'
                    listscanfile= os.listdir(namedirtopcf)
                    classdirec=2
        #    for ll in patient_list2:
                elif ll.find('.dcm',0)>0:
        #            print 'it is not a dir'
                    listscanfile=listscanfile1
                    namedirtopcf=namedirtopcf1
#                    print 'write classider'
                    classdirec=1
                    break
                
            ldcm=[]
            lscanumber=[]
            for ll in listscanfile:
             if  ll.lower().find('.dcm',0)>0:
                ldcm.append(ll)
                endnumslice=ll.find('.dcm')   
                posend=endnumslice
                while ll.find('-',posend)==-1:
                    posend-=1
                debnumslice=posend+1
                scann=int(ll[debnumslice:endnumslice])
                lscanumber.append(scann)
            numberFile=len(ldcm)    
            
            if retou==1:
                print '2',spdir,path_patient
                destdir = os.path.join(spdir,f)
                print destdir
                remove_folder(destdir)    
                os.mkdir(destdir) 
                destdirlung = os.path.join(destdir,lungmask)
#                destdirlungbmp = os.path.join(destdirlung,lungmaskbmp)
#                remove_folder(destdirlungbmp) 
                remove_folder(destdirlung) 
                os.mkdir(destdirlung)                    
#                os.mkdir(destdirlungbmp) 
                print destdirlung
                #directory for scan in bmp
                bmp_dir = os.path.join(namedirtopcf, scanbmp)
                remove_folder(bmp_dir)    
                os.mkdir(bmp_dir) 
                #directory for lung mask
                lung_dir = os.path.join(namedirtopcf, lungmask)
                lung_bmp_dir = os.path.join(lung_dir, lungmaskbmp)
#                if os.path.exists(lung_dir)== False:
#                   os.mkdir(lung_dir)
#                if os.path.exists(lung_bmp_dir)== False:
#                   os.mkdir(lung_bmp_dir)
                listlung= os.listdir(lung_dir)
                #directory for pickle from cnn and status
                for nset in listset:
                    
                    pickledir = os.path.join( namedirtopcf,nset)
#                    print pickledir
                    remove_folder(pickledir)
                    os.mkdir(pickledir) 
            #directory for picklefile lung
                pickledir_lung = os.path.join( namedirtopcf,picklefilelung)
                remove_folder(pickledir_lung)
                os.mkdir(pickledir_lung)
                #directory for bpredicted images
                predictout_f_dir = os.path.join( namedirtopcf,predictout)
                remove_folder(predictout_f_dir)
                os.mkdir(predictout_f_dir)
                
                predictout_f_dir_bg = os.path.join( predictout_f_dir,vbg)
                remove_folder(predictout_f_dir_bg)
                os.mkdir(predictout_f_dir_bg)  
                
                predictout_f_dir_th = os.path.join( predictout_f_dir,str(thrproba))
                remove_folder(predictout_f_dir_th)
                os.mkdir(predictout_f_dir_th) 
                
                #directory for the pavaement in jpeg                
                jpegpathf = os.path.join( namedirtopcf,jpegpath)
                remove_folder(jpegpathf)    
                os.mkdir(jpegpathf)
                

                Nset=numberFile/3
                print 'total number of scans: ',numberFile, 'in each set: ', Nset
                upperset=[]
                middleset=[]
                lowerset=[]
                allset=[]
                for scanumber in range(0,numberFile):
        #            print scanumber
                    if scanumber%subs==0:
                        
                        allset.append(lscanumber[scanumber])
#                        print 'loop',scanumber
                        
                        if scanumber < Nset:
                            upperset.append(lscanumber[scanumber])
                        elif scanumber < 2*Nset:
                            middleset.append(lscanumber[scanumber])
                        else:
                            lowerset.append(lscanumber[scanumber])
                        lungSegment['upperset']=upperset
                        lungSegment['middleset']=middleset
                        lungSegment['lowerset']=lowerset
                        lungSegment['allset']=allset
                       
                        scanfile=ldcm[scanumber] 
                        scn=rsliceNum(scanfile,'-','.dcm')
                        for i in listlung:
                           if sepIsDash:
                               scanlung=rsliceNum(i,'-','.dcm')
                               
                               
                           else:
                               scanlung=rsliceNum(i,'_','.dcm')
                           if scanlung==scn:
                               if sepIsDash:
                                   endnumslice=i.find('.dcm')  
                                   core=i[0:endnumslice]
                                   print core
                                   coref=core+'_'+str(scn)+'.dcm'
                                   dst=os.path.join(destdirlung,coref)
                               else:
                                   dst=os.path.join(destdirlung,i)
                               src=os.path.join(lung_dir,i)  
                          
#                        print scanfile,namedirtopcf
                               shutil.copyfile(src, dst)
                               break
                                   
                
                        print scanfile, scn
                        src=os.path.join(namedirtopcf,scanfile)
                        dst=os.path.join(destdir,scanfile)
#                        print scanfile,namedirtopcf
                        shutil.copyfile(src, dst)
                        
#           
                
          
                
#                uiptree(namedirtopcf,'set2')
#                classpatch(namedirtopcf)
            print('completed on: ',f)    
            print('================================================')  
            print('================================================') 
       
       (top, tail)= os.path.split(path_patient)
       for widget in cadrestatus.winfo_children():       
                widget.destroy()
       wcadrewait = Label(cadrestatus, text="completed for "+tail,fg='darkgreen',bg='lightgreen',width=85)
       wcadrewait.pack()

       runf('set2')
    else:
    #            print 'path patient does not exist'
        wer = Label(cadrestatus, text="path for patients does not exist",\
               fg='red',bg='yellow',width=85)
        wer.pack(side=TOP,fill='both')
        bouton1_run = Button(cadrestatus, text="continue", fg='red',\
              bg='yellow',command= lambda: runl1())
        bouton1_run.pack()


def runl1 ():
    for widget in cadrelistpatient.winfo_children():
               widget.destroy()
    for widget in cadreparam.winfo_children():
               widget.destroy()
    for widget in cadrestatus.winfo_children():
                widget.destroy()
    for widget in cadretop.winfo_children():
                widget.destroy()
    for widget in cadrerun.winfo_children():
                widget.destroy()
    for widget in cadrestat.winfo_children():
                widget.destroy()
    for widget in cadreim.winfo_children():
                widget.destroy()
#    for widget in cadrepn.winfo_children():
#                widget.destroy()
    runl()

def chp(newp):
    global varglobal
    varglobal=(thrpatch,thrproba,newp,subsdef)
    for widget in cadrelistpatient.winfo_children():
               widget.destroy()
    for widget in cadreparam.winfo_children():
               widget.destroy()
    for widget in cadrestatus.winfo_children():
                widget.destroy()
    for widget in cadretop.winfo_children():
                widget.destroy()
    for widget in cadrerun.winfo_children():
                widget.destroy()
    for widget in cadrestat.winfo_children():
                widget.destroy()
    for widget in cadreim.winfo_children():
                widget.destroy()
#    for widget in cadrepn.winfo_children():
#                widget.destroy()
   
#    print varglobal
    runl()




def runl ():
    global path_patient,varglobal
    runalready=False
#    print path_patient  varglobal=(thrpatch,thrproba,path_patient,subsdef)
    bouton_quit = Button(cadretop, text="Quit", command= quit,bg='red',fg='yellow')
    bouton_quit.pack(side="top")
    separator = Frame(cadretop,height=2, bd=10, relief=SUNKEN)
    separator.pack(fill=X)
    w = Label(cadretop, text="path for patients:")
    w.pack(side=LEFT,fill='both')
    
    clepp = StringVar()
    e = Entry(cadretop, textvariable=clepp,width=80)
    e.delete(0, END)
    e.insert(0, varglobal[2])
#    e.insert(0, workingdir)
    e.pack(side=LEFT,fill='both',expand=1)
    boutonp = Button(cadretop, text='change patient dir',command= lambda : chp(clepp.get()),bg='green',fg='blue')
    boutonp.pack(side=LEFT)
##   
#    print varglobal
    if os.path.exists(varglobal[2]):
        pl=os.listdir(varglobal[2])
        ll = Label(cadrelistpatient, text='list of patient(s):')
        ll.pack()
        tow=''
        for l in pl:
            ld=os.path.join(varglobal[2],l)
            if os.path.isdir(ld):
                tow =tow+l+' - '
                pdir=os.path.join(ld,picklefile['set2'])
                if os.path.exists(pdir):
                    runalready=True
                else:
                    psp=os.listdir(ld)
                    for ll in psp:
                        if ll.find('.dcm')<0:
                            pdir1=os.path.join(ld,ll)
                            pdir=os.path.join(pdir1,picklefile['set2'])
                            if os.path.exists(pdir):
                                runalready=True
                            
            ll = Label(cadrelistpatient, text=tow,fg='blue')
            
        ll.pack(side =TOP)
       
             

    else:     
        print 'do not exist'
        ll = Label(cadrelistpatient, text='path_patient does not exist:',fg='red',bg='yellow')
        ll.pack()

    separator = Frame(cadrelistpatient,height=2, bd=10, relief=SUNKEN)
    separator.pack(fill=X)

    wcadre5 = Label(cadreparam, text="subsample:")
    wcadre5.pack(side=LEFT)
    clev5 = IntVar()
    e5 = Entry(cadreparam, textvariable=clev5,width=5)
    e5.delete(0, END)
    e5.insert(0, str(varglobal[3]))
    e5.pack(fill='x',side=LEFT)
    wcadresep = Label(cadreparam, text=" | ",bg='purple')
    wcadresep.pack(side=LEFT)  

    wcadre6 = Label(cadreparam, text="patch ovelapp [0-1]:")
    wcadre6.pack(side=LEFT)    
    clev6 = DoubleVar()
    e6 = Entry(cadreparam, textvariable=clev6,width=5)
    e6.delete(0, END)
    e6.insert(0, str(varglobal[0]))    
    e6.pack(fill='x',side=LEFT)
    wcadresep = Label(cadreparam, text=" | ",bg='purple')
    wcadresep.pack(side=LEFT) 

    wcadre7 = Label(cadreparam, text="predict proba acceptance[0-1]:")
    wcadre7.pack(side=LEFT)
    clev7 = DoubleVar()
    e7 = Entry(cadreparam, textvariable=clev7,width=5)
    e7.delete(0, END)
    e7.insert(0, str(varglobal[1]))
    e7.pack(fill='x',side=LEFT)
    wcadresep = Label(cadreparam, text=" | ",bg='purple')   
    wcadresep.pack(side=LEFT)
    
#    retour0=IntVar(cadreparam)
#    bouton0 = Radiobutton(cadreparam, text='run predict',variable=retour0,value=1,bd=2)
#    bouton0.pack(side=RIGHT)
#    if runalready:
#         bouton1 = Radiobutton(cadreparam, text='visu only',variable=retour0,value=0,bd=2)
#         bouton1.pack(side=RIGHT)
#    print runalready
    if runalready:
       bouton_run1 = Button(cadreparam, text="Run visu", bg='green',fg='blue',\
             command= lambda: runpredict(clepp.get(),clev5.get(),clev6.get(),clev7.get(),0))
       bouton_run1.pack(side=RIGHT)
    bouton_run2 = Button(cadreparam, text="Run predict", bg='green',fg='blue',\
             command= lambda: runpredict(clepp.get(),clev5.get(),clev6.get(),clev7.get(),1))
    bouton_run2.pack(side=RIGHT)
#    separator = Frame(cadretop,height=2, bd=10, relief=SUNKEN)
#    separator.pack(fill=X, padx=5, pady=2)


##########################################################
    
t = datetime.datetime.now()
today = str('date: '+dd(t.month)+'-'+dd(t.day)+'-'+str(t.year)+\
'_'+dd(t.hour)+':'+dd(t.minute)+':'+dd(t.second))

print today


quitl=False
patchi=False
listelabelfinal={}
oldc=0
#imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)

fenetre = Tk()
fenetre.title("predict")
fenetre.geometry("700x800+100+50")



cadretop = LabelFrame(fenetre, width=700, height=20, text='top',borderwidth=5,bg="purple",fg='yellow')
cadretop.grid(row=0,sticky=NW)
cadrelistpatient = LabelFrame(fenetre, width=700, height=20, text='list',borderwidth=5,bg="purple",fg='yellow')
cadrelistpatient.grid(row=1,sticky=NW)
cadreparam = LabelFrame(fenetre, width=700, height=20, text='param',borderwidth=5,bg="purple",fg='yellow')
cadreparam.grid(row=2,sticky=NW)
cadrestatus = LabelFrame(fenetre,width=700, height=20,text="status run",bg='purple',fg='yellow')
cadrestatus.grid(row=3,sticky=NW)
cadrerun = LabelFrame(fenetre,text="select a patient",width=700, height=20,fg='yellow',bg='purple')
cadrerun.grid(row=4,sticky=NW)
#cadrepn = LabelFrame(fenetre,text="patient name list:",width=700, height=20,bg='purple',fg='yellow')
#cadrepn.grid(row=5,sticky=NW)
cadrestat=LabelFrame(fenetre,text="statistic", width=350,height=20,fg='yellow',bg='purple')
cadrestat.grid(row=6,  sticky=NW )
cadreim=LabelFrame(fenetre,text="images", width=350,height=20,fg='yellow',bg='purple')
cadreim.grid(row=6,  sticky=E)
    
#setva()
runl()


#dataset_list=[]
#nameset_list=[]
#proba=[]
fenetre.mainloop()

#visuinter()
#errorfile.close() 
