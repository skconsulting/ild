# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:21:22 2017

@author: sylvain
version 1.0
"""
from param_pix_p import *
from tdGenePredictGui import *


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

def tagviewn(fig,label,pro,nbr,x,y):
    """write text in image according to label and color"""

    col=classifc[label]
#    print col, label
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



def drawpatch(t,dx,dy,slicenumber,va,patch_list_cross_slice):
    imgn = np.zeros((dx,dy,3), np.uint8)
    ill = 0
#    endnumslice=k.find('.bmp')
#    slicenumber=slnum
    th=t/100.0
    listlabel={}
    listlabelaverage={}
#    print slicenumber,th
    for ll in patch_list_cross_slice[slicenumber]:

#            print ll
#            slicename=ll[0]
            xpat=ll[0][0]
            ypat=ll[0][1]
        #we find max proba from prediction
            proba=ll[1]

            prec, mprobai = maxproba(proba)

            classlabel=fidclass(prec,classif)
            classcolor=classifc[classlabel]


            if mprobai >th and classlabel not in excluvisu and va[classlabel]==True:
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
    ts='Threshold:'+str(t)

    cv2.putText(imgn,ts,(0,50),cv2.FONT_HERSHEY_PLAIN,0.7,white,1)
    return imgn


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
    for pat in classif :
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
                      lungSegment,tabMed,thrprobaUIP,patch_list_cross_slice_sub):
#def openfichiervolume(listHug,path_patient,patch_list_cross_slice,
#                      lungSegment,tabMed,thrprobaUIP,patch_list_cross_slice_sub):
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichiervolume start',path_patient
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
    voltotal=int(round(((dictPS['all'][0]+dictPS['all'][1])*volelem),0))

    for patt in classif:
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
    cv2.createTrackbar( 'None','SliderVol',0,1,nothings)
    viewasked={}
    for key in classif:
        viewasked[key]=True
        cv2.createTrackbar( key,'SliderVol',0,1,nothings)
    imgbackg = np.zeros((dimtabx,dimtaby,3), np.uint8)
    posrc=0
    xr=800
    xrn=xr+10
    for key1 in classif:
        yr=15*posrc

        yrn=yr+10
        cv2.rectangle(imgbackg, (xr, yr),(xrn,yrn), classifc[key1], -1)
        cv2.putText(imgbackg,key1,(xr+15,yr+10),cv2.FONT_HERSHEY_PLAIN,1.0,classifc[key1],1 )
        dictpostotal[key1]=(xr-130,yr+10)
        posrc+=1
    #for total volume
    posrc+=1
    yr=15*posrc
    yrn=yr+10
    cv2.rectangle(imgbackg, (xr, yr),(xrn,yrn), white, -1)
    cv2.putText(imgbackg,key1,(xr+15,yr+10),cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
    dictpostotal['total']=(xr-130,yr+10)
    
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

    while(1):
#            print "corectnumber",corectnumber

            cv2.setMouseCallback('imageVol',draw_circle,imgtext)
            tl = cv2.getTrackbarPos('Threshold','SliderVol')
            allview = cv2.getTrackbarPos('All','SliderVol')
            noneview = cv2.getTrackbarPos('None','SliderVol')

            for key1 in classif:
                s = cv2.getTrackbarPos(key1,'SliderVol')
                if allview==1:
                     viewasked[key1]=True
                elif noneview ==1:
                    viewasked[key1]=False
                elif s==0:
                    viewasked[key1]=False
                else:
                     viewasked[key1]=True
#                     print key1
            dictP = {}  # dictionary with patch in lung segment    
            dictSubP = {}  # dictionary with patch in subpleural
            dictSurf = {}  # dictionary with patch volume in percentage
            for patt in classif:
                dictP = initdictP(dictP, patt)
                dictSubP = initdictP(dictSubP, patt)
            tl=tl/100.0
            thrprobaUIP=tl
            dictP, dictSubP, dictSurf= uipTree(dirf,patch_list_cross_slice,lungSegment,tabMed,dictPS,
                                               dictP,dictSubP,dictSurf,thrprobaUIP,patch_list_cross_slice_sub)
#            filet= open ('a.txt',"w")
#            filet.write(str(dictP))
#            filet.close()
#            break
            surfmax={}
            patmax={}
            
            imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)
            img = np.zeros((dimtabx,dimtaby,3), np.uint8)
            cv2.putText(imgtext,'Threshold : '+str(tl),(50,50),cv2.FONT_HERSHEY_PLAIN,1,yellow,1,cv2.LINE_AA)
            if allview==1:

                for patt in classif:
                    vol=int(round(((dictP[patt]['all'][0]+dictP[patt]['all'][1] )*volelem),0))    

#                    print patt,vol
                    cv2.putText(imgtext,'Vol ml: '+str(vol),(dictpostotal[patt][0],dictpostotal[patt][1]),
                                cv2.FONT_HERSHEY_PLAIN,1.0,classifc[patt],1 )
                #total volume
                cv2.putText(imgtext,'Total ml: '+str(voltotal),(dictpostotal['total'][0],dictpostotal['total'][1]),
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
                    for pat in classif:
#                        img = np.zeros((dimtabx,dimtaby,3), np.uint8)
                        if viewasked[pat]==True:
                             vol=int(round(((dictP[pat]['all'][0]+dictP[pat]['all'][1] )*volelem),0)) 
#                             vol=int((dictP[pat]['all'][0]+dictP[pat]['all'][1]) *volelem)         
#                    print patt,vol
                             cv2.putText(imgtext,'Vol ml: '+str(vol),(dictpostotal[pat][0],dictpostotal[pat][1]),
                                cv2.FONT_HERSHEY_PLAIN,1.0,classifc[pat],1 )
                             if dictSurf[pat][i]>0:
                                 colori=classifc[pat]
                             else:
                                 colori=grey
                             lungtw=colorimage(dictPosImage[i],colori)
                             img=cv2.add(img,lungtw)
                             cv2.rectangle(imgtext,(dictPosTextImage[i][0],dictPosTextImage[i][1]-15),(dictPosTextImage[i][0]+55,dictPosTextImage[i][1]),white,-1)
                             cv2.putText(imgtext,str(dictSurf[pat][i])+'%',(dictPosTextImage[i][0],dictPosTextImage[i][1]),cv2.FONT_HERSHEY_PLAIN,1.2,grey,1,)
#            break
                cv2.putText(imgtext,'Total ml: '+str(voltotal),(dictpostotal['total'][0],dictpostotal['total'][1]),
                                cv2.FONT_HERSHEY_PLAIN,1.0,white,1 )
            imgtowrite=cv2.add(imgtext,imgbackg)
            imgray = cv2.cvtColor(imgtowrite,cv2.COLOR_BGR2GRAY)
            np.putmask(imgray,imgray>0,255)
            nthresh=cv2.bitwise_not(imgray)
            vis1=cv2.bitwise_and(img,img,mask=nthresh)
            imgtowrite=cv2.add(vis1,imgtowrite)
#            imgtowrite=cv2.add(imgtext,imgtowrite)
            imgtowrite=cv2.cvtColor(imgtowrite,cv2.COLOR_BGR2RGB)
            cv2.imshow('imageVol',imgtowrite)

            if quitl or cv2.waitKey(20) & 0xFF == 27 :
    #            print 'on quitte', quitl
                break
    quitl=False
    #    print 'on quitte 2'
    cv2.destroyWindow("imageVol")
    cv2.destroyWindow("SliderVol")

    return ''


def openfichier(ti,datacross,path_img,thrprobaUIP,patch_list_cross_slice):
    global  quitl,dimtabx,dimtaby,patchi,ix,iy
    print 'openfichier start'
    quitl=False
    
    slnt=datacross[0]
    dimtabx=datacross[1]
    dimtaby=datacross[2]

    patchi=False
    ix=0
    iy=0
    if ti =="cross view" or ti =="merge view":
        sn=scan_bmp
        corectnumber=1
#        psroi=os.path.join(path_img,sroi)
    else:
        sn=transbmp
        corectnumber=0

    pdirk = os.path.join(path_img,source_name)
    pdirk = os.path.join(pdirk,sn)
    list_image={}
    cdelimter='_'
    extensionimage='.'+typei

    limage=[name for name in os.listdir(pdirk) if name.find('.'+typei,1)>0 ]
    if len(limage)==0:
         limage=[name for name in os.listdir(pdirk) if name.find('.'+typei1,1)>0 ]
         extensionimage='.'+typei1
        
    if ((ti =="cross view" or ti =="merge view") and len(limage)+1==slnt) or ti =="front view":

        for iimage in limage:

            sln=rsliceNum(iimage,cdelimter,extensionimage)
            list_image[sln]=iimage

        image0=os.path.join(pdirk,list_image[slnt/2])
        img = cv2.imread(image0,1)
    #    cv2.imshow('cont',img)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()

        imgtext = np.zeros((dimtabx,dimtaby,3), np.uint8)

        cv2.namedWindow('imagecr',cv2.WINDOW_NORMAL)
        cv2.namedWindow("Slidercr",cv2.WINDOW_NORMAL)

        cv2.createTrackbar( 'Brightness','Slidercr',0,100,nothing)
        cv2.createTrackbar( 'Contrast','Slidercr',50,100,nothing)
        cv2.createTrackbar( 'Threshold','Slidercr',int(thrprobaUIP*100),100,nothing)
        cv2.createTrackbar( 'Flip','Slidercr',slnt/2,slnt-2,nothings)
        cv2.createTrackbar( 'All','Slidercr',1,1,nothings)
        cv2.createTrackbar( 'None','Slidercr',0,1,nothings)
        viewasked={}
        for key1 in classif:
#            print key1
            viewasked[key1]=True
            cv2.createTrackbar( key1,'Slidercr',0,1,nothings)
#        return

        while(1):
#            print "corectnumber",corectnumber
            cv2.setMouseCallback('imagecr',draw_circle,img)
            c = cv2.getTrackbarPos('Contrast','Slidercr')
            l = cv2.getTrackbarPos('Brightness','Slidercr')
            tl = cv2.getTrackbarPos('Threshold','Slidercr')
            fl = cv2.getTrackbarPos('Flip','Slidercr')
            allview = cv2.getTrackbarPos('All','Slidercr')
            noneview = cv2.getTrackbarPos('None','Slidercr')

            for key2 in classif:
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


    #        img,pdirk= opennew(dirk, fl,L)
    #        print pdirk
            slicenumber=fl+corectnumber
            imagel=os.path.join(pdirk,list_image[slicenumber])
            img = cv2.imread(imagel,1)
            
            imglumi=lumi(img,l)
            imcontrast=contrasti(imglumi,c)
            if int(slicenumber) ==13:
                print img[40][40],imglumi[40][40],imcontrast[40][40]
                
            imcontrast=cv2.cvtColor(imcontrast,cv2.COLOR_BGR2RGB)

    #        print 'imcontrast',imcontrast.shape, imcontrast.dtype
            imgn= drawpatch(tl,dimtabx,dimtaby,slicenumber,viewasked,patch_list_cross_slice)

            imgngray = cv2.cvtColor(imgn,cv2.COLOR_BGR2GRAY)
            np.putmask(imgngray,imgngray>0,255)
    #        print 'imgray',imgngray.shape, imgngray.dtype
            mask_inv = cv2.bitwise_not(imgngray)
#            print 'mask_inv',mask_inv.shape, mask_inv.min(),mask_inv.max()
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

            cv2.imshow('imagecr',imgtoshow)


            if patchi :
                print 'retrieve patch asked'
                imgtext= retrievepatch(ix,iy,slicenumber,dimtabx,dimtaby,patch_list_cross_slice)
#                imgtext=retrievepatch(ix,iy,fl+corectnumber,preprob,listnamepatch,dimtabx,dimtaby)
                patchi=False

            if quitl or cv2.waitKey(20) & 0xFF == 27 :
    #            print 'on quitte', quitl
                break
        quitl=False
    #    print 'on quitte 2'
        cv2.destroyWindow("imagecr")
        cv2.destroyWindow("Slidercr")

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
    for key1 in classif:
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
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,patch_list_cross_slice)
            
    elif viewstyle=='front view':
            datarep= pickle.load( open( os.path.join(path_data_dir,"datafront"), "rb" ))
            patch_list_front_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_front_slice"), "rb" ))
            thrprobaUIP=float(indata['thrprobaUIP'])
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,patch_list_front_slice)
            
    elif viewstyle=='volume view from cross':

            patch_list_cross_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice"), "rb" ))
            patch_list_cross_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_cross_slice_sub"), "rb" ))
            lungSegment= pickle.load( open( os.path.join(path_data_dir,"lungSegment"), "rb" ))
            tabMed= pickle.load( open( os.path.join(path_data_dir,"tabMed"), "rb" ))
            thrprobaUIP=float(indata['thrprobaUIP'])
            
            messageout = openfichiervolume(listHug,path_patient,patch_list_cross_slice,
                      lungSegment,tabMed,thrprobaUIP,patch_list_cross_slice_sub)
    
    elif viewstyle=='volume view from front':
            patch_list_front_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_front_slice"), "rb" ))
            patch_list_front_slice_sub= pickle.load( open( os.path.join(path_data_dir,"patch_list_front_slice_sub"), "rb" ))
            lungSegmentfront= pickle.load( open( os.path.join(path_data_dir,"lungSegmentfront"), "rb" ))
            tabMedfront= pickle.load( open( os.path.join(path_data_dir,"tabMedfront"), "rb" ))
            thrprobaUIP=float(indata['thrprobaUIP'])
            
            messageout = openfichiervolume(listHug,path_patient,patch_list_front_slice,
                      lungSegmentfront,tabMedfront,thrprobaUIP,patch_list_front_slice_sub)

    elif viewstyle=='merge view':

            thrprobaUIP=float(indata['thrprobaUIP'])
            datarep= pickle.load( open( os.path.join(path_data_dir,"datacross"), "rb" ))
#            patch_list_merge= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge"), "rb" ))
#            proba_merge= pickle.load( open( os.path.join(path_data_dir,"proba_merge"), "rb" ))
            patch_list_merge_slice= pickle.load( open( os.path.join(path_data_dir,"patch_list_merge_slice"), "rb" ))
            messageout=openfichier(viewstyle,datarep,patient_path_complet,thrprobaUIP,patch_list_merge_slice)
    
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
