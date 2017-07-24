# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:48:43 2017

@author: sylvain
tool for roi generation
"""
from param_pix_r import *


pattern=''

quitl=False
images={}
tabroi={}
tabroifinal={}
tabroinumber={}

def contour3(vis,im,l):
    col=classifc[l]
    visi = np.zeros((dimtabx,dimtaby,3), np.uint8)
    vis2= np.zeros((dimtabx,dimtaby,3), np.uint8)
    imagemax= cv2.countNonZero(np.array(im))
    if imagemax>0:
        mgray = cv2.cvtColor(vis,cv2.COLOR_BGR2GRAY)
        np.putmask(mgray,mgray>0,255)
        nthresh=cv2.bitwise_not(mgray)

        cv2.fillPoly(visi, [np.array(im)],col)
        vis1=cv2.bitwise_and(visi,visi,mask=nthresh)
        vis2=cv2.add(vis,vis1)
    return vis2

def click_and_crop(event, x, y, flags, param):
    global quitl,pattern,dirpath_patient,dirroit

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(menus, (150,12), (370,32), black, -1)
        posrc=0
        print x,y
        for key1 in classif:
            labelfound=False
            xr=5
            yr=15*posrc
            xrn=xr+10
            yrn=yr+10
            if x>xr and x<xrn and y>yr and y< yrn:

                print 'this is',key1
                pattern=key1
                cv2.rectangle(menus, (200,0), (210,10), classifc[pattern], -1)
                cv2.rectangle(menus, (212,0), (340,12), black, -1)
                cv2.putText(menus,key1,(215,10),cv2.FONT_HERSHEY_PLAIN,0.7,classifc[key1],1 )
                labelfound=True
                break
            posrc+=1

        if  x> zoneverticalgauche[0][0] and y > zoneverticalgauche[0][1] and x<zoneverticalgauche[1][0] and y<zoneverticalgauche[1][1]:
            print 'this is in menu'
            labelfound=True

        if  x> zoneverticaldroite[0][0] and y > zoneverticaldroite[0][1] and x<zoneverticaldroite[1][0] and y<zoneverticaldroite[1][1]:
            print 'this is in menu'
            labelfound=True

        if  x> zonehorizontal[0][0] and y > zonehorizontal[0][1] and x<zonehorizontal[1][0] and y<zonehorizontal[1][1]:
            print 'this is in menu'
            labelfound=True

        if x>posxdel and x<posxdel+10 and y>posydel and y< posydel+10:
            print 'this is suppress'
            suppress()
            labelfound=True

        if x>posxquit and x<posxquit+10 and y>posyquit and y< posyquit+10:
            print 'this is quit'
            quitl=True
            labelfound=True

        if x>posxdellast and x<posxdellast+10 and y>posydellast and y< posydellast+10:
            print 'this is delete last'
            labelfound=True
            dellast()

        if x>posxdelall and x<posxdelall+10 and y>posydelall and y< posydelall+10:
            print 'this is delete all'
            labelfound=True
            delall()

        if x>posxcomp and x<posxcomp+10 and y>posycomp and y< posycomp+10:
            print 'this is completed for all'
            labelfound=True
            completed(imagename,dirpath_patient,dirroit)

        if x>posxreset and x<posxreset+10 and y>posyreset and y< posyreset+10:
            print 'this is reset'
            labelfound=True
            reseted()
        if x>posxvisua and x<posxvisua+10 and y>posyvisua and y< posyvisua+10:
            print 'this is visua'
            labelfound=True
            visua()
        if x>posxeraseroi and x<posxeraseroi+10 and y>posyeraseroi and y< posyeraseroi+10:
            print 'this is erase roi'
            labelfound=True
            eraseroi(imagename,dirpath_patient)

        if x>posxlastp and x<posxlastp+10 and y>posylastp and y< posylastp+10:
            print 'this is last point'
            labelfound=True
            closepolygon()



        if not labelfound:
            print 'add point',pattern
            if len(pattern)>0:
#                print 'len pattern >0'
#                print 'fxs,x0',fxs,x0new
#                global fxs,x0new,y0new
                xnew=int((x+x0new)/fxs)
                ynew=int((y+y0new)/fxs)
#                print x,y,xnew,ynew
                numeropoly=tabroinumber[pattern][scannumber]
#                print 'length last pattent',len(tabroi[pattern][scannumber][numeropoly])
                tabroi[pattern][scannumber][numeropoly].append((xnew, ynew))
#                print numeropoly, tabroi[pattern][scannumber][numeropoly]
                cv2.rectangle(images[scannumber], (xnew,ynew),
                              (xnew,ynew), classifc[pattern], 1)

                for l in range(0,len(tabroi[pattern][scannumber][numeropoly])-1):
                    cv2.line(images[scannumber], (tabroi[pattern][scannumber][numeropoly][l][0],tabroi[pattern][scannumber][numeropoly][l][1]),
                              (tabroi[pattern][scannumber][numeropoly][l+1][0],tabroi[pattern][scannumber][numeropoly][l+1][1]), classifc[pattern], 1)
                    l+=1
#                cv2.imshow('images',images[scannumber])
            else:
                cv2.rectangle(menus, (212,0), (340,12), black, -1)
                cv2.putText(menus,'No pattern selected',(215,10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

def closepolygon():
    numeropoly=tabroinumber[pattern][scannumber]
    if len(tabroi[pattern][scannumber][numeropoly])>0:
        numeropoly+=1
        print 'numeropoly',numeropoly
        tabroinumber[pattern][scannumber]=numeropoly
        tabroi[pattern][scannumber][numeropoly]=[]
    else:
        'wait for new point'
    cv2.putText(menus,'polygone closed',(215,20),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

def suppress():
    numeropoly=tabroinumber[pattern][scannumber]
    lastp=len(tabroi[pattern][scannumber][numeropoly])
    if lastp>0:
        cv2.line(images[scannumber], (tabroi[pattern][scannumber][numeropoly][lastp-2][0],tabroi[pattern][scannumber][numeropoly][lastp-2][1]),
                 (tabroi[pattern][scannumber][numeropoly][lastp-1][0],tabroi[pattern][scannumber][numeropoly][lastp-1][1]), black, 1)
        tabroi[pattern][scannumber][numeropoly].pop()
        for l in range(0,len(tabroi[pattern][scannumber][numeropoly])-1):
            cv2.line(images[scannumber], (tabroi[pattern][scannumber][numeropoly][l][0],tabroi[pattern][scannumber][numeropoly][l][1]),
                     (tabroi[pattern][scannumber][numeropoly][l+1][0],tabroi[pattern][scannumber][numeropoly][l+1][1]), classifc[pattern], 1)
            l+=1
    cv2.putText(menus,'delete last entry',(215,20),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

def completed(imagename,dirpath_patient,dirroit):
    print 'completed start'
    for key in classif:
        numeropoly=tabroinumber[key][scannumber]
        for n in range (0,numeropoly+1):
            if len(tabroi[key][scannumber][n])>0:
                for l in range(0,len(tabroi[key][scannumber][n])-1):
                        cv2.line(images[scannumber], (tabroi[key][scannumber][n][l][0],tabroi[key][scannumber][n][l][1]),
                                      (tabroi[key][scannumber][n][l+1][0],tabroi[key][scannumber][n][l+1][1]), black, 1)

#            vis=contour3(tabroi[key][scannumber],key,dimtabx,dimtaby)
                images[scannumber]=contour3(images[scannumber],tabroi[key][scannumber][n],key)
#                cv2.imshow('images',images[scannumber])
                tabroi[key][scannumber][n]=[]
                tabroifinal[key][scannumber]=cv2.add(images[scannumber],tabroifinal[key][scannumber])

        tabroinumber[key][scannumber]=0

        imgray = cv2.cvtColor(tabroifinal[key][scannumber],cv2.COLOR_BGR2GRAY)

#        cv2.imshow('a',tabroifinal[key][scannumber])
        imagemax= cv2.countNonZero(imgray)
#        print key,imagemax
        if imagemax>0:
            dirroi=os.path.join(dirpath_patient,key)
            if not os.path.exists(dirroi):
                os.mkdir(dirroi)
#                print dirroi
#                print imagename,scannumber
            posext=imagename.find('.'+typei)
            imgcoreScans=imagename[0:posext]+'.'+typei
            imgcoreScan=os.path.join(dirroi,imgcoreScans)
            imgcoreRoi=os.path.join(dirroit,imgcoreScans) 
            if os.path.exists(imgcoreScan):
                cv2.putText(menus,'ROI '+key+' slice:'+str(scannumber)+' overwritten',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
            tabtowrite=cv2.cvtColor(tabroifinal[key][scannumber],cv2.COLOR_BGR2RGB)
            cv2.imwrite(imgcoreScan,tabtowrite)
            mroi=cv2.imread(imgcoreRoi,1)
            mroiaroi=cv2.addWeighted(mroi,1,tabtowrite,0.5,0)
            cv2.imwrite(imgcoreRoi,mroiaroi)
            
            cv2.putText(menus,'Slice ROI stored',(215,20),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
    images[scannumber]=np.zeros((dimtabx,dimtaby,3), np.uint8)


def visua():
    for key in classif:
        numeropoly=tabroinumber[key][scannumber]
        for n in range (0,numeropoly+1):
            if len(tabroi[key][scannumber][n])>0:
                    for l in range(0,len(tabroi[key][scannumber][n])-1):
                        cv2.line(images[scannumber], (tabroi[key][scannumber][n][l][0],tabroi[key][scannumber][n][l][1]),
                                      (tabroi[key][scannumber][n][l+1][0],tabroi[key][scannumber][n][l+1][1]), black, 1)

                    images[scannumber]=contour3(images[scannumber],tabroi[key][scannumber][n],key)
    cv2.putText(menus,' Visualization ROI',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

def eraseroi(imagename,dirpath_patient):
    print 'this is erase roi'
    if len(pattern)>0:
        tabroifinal[pattern][scannumber]=np.zeros((dimtabx,dimtaby,3), np.uint8)
        images[scannumber]=np.zeros((dimtabx,dimtaby,3), np.uint8)
        dirroi=os.path.join(dirpath_patient,pattern)
        imgcoreScan=os.path.join(dirroi,imagename)
#        tabroifinal[pattern][scannumber]=cv2.add(images[scannumber],tabroifinal[key][scannumber])
        if os.path.exists(imgcoreScan):
            os.remove(imgcoreScan)
            cv2.putText(menus,'ROI '+pattern+' slice:'+str(scannumber)+' erased',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
        else:
                    cv2.putText(menus,'ROI '+pattern+' slice:'+str(scannumber)+' not exist',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
    else:
        cv2.putText(menus,' no pattern defined',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )



#    vis=contour4(images[scannumber],pattern,dimtabx,dimtaby)
#    images[scannumber]=cv2.add(vis, images[scannumber])


def reseted():
#    global images
    for key in classif:
        numeropoly=tabroinumber[key][scannumber]
        for n in range (0,numeropoly+1):
            if len(tabroi[key][scannumber][n])>0:
                for l in range(0,len(tabroi[key][scannumber][n])-1):

    #                tabroifinal[pattern][tabroi[pattern][scannumber][l][0]][tabroi[pattern][scannumber][l][1]]=classifc[pattern]
                    cv2.line(images[scannumber], (tabroi[key][scannumber][n][l][0],tabroi[key][scannumber][n][l][1]),
                                  (tabroi[key][scannumber][n][l+1][0],tabroi[key][scannumber][n][l+1][1]), black, 1)
                tabroi[key][scannumber][n]=[]
#        tabroifinal[key][scannumber]=np.zeros((dimtabx,dimtaby,3), np.uint8)
    images[scannumber]=np.zeros((dimtabx,dimtaby,3), np.uint8)
    cv2.putText(menus,' Delete all drawings',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

def dellast():
#    global images
    numeropoly=tabroinumber[pattern][scannumber]
    if len(tabroi[pattern][scannumber][numeropoly])>0:
#        print 'len>0'
        for l in range(0,len(tabroi[pattern][scannumber][numeropoly])-1):
            cv2.line(images[scannumber], (tabroi[pattern][scannumber][numeropoly][l][0],tabroi[pattern][scannumber][numeropoly][l][1]),
                     (tabroi[pattern][scannumber][numeropoly][l+1][0],tabroi[pattern][scannumber][numeropoly][l+1][1]), black, 1)
        tabroi[pattern][scannumber][numeropoly]=[]
#        tabroinumber[pattern][scannumber]=max(numeropoly-1,0)
    elif numeropoly >0 :
#        print 'len=0 and num>0'
        numeropoly=numeropoly-1
        tabroinumber[pattern][scannumber]=numeropoly
        for l in range(0,len(tabroi[pattern][scannumber][numeropoly])-1):
            cv2.line(images[scannumber], (tabroi[pattern][scannumber][numeropoly][l][0],tabroi[pattern][scannumber][numeropoly][l][1]),
                     (tabroi[pattern][scannumber][numeropoly][l+1][0],tabroi[pattern][scannumber][numeropoly][l+1][1]), black, 1)
        tabroi[pattern][scannumber][numeropoly]=[]
    else:
        print'length null'
    cv2.putText(menus,' Delete last polygon',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

def delall():
#    global images
    print 'delall', pattern
    numeropoly=tabroinumber[pattern][scannumber]
#    for key in classif:
#        numeropoly=tabroinumber[key][scannumber]
#        print key, numeropoly
    for n in range (0,numeropoly+1):
            for l in range(0,len(tabroi[pattern][scannumber][n])-1):
                cv2.line(images[scannumber], (tabroi[pattern][scannumber][n][l][0],tabroi[pattern][scannumber][n][l][1]),
                    (tabroi[pattern][scannumber][n][l+1][0],tabroi[pattern][scannumber][n][l+1][1]), black, 1)
            tabroi[pattern][scannumber][n]=[]
    tabroinumber[pattern][scannumber]=0
#    images[scannumber]=np.zeros((dimtabx,dimtaby,3), np.uint8)
    cv2.putText(menus,' Delete all polygons',(150,30),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

def contrasti(im,r):

     r1=0.5+r/100.0
     im=im.astype(np.uint16)
     tabi1=im*r1
     tabi2=np.clip(tabi1,0,imageDepth)
     tabi3=tabi2.astype(np.uint8)
     return tabi3

def lumi(tabi,r):
    r1=r
    tabi1=tabi.astype(np.uint16)
    tabi1=tabi1+r1
    tabi2=np.clip(tabi1,0,imageDepth)
    tabi3=tabi2.astype(np.uint8)
    return tabi3

def rsliceNum(s,c,e):
    ''' look for  afile according to slice number'''
    #s: file name, c: delimiter for snumber, e: end of file extension
    endnumslice=s.find(e)
    posend=endnumslice
    while s.find(c,posend)==-1:
        posend-=1
    debnumslice=posend+1
    return int((s[debnumslice:endnumslice]))


def zoomfunction(im,z,px,py):
    global fxs,x0new,y0new
    fxs=1+(z/50.0)
    imgresize=cv2.resize(im,None,fx=fxs,fy=fxs,interpolation=cv2.INTER_LINEAR)
    dimtabxn=imgresize.shape[0]
    dimtabyn=imgresize.shape[1]
    px0=((dimtabxn-dimtabx)/2)*px/50
    py0=((dimtabyn-dimtaby)/2)*py/50
#    print px0,py0
#    x0=max(0,((dimtabxn-dimtabx)/2)-px0)
    x0=max(0,px0)
#    y0=max(0,((dimtabyn-dimtaby)/2)-py0)
    y0=max(0,py0)
    x1=min(dimtabxn,x0+dimtabx)
    y1=min(dimtabyn,y0+dimtaby)
#    print dimtabxn,x0,x1
    crop_img=imgresize[y0:y1,x0:x1]
#    print crop_img.shape
#    tabi = np.array(im)
    x0new=x0
    y0new=y0
    return crop_img


def loop(slnt,pdirk,dirpath_patient,dirroi):
    global quitl,scannumber,imagename
    quitl=False
    list_image={}
    cdelimter='_'
    extensionimage='.'+typei
    limage=[name for name in os.listdir(pdirk) if name.find('.'+typei,1)>0 ]

    if len(limage)+1==slnt:
#        print 'good'
#
        for iimage in range(0,slnt-1):
    #        print iimage
            s=limage[iimage]
                #s: file name, c: delimiter for snumber, e: end of file extension
            sln=rsliceNum(s,cdelimter,extensionimage)
            list_image[sln]=s
        fl=slnt/2
        imagename=list_image[fl+1]
        imagenamecomplet=os.path.join(pdirk,imagename)
        image = cv2.imread(imagenamecomplet,cv2.IMREAD_ANYDEPTH)
        image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.namedWindow("Slider2",cv2.WINDOW_NORMAL)

        cv2.createTrackbar( 'Brightness','Slider2',0,100,nothing)
        cv2.createTrackbar( 'Contrast','Slider2',50,100,nothing)
        cv2.createTrackbar( 'Flip','Slider2',slnt/2,slnt-2,nothing)
        cv2.createTrackbar( 'Zoom','Slider2',0,100,nothing)
        cv2.createTrackbar( 'Panx','Slider2',50,100,nothing)
        cv2.createTrackbar( 'Pany','Slider2',50,100,nothing)
        cv2.setMouseCallback("image", click_and_crop)

    while True:

        key = cv2.waitKey(100) & 0xFF
        if key != 255:
            print key

        if key == ord("c"):
                print 'completed'
                completed(imagename,dirpath_patient,dirroi)

        elif key == ord("d"):
                print 'delete entry'
                suppress()

        elif key == ord("l"):
                print 'delete last polygon'
                dellast()

        elif key == ord("a"):
                print 'delete all'
                delall()

        elif key == ord("r"):
                print 'reset'
                reseted()

        elif key == ord("v"):
                print 'visualize'
                visua()
        elif key == ord("e"):
                print 'erase'
                eraseroi(imagename,dirpath_patient)
        elif key == ord("f"):
                print 'close polygone'
                closepolygon()
        elif key == ord("q")  or quitl or cv2.waitKey(20) & 0xFF == 27 :
               print 'on quitte', quitl
               cv2.destroyAllWindows()
               break
        c = cv2.getTrackbarPos('Contrast','Slider2')
        l = cv2.getTrackbarPos('Brightness','Slider2')
        fl = cv2.getTrackbarPos('Flip','Slider2')
        z = cv2.getTrackbarPos('Zoom','Slider2')
        px = cv2.getTrackbarPos('Panx','Slider2')
        py = cv2.getTrackbarPos('Pany','Slider2')
#        print fl
        scannumber=fl+1
        imagename=list_image[scannumber]
        imagenamecomplet=os.path.join(pdirk,imagename)
        image = cv2.imread(imagenamecomplet,cv2.IMREAD_ANYDEPTH)
        image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        image=zoomfunction(image,z,px,py)
        imagesview=zoomfunction(images[scannumber],z,px,py)

#        initmenus(slnt,dirpath_patient,z)
        populate(dirpath_patient,slnt)
        imglumi=lumi(image,l)
        image=contrasti(imglumi,c)
#        print image.shape,images[scannumber].shape
        imageview=cv2.add(image,imagesview)
        imageview=cv2.add(imageview,menus)
        for key1 in classif:
                tabroifinalview=zoomfunction(tabroifinal[key1][scannumber],z,px,py)
                imageview=cv2.add(imageview,tabroifinalview)
        imageview=cv2.cvtColor(imageview,cv2.COLOR_BGR2RGB)
        cv2.imshow("image", imageview)

def tagviews (tab,t0,x0,y0,t1,x1,y1,t2,x2,y2,t3,x3,y3,t4,x4,y4,t5,x5,y5):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    col=white
    viseg=cv2.putText(tab,t0,(x0, y0), font,0.3,col,1)
    viseg=cv2.putText(viseg,t1,(x1, y1), font,0.3,col,1)
    viseg=cv2.putText(viseg,t2,(x2, y2), font,0.3,col,1)

    viseg=cv2.putText(viseg,t3,(x3, y3), font,0.3,col,1)
    viseg=cv2.putText(viseg,t4,(x4, y4), font,0.3,col,1)
    viseg=cv2.putText(viseg,t5,(x5, y5), font,0.3,col,1)
    return viseg



def genebmp(fn,nosource,dirroi):
    """generate patches from dicom files"""

    print ('load dicom files in :',fn)
    (top,tail) =os.path.split(fn)
    (top1,tail1) =os.path.split(top)

#    print 'fn' ,fn
    if not os.path.exists(fn):
        print 'path does not exists'

    fmbmpbmp=os.path.join(fn,scan_bmp)

    remove_folder(fmbmpbmp)
    os.mkdir(fmbmpbmp)
    if nosource:
        fn=top
     
    listdcm=[name for name in  os.listdir(fn) if name.lower().find('.dcm')>0]

    FilesDCM =(os.path.join(fn,listdcm[0]))
    RefDs = dicom.read_file(FilesDCM)
    dsr= RefDs.pixel_array
    dimtabx=dsr.shape[0]
    dimtaby=dsr.shape[1]
#    print dimtabx, dimtaby
    slnt=0
    for l in listdcm:

        FilesDCM =(os.path.join(fn,l))
        RefDs = dicom.read_file(FilesDCM)
        slicenumber=int(RefDs.InstanceNumber)
        if slicenumber> slnt:
            slnt=slicenumber

#    print 'number of slices', slnt
    slnt=slnt+1
    for l in listdcm:
#        print l
        FilesDCM =(os.path.join(fn,l))
        RefDs = dicom.read_file(FilesDCM)
        slicenumber=int(RefDs.InstanceNumber)

        dsr= RefDs.pixel_array
        dsr= dsr-dsr.min()
        c=float(imageDepth)/dsr.max()
        dsr=dsr*c
        dsr = dsr.astype('uint8')
        resized_image = cv2.resize(dsr, (dimtabx, dimtaby))

        endnumslice=l.find('.dcm')
        imgcoreScan=l[0:endnumslice]+'_'+str(slicenumber)+'.'+typei
        bmpfile=os.path.join(fmbmpbmp,imgcoreScan)
        roibmpfile=os.path.join(dirroi,imgcoreScan)

        t2='Prototype Not for medical use'
        t1='Pt: '+tail1
        t0='CONFIDENTIAL'
        t3='Scan: '+str(slicenumber)

        t4=time.asctime()
        t5=''
        resized_image=tagviews(resized_image,t0,dimtabx-80,dimtaby-10,t1,0,dimtaby-20,t2,dimtabx-250,dimtaby-10,
                     t3,0,dimtaby-30,t4,0,dimtaby-10,t5,0,dimtaby-20)
        cv2.imwrite(bmpfile,resized_image)
        cv2.imwrite(roibmpfile,resized_image)
#        cv2.imshow('dd',dsr)
    return slnt,dimtabx,dimtaby

def nothing(x):
    pass


def menudraw(slnt):
    global refPt, cropping,pattern,x0,y0,quitl,tabroi,tabroifinal,menus,images
    global posxdel,posydel,posxquit,posyquit,posxdellast,posydellast,posxdelall,posydelall
    global posxcomp,posycomp,posxreset,posyreset,posxvisua,posyvisua,posxeraseroi,posyeraseroi,posxlastp,posylastp
    posrc=0
    for key1 in classif:
        tabroi[key1]={}
        tabroifinal[key1]={}
        tabroinumber[key1]={}
        for sln in range(1,slnt):
            tabroinumber[key1][sln]=0
            tabroi[key1][sln]={}
            tabroi[key1][sln][0]=[]
            tabroifinal[key1][sln]= np.zeros((dimtabx,dimtaby,3), np.uint8)

        xr=5
        yr=15*posrc
        xrn=xr+10
        yrn=yr+10
        cv2.rectangle(menus, (xr, yr),(xrn,yrn), classifc[key1], -1)
        cv2.putText(menus,key1,(xr+15,yr+10),cv2.FONT_HERSHEY_PLAIN,0.7,classifc[key1],1 )
        posrc+=1

    posxdel=dimtabx-20
    posydel=15
    cv2.rectangle(menus, (posxdel,posydel),(posxdel+10,posydel+10), white, -1)
    cv2.putText(menus,'(d) del',(posxdel-40, posydel+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

    posxquit=dimtabx-20
    posyquit=dimtaby-50
    cv2.rectangle(menus, (posxquit,posyquit),(posxquit+10,posyquit+10), red, -1)
    cv2.putText(menus,'(q) quit',(posxquit-45, posyquit+10),cv2.FONT_HERSHEY_PLAIN,0.7,red,1 )

    posxdellast=dimtabx-20
    posydellast=30
    cv2.rectangle(menus, (posxdellast,posydellast),(posxdellast+10,posydellast+10), white, -1)
    cv2.putText(menus,'(l) del last',(posxdellast-68, posydellast+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

    posxdelall=dimtabx-20
    posydelall=45
    cv2.rectangle(menus, (posxdelall,posydelall),(posxdelall+10,posydelall+10), white, -1)
    cv2.putText(menus,'(e) del all',(posxdelall-57, posydelall+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

    posxcomp=dimtabx-20
    posycomp=75
    cv2.rectangle(menus, (posxcomp,posycomp),(posxcomp+10,posycomp+10), white, -1)
    cv2.putText(menus,'(c) completed',(posxcomp-85, posycomp+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

    posxreset=dimtabx-20
    posyreset=90
    cv2.rectangle(menus, (posxreset,posyreset),(posxreset+10,posyreset+10), white, -1)
    cv2.putText(menus,'(r) reset',(posxreset-55, posyreset+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

    posxvisua=dimtabx-20
    posyvisua=60
    cv2.rectangle(menus, (posxvisua,posyvisua),(posxvisua+10,posyvisua+10), white, -1)
    cv2.putText(menus,'(v) visua',(posxvisua-55, posyvisua+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

    posxeraseroi=dimtabx-20
    posyeraseroi=105
    cv2.rectangle(menus, (posxeraseroi,posyeraseroi),(posxeraseroi+10,posyeraseroi+10), white, -1)
    cv2.putText(menus,'(e) eraseroi',(posxeraseroi-75, posyeraseroi+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

    posxlastp=dimtabx-20
    posylastp=120
    cv2.rectangle(menus, (posxlastp,posylastp),(posxlastp+10,posylastp+10), white, -1)
    cv2.putText(menus,'(f) last p',(posxlastp-75, posylastp+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

#
#
def populate(pp,sl):
#    print 'populate'
    for key in classif:
        dirroi=os.path.join(pp,key)
#        print dirroi,sl
        if os.path.exists(dirroi):
                listroi =[name for name in  os.listdir(dirroi) if name.lower().find('.'+typei)>0]
                for roiimage in listroi:
#                    tabroi[key][sl]=genepoly(os.path.join(dirroi,roiimage))
                    img=os.path.join(dirroi,roiimage)
                    imageroi= cv2.imread(img,1)
#                    imageroi=zoomfunction(imageroi,z)
                    cdelimter='_'
                    extensionimage='.'+typei
                    slicenumber=rsliceNum(roiimage,cdelimter,extensionimage)
                    imageview=cv2.cvtColor(imageroi,cv2.COLOR_RGB2BGR)
                    tabroifinal[key][slicenumber]=imageview
#                    cv2.imshow('imageroi',imageroi)



def initmenus(slnt,dirpath_patient):
    global menus,imageview,zoneverticalgauche,zoneverticaldroite,zonehorizontal

    menus=np.zeros((dimtabx,dimtaby,3), np.uint8)
    for i in range(1,slnt):
        images[i]=np.zeros((dimtabx,dimtaby,3), np.uint8)
    imageview=np.zeros((dimtabx,dimtaby,3), np.uint8)
    menudraw(slnt)

    zoneverticalgauche=((0,0),(25,dimtaby))
    zonehorizontal=((0,0),(dimtabx,20))
    zoneverticaldroite=((dimtabx-25,0),(dimtabx,dimtaby))



def openfichierroi(patient,patient_path_complet):
    global dirpath_patient,dimtabx,dimtaby,dirroit
    dirpath_patient=os.path.join(patient_path_complet,patient)
    dirsource=os.path.join(dirpath_patient,source_name)
    dirroit=os.path.join(dirpath_patient,roi_name)
    if not os.path.exists(dirsource):
         os.mkdir(dirsource)
    if not os.path.exists(dirroit):
         os.mkdir(dirroit)
    listdcm=[name for name in os.listdir(dirsource) if name.find('.dcm')>0]
    nosource=True
    if len(listdcm)>0:
        nosource=False
    slnt,dimtabx,dimtaby=genebmp(dirsource,nosource,dirroit)
    dirsourcescan=os.path.join(dirsource,scan_bmp)
    initmenus(slnt,dirpath_patient)
    loop(slnt,dirsourcescan,dirpath_patient,dirroit)
    return 'completed'