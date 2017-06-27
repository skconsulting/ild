# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017

@author: sylvain
"""

#from __future__ import print_function
from param_pix import *

predict_source='predict_dum'
#predict_source='predict_new'
#predict_source='predict_1'



#pickel_train='pickle_lu_f'
pickel_train='pickle_S3'

thrproba=0.7
ldummy=True

attn=0.4 #attenuation color
if ldummy:
    print 'mode dummy'
cwd=os.getcwd()

(cwdtop,tail)=os.path.split(cwd)
namedirtopc=os.path.join(cwdtop,predict_source)

pickle_dir_train=os.path.join(cwdtop,pickel_train)

predict_result='predict_result'


def load_model_set(pickle_dir_train):
    print('Loading saved weights...')
    print('-'*30)
    listmodel=[name for name in os.listdir(pickle_dir_train) if name.find('weights')==0]
#    print listmodel

    ordlist=[]
    for name in listmodel:
        nfc=os.path.join(pickle_dir_train,name)
        nbs = os.path.getmtime(nfc)
#        print name,nbs,type(nbs)
        
#        posp=name.find('-')+1
#        post=name.find('.h')
#        numepoch=float(name[posp:post])
        tt=(name,nbs)
#        ordlist.append (tt)
        ordlist.append(tt)

    ordlistc=sorted(ordlist,key=lambda col:col[1],reverse=True)

    namelast=ordlistc[0][0]

    namelastc=os.path.join(pickle_dir_train,namelast)
    print 'last weights :',namelast

    return namelastc


def predictr(X_predict):

    print('-'*30)
    print('Predicting masks on test data...')

    imgs_mask_test = model.predict(X_predict, verbose=1,batch_size=1)
#   
#    print imgs_mask_test.shape
    return imgs_mask_test

def tagviewn(tab,label,x,y):
    """write text in image according to label and color"""
    col=classifc[label]
    font = cv2.FONT_HERSHEY_SIMPLEX
#    print col, label
    labnow=classif[label]

    deltax=130*((labnow)//10)
    deltay=11*((labnow)%10)
#    gro=-x*0.0027+1.2
    gro=0.3
#    print x+deltax,y+deltay,label
    viseg=cv2.putText(tab,label,(x+deltax, y+deltay+10), font,gro,col,1)
    return viseg


def visu(namedirtopcf,imgs_mask_test,num_list,dimtabx,dimtaby,tabscan,sroidir):
    print 'visu starts'
#    imclass = np.argmax(imgs_mask_test, axis=3)[0,:,:]
#    imclass1 = np.argmax(imgs_mask_test, axis=3)
#    print imclass.shape
#    print imclass.min(), imclass.max()
#    print imclass1.shape
#    print imclass1.min(), imclass1.max()
#    print imgs_mask_test.shape
    
#    o=normi(imclass)
#    masked_imclass = np.ma.masked_where(imclass == 0, imclass)
#    
#    cv2.imshow('imclass',o)
##            cv2.imshow('tabroif',f)
##            cv2.imwrite('a.bmp',o)
##            cv2.imwrite('b.bmp',roif)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    roip=False
    if os.path.exists(sroidir):
        listsroi=os.listdir(sroidir)
        roip=True
        tabroi={}
        for roil in listsroi:
            posu=roil.find('_')+1
            posp=roil.find('.')
            nums=int(roil[posu:posp])
            tabroi[nums]=os.path.join(sroidir,roil)
    
    imgs = np.zeros((imgs_mask_test.shape[0], dimtabx, dimtaby,3), dtype=np.uint8)
#    imgpat = np.zeros((dimtabx, dimtaby,3), dtype=np.uint8)
    
    for i in range (imgs_mask_test.shape[0]):
        patlist=[]
        imi=imgs_mask_test[i]
        imclass=np.argmax(imi, axis=2).astype(np.uint8)
        imamax=np.amax(imi, axis=2)
        np.putmask(imamax,imamax>=thrproba,255)
        np.putmask(imamax,imamax<thrproba,0)
        imamax=imamax.astype(np.uint8)
#        print type(imamax[0][0])
        imclass=np.bitwise_and(imamax,imclass)
#        plt.hist(imamax.flatten(), bins=80, color='c')
#        plt.xlabel("proba imi")
#        plt.ylabel("Frequency")
#        plt.show()

#        print 'im1',imclass[200][200],imclass.min(),imclass.max(), type(imclass[0][0]), np.unique(imclass)
        imclassc = np.expand_dims(imclass,2)
#        print 'im2',imclassc[200][200],imclassc.min(),imclassc.max(),type(imclassc[0][0][0]), np.unique(imclassc)
        imclassc=np.repeat(imclassc,3,axis=2) 
#        print 'im3',imclassc[200][200],imclassc.min(),imclassc.max(),type(imclassc[0][0][0]), np.unique(imclassc)

        
#        imclassc= cv2.cvtColor(imclass,cv2.COLOR_GRAY2BGR)
#        imclassc=np.repeat(imclass,3,axis=2) 
        

        for key,value in classif.items():
            
            if key not in classifnotvisu:
                imcc=imclassc.copy()
                bl=(value,value,value)
                blc=[]
                zz=classifc[key]
#                print zz
                for z in range(3):
                    blc.append(int(zz[z]))
#                print imcc[200][200][z]*0.5)
#                print blc
        
#                print key,bl
#                print imcc[200][200]
                np.putmask(imcc,imcc!=bl,black)
#                print imcc[200][200]
    #        print 'im4',imclassc[200][200],imclassc.min(),imclassc.max(),type(imclassc[0][0][0]), np.unique(imclassc)
                np.putmask(imcc,imcc==bl,blc)
#                print imcc[200][200]
                if imcc.max()>0:
                    patlist.append(key)               
                imgs[i]=cv2.add(imgs[i],imcc)
                for p in patlist:
                    delx=int(dimtaby*0.6-120)
                    imgs[i]=tagviewn(imgs[i],p,delx,0)
#                print imgs[i][200][200]
#        print 'im5',imclassc[200][200],imclassc.min(),imclassc.max(),type(imclassc[0][0][0]), np.unique(imclassc)
        
        
#        np.putmask(imclass,imclass==2,100)
        
#        print 'im3',imclassc.min(),imclassc.max()

#        masked_imclass=np.where(imclass==2)
#        print masked_imclass.min(),masked_imclass.max()
#        print np.unique(masked_imclass)
#        masked_imclass = np.ma.masked_where(imclass == 0, imclass)
#        masked_imclass = np.ma.masked_where(imclass == 1, imclass)
#        print imclas[][200]
#        print masked_imclass.min(),masked_imclass.max()
        
#        imi1=np.moveaxis(imi,0,2)
#        imi=imi.reshape(imi.shape[1],imi.shape[2],imi.shape[0])
#        print imi[0][0]
#        print imi1[0][0]
#        ooo
#        patlist=[]
#        for j in range (0,dimtabx):
#            for k in range(dimtaby):
#                proba=imi[j][k]
##                print proba
##                print imi[j][k]
#
#                numpat=argmax(proba)
#                pr=amax(proba)
##                print  numpat
#          
##                ooo
##                print numpat
#                pat=fidclass(numpat,classif)
##                print pat
#                patlistf.append(pat)
#                if pat not in classifnotvisu:
#                    if pat not in patlist:
#                        patlist.append(pat)
#                    if pr> thrproba:
#                        colnew=classifc[pat]
#                        colfin=[]
#                        for z in range(3):
#                            colfin.append(int(colnew[z]*0.2))
#                        imgs[i][j][k]=colfin
#                
#        for p in patlist:
#            delx=int(dimtaby*0.6-120)
#            imgs[i]=tagviewn(imgs[i],p,delx,0)

        uniquelbls = np.unique(patlist)
        print 'list of patterns',uniquelbls
        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)
        pred_dir = 'preds'
        pred_dir=os.path.join(namedirtopcf,predict_result)
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
    print num_list
    for image, image_id in zip(imgs, num_list):
        if roip:
            imgn=cv2.imread(tabroi[image_id])
            imgnc=cv2.resize(imgn,(dimtabx, dimtaby),interpolation=cv2.INTER_LINEAR)
            imgnc= cv2.cvtColor(imgnc,cv2.COLOR_RGB2BGR)
        else:
            print 'no roi'
            imgn=normi(tabscan[image_id])
            imgnc= cv2.cvtColor(imgn,cv2.COLOR_GRAY2BGR)
            imgnc=(imgnc*attn).astype(np.uint8)
#            cv2.imshow('tabscan[image_id]',tabscan[image_id])
#            cv2.imshow('imgn',imgn)
#            cv2.imshow('imgnc',imgnc)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            
        image2=cv2.add(image,imgnc)
#        image2=image
        plt.figure(figsize = (10, 7))
        plt.subplot(1,3,1)
        plt.title('source')
        plt.imshow(imgnc)
        
        plt.subplot(1,3,2)
        plt.title('predict')
        plt.imshow( image )
        
        plt.subplot(1,3,3)
        plt.title('source+predict')
        plt.imshow( image2 )
        plt.imshow( imclassc, alpha=0.5 )
        imsave(os.path.join(pred_dir, str(image_id) + '.bmp'), image2)


def tagviews(tab,text,x,y):
    """write simple text in image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    viseg=cv2.putText(tab,text,(x, y), font,0.3,white,1)
    return viseg

def genebmp(dirName,fileList,slnt,hug):
    """generate patches from dicom files and sroi"""
    print ('generate  bmp files from dicom files in :',f)
    sliceok=[]
    (top,tail)=os.path.split(dirName)
#    global constPixelSpacing, dimtabx,dimtaby
    #directory for patches
    bmp_dir = os.path.join(dirName, bmpname)
    if os.path.exists(bmp_dir):

        shutil.rmtree(bmp_dir)

    os.mkdir(bmp_dir)
    if hug:
        lung_dir = os.path.join(dirName, lungmask)
        if not os.path.exists(lung_dir):
            lung_dir = os.path.join(dirName, lungmask1)
        
    else:
        (top,tail)=os.path.split(dirName)
        lung_dir = os.path.join(top, lungmask)
        if not os.path.exists(lung_dir):
            lung_dir = os.path.join(top, lungmask1)
        
    lung_bmp_dir = os.path.join(lung_dir,lungmaskbmp)
    remove_folder(lung_bmp_dir)
    os.mkdir(lung_bmp_dir)
    
    lunglist = [name for name in os.listdir(lung_dir) if ".dcm" in name.lower()]

    tabscan=np.zeros((slnt,image_rows,image_cols),np.int16)
    tabslung=np.zeros((slnt,image_rows,image_cols),np.uint8)
#    os.listdir(lung_dir)
    for filename in fileList:
#            print(filename)
            FilesDCM =(os.path.join(dirName,filename))
            RefDs = dicom.read_file(FilesDCM)
            dsr= RefDs.pixel_array
            dsr=dsr.astype('int16')

            scanNumber=int(RefDs.InstanceNumber)
            if scanNumber not in sliceok:
                sliceok.append(scanNumber)
            endnumslice=filename.find('.dcm')
            imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)+'.'
#            bmpfile=os.path.join(dirFilePbmp,imgcore)
            dsr[dsr == -2000] = 0
            intercept = RefDs.RescaleIntercept
#            print intercept
            slope = RefDs.RescaleSlope
            if slope != 1:
                dsr = slope * dsr.astype(np.float64)
                dsr = dsr.astype(np.int16)

            dsr += np.int16(intercept)
            dsr = dsr.astype('int16')

            dsr=cv2.resize(dsr,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)

            dsrforimage=normi(dsr)
#            dsrforimage = bytescale(dsr, low=0, high=255)
            

            tabscan[scanNumber]=dsr

            imgcored=imgcoredeb+typei
            bmpfiled=os.path.join(bmp_dir,imgcored)
            textw='n: '+tail+' scan: '+str(scanNumber)
#            print imgcoresroi,bmpfileroi
            dsrforimage=tagviews(dsrforimage,textw,0,20)
#            dsrforimage=cv2.cvtColor(dsrforimage,cv2.COLOR_GRAY2BGR)
            cv2.imwrite (bmpfiled, dsrforimage,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
            

           


    for lungfile in lunglist:
#             print(lungfile)
#             if ".dcm" in lungfile.lower():  # check whether the file's DICOM
                FilesDCM =(os.path.join(lung_dir,lungfile))
                RefDs = dicom.read_file(FilesDCM)
                dsr= RefDs.pixel_array
                dsr=dsr.astype('int16')
#                fxs=float(RefDs.PixelSpacing[0])/avgPixelSpacing
#                print 'fxs',fxs
                scanNumber=int(RefDs.InstanceNumber)
                if scanNumber in sliceok:
                    endnumslice=filename.find('.dcm')
                    imgcoredeb=filename[0:endnumslice]+'_'+str(scanNumber)+'.'
                    imgcore=imgcoredeb+typei
                    bmpfile=os.path.join(lung_bmp_dir,imgcore)
                    dsr=normi(dsr)
                    dsrresize=cv2.resize(dsr,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)
    
                    cv2.imwrite (bmpfile, dsrresize)
    #                np.putmask(dsrresize,dsrresize==1,0)
                    np.putmask(dsrresize,dsrresize>0,1)
                    tabslung[scanNumber]=dsrresize

    return tabscan,tabslung


def preparscan(tabscan,tabslung):
#    (top,tail)=os.path.split(namedirtopcf)
    scan_list=[]
    num_list=[]
    for num in numsliceok:
        tabl=tabslung[num].copy()
        scan=tabscan[num].copy()
#        print scan.min(),scan.max()
        
        tablc=tabl.astype(np.int16)
        taba=cv2.bitwise_and(scan,scan,mask=tabl)
        np.putmask(tablc,tablc==0,-1000)
        np.putmask(tablc,tablc==1,0)
        tabab=cv2.bitwise_or(taba,tablc) 
        tababn=norm(tabab)
        
#        cv2.imshow('tababn',normi(tababn))
###            cv2.imshow('tabroif',f)
###            cv2.imwrite('a.bmp',o)
###            cv2.imwrite('b.bmp',roif)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
#    
        scan_list.append(tababn)
#        print tabab.min(),tabab.max()
        num_list.append(num)
    X_train = np.asarray(np.expand_dims(scan_list,3)) 
    if num_bit ==3:
     X_train=np.repeat(X_train,3,axis=3)     
   
    return X_train,num_list


def preparscandummy(namedirtopcf):
    print 'pre dummy',namedirtopcf
    
    listimage =[name for name in os.listdir(namedirtopcf) if name.find('.bmp')>0]
#    (top,tail) =os.split(namedirtopcf)
    print listimage
    tabscan=np.zeros((100,image_rows,image_cols),np.uint8)
    scan_list=[]
    num_list=[]
    for i in listimage:
        imgf=os.path.join(namedirtopcf,i)
        if num_bit==3:
            img=cv2.imread(imgf)
        else:
            img=cv2.imread(imgf,0)
            
        scan_list.append(img)
        print img.shape, img.min(), img.max()
        
        num=rsliceNum(imgf,'_','.bmp')
#        print tabab.min(),tabab.max()
#        print num
        num_list.append(num)
        tabscan[num]=img
#        print img.shape, img.min(), img.max()
#        print  tabscan[num].shape,  tabscan[num].min(),  tabscan[num].max()
#        
#        print num
#        cv2.imshow('tabscan[num]',normi(tabscan[num]))
#        cv2.imshow('img',normi(img))
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    X_train = np.asarray(scan_list) 
    print X_train.shape
    X_train=preprocess_batch(X_train)
    if num_bit==1:
        X_train = np.expand_dims(X_train,3) 
#    print X_train.shape
    return X_train,num_list,tabscan




listdirc= (os.listdir(namedirtopc))


def load_train_data(numpidir):

    class_weights=pickle.load(open( os.path.join(numpidir,"class_weights.pkl"), "rb" ))
    clas_weigh_l=[]
    num_class=len(class_weights)
    print 'number of classes :',num_class
    for i in range (0,num_class):
#            print i,class_weights[i]
            clas_weigh_l.append(class_weights[i])
    print 'weights for classes:'
    for i in range (0,num_class):
                    print i, clas_weigh_l[i]
    class_weights_r=np.array(clas_weigh_l)
#    numpatl={}
#    for pat in usedclassif:
#        numpatl[pat]=0
##        print fidclass(classif[pat],classif)
##        print pat, numpatl[pat]
#    y_test1=np.moveaxis(y_test,1,3)
#    for i in range (y_test1.shape[0]):
#        for j in range (0,y_test1.shape[1]):
#            for k in range(y_test1.shape[2]):
#                proba=y_test1[i][j][k]
#                numpat=argmax(proba)     
#                pat=fidclass(numpat,classif)
#                numpatl[pat]+=1
#     
#    for pat in usedclassif:
#        print pat,numpatl[pat]
#    tot=numpatl['back_ground']*1.0
#    for pat in usedclassif:
#        print pat,tot/numpatl[pat]
#        

   
  
    return class_weights_r,num_class

def loadmodel():

    weights,num_class=load_train_data(pickle_dir_train)
 
#    model = get_unet(num_class,image_rows,image_cols)
#    model = get_model(num_class,image_rows,image_cols,weights)
    model = get_model(num_class,num_bit,image_rows,image_cols,False,weights)
#    mloss = weighted_categorical_crossentropy(weights).myloss
#    model.compile(optimizer=Adam(lr=1e-5), loss=mloss, metrics=['categorical_accuracy'])
#    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    namelastc=load_model_set(pickle_dir_train)
    model.load_weights(namelastc)
    return model
#model =load_model_set(pickle_dir_train)

def genepara(fileList,namedir):
    fileList =[name for name in  os.listdir(namedir) if ".dcm" in name.lower()]
    slnt=0
    for filename in fileList:
        FilesDCM =(os.path.join(namedir,filename))
        RefDs = dicom.read_file(FilesDCM)
        scanNumber=int(RefDs.InstanceNumber)
        if scanNumber not in numsliceok:

                numsliceok.append(scanNumber)
        if scanNumber>slnt:
            slnt=scanNumber
    print 'number of slices', slnt
    slnt=slnt+1
    return slnt



model=loadmodel()

def fullprint(*args, **kwargs):
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold='nan')
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)

print('work in directory:',namedirtopc)
for f in listdirc:
    print '-----------'
    print('work on:',f)
    print '-----------'
    numsliceok=[]
    namedirtopcf=os.path.join(namedirtopc,f)
    sroidir=os.path.join(namedirtopcf,sroi)
    contenudir = [name for name in os.listdir(namedirtopcf) if name.find('.dcm')>0]
    
    if not ldummy:
        if len(contenudir)>0:
            slnt = genepara(contenudir,namedirtopcf)
            tabscan,tabslung=genebmp(namedirtopcf,contenudir,slnt,True)
        else:
              namedirtopcfs=os.path.join(namedirtopcf,sourcedcm)
              contenudir = [name for name in os.listdir(namedirtopcfs) if name.find('.dcm')>0]
              slnt = genepara(contenudir,namedirtopcfs)
              tabscan,tabslung=genebmp(namedirtopcfs,contenudir,slnt,False)
        X_predict,num_list=preparscan(tabscan,tabslung)       
    else:
         X_predict,num_list,tabscan=preparscandummy(namedirtopcf) 
    print 'Xpredict :', X_predict.shape
    print 'Xpredict min max :', X_predict.min(),X_predict.max()
#    
    imgs_mask_test=predictr(X_predict)
#    imgs_mask_test=np.flip(imgs_mask_test,3)
    print 'imgs_mask_test shape :', imgs_mask_test.shape
    print 'imgs_mask_test[0][100][100]',imgs_mask_test[0][100][100]
    print 'X_predict[0][100][100]',X_predict[0][100][100]
    print 'imgs_mask_test[0][0][200]',imgs_mask_test[0][0][200]
    
#    print 'X_predict shape :', X_predict.shape
#    print 'X_predict[0][50][200]',X_predict[0][50][0]
#    print 'X_predict[0][0][200]',X_predict[0][0][200]
    
    
    imamax=np.amax(imgs_mask_test[0], axis=2)
#    print 'imamax min max',imamax.min(), imamax.max(),imamax[100][200]
    plt.hist(imamax.flatten(), bins=80, color='c')
    plt.xlabel("proba")
    plt.ylabel("Frequency")
    plt.show()
#    np.putmask(imamax,imamax>0.9,1)
#    np.putmask(imamax,imamax<=0.9,0)
#    print 'imamax[50][0]',imamax[50][0]
#    print 'imamax[0][200]',imamax[0][200]
    
#    np.putmask(imamax,imamax<=0.7,0)
#    np.putmask(imamax,imamax>0.7,1)

#    imclass = np.argmax(imgs_mask_test[0], axis=2)
#    print 'imclass[100][200]',imclass[100][200]
#    print 'imclass[0][0]',imclass[0][0]
#    cv2.imwrite('a.bmp',normi(imclass))
#    cv2.imwrite('b.bmp',imclass*100)
#    cv2.imwrite('c.bmp',normi(imamax))
#    cv2.imwrite('d.bmp',imamax)
#    cv2.imwrite('e.bmp',X_predict[0])
#    print 'imclass shape :',imclass.shape    
#    plt.figure(figsize = (10, 5))
#    plt.subplot(1,2,1)
#    plt.title('imclass')
#    plt.imshow( imclass )
#    plt.subplot(1,2,2)
#    plt.title('image')
#    plt.imshow( np.asarray(crpim) )
#    masked_imclass = np.ma.masked_where(imclass == 0, imclass)
#    plt.imshow( X_predict[0][:,:,0] )
#    print X_predict[0][:,:,0].shape
   
    visu(namedirtopcf,imgs_mask_test,num_list,image_cols,image_rows,tabscan,sroidir)
    
    