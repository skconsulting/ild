# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017
training of CNN using a generator
@author: sylvain
"""

from param_pix import cwdtop,num_bit,image_rows,image_cols,hugeClass,classif
from param_pix import get_model,evaluate,generandom,geneaug
#from param_pix import normi


import cPickle as pickle
import datetime
#import matplotlib.pyplot as plt
import numpy as np
import os
import random
#import cv2
import sys


from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger,EarlyStopping
from keras.utils import np_utils
import keras
print ' keras.backend.image_data_format :',keras.backend.image_data_format()

####################################################################################
#smallvalue=['HC','reticulation','bronchiectasis','GGpret']
nameHug='IMAGEDIR'

toppatch= 'TOPROI' #for scan classified ROI
#extendir='T_2'  #for scan classified ROI
extendir='0'  #for scan classified ROI

nametop='TRAIN_SET'
pickel_top='pickle' #path to get input data
pickel_ext='train_set'  #path to get input data
pickel_ext_set='unet0'  #path to get input data


numpidirvalrefend='pickle_train_val_0'

trainSetSize=5500# default 4785 for chu+HUG,  , number of images for each epoch, multiple of batch_size
nb_epoch=30 # default 35 number of epoch for each set
batch_size= 6#default 4 batch size (gpu ram dependant) 8 for sk3, 6 for unet, 4 for sk4
turnNumber=1 #22 number of turn of trainset size 216
numgen=-1  #to restart after crash -1 means start new


#all in percent
maxshiftv=0
maxshifth=0
maxrot=7 #7
maxresize=0
maxscaleint=0
maxmultint=20

notToAug=['']

#calculate=True #to be put to False for actual training

calculate=False #to be put to False for actual training

#################################################################################


roipicklepath = 'roipicklepatches'
patchtoppath=os.path.join(cwdtop,nameHug)
patchtoppath=os.path.join(patchtoppath,toppatch+'_'+extendir)

picklepathdir =os.path.join(patchtoppath,roipicklepath)
print ' path for data input :',picklepathdir


def caltime(i):
    if i<60:
        return str(i)+'s'
    elif i<3600:
        nbmin=i/60
        nbsecond=i-(nbmin*60)
        return str(nbmin)+'m '+str(nbsecond)+'s'
    elif i<24*3600:
        nbheure=i/3600
        nbmin=(i-(nbheure*3600))/60
        nbsecond=i-(nbheure*3600)-nbmin*60
        return str(nbheure)+'h ' +str(nbmin)+'m '+str(nbsecond)+'s'
    else:
        nbdays=i/(24*3600)
        nbheure=(i -(nbdays*24*3600))/3600
        nbmin=(i-(nbdays*24*3600)-(nbheure*3600))/60
        nbsecond=i-(nbdays*24*3600)-(nbheure*3600)-(nbmin*60)
        return str(nbdays)+'d '+str(nbheure)+'h ' +str(nbmin)+'m '+str(nbsecond)+'s'
    
toe=2.03
toe=113 #sk4
print 'number of numgen for 1 turn:',1.0*trainSetSize/batch_size
print 'evaluation time:'
print 'time for one epoch :',str(toe)+'s'
#print 'time for one set:',caltime(int(toe*nb_epoch))
print 'time for one turn :',caltime(int(toe*nb_epoch))
print 'total time for ',turnNumber,' turns :',caltime(int(toe*nb_epoch*turnNumber))

##############################################################
validationdir='V' #validation set directory
numpidirvalref=os.path.join(cwdtop,nametop)
numpidirvalref=os.path.join(numpidirvalref,numpidirvalrefend)
numpidirvalref=os.path.join(numpidirvalref,validationdir)
print 'validation path: ', numpidirvalref

pickle_store_ext= 'pickle' #path to store csv with train status
#pickel_dirsource='pickle_ILD_TXT'


pickel_dirsource=pickel_top+'_'+pickel_ext+'_'+pickel_ext_set
#pickel_dirsource='pickle_UIP'

t = datetime.datetime.now()
today = 'd_'+str(t.month)+'-'+str(t.day)+'-'+str(t.year)+'_'+str(t.hour)+'_'+str(t.minute)

pickel_train=pickel_dirsource
 

pickle_dir=os.path.join(cwdtop,nametop)
pickle_dir=os.path.join(pickle_dir,pickel_dirsource)
pickle_store= os.path.join(pickle_dir,pickle_store_ext)

print 'source of val data input',pickle_dir
print 'directory for result',pickle_store


if not os.path.exists(pickle_dir):
    os.mkdir(pickle_dir)
if not os.path.exists(pickle_store):
    os.mkdir(pickle_store)

pickle_dir_train=os.path.join(cwdtop,pickel_train)

def load_train_val(numpidir):

    X_test = pickle.load( open( os.path.join(numpidir,"X_test.pkl"), "rb" ))
    y_test = pickle.load( open( os.path.join(numpidir,"Y_test.pkl"), "rb" ))
    DIM_ORDERING=keras.backend.image_data_format()
    if DIM_ORDERING == 'channels_first':
          num_class= y_test.shape[1]
          img_rows=X_test.shape[2]
          img_cols=X_test.shape[3]
          num_images=X_test.shape[0]
    else:
        num_class= y_test.shape[3]
        img_rows=X_test.shape[1]
        img_cols=X_test.shape[2]
        num_images=X_test.shape[0]

#    num_class= 4
#    img_rows=X_test.shape[2]
#    img_cols=X_test.shape[3]
#    num_images=X_test.shape[0]
#    for key,value in class_weights.items():
#        print key, value
#    print X_test.min(),X_test.max()
#    print y_test.min(),y_test.max()
#    print 'loadtrainval',y_test.shape
    
    return X_test, y_test,num_class,img_rows,img_cols,num_images
        


def load_weight(numpidir):
    class_weights=pickle.load(open( os.path.join(numpidir,"class_weights.pkl"), "rb" ))
    num_class= len(class_weights)

    print 'number of classes :',num_class
  
    clas_weigh_l=[]
    for i in range (0,num_class):
#            print i,class_weights[i]
            clas_weigh_l.append(class_weights[i])
    print 'weights for classes:'
    for i in range (0,num_class):
                    print i, clas_weigh_l[i]
    class_weights_r=np.array(clas_weigh_l)

    return num_class,class_weights_r,class_weights


def store_model(model,it):
    name_model=os.path.join(pickle_store,'weights_'+str(today)+'_'+str(it)+'model.hdf5')
    model.save_weights(name_model)

def load_model_set(pickle_dir_train,filew):
    listmodel=[name for name in os.listdir(pickle_dir_train) if name.find('weights')==0]
    print 'load_model',pickle_dir_train
    ordlist=[]
    for name in listmodel:
        nfc=os.path.join(pickle_dir_train,name)
        nbs = os.path.getmtime(nfc)
        tt=(name,nbs)
        ordlist.append(tt)
    ordlistc=sorted(ordlist,key=lambda col:col[1],reverse=True)
    namelast=ordlistc[0][0]
    namelastc=os.path.join(pickle_dir_train,namelast)
    print 'last weights :',namelast   
    filew.write ('last weights :'+namelast+'\n')
    return namelastc


def  readclasses(pat,namepat,indexpat,scaleint,multint,rotimg,resiz,shiftv,shifth):
# scan,mask=readclasses(pat,listscaninroi[pat],indexpat,scaleint,rotimg,resiz,shiftv,shifth)  
    patpick=os.path.join(picklepathdir,pat)
    patpick=os.path.join(patpick,namepat[indexpat])
    readpkl=pickle.load(open(patpick, "rb"))
                          
    scanr=readpkl[0]
    scan=geneaug(scanr,scaleint,multint,rotimg,resiz,shiftv,shifth,False)

    maskr=readpkl[1]
    mask=geneaug(maskr,0,0,rotimg,resiz,shiftv,shifth,True)

    minr=maskr.min()
    manr=maskr.max()
    minm=mask.min()
    manm=mask.max()
    mansr=scanr.max()
    mansm=scan.max()
    if manm==0 or manr==0:
        print 'error label nul'
        sys.exit()
    if mansr==0 or mansm==0:
        print 'error scan nul'
        sys.exit()
    if minr != minm or manr!=manm or manr==0 or manm==0:
        print 'error label'
        print minr,minm,manr,manm,pat ,indexpat
#        cv2.imwrite('images.bmp',normi(scanr))
#        cv2.imwrite('imaget.bmp',normi(scan))
#        cv2.imwrite('labels.bmp',normi( maskr))
#        cv2.imwrite('labelt.bmp',normi( mask))
        sys.exit()
#    mask=geneaug(maskr,0,rotimg,resiz,shiftv,shifth,False)
    
    return scan, mask  

def readclasses2(num_classes,X_testi,y_testi):
#    print 'start readclasses2'
    DIM_ORDERING=keras.backend.image_data_format()
    y_test = np.array(y_testi)
   
    lytest=y_test.shape[0]
    ytestr=np.zeros((lytest,image_rows, image_cols,int(num_classes)),np.uint8)
#    print y_test.shape
    for i in range (lytest):
        for j in range (0,image_rows):
#            print i,j,y_test[i][j]
            ytestr[i][j] = np_utils.to_categorical(y_test[i][j], num_classes)
            
            
    if DIM_ORDERING == 'channels_first':
            X_test = np.asarray(np.expand_dims(X_testi,1)) 
            ytestr=np.moveaxis(ytestr,-1,1)
    else:
            X_test = np.asarray(np.expand_dims(X_testi,3))  
  
#    print 'rdclass2',X_test.shape
#    print 'rdclass2',ytestr.shape
    return  X_test,  ytestr  


def gen_random_image(numclass,listroi,listscaninroi,indexpatc):
    global classnumber,numgen
    numgen+=1
#    numgen=numgen%(maximage*numclass)
#    print 'numgen',numgen
    pat =listroi[numgen%numclass]
    numberscan=classnumber[pat]
    if  pat in hugeClass:
        indexpat =  random.randint(0, numberscan-1)                       
    else:                                                   
        indexpat =  indexpatc[pat]%numberscan

    indexpatc[pat]=indexpat+1
    if pat in notToAug:
        keepaenh=0
    else:
        keepaenh=1
    scaleint,multint,rotimg,resiz,shiftv,shifth=generandom(maxscaleint,
                            maxmultint,maxrot,maxresize,maxshiftv,maxshifth,keepaenh)
    scan,mask=readclasses(pat,listscaninroi[pat],indexpat,scaleint,multint,rotimg,resiz,shiftv,shifth) 
#    scaleint,rotimg,resiz,shiftv,shifth=generandom(keepaenh)

    
#    print  pat,scaleint,rotimg,resiz,shiftv,shifth

#    indexaug = random.randint(0, 7)
#    print '\n'
#    print numgen ,pat,indexpat,numberscan
#    scan,mask=readclasses(pat,listscaninroi[pat],indexpat,scaleint,rotimg,resiz,shiftv,shifth)  
    return scan,mask,indexpatc
    #        label_list.append(mask)


    
def batch_generator(batch_size,numclass,num_classes,listroi,listscaninroi,indexpatc):
    while True:
        image_list = []
        mask_list = []
        for i in range(batch_size):
#            print indexpatc['ground_glass']
            img, mask,indexpatc = gen_random_image(numclass,listroi,listscaninroi,indexpatc)
            image_list.append(img)
            mask_list.append(mask)
            
        X_test,  ytestr  =readclasses2(num_classes,image_list,mask_list)
#        print X_test.shape
#        print ytestr.shape
        yield  X_test,  ytestr          

def train():
    global classnumber,numgen
    print('-'*30)
#    print('Loading train data...')
#    print('-'*30)
    tn = datetime.datetime.now()
    todayf ='f'+'_'+str(tn.month)+'_'+str(tn.day)+'_'+str(tn.year)+'_'+str(tn.hour)+'_'+str(tn.minute)+'stat.txt'
    todayf=os.path.join(pickle_store,todayf)
    filew=open (todayf,'w')

    todayn = 'month:'+str(tn.month)+'-day:'+str(tn.day)+'-year:'+str(tn.year)+' - '+str(tn.hour)+'h'+str(tn.minute)+'m'+'\n'
    filew.write('start of training :'+todayn)

#    listplickle=[name for name in os.walk(pickle_dir).next()[1] if name != validationdir]
#    lnumpi=len(listplickle)
    
    num_class,weights,class_weights = load_weight(pickle_dir)

    print('Creating and compiling model...')
    print('-'*30)
    weightedloss=True
#    model = get_model(num_class,num_bit,image_rows,image_cols,False,weights,weightedloss)    
#    sys.exit()
    numpidirval =os.path.join(pickle_dir,validationdir)
    x_val, y_val, num_class,img_rows,img_cols,num_images = load_train_val(numpidirval)
#    print img_rows
#    print image_rows
#    print img_cols
#    print image_cols
    assert img_rows==image_rows,"dimension mismatch"
    assert img_cols==image_cols,"dimension mismatch"
    
#    listmodel=[name for name in os.listdir(pickle_store) if name.find('weights')==0]
#    if len(listmodel)>0:
#
#         print 'load weight found from last training'
#         filew.write ('load weight found from last training\n')
#         namelastc=load_model_set(pickle_store,filew)         
#         model.load_weights(namelastc)  
#    else:
#         print 'first training to be run'
#         filew.write ('first training to be run\n')
#    
    today = str('m'+str(t.month)+'_d'+str(t.day)+'_y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m')
    rese=os.path.join(pickle_store,str(today)+'_e.csv')
    listroi=[name for name in os.listdir(picklepathdir)]

    numclass=len(listroi)
    print '-----------'
    print'number of classes in scan:', numclass
    print '-----------'
    print 'list of classes:'
    for i in listroi:
        print i

#    assert numclass+2==num_class,"dimension mismatch"
    listscaninroi={}
    classnumber={}
    for pat in classif:
        classnumber[pat]=0

    indexpatc={}
    for j in listroi:
        indexpatc[j]=0

    totalimages=0
    maximage=0

    for pat in listroi:   
        listscaninroi[pat]=os.listdir(os.path.join(picklepathdir,pat))
        classnumber[pat]=len(listscaninroi[pat])
#        print pat,classnumber[pat]
        if pat not in hugeClass:
            if classnumber[pat]>maximage:
                maximage=classnumber[pat]
                ptmax=pat
        totalimages=totalimages+classnumber[pat]
        
    print '-----------'    
    for pat in classif:
        print 'number of data in:',pat, ' : ',classnumber[pat]
    for pat in hugeClass:
        print 'number of data in hugeClass:',pat, classnumber[pat]
  
    print '-----------------------'

    print 'x_val min max :',x_val.min(),x_val.max()
    print 'shape x_val :',x_val.shape
    print 'shape y_val :',y_val.shape
    print 'image number of rows :',img_rows
    print 'image number of columns :',img_cols
    print 'image width:', num_bit  
    print ( 'number of numgen for 1 turn:',str(1.0*trainSetSize/batch_size) )
    print 'number of validation images:', num_images
    print 'maximum data in one pat:',maximage,' in ',ptmax
    print 'number of classes:', num_class
    print 'starting number :'+str(numgen+1)
    print( 'equivalent number of images from starting number :'+str((numgen+1)/(trainSetSize*batch_size)))
    print 'number of turns:',  turnNumber 
    print 'size of train set:',trainSetSize
    print 'batch_size  :',batch_size
    print 'total number of images for training:',turnNumber*trainSetSize 
    print 'number total of scan images:',totalimages
    print 'comparison with total number of images: :',1.0*(turnNumber*trainSetSize) /totalimages
    print 'number of epoch per subset:',nb_epoch
       
    print('-'*30)
    filew.write ( ' path for data input :'+picklepathdir+'\n')
    filew.write (  'directory for result'+pickle_store+'\n')
    filew.write ('x_val min max :'+str(x_val.min())+' '+str(x_val.max())+'\n')
    filew.write ('shape x_val :'+str(x_val.shape)+'\n')
    filew.write ('shape y_val :'+str(y_val.shape)+'\n')
    filew.write ('image number of rows :'+str(img_rows)+'\n')
    filew.write ('image number of columns :'+str(img_cols)+'\n')
    filew.write ( 'image width:'+ str(num_bit) +'\n')
    filew.write ( 'number of numgen for 1 turn:'+str(1.0*trainSetSize/batch_size)+'\n')
    filew.write ( 'number of validation images:'+ str(num_images)+'\n')
    filew.write ( 'maximum data in one pat:'+str(maximage)+' in '+ptmax+'\n')
    filew.write ('number of classes:'+ str(num_class)+'\n')
    filew.write ( 'starting number :'+str(numgen+1)+'\n')
    filew.write ( 'equivalent number of images from starting number :'+str((numgen+1)/(trainSetSize*batch_size))+'\n')

    filew.write ('number of turns:'+str(  turnNumber)+ '\n')
    filew.write ( 'size of train set:'+str(trainSetSize)+'\n')
    filew.write ( 'batch_size  :'+str(batch_size)+'\n')
    filew.write (  'total number of images for training:'+str(turnNumber*trainSetSize)  +'\n') 
    filew.write ( 'number total of scan images:'+str(totalimages)+'\n')
    filew.write (  'comparison with total number of images: :'+
                 str(1.0*(turnNumber*trainSetSize) /totalimages) +'\n') 
    filew.write ( 'number of epoch per subset :'+str(nb_epoch)+'\n')

    filew.write('------------------------------------------\n')
    filew.close()
    if calculate:
        return
    
    
    early_stopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1,min_delta=0.01,mode='auto')      
#    early_stopping=EarlyStopping(monitor='val_fn', patience=10, verbose=1,min_delta=0.01,mode='auto')                                     
                               
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                                              patience=5, min_lr=1e-6,verbose=1,mode='auto')#init 5
    model_checkpoint = ModelCheckpoint(os.path.join(pickle_store,'weights_'+today+'.{epoch:02d}-{val_loss:.3f}.hdf5'), 
                                monitor='val_loss', save_best_only=True,save_weights_only=True,mode='auto') 
# to be used with custom metrics
#    model_checkpoint = ModelCheckpoint(os.path.join(pickle_store,'weights_'+today+'.{epoch:02d}-{val_loss:.3f}.hdf5'), 
#                                monitor='val_fn', save_best_only=True,save_weights_only=True,mode='max')        

    csv_logger = CSVLogger(rese,append=True)
    print('Fitting model...')          
    tn = datetime.datetime.now()
   
#    ooo
    DIM_ORDERING=keras.backend.image_data_format()
    print DIM_ORDERING

    for i in range (0,turnNumber):
        print 'turn:',i
        nb_epoch_i_p=(i+1)*nb_epoch
        init_epoch_i_p=i*nb_epoch
        todayn = 'month:'+str(tn.month)+'-day:'+str(tn.day)+'-year:'+str(tn.year)+' - '+str(tn.hour)+'h'+str(tn.minute)+'m'
        filew=open (todayf,'a')
        filew.write('start turn: '+ str(i)+'\n')
        filew.write('start epoch  :'+str(init_epoch_i_p)+ ' at '+todayn+' for max epoch: '+str(nb_epoch_i_p)+'\n')
        filew.write('------------------------------------------\n')
        
        listmodel=[name for name in os.listdir(pickle_store) if name.find('weights')==0]
        if len(listmodel)>0:
             print 'load weight found from last training'
             filew.write ('load weight found from last training\n')
             namelastc=load_model_set(pickle_store,filew)         
             
        else:
             print 'first training to be run'
             filew.write ('first training to be run\n')
             namelastc='NAN'
        model = get_model(num_class,num_bit,image_rows,image_cols,False,weights,weightedloss,namelastc)        
        filew.close()
        print('-'*30)
        json_string = model.to_json()
        pickle.dump(json_string,open( os.path.join(pickle_store,'CNN.h5'), "wb"),protocol=-1)
        model.summary()
        
        history = model.fit_generator(
        generator=batch_generator(batch_size,numclass,num_class,listroi,listscaninroi,indexpatc),
                epochs=nb_epoch_i_p,
                initial_epoch=init_epoch_i_p,
                steps_per_epoch=trainSetSize//batch_size,
#                sample_weight=class_weights,
                validation_data=(x_val,y_val),
                verbose=2,
                callbacks=[model_checkpoint,reduce_lr,csv_logger,early_stopping] ,
                max_queue_size=batch_size)   
        
        tn = datetime.datetime.now()

        todayn = 'month:'+str(tn.month)+'_day:'+str(tn.day)+'_year:'+str(tn.year)+'_'+str(tn.hour)+'h'+str(tn.minute)+'m'
        filew=open (todayf,'a')
        filew.write('end turn: '+ str(i)+'\n')
        filew.write('end epoch  :'+str(nb_epoch_i_p)+ ' at '+todayn+' for max epoch: '+str(turnNumber*nb_epoch)+'\n')
        filew.write('for restart  :'+str(numgen)+'\n')
        print('end epoch  :'+str(nb_epoch_i_p)+ ' at '+todayn+' for max epoch: '+str(turnNumber*nb_epoch)+'\n')    
    
        print('Predict model...')
        print('-'*30)
        y_score = model.predict(x_val, batch_size=batch_size,verbose=1)
        print 'y_score.shape',y_score.shape

        if DIM_ORDERING == 'channels_first':
            yvf= np.argmax(y_val, axis=1).flatten()
            #            print yvf[0]
            ysf=  np.argmax(y_score, axis=1).flatten()  
        else:
            yvf= np.argmax(y_val, axis=3).flatten()
#            print yvf[0], np.argmax(y_val, axis=-1)
            ysf=  np.argmax(y_score, axis=3).flatten()  
        fscore, acc, cm = evaluate(yvf,ysf,num_class)
        print cm
        print('Val F-score: '+str(fscore)+'\tVal acc: '+str(acc))         

        print '---------------'
        tn = datetime.datetime.now()
        todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
        filew.write('  finished at :'+todayn)
        
        filew.write('  f-score is : '+ str(fscore)+'\n')
        filew.write('  accuray is : '+ str(acc)+'\n')
        filew.write('  confusion matrix\n')
        n= cm.shape[0]
        for cmi in range (0,n):     
                        filew.write('  ')
                        for j in range (0,n):
                            filew.write(str(cm[cmi][j])+' ')
                        filew.write('\n')
                    
        filew.write('------------------------------------------\n')
        filew.close()
    
    
        x_val, y_val, num_class,img_rows,img_cols,num_images = load_train_val(numpidirvalref)
        print('Predict model from val...')
        print('-'*30)
        y_score = model.predict(x_val, batch_size=batch_size,verbose=1)
        print 'y_score.shape',y_score.shape
    
        if DIM_ORDERING == 'channels_first':
            yvf= np.argmax(y_val, axis=1).flatten()
            #            print yvf[0]
            ysf=  np.argmax(y_score, axis=1).flatten()  
        else:
            yvf= np.argmax(y_val, axis=3).flatten()
    #            print yvf[0], np.argmax(y_val, axis=-1)
            ysf=  np.argmax(y_score, axis=3).flatten()  
        fscore, acc, cm = evaluate(yvf,ysf,num_class)
        print cm
        print('Val F-score: '+str(fscore)+'\tVal acc: '+str(acc))         
    
        print '---------------'
        tn = datetime.datetime.now()
        todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
        filew=open (todayf,'a')
        filew.write('  finished at :'+todayn)
        
        filew.write('  f-score is : '+ str(fscore)+'\n')
        filew.write('  accuray is : '+ str(acc)+'\n')
        filew.write('  confusion matrix\n')
        n= cm.shape[0]
        for cmi in range (0,n):     
                        filew.write('  ')
                        for j in range (0,n):
                            filew.write(str(cm[cmi][j])+' ')
                        filew.write('\n')
                    
        filew.write('------------------------------------------\n')
        print ('------------------------')      

    print ('------------------------')  
         
    tn = datetime.datetime.now()
    todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
    filew.write('completed at :'+todayn)
    print ('completed at :'+todayn)
    filew.close()
                 
train()