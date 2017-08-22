# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017
training of CNN
@author: sylvain
"""

from param_pix import cwdtop,num_bit,image_rows,image_cols
from param_pix import get_model,evaluate,norm

import cPickle as pickle
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger,EarlyStopping
from keras.utils import np_utils
####################################################################################

nameHug='IMAGEDIR'

toppatch= 'TOPROI' #for scan classified ROI
extendir='essai'  #for scan classified ROI
#extendir='0'  #for scan classified ROI


roipicklepath = 'roipicklepatches'
patchtoppath=os.path.join(cwdtop,nameHug)
patchtoppath=os.path.join(patchtoppath,toppatch+'_'+extendir)

picklepathdir =os.path.join(patchtoppath,roipicklepath)
print ' path for data input :',picklepathdir


maxepoch=5 # number of images for each epoch * batch_size
nb_epoch=3 #number of epoch for each set
batch_size=2 #batch size (gpu ram dependant)

pickel_top='pickle' #path to get input data
#pickel_ext='lu_f6'
#pickel_ext='S3'
#pickel_ext='ILD_TXT'
#pickel_ext='ROI_ILD1'  #path to get input data
#pickel_ext='ROI_ILD_TXT'  #path to get input data
pickel_ext='train_set'  #path to get input data
pickel_ext_set='3'  #path to get input data



##############################################################
validationdir='V' #validation set directory
pickle_store_ext= 'pickle' #path to store csv with train status
#pickel_dirsource='pickle_ILD_TXT'


pickel_dirsource=pickel_top+'_'+pickel_ext+'_'+pickel_ext_set
#pickel_dirsource='pickle_UIP'

t = datetime.datetime.now()
today = 'd_'+str(t.month)+'-'+str(t.day)+'-'+str(t.year)+'_'+str(t.hour)+'_'+str(t.minute)

pickel_train=pickel_dirsource
 

pickle_dir=os.path.join(cwdtop,pickel_dirsource)
pickle_store= os.path.join(pickle_dir,pickle_store_ext)

print 'source of data input',pickle_dir
print 'directory for result',pickle_store
if not os.path.exists(pickle_dir):
    os.mkdir(pickle_dir)
if not os.path.exists(pickle_store):
    os.mkdir(pickle_store)

pickle_dir_train=os.path.join(cwdtop,pickel_train)

def load_train_val(numpidir):

    X_test = pickle.load( open( os.path.join(numpidir,"X_test.pkl"), "rb" ))
    y_test = pickle.load( open( os.path.join(numpidir,"Y_test.pkl"), "rb" ))
    num_class= y_test.shape[3]
#    num_class= 4
    img_rows=X_test.shape[1]
    img_cols=X_test.shape[2]
    num_images=X_test.shape[0]
#    for key,value in class_weights.items():
#        print key, value
    print X_test.min(),X_test.max()
    print y_test.min(),y_test.max()
    
    return X_test, y_test,num_class,img_rows,img_cols,num_images
        

def load_train_data(numpidir):
    X_train = pickle.load( open( os.path.join(numpidir,"X_train.pkl"), "rb" ))
    y_train = pickle.load( open( os.path.join(numpidir,"y_train.pkl"), "rb" ))
    num_class= y_train.shape[3]
#    num_class= 4
    img_rows=X_train.shape[1]
    img_cols=X_train.shape[2]
    num_images=X_train.shape[0]
#    for key,value in class_weights.items():
#        print key, value
    
    return X_train, y_train,num_class,img_rows,img_cols,num_images


def load_weight(numpidir):
    class_weights=pickle.load(open( os.path.join(numpidir,"class_weights.pkl"), "rb" ))
    num_class= len(class_weights)

    print 'number of classes :',num_class
  
#    for key,value in class_weights.items():
#        print key, value

    clas_weigh_l=[]
    for i in range (0,num_class):
#            print i,class_weights[i]
            clas_weigh_l.append(class_weights[i])
    print 'weights for classes:'
    for i in range (0,num_class):
                    print i, clas_weigh_l[i]
    class_weights_r=np.array(clas_weigh_l)
#    print class_weights_r
#    ooo
    return num_class,class_weights_r,class_weights


def store_model(model,it):
    name_model=os.path.join(pickle_store,'weights_'+str(today)+'_'+str(it)+'model.hdf5')
    model.save_weights(name_model)

def load_model_set(pickle_dir_train):
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
    return namelastc

#def generate_arrays_from_file(path):
#    while 1:
#        f = open(path)
#        for line in f:
#            # create Numpy arrays of input data
#            # and labels, from each line in the file
#            x, y = process_line(line)
#            yield (x, y)
#    f.close()

def geneaug(image,tt):
    if tt==0:
        imout=image
    elif tt==1:
    # 1 90 deg
        imout = np.rot90(image)
    elif tt==2:
    #2 180 deg
        imout = np.rot90(np.rot90(image))
    elif tt==3:
    #3 270 deg
        imout = np.rot90(np.rot90(np.rot90(image)))
    elif tt==4:
    #4 flip fimage left-right
            imout=np.fliplr(image)
    elif tt==5:
    #5 flip fimage left-right +rot 90
        imout = np.rot90(np.fliplr(image))
    elif tt==6:
    #6 flip fimage left-right +rot 180
        imout = np.rot90(np.rot90(np.fliplr(image)))
    elif tt==7:
    #7 flip fimage left-right +rot 270
        imout = np.rot90(np.rot90(np.rot90(np.fliplr(image))))
    elif tt==8:
    # 8 flip fimage up-down
        imout = imout=np.flipud(image)
    elif tt==9:
    #9 flip fimage up-down +rot90
        imout = np.rot90(np.flipud(image))
    elif tt==10:
    #10 flip fimage up-down +rot180
        imout = np.rot90(np.rot90(np.flipud(image)))
    elif tt==11:
    #11 flip fimage up-down +rot270
        imout = np.rot90(np.rot90(np.rot90(np.flipud(image))))

    return imout

def readclasses(pat,namepat,indexpat,indexaug):

    patpick=os.path.join(picklepathdir,pat)
    patpick=os.path.join(patpick,namepat[indexpat])
    readpkl=pickle.load(open(patpick, "rb"))
                                            
    scanr=readpkl[0][0]
    scan=geneaug(scanr,indexaug)
    maskr=readpkl[1][0]
    mask=geneaug(maskr,indexaug)
#    cv2.imshow(pat+str(indexpat)+'mask',normi(mask))
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    scanm=norm(scan)

    return scanm, mask  

def readclasses2(num_classes,X_testi,y_testi):

    X_test = np.asarray(np.expand_dims(X_testi,3))  
    y_test = np.array(y_testi)
#    print X_test.shape
#    print y_test.shape
    
    lytest=y_test.shape[0]
    ytestr=np.zeros((lytest,image_rows, image_cols,int(num_classes)),np.uint8)
    for i in range (lytest):
        for j in range (0,image_rows):
            ytestr[i][j] = np_utils.to_categorical(y_test[i][j], num_classes)
      
    return  X_test,  ytestr  



def gen_random_image(maximage,numclass,numgen,listroi,listscaninroi,indexpatc):
    numgen+=1
    numgen=numgen%(maximage*numclass)
#    print numgen,numgen%numclass

    pat =listroi[numgen%numclass]
    numberscan=len(listscaninroi[pat])
    indexpatc[pat] =  indexpatc[pat]%numberscan
    indexpat=indexpatc[pat]
    indexpatc[pat]=indexpatc[pat]+1

    indexaug = random.randint(0, 11)
#        print numgen ,pat,indexpat,numberscan
    scan,mask=readclasses(pat,listscaninroi[pat],indexpat,indexaug)  
    return scan,mask,numgen
    #        label_list.append(mask)


    
def batch_generator(batch_size,maximage,numclass,num_classes,listroi,listscaninroi,indexpatc,numgen):
    while True:
        image_list = []
        mask_list = []
        for i in range(batch_size):
            img, mask,numgen = gen_random_image(maximage,numclass,numgen,listroi,listscaninroi,indexpatc)
            
#            ooo
            image_list.append(img)
            mask_list.append(mask)
            
        X_test,  ytestr  =readclasses2(num_classes,image_list,mask_list)
#        print X_test.shape
#        print ytestr.shape
        yield  X_test,  ytestr  
        """Here is an example of fit_generator():
model.fit_generator(generator(), samples_per_epoch=50, nb_epoch=10)
Breaking it down:
generator() generates batches of samples indefinitely
sample_per_epoch number of samples you want to train in each epoch
nb_epoch number of epochs
As you can manually define sample_per_epoch and nb_epoch , you have to provide codes for generator . Here is an example:
Assume features is an array of data with shape (100,64,64,3) and labels is an array of data with shape (100,1). We use data from features and labels to train our model.
def generator(features, labels, batch_size):
 # Create empty arrays to contain batch of features and labels#
 batch_features = np.zeros((batch_size, 64, 64, 3))
 batch_labels = np.zeros((batch_size,1))
 while True:
   for i in range(batch_size):
     # choose random index in features
     index= random.choice(len(features),1)
     batch_features[i] = some_processing(features[index])
     batch_labels[i] = labels[index]
   yield batch_features, batch_labels
With the generator above, if we define batch_size = 10 , that means it will randomly taking out 10 samples from features and labels to feed into each epoch until an epoch hits 50 sample limit. Then fit_generator() destroys the used data and move on repeating the same process in new epoch.
One great advantage about fit_generator() besides saving memory is user 
        """        

def train():
    
    print('-'*30)
    print('Loading train data...')
    print('-'*30)
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

    model = get_model(num_class,num_bit,image_rows,image_cols,False,weights)
    
    numpidirval =os.path.join(pickle_dir,validationdir)
    x_val, y_val, num_class,img_rows,img_cols,num_images = load_train_val(numpidirval)
    assert img_rows==image_rows,"dimension mismatch"
    assert img_cols==image_cols,"dimension mismatch"
    
    listmodel=[name for name in os.listdir(pickle_dir) if name.find('weights')==0]
    if len(listmodel)>0:
         print 'load weight found from last training'
         namelastc=load_model_set(pickle_dir)         
         model.load_weights(namelastc)  
    else:
         print 'first training to be run'

    rese=os.path.join(pickle_store,str(today)+'_e.csv')
#    resBest=os.path.join(pickle_store,str(today)+'_Best.csv')
#    open(rese, 'a').write('Epoch, Val_fscore, Val_acc, Train_loss, Val_loss\n')
#    open(resBest, 'a').write('Epoch, Val_fscore, Val_acc, Train_loss, Val_loss\n')
    
    print '-----------------------'

    print 'x_val min max :',x_val.min(),x_val.max()
    print 'shape x_val :',x_val.shape
    print 'shape y_val :',y_val.shape
    print 'number of classes:', num_class
    print 'image number of rows :',img_rows
    print 'image number of columns :',img_cols
    print 'number of epoch for all subset :',maxepoch
    print 'number of epoch per subset:',nb_epoch
    print 'batch_size  :',batch_size
    print 'number of images validation:', num_images
    print 'image width:', num_bit                              
    print('-'*30)
    
    filew.write ('x_val min max :'+str(x_val.min())+' '+str(x_val.max())+'\n')
    filew.write ('shape x_val :'+str(x_val.shape)+'\n')
    filew.write ('shape y_val :'+str(y_val.shape)+'\n')
    filew.write ('number of classes:'+ str(num_class)+'\n')
    filew.write ('image number of rows :'+str(img_rows)+'\n')
    filew.write ('image number of columns :'+str(img_cols)+'\n')
    filew.write ( 'number of epoch for all subset :'+str(maxepoch)+'\n')
    filew.write ( 'number of epoch per subset :'+str(nb_epoch)+'\n')
    filew.write ( 'batch_size  :'+str(batch_size)+'\n')
    filew.write ( 'number of images validation:'+ str(num_images)+'\n')

    filew.write ( 'image width:'+ str(num_bit) +'\n')    
    filew.write('------------------------------------------\n')
    filew.close()
    
    listroi=[name for name in os.listdir(picklepathdir)]

    numclass=len(listroi)
    print '-----------'
    print'number of classes in scan:', numclass
    print '-----------'

#    assert numclass+2==num_class,"dimension mismatch"
    listscaninroi={}

    indexpatc={}
    for j in listroi:
        indexpatc[j]=0

    totalimages=0
    maximage=0

    for c in listroi:
        listscaninroi[c]=os.listdir(os.path.join(picklepathdir,c))
        numberscan=len(listscaninroi[c])
        if numberscan>maximage:
            maximage=numberscan
            ptmax=c
    totalimages+=numberscan
    
    print 'number total of scan images:',totalimages
    print 'maximum data in one pat:',maximage,' in ',ptmax
    print '-----------'
    numgen=-1
    
    early_stopping=EarlyStopping(monitor='val_loss', patience=25, verbose=1,min_delta=0.005,mode='min')                                     
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                              patience=5, min_lr=1e-5,verbose=1)
    model_checkpoint = ModelCheckpoint(os.path.join(pickle_dir,'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), 
                                monitor='val_loss', save_best_only=True,save_weights_only=True)       

    csv_logger = CSVLogger(rese,append=True)
    print('Fitting model...')          
    tn = datetime.datetime.now()
    todayn = 'month:'+str(tn.month)+'-day:'+str(tn.day)+'-year:'+str(tn.year)+' - '+str(tn.hour)+'h'+str(tn.minute)+'m'
    filew=open (todayf,'a')
    filew.write('start epoch  :'+str(nb_epoch)+ ' at '+todayn+' for max epoch: '+str(maxepoch*nb_epoch)+'\n')
    filew.write('------------------------------------------\n')
    filew.close()
    print('-'*30)
#    ooo
    history = model.fit_generator(
    generator=batch_generator(batch_size,maximage,numclass,num_class,listroi,listscaninroi,indexpatc,numgen),
    epochs=nb_epoch,
    steps_per_epoch=maxepoch*batch_size,
    validation_data=(x_val,y_val),
    verbose=1,
    callbacks=[model_checkpoint,reduce_lr,csv_logger,early_stopping]  )
    
    
    tn = datetime.datetime.now()
    todayn = 'month:'+str(tn.month)+'-day:'+str(tn.day)+'-year:'+str(tn.year)+' - '+str(tn.hour)+'h'+str(tn.minute)+'m'
    filew=open (todayf,'a')
    filew.write('end epoch  :'+str(nb_epoch)+ ' at '+todayn+' for max epoch: '+str(maxepoch*nb_epoch)+'\n')
    print('end epoch  :'+str(nb_epoch)+ ' at '+todayn+' for max epoch: '+str(maxepoch*nb_epoch)+'\n')    

    print('Predict model...')
    print('-'*30)
    y_score = model.predict(x_val, batch_size=batch_size,verbose=1)
#                print 'y_score.shape',y_score.shape
    yvf= np.argmax(y_val, axis=3).flatten()
    #            print yvf[0]
    ysf=  np.argmax(y_score, axis=3).flatten()   
    #            print ysf[0]     
    #            print type(ysf[0])                
    #            print(history.history.keys())
                
    #            score = model.evaluate(x_val, y_val, verbose=0)  
    #            print 'Test score:', score[0]
    #            print 'Test accuracy:', score[1]
    
    #            output = model.predict_proba(x_val, verbose=0)
    #            output = output.reshape((output.shape[0], img_rows, img_cols, num_class))
    
    #            plot_results(output)
    
    fscore, acc, cm = evaluate(yvf,ysf,num_class)
    print cm
    print('Val F-score: '+str(fscore)+'\tVal acc: '+str(acc))         
#            open(rese, 'a').write(str(str(it)+', '+str(fscore)+', '+str(acc)+', '+str(np.max(history.history['loss']))+', '+str(np.max(history.history['val_loss']))+'\n'))
#                print 'fscore :',fscore
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
    print ('------------------------')               
    tn = datetime.datetime.now()
    todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
    filew.write('completed at :'+todayn)
    print ('completed at :'+todayn)
    filew.close()
       
                  
train()