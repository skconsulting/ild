# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017
@author: sylvain
"""

from param_pix import get_model,image_rows,image_cols,num_bit,evaluate

import datetime
import os
import cPickle as pickle
import numpy as np
import keras
import pandas as pd

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger,EarlyStopping
from keras import backend as K
K.set_image_dim_ordering('tf')

nepochs=5
nameHug='TRAIN_SET'
subHug='pickle_train_set'
extension='p'
#nameHug='CHU'
#nameHug='DUMMY'
dataindir='V'


pickle_store_ext= 'pickle'

################################

cwd=os.getcwd()
#
(cwdtop,tail)=os.path.split(cwd)

pickel_dirsource=os.path.join(cwdtop,nameHug)
pickel_dirsource=os.path.join(pickel_dirsource,subHug+'_'+extension)

pickel_dirdata=os.path.join(pickel_dirsource,dataindir) #directory to load data
pickel_dirout=os.path.join(pickel_dirsource,pickle_store_ext) #directory to store weights and results
if not os.path.exists(pickel_dirout):
    os.mkdir(pickel_dirout)


t = datetime.datetime.now()
today = 'd_'+str(t.month)+'-'+str(t.day)+'-'+str(t.year)+'_'+str(t.hour)+'_'+str(t.minute)

#######################################################

def load_train_val(numpidir):

    X_test = pickle.load( open( os.path.join(numpidir,"X_test.pkl"), "rb" ))
    y_test = pickle.load( open( os.path.join(numpidir,"Y_test.pkl"), "rb" ))
    X_train = pickle.load( open( os.path.join(numpidir,"X_train.pkl"), "rb" ))
    y_train = pickle.load( open( os.path.join(numpidir,"y_train.pkl"), "rb" ))
    DIM_ORDERING=keras.backend.image_data_format()
    if DIM_ORDERING == 'channels_first':
          num_class= y_test.shape[1]
          img_rows=X_test.shape[2]
          img_cols=X_test.shape[3]

    else:
        num_class= y_test.shape[3]
        img_rows=X_test.shape[1]
        img_cols=X_test.shape[2]
    num_val_images=X_test.shape[0]
    num_train_images=X_test.shape[0]

    return X_test, y_test,X_train,y_train,num_class,img_rows,img_cols,num_val_images,num_train_images

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

#
#def store_model(model,it):
#    name_model=os.path.join(pickle_store,'weights_'+str(today)+'_'+str(it)+'model.hdf5')
#    model.save_weights(name_model)

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


def train():
    tn = datetime.datetime.now()
    todayf ='f'+'_'+str(tn.month)+'_'+str(tn.day)+'_'+str(tn.year)+'_'+str(tn.hour)+'_'+str(tn.minute)+'stat.txt'
    todayf=os.path.join(pickel_dirout,todayf)
    filew=open (todayf,'w')

    todayn = 'month:'+str(tn.month)+'-day:'+str(tn.day)+'-year:'+str(tn.year)+' - '+str(tn.hour)+'h'+str(tn.minute)+'m'+'\n'
    filew.write('start of training :'+todayn)
    
    print('-'*30)
    print('Loading train data...')
    print('-'*30)
    

    num_class,class_weights_r,class_weights = load_weight(pickel_dirsource)
    print('Creating and compiling model...')
    print('-'*30)
    
    model = get_model(num_class,num_bit,image_rows,image_cols,False,class_weights_r)

    listmodel=[name for name in os.listdir(pickel_dirout) if name.find('weights')==0]
    if len(listmodel)>0:
         print 'weight  found'
         namelastc=load_model_set(pickel_dirout)         
         model.load_weights(namelastc)  
    else:
         print 'no weight found'

    X_test, y_test,X_train,y_train,num_class1,img_rows,img_cols,num_val_images,num_train_images = load_train_val(pickel_dirdata)
    assert img_rows==image_rows,"dimension mismatch"
    assert img_cols==image_cols,"dimension mismatch"
    assert num_class==num_class1,"num class mismatch"

    
    today = str('m'+str(t.month)+'_d'+str(t.day)+'_y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m')
    rese=os.path.join(pickel_dirout,str(today)+'_e.csv')

    early_stopping=EarlyStopping(monitor='val_loss', patience=150, verbose=1,min_delta=0.005,mode='min')                                     
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                              patience=50, min_lr=1e-5,verbose=1)#init 5
    model_checkpoint = ModelCheckpoint(os.path.join(pickel_dirout,'weights_'+today+'.{epoch:02d}-{val_loss:.2f}.hdf5'), 
                                monitor='val_loss', save_best_only=True,save_weights_only=True) 
    csv_logger = CSVLogger(rese,append=True)
    
    
    print 'shape x_train :',X_train.shape
    print 'shape y_train :',y_train.shape
    print 'shape X_test :',X_test.shape
    print 'shape y_test :',y_test.shape
    print('-'*30)
    print 'number of validation images:', num_val_images
    print 'number of train images:', num_train_images
    print 'image width:', num_bit
    print 'number of classes:', num_class
    print 'image number of rows :',img_rows
    print 'image number of columns :',img_cols
    print 'number of epochs:',nepochs
    print('-'*30)
    filew.write ( ' path for data input :'+pickel_dirdata+'\n')
    filew.write (  'directory for result'+pickel_dirout+'\n')
    filew.write ('shape x_train :'+str(X_train.shape)+'\n')
    filew.write ('shape y_train :'+str(y_train.shape)+'\n')
    filew.write ('image number of rows :'+str(img_rows)+'\n')
    filew.write ('image number of columns :'+str(img_cols)+'\n')
    filew.write ( 'image width:'+ str(num_bit) +'\n')    
    filew.write ( 'number of  of epochs:'+ str(nepochs)+'\n')
    filew.close()

                             
       
    print('Fitting model...')
    print('-'*30)
    print X_train[0].min(), X_train[0].max()

    debug=False
    if debug:
        xt=  X_train
        yt= y_train
        np.set_printoptions(threshold=np.nan)
        print 'xt', xt.shape
        print 'yt', yt.shape
        print xt[0].min(), xt[0].max()
        print yt[0].min(), yt[0].max()
        print xt[0][150][0]
        print yt[0][150][0]
        print xt[0][0][0]
        print yt[0][0][0]

        
        plt.figure(figsize = (10, 5))
        #    plt.subplot(1,3,1)
        #    plt.title('image')
        #    plt.imshow( np.asarray(crpim) )
        plt.subplot(1,2,1)
        plt.title('image')
        plt.imshow( xt[0][:,:,0] )
        plt.subplot(1,2,2)
        plt.title('label')
        plt.imshow( np.argmax(yt[0],axis=2) )
        plt.show()
        ooo

  
    history= model.fit(X_train, y_train, batch_size=1, epochs=nepochs, verbose =1,
                      validation_data=(X_test,y_test), shuffle=True,
                      callbacks=[model_checkpoint,reduce_lr,csv_logger,early_stopping]  )
    print('Predict model...')
    print('-'*30)
    y_score = model.predict(X_test, batch_size=1,verbose=1)
    print y_score.shape
    yvf= np.argmax(y_test, axis=3).flatten()
    ysf=  np.argmax(y_score, axis=3).flatten()   

    fscore, acc, cm = evaluate(yvf,ysf,num_class)
 
    print('Val F-score: '+str(fscore)+'\tVal acc: '+str(acc))         
    print 'fscore :',fscore
    print cm
    tn = datetime.datetime.now()
    todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
    filew=open (todayf,'w')
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
#    pd.DataFrame(history.history).to_csv(os.path.join(pickel_dirout,'sknet.csv'), index=False)


train()