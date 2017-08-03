# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017
training of CNN
@author: sylvain
"""

from param_pix import cwdtop,num_bit,image_rows,image_cols
from param_pix import get_model,evaluate

import cPickle as pickle
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os


from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger,EarlyStopping

maxepoch=2 # number of time the full number of data in all set is used
nb_epoch=2 #number of poch for each set
batch_size=2 #batch size (gpu ram dependant)

pickel_top='pickle' #path to get input data
#pickel_ext='lu_f6'
#pickel_ext='S3'
#pickel_ext='ILD_TXT'
#pickel_ext='ROI_ILD1'  #path to get input data
#pickel_ext='ROI_ILD_TXT'  #path to get input data
pickel_ext='train_set'  #path to get input data
pickel_ext_set='1'  #path to get input data



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
pickle_store= os.path.join(cwdtop,pickle_store_ext)
print 'source of data input',pickle_dir
pickle_dir_train=os.path.join(cwdtop,pickel_train)


def load_train_val(numpidir):

    X_test = pickle.load( open( os.path.join(numpidir,"X_test.pkl"), "rb" ))
    y_test = pickle.load( open( os.path.join(numpidir,"y_test.pkl"), "rb" ))
    num_class= y_test.shape[3]
#    num_class= 4
    img_rows=X_test.shape[1]
    img_cols=X_test.shape[2]
    num_images=X_test.shape[0]
#    for key,value in class_weights.items():
#        print key, value
    
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


def train():
    
    print('-'*30)
    print('Loading train data...')
    print('-'*30)
    filew=open (pickle_dir+'/status.txt','a')
    tn = datetime.datetime.now()
    todayn = 'month:'+str(tn.month)+'-day:'+str(tn.day)+'-year:'+str(tn.year)+' - '+str(tn.hour)+'h'+str(tn.minute)+'m'+'\n'
    filew.write('start of training :'+todayn)

    listplickle=[name for name in os.walk(pickle_dir).next()[1] if name != validationdir]
    print 'list of subsets :',listplickle

    num_class,weights,class_weights = load_weight(pickle_dir)
    print('Creating and compiling model...')
    print('-'*30)

    model = get_model(num_class,num_bit,image_rows,image_cols,False,weights)
    
    
#    maxf         = 0
#    maxacc       = 0
#    maxit        = 0
#    maxtrainloss = 0
#    maxvaloss    = np.inf
#    best_model   = model
#    tolerance =1.005
#    it           = 0    
    p            = 0
    
    
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
    print 'shape x_val :',x_val.shape
    print 'shape y_val :',y_val.shape
    filew.write('shape x_val :'+str(x_val.shape)+'\n')
    filew.write('shape y_val :'+str(y_val.shape)+'\n')
    filew.write('------------------------------------------\n')
    while p < maxepoch:
            p += 1
            print('Epoch: ' + str(p), 'max epoch:',maxepoch)            
            tn = datetime.datetime.now()
            todayn = 'month:'+str(tn.month)+'-day:'+str(tn.day)+'-year:'+str(tn.year)+' - '+str(tn.hour)+'h'+str(tn.minute)+'m'
            filew.write('start epoch  :'+str(p)+ ' at '+todayn+' for max epoch: '+str(maxepoch)+'\n')
            filew.write('------------------------------------------\n')
          
            lnumpi=len(listplickle)
            i=0
            for numpi in listplickle:
                tn = datetime.datetime.now()
                todayn = 'month:'+str(tn.month)+'-day:'+str(tn.day)+'-year:'+str(tn.year)+' at '+str(tn.hour)+'h'+str(tn.minute)+'m'
                filew.write('  start subset :'+numpi+' at:' +todayn+' for max subset: '+str(lnumpi-1)+'\n')

                i+=1
                nb_epoch_i_p=(i+(p-1)*(lnumpi))*nb_epoch
                init_epoch_i_p=((i-1)+(p-1)*(lnumpi))*nb_epoch
                print 'work on subset :',numpi, ' for max subset: '+str(lnumpi-1)
#                print i,p,nb_epoch,lnumpi,nb_epoch_i_p,init_epoch_i_p
                
                numpidir =os.path.join(pickle_dir,numpi)
                x_train, y_train,  num_class,img_rows,img_cols,num_images = load_train_data(numpidir)
                assert img_rows==image_rows,"dimension mismatch"
                assert img_cols==image_cols,"dimension mismatch"
               
                print 'shape x_train :',x_train.shape
                print 'shape y_train :',y_train.shape
                filew.write('  shape x_train :'+str(x_train.shape)+'\n')
                filew.write('  shape y_train :'+str(y_train.shape)+'\n')

                print('-'*30)
                print 'number of images:', num_images
                print 'image width:', num_bit
                print 'number of classes:', num_class
                print 'image number of rows :',img_rows
                print 'image number of columns :',img_cols
                print 'number of epoch per subset :',maxepoch
                print 'number of epoch  :',nb_epoch
                print 'batch_size  :',batch_size
                print('-'*30)
        
                early_stopping=EarlyStopping(monitor='val_loss', patience=15, verbose=0)                     
                model_checkpoint = ModelCheckpoint(os.path.join(pickle_dir,'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), 
                                monitor='val_loss', save_best_only=True,save_weights_only=True)       
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                      patience=5, min_lr=0.001)
                csv_logger = CSVLogger(rese,append=True)
                print('Fitting model...')
                print('-'*30)
                print x_train[0].min(), x_train[0].max()
        
                debug=False
                if debug:
                    xt=  x_train
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
                          
                    plt.figure(figsize = (5, 5))
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
                    filew.close()
                    return
                    

                
#                history= 
                """
                model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch_i_p, verbose =1,
                      validation_data=(x_val,y_val), shuffle=True, initial_epoch=init_epoch_i_p,
#                      class_weight=class_weights,
                      callbacks=[model_checkpoint,reduce_lr,csv_logger,early_stopping]  )
                """
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
                for i in range (0,n): 

                    filew.write('  ')
                    for j in range (0,n):
                        filew.write(str(cm[i][j])+' ')
                    filew.write('\n')
                filew.write('------------------------------------------\n')
    tn = datetime.datetime.now()
    todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
    filew.write('completed at :'+todayn)
    print ('completed at :'+todayn)
    filew.close()
       
                  
train()