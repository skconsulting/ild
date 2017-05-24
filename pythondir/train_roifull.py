# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017
@author: sylvain
"""

from param_pix import *
import os
import keras
import theano
from itertools import product
import numpy as np
from keras.models import Model

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D ,Dense,Flatten,Dropout
from keras.optimizers import Adam,Adagrad
from keras.callbacks import ModelCheckpoint
from keras import backend as K
#import sklearn.metrics as metrics
from keras.models import load_model
import cPickle as pickle
import datetime
K.set_image_dim_ordering('th')
from numpy import argmax,amax
print keras.__version__
print theano.__version__
maxepoch=10

pickel_dirsource='pickle_ILD_TXT'
#pickel_dirsource='pickle_ILD6'

t = datetime.datetime.now()
today = 'd_'+str(t.month)+'-'+str(t.day)+'-'+str(t.year)+'_'+str(t.hour)+'_'+str(t.minute)

pickel_train=pickel_dirsource
 

cwd=os.getcwd()

(cwdtop,tail)=os.path.split(cwd)
pickle_dir=os.path.join(cwdtop,pickel_dirsource)
print 'source',pickle_dir
pickle_dir_train=os.path.join(cwdtop,pickel_train)

def load_train_data(numpidir):
    X_train = pickle.load( open( os.path.join(numpidir,"X_train.pkl"), "rb" ))
    y_train = pickle.load( open( os.path.join(numpidir,"y_train.pkl"), "rb" ))
    X_test = pickle.load( open( os.path.join(numpidir,"X_test.pkl"), "rb" ))
    y_test = pickle.load( open( os.path.join(numpidir,"y_test.pkl"), "rb" ))
    class_weights=pickle.load(open( os.path.join(numpidir,"class_weights.pkl"), "rb" ))
    num_class= y_train.shape[1]
    img_rows=X_train.shape[2]
    img_cols=X_train.shape[3]
    num_images=X_train.shape[0]
    for key,value in class_weights.items():
        print key, value
    print num_class
   
    clas_weigh_l=[]
    for i in range (0,num_class):
#            print i,class_weights[i]
            clas_weigh_l.append(class_weights[i])
    print 'weights for classes:'
    for i in range (0,num_class):
                    print i, clas_weigh_l[i]
    class_weights_r=np.array(clas_weigh_l)
    return X_train, y_train, X_test, y_test,num_class,img_rows,img_cols,num_images,class_weights_r
#
#smooth = 1.
#
#def dice_coef(y_true, y_pred):
#    y_true_f = K.flatten(y_true)
#    y_pred_f = K.flatten(y_pred)
#    intersection = K.sum(y_true_f * y_pred_f)
#    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#

#def load_model_set(pickle_dir_train):
#    listmodel=[name for name in os.listdir(pickle_dir_train) if name.find('weights')==0]
#
#    ordlist=[]
#    for name in listmodel:
#        nfc=os.path.join(pickle_dir_train,name)
#        nbs = os.path.getmtime(nfc)
#        tt=(name,nbs)
#        ordlist.append(tt)
#
#    ordlistc=sorted(ordlist,key=lambda col:col[1],reverse=True)
#
#    namelast=ordlistc[0][0]
#
#    namelastc=os.path.join(pickle_dir_train,namelast)
#    print 'last weights :',namelast
#    model=load_model(namelastc)
#    return model

def store_model(model,it):
    name_model=os.path.join('../pickle','weights_'+str(today)+'_'+str(it)+'model.hdf5')
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
#    listplickle=os.listdir(pickle_dir)
#    print pickle_dir
    listplickle=os.walk(pickle_dir).next()[1]
    print listplickle
#    numpidirb=pickle_dir
    fp=True
    for numpi in listplickle:
    
#        print numpi, numpidirb
        numpidir =os.path.join(pickle_dir,numpi)
        x_train, y_train, x_val, y_val, num_class,img_rows,img_cols,num_images,weights = load_train_data(numpidir)
       
        if fp:
            print 'shape x_train :',x_train.shape
            print 'shape y_train :',y_train.shape
            print 'shape x_val :',x_val.shape
            print 'shape y_val :',y_val.shape
            print('-'*30)
            print 'number of images:', num_images
            print 'number of classes:', num_class
            print 'image number of rows :',img_rows
            print 'image number of columns :',img_cols
            print('-'*30)
            fp=False
        print('Creating and compiling model...')
        print('-'*30)
        
       
#            listmodel=[name for name in os.listdir(numpidir) if name.find('weights')==0]
#            if len(listmodel)==0:
#                print 'first pass'
#        model = get_unet(num_class,img_rows,img_cols,class_weights)
#        model = get_unet(num_class,image_rows,image_cols)
        model = get_model(num_class,img_rows,img_cols,weights)
#        mloss = weighted_categorical_crossentropy(weights).myloss
#        model.compile(optimizer=Adam(lr=1e-5), loss=mloss, metrics=['categorical_accuracy'])
#        model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])


#        numpidirb=os.path.join(pickle_dir,numpi)
        listmodel=[name for name in os.listdir(numpidir) if name.find('weights')==0]
        if len(listmodel)>0:
             namelastc=load_model_set(numpidir)
             model.load_weights(namelastc)        
        
        model_checkpoint = ModelCheckpoint(os.path.join(numpidir,'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', save_best_only=True,save_weights_only=True)
        rese=os.path.join('../pickle',str(today)+'_e.csv')
        resBest=os.path.join('../pickle',str(today)+'_Best.csv')
        open(rese, 'a').write('Epoch, Val_fscore, Val_acc, Train_loss, Val_loss\n')
        open(resBest, 'a').write('Epoch, Val_fscore, Val_acc, Train_loss, Val_loss\n')

        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        
        tolerance =1.005

        maxf         = 0
        maxacc       = 0
        maxit        = 0
        maxtrainloss = 0
        maxvaloss    = np.inf
        best_model   = model
        it           = 0    
        p            = 0
        while p < maxepoch:
            p += 1
            print('Epoch: ' + str(it))
            history= model.fit(x_train, y_train, batch_size=1, epochs=1, verbose =1,
                      validation_data=(x_val,y_val), shuffle=True,
                      callbacks=[model_checkpoint]  )
            print('Predict model...')
            print('-'*30)
            y_score = model.predict(x_val, batch_size=1,verbose=1)
            yvf= np.argmax(y_val, axis=1).flatten()
#            print yvf[0]
            ysf=  np.argmax(y_score, axis=1).flatten()   
#            print ysf[0]     
#            print type(ysf[0])                
#            print(history.history.keys())
            fscore, acc, cm = evaluate(yvf,ysf,num_class)
            print('Val F-score: '+str(fscore)+'\tVal acc: '+str(acc))         
            open(rese, 'a').write(str(str(it)+', '+str(fscore)+', '+str(acc)+', '+str(np.max(history.history['loss']))+', '+str(np.max(history.history['val_loss']))+'\n'))
            print 'fscore :',fscore
        # check if current state of the model is the best and write evaluation metrics to file
            if fscore > maxf*tolerance:  # if fscore > maxf*params['tolerance']:
                print 'fscore is bigger than last iterations + 0.5%'
                #p            = 0  # restore patience counter
                best_model   = model  # store current model state
                maxf         = fscore 
                maxacc       = acc
                maxit        = it
                maxtrainloss = np.max(history.history['loss'])
                maxvaloss    = np.max(history.history['val_loss'])
    
    #            print(np.round(100*cm/np.sum(cm,axis=1)))
    #            print 'cm'
                print cm
    #            print 'cmfloat'
#                print(np.round(100.0*cm/np.sum(cm,axis=1).astype(float),1))

                open(resBest, 'a').write(str(str(maxit)+', '+str(maxf)+', '+str(maxacc)+', '+str(maxtrainloss)+', '+str(maxvaloss)+'\n'))
                store_model(best_model,it)
            it += 1
        
train()