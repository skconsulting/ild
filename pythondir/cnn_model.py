'''
This is a part of the supplementary material uploaded along with 
the manuscript:

    "Lung Pattern Classification for Interstitial Lung Diseases Using a Deep Convolutional Neural Network"
    M. Anthimopoulos, S. Christodoulidis, L. Ebner, A. Christe and S. Mougiakakou
    IEEE Transactions on Medical Imaging (2016)
    http://dx.doi.org/10.1109/TMI.2016.2535865

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

For more information please read the README file. The files can also 
be found at: https://github.com/intact-project/ild-cnn

modified by S. Kritter
version 1.1
28 july 2017

'''
# debug
# from ipdb import set_trace as bp
from param_pix_t import modelname
import ild_helpers as H
import datetime
import cPickle as pickle
import cv2
import os
import numpy as np
import sys
#import h5py

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger,EarlyStopping
from keras.models import model_from_json
from keras.optimizers import Adam

def get_FeatureMaps(L, policy, constant=17):
    return {
        'proportional': (L+1)**2,
        'static': constant,
    }[policy]

def get_Obj(obj):
    return {
        'mse': 'MSE',
        'ce': 'categorical_crossentropy',
    }[obj]

def get_model(input_shape, output_shape, params,filew):

    print('compiling model...')
        
    # Dimension of The last Convolutional Feature Map (eg. if input 32x32 and there are 5 conv layers 2x2 fm_size = 27)
    fm_size = input_shape[-1] - params['cl']
    print 'fm_size : ', fm_size
    print 'input_shape[-1] : ', input_shape[-1]
    print 'number of convolutional layers : ', params['cl']
    
    # Tuple with the pooling size for the last convolutional layer using the params['pf']
    pool_siz = (np.round(fm_size*params['pf']).astype(int), np.round(fm_size*params['pf']).astype(int))
    
    # Initialization of the model
    model = Sequential()

    # Add convolutional layers to model
    # model.add(Convolution2D(params['k']*get_FeatureMaps(1, params['fp']), 2, 2, init='orthogonal', activation=LeakyReLU(params['a']), input_shape=input_shape[1:]))
    # added by me
    model.add(Conv2D(params['k']*get_FeatureMaps(1, params['fp']), (2, 2), kernel_initializer='orthogonal', input_shape=input_shape[1:]))
    # model.add(Activation('relu'))
    #SK
    model.add(LeakyReLU(alpha=params['a']))
#    model.add(PReLU(init='zero', weights=None))
    print 'Layer 1 parameters settings:'
    print 'number of filters to be used : ', params['k']*get_FeatureMaps(1, params['fp'])
    print 'kernel size : 2 x 2' 
    print 'input_shape of tensor is : ', input_shape[1:]


    for i in range(2, params['cl']+1):
        print i, params['k']*get_FeatureMaps(i, params['fp'])
        # model.add(Convolution2D(params['k']*get_FeatureMaps(i, params['fp']), 2, 2, init='orthogonal', activation=LeakyReLU(params['a'])))
        model.add(Conv2D(params['k']*get_FeatureMaps(i, params['fp']), (2, 2), kernel_initializer='orthogonal'))
        # model.add(Activation('relu'))
        model.add(LeakyReLU(alpha=params['a']))
        print 'Layer',  i, ' parameters settings:'
        print 'number of filters to be used : ', params['k']*get_FeatureMaps(i, params['fp'])
        print 'kernel size : 2 x 2' 


    # Add Pooling and Flatten layers to model
    print 'entering 2D Pooling layer'
    print 'Pooling : ', params['pt']
    print 'pool_size : ', pool_siz
    if params['pt'] == 'Avg':
        model.add(AveragePooling2D(pool_size=pool_siz))
    elif params['pt'] == 'Max':
        model.add(MaxPooling2D(pool_size=pool_siz))
    else:
        sys.exit("Wrong type of Pooling layer")

    model.add(Flatten())

    # dropout is 50% or do=0.5
    model.add(Dropout(params['do']))

    # Add Dense layers and Output to model
    # model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp']))/params['pf']*6, init='he_uniform', activation=LeakyReLU(0)))
    model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp']))/params['pf']*6, kernel_initializer='he_uniform'))
    
    print 'output_dimension : ', int(params['k']*get_FeatureMaps(params['cl'], params['fp']))/params['pf']*6

    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=params['a']))
    model.add(Dropout(params['do']))

    
    # model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp']))/params['pf']*2, init='he_uniform', activation=LeakyReLU(0)))
    model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp']))/params['pf']*2, kernel_initializer='he_uniform'))
    #  model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=params['a']))
    model.add(Dropout(params['do']))
    model.add(Dense(output_shape[1], kernel_initializer='he_uniform', activation='softmax'))
  



#sk modif for decay, works only with ADAM

#    decay = 1.e-6
#    lr=0.001
#    lr =1.0
    learning_rate=1e-4
    # Compile model and select optimizer and objective function
    if params['opt'] not in ['Adam', 'Adagrad', 'SGD']:
        sys.exit('Wrong optimizer: Please select one of the following. Adam, Adagrad, SGD')
    if get_Obj(params['obj']) not in ['MSE', 'categorical_crossentropy']:
        sys.exit('Wrong Objective: Please select one of the following. MSE, categorical_crossentropy')
#    model.compile(optimizer=params['opt'], loss=get_Obj(params['obj']), metrics=['categorical_accuracy'])
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    filew.write ('learning rate:'+str(learning_rate)+'\n')
    print ('learning rate:'+str(learning_rate))

#    optimizer=keras.optimizers.Adam(lr=lr,decay=decay)
#    model.compile(optimizer=optimizer, loss=get_Obj(params['obj']))

    return model

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


def train(x_train, y_train, x_val, y_val, params,eferror,patch_dir_store):
    ''' TODO: documentation '''

    filew = open(eferror, 'a')
    # Parameters String used for saving the files
    parameters_str = str('_d' + str(params['do']).replace('.', '') +
                         '_a' + str(params['a']).replace('.', '') + 
                         '_k' + str(params['k']).replace('.', '') + 
                         '_c' + str(params['cl']).replace('.', '') + 
                         '_s' + str(params['s']).replace('.', '') + 
                         '_pf' + str(params['pf']).replace('.', '') + 
                         '_pt' + params['pt'] +
                         '_fp' + str(params['fp']).replace('.', '') +
                         '_opt' + params['opt'] +
                         '_obj' + params['obj'])

    # Printing the parameters of the model
    print('[Dropout Param] \t->\t'+str(params['do']))
    print('[Alpha Param] \t\t->\t'+str(params['a']))
    print('[Multiplier] \t\t->\t'+str(params['k']))
    print('[Patience] \t\t->\t'+str(params['patience']))
    print('[Tolerance] \t\t->\t'+str(params['tolerance']))
    print('[Input Scale Factor] \t->\t'+str(params['s']))
    print('[Pooling Type] \t\t->\t'+ params['pt'])
    print('[Pooling Factor] \t->\t'+str(str(params['pf']*100)+'%'))
    print('[Feature Maps Policy] \t->\t'+ params['fp'])
    print('[Optimizer] \t\t->\t'+ params['opt'])
    print('[Objective] \t\t->\t'+ get_Obj(params['obj']))
    print('[Results filename] \t->\t'+str(params['res_alias']+parameters_str+'.txt'))
    
    
    filew.write('[Dropout Param] \t->\t'+str(params['do'])+'\n')
    filew.write('[Alpha Param] \t\t->\t'+str(params['a'])+'\n')
    filew.write('[Multiplier] \t\t->\t'+str(params['k'])+'\n')
    filew.write('[Patience] \t\t->\t'+str(params['patience'])+'\n')
    filew.write('[Tolerance] \t\t->\t'+str(params['tolerance'])+'\n')
    filew.write('[Input Scale Factor] \t->\t'+str(params['s'])+'\n')
    filew.write('[Pooling Type] \t\t->\t'+ params['pt']+'\n')
    filew.write('[Pooling Factor] \t->\t'+str(str(params['pf']*100)+'%')+'\n')
    filew.write('[Feature Maps Policy] \t->\t'+ params['fp']+'\n')
    filew.write('[Optimizer] \t\t->\t'+ params['opt']+'\n')
    filew.write('[Objective] \t\t->\t'+ get_Obj(params['obj'])+'\n')
    filew.write('[Results filename] \t->\t'+str(params['res_alias']+parameters_str+'.txt')+'\n')
    


    # Rescale Input Images
    if params['s'] != 1:
        print('\033[93m'+'Rescaling Patches...'+'\033[0m')
        x_train = np.asarray(np.expand_dims([cv2.resize(x_train[i, 0, :, :], (0,0), fx=params['s'], fy=params['s']) for i in xrange(x_train.shape[0])], 1))
        x_val = np.asarray(np.expand_dims([cv2.resize(x_val[i, 0, :, :], (0,0), fx=params['s'], fy=params['s']) for i in xrange(x_val.shape[0])], 1))
        print('\033[92m'+'Done, Rescaling Patches'+'\033[0m')
        print('[New Data Shape]\t->\tX: '+str(x_train.shape))

    print 'x_shape is: ', x_train.shape
    print ('x min max is : '+ str(x_train.min())+' '+str(x_train.max()))
    filew.write('x_shape is : '+ str(x_train.shape)+'\n')
    filew.write('x min max is : '+ str(x_train.min())+' '+str(x_train.max())+'\n')
    
    listmodel=[name for name in os.listdir(patch_dir_store) if name.find('weights')==0]
#    model = get_model(x_train.shape, y_train.shape, params)
    

    if len(listmodel)>0:
         json_string1=pickle.load( open(os.path.join(patch_dir_store,modelname), "rb"))
         model = model_from_json(json_string1)
         learning_rate=1e-4
         filew.write ('learning rate:'+str(learning_rate)+'\n')
         print ('learning rate:'+str(learning_rate))
#         model.compile(optimizer=params['opt'], loss=get_Obj(params['obj']), metrics=['categorical_accuracy'])
         model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

         namelastc=load_model_set(patch_dir_store) 
         print 'load weight found from last training',namelastc
         filew.write('load weight found from last training\n'+namelastc+'\n')
#         model= load_model(namelastc)
         model.load_weights(namelastc)  



    else:
         print 'first training to be run'
         filew.write('first training to be run\n')
         model = get_model(x_train.shape, y_train.shape, params,filew)
    
    filew.write ('-----------------\n')
    nb_epoch_i_p=params['patience']

    # Open file to write the results
    rese=os.path.join(patch_dir_store,params['res_alias']+parameters_str+'.csv')

    print ('starting the loop of training with number of patience = ', params['patience'])
    t = datetime.datetime.now()
#    todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n' 
    todayn = str('m'+str(t.month)+'_d'+str(t.day)+'_y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'+'\n')
    today = str('m'+str(t.month)+'_d'+str(t.day)+'_y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m')
    filew.write ('starting the loop of training with number of patience = '+ str(params['patience'])+'\n')
    filew.write('started at :'+todayn)

    early_stopping=EarlyStopping(monitor='val_loss', patience=15, verbose=1,min_delta=0.005,mode='min')                     
    model_checkpoint = ModelCheckpoint(os.path.join(patch_dir_store,'weights_'+today+'.{epoch:02d}-{val_loss:.2f}.hdf5'), 
                                monitor='val_loss', save_best_only=True,save_weights_only=True)       
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                      patience=5, min_lr=1e-5,verbose=1)
    csv_logger = CSVLogger(rese,append=True)
    filew.close()
    model.fit(x_train, y_train, batch_size=500, epochs=nb_epoch_i_p, verbose =1,
                      validation_data=(x_val,y_val),
#                      class_weight=class_weights,
                      callbacks=[model_checkpoint,reduce_lr,csv_logger,early_stopping]  )
        # Evaluate models
    filew = open(eferror, 'a')
    
    print 'predict model'
    y_score = model.predict(x_val, batch_size=1050)

    fscore, acc, cm = H.evaluate(np.argmax(y_val, axis=1), np.argmax(y_score, axis=1))
    print('Val F-score: '+str(fscore)+'\tVal acc: '+str(acc))
    print cm

    print '---------------'
    t = datetime.datetime.now()
#    todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
    todayn = str('m'+str(t.month)+'_d'+str(t.day)+'_y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'+'\n')

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
    print ('------------------------')
    return model


def prediction(X_test, y_test, params):
    f=open ('../pickle/res.txt','w')
#    model = H.load_model()
    model= load_model('../pickle/ILD_CNN_model.h5')
    #sk modif
#    model.compile(optimizer='Adam', loss=get_Obj(params['obj']))

    y_classes = model.predict_classes(X_test, batch_size=100)
    y_val_subset = y_classes[:]
    y_test_subset = y_test[:]

    # argmax functions shows the index of the 1st occurence of the highest value in an array
#    y_actual = np.argmax(y_val_subset)
#    y_predict = np.argmax(y_test_subset)
    
    fscore, acc, cm = H.evaluate(y_test_subset, y_val_subset)

    print 'f-score is : ', fscore
    print 'accuray is : ', acc
    print 'confusion matrix'
    print cm
    f.write('f-score is : '+ str(fscore)+'\n')
    f.write( 'accuray is : '+ str(acc)+'\n')
    f.write('confusion matrix\n')
    n= cm.shape[0]
    for i in range (0,n):
        for j in range (0,n):
           f.write(str(cm[i][j])+' ')
        f.write('\n')
    f.close()
    open('../' + 'TestLog.csv', 'a').write(str(params['res_alias']) + ', ' + str(str(fscore) + ', ' + str(acc)+'\n'))
    return