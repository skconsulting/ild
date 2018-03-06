# -*- coding: utf-8 -*-
# debug
# from ipdb import set_trace as bp
from param_pix_t import modelname,learning_rate,fidclass
from param_pix_t import classif,hugeClass,toAug,generandom,geneaug,norm

import ild_helpers as H
import datetime
import cPickle as pickle
#import cv2
import os
import numpy as np
import random
import sys
#import h5py
import keras
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger,EarlyStopping
#from keras.models import model_from_json
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras import applications
#from keras import regularizers
from keras.utils import np_utils

from keras.layers import Input, concatenate,  Conv2DTranspose
#from keras.layers import UpSampling2D


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

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def vgg16_model(input_shape, output_shape, params,filew,patch_dir_store,numbits):
    num_class=output_shape[-1]
    dimpx=input_shape[-1]
    INP_SHAPE = (numbits, dimpx, dimpx)
    dim_org=keras.backend.image_data_format()
    """VGG 16 Model for Keras

    Model Schema is based on 
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

    ImageNet Pretrained Weights 
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of categories for our classification task
    """
    padding='same'
    kernel_size=(3,3)
    model = Sequential()
    model.add(Conv2D(64, kernel_size, activation='relu', padding=padding , data_format=dim_org,input_shape=INP_SHAPE))
    model.add(Conv2D(64, kernel_size, activation='relu', padding=padding,data_format=dim_org ))
    model.add(MaxPooling2D((2,2), strides=(2,2),data_format=dim_org))
    
    model.add(Conv2D(128, kernel_size, activation='relu', padding=padding,data_format=dim_org ))
    model.add(Conv2D(128, kernel_size, activation='relu', padding=padding,data_format=dim_org ))
    model.add(MaxPooling2D((2,2), strides=(2,2),data_format=dim_org))
    
    model.add(Conv2D(256, kernel_size, activation='relu', padding=padding,data_format=dim_org ))
    model.add(Conv2D(256, kernel_size, activation='relu', padding=padding,data_format=dim_org ))
    model.add(MaxPooling2D((2,2), strides=(2,2),data_format=dim_org))
    
    model.add(Conv2D(512, kernel_size, activation='relu', padding=padding,data_format=dim_org ))
    model.add(Conv2D(512, kernel_size, activation='relu', padding=padding,data_format=dim_org ))
    model.add(Conv2D(512, kernel_size, activation='relu', padding=padding,data_format=dim_org ))
    model.add(MaxPooling2D((2,2), strides=(2,2),data_format=dim_org))
    
    model.add(Conv2D(512, kernel_size, activation='relu', padding=padding,data_format=dim_org ))
    model.add(Conv2D(512, kernel_size, activation='relu', padding=padding,data_format=dim_org ))
    model.add(Conv2D(512, kernel_size, activation='relu', padding=padding,data_format=dim_org ))
#    model.add(MaxPooling2D((2,2), strides=(2,2),data_format=dim_org))
    model.add(AveragePooling2D(pool_size=(2,2),data_format=dim_org))
#    
    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    model.load_weights('vgg16_weights.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_class, activation='softmax'))

    for layer in model.layers[:-5]:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    filew.write ('learning rate:'+str(learning_rate)+'\n')
    print ('learning rate:'+str(learning_rate))
    json_string = model.to_json()
    pickle.dump(json_string, open(os.path.join(patch_dir_store,modelname), "wb"),protocol=-1)
    orig_stdout = sys.stdout
    f = open(os.path.join(patch_dir_store,'modelvgg16.txt'), 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()
    print model.layers[-1].output_shape #== (None, 16, 16, 21)

    return model




def vgg6net(input_shape, output_shape, params,filew,patch_dir_store,numbits):
    print 'input shape:',input_shape,'output shape:',output_shape
   
    num_class=output_shape[-1]
    dimpx=input_shape[-1]
    INP_SHAPE = (numbits, dimpx, dimpx)
    base_model = applications.VGG16(weights='imagenet', include_top=False, 
                                    input_shape=INP_SHAPE,
                                     pooling='avg',
                                    )
    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(256, activation='relu'))
    add_model.add(Dense(num_class, activation='sigmoid'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    
#    for layer in model.layers[:25]:
#        layer.trainable = False
    
    for layer in model.layers[:-5]:
        layer.trainable = False
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    filew.write ('learning rate:'+str(learning_rate)+'\n')
    print ('learning rate:'+str(learning_rate))
    json_string = model.to_json()
    pickle.dump(json_string, open(os.path.join(patch_dir_store,modelname), "wb"),protocol=-1)
    orig_stdout = sys.stdout
    f = open(os.path.join(patch_dir_store,'modelvgg16.txt'), 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()
    print model.layers[-1].output_shape #== (None, 16, 16, 21)
    return model
    
    
    
    

def get_modelunet(input_shape, output_shape, params,filew,patch_dir_store):
    num_class=output_shape[-1]
    
    dimpx=input_shape[-1]
    dim_org=keras.backend.image_data_format()
    INP_SHAPE = (1, dimpx, dimpx)
    kernel_size=(2,2)
    pool_siz=(2,2)
#    inputs = Input((img_rows, img_cols, 1))
    inputs = Input(INP_SHAPE)
    conv1 = Conv2D(32, kernel_size, activation='relu', padding='same',data_format=dim_org)(inputs)
    conv1 = Conv2D(32, kernel_size, activation='relu', padding='same',data_format=dim_org)(conv1)
    pool1 = MaxPooling2D(pool_size=pool_siz)(conv1)
    conv2 = Conv2D(64, kernel_size, activation='relu', padding='same',data_format=dim_org)(pool1)
    conv2 = Conv2D(64, kernel_size, activation='relu', padding='same',data_format=dim_org)(conv2)
    pool2 = MaxPooling2D(pool_size=pool_siz)(conv2)
    conv3 = Conv2D(128, kernel_size, activation='relu', padding='same',data_format=dim_org)(pool2)
    conv3 = Conv2D(128, kernel_size, activation='relu', padding='same',data_format=dim_org)(conv3)
    pool3 = MaxPooling2D(pool_size=pool_siz)(conv3)
    conv4 = Conv2D(256, kernel_size, activation='relu', padding='same',data_format=dim_org)(pool3)
    conv4 = Conv2D(256, kernel_size, activation='relu', padding='same',data_format=dim_org)(conv4)
    pool4 = MaxPooling2D(pool_size=pool_siz)(conv4)
    conv5 = Conv2D(512, kernel_size, activation='relu', padding='same',data_format=dim_org)(pool4)
    conv5 = Conv2D(512, kernel_size, activation='relu', padding='same',data_format=dim_org)(conv5)
#    up6 = concatenate([UpSampling2D(size=(2, 2),data_format=dim_org)(conv5), conv4],axis=1)
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='valid',data_format=dim_org)(conv5), conv4], axis=1)
    conv6 = Conv2D(256, kernel_size, activation='relu', padding='same',data_format=dim_org)(up6)
    conv6 = Conv2D(256, kernel_size, activation='relu', padding='same',data_format=dim_org)(conv6)
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid',data_format=dim_org)(conv6), conv3], axis=1)
#    up7 = concatenate([UpSampling2D(size=(2, 2),data_format=dim_org)(conv6), conv3], axis=1)

    conv7 = Conv2D(128, kernel_size, activation='relu', padding='same',data_format=dim_org)(up7)
    conv7 = Conv2D(128, kernel_size, activation='relu', padding='same',data_format=dim_org)(conv7)
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid',data_format=dim_org)(conv7), conv2], axis=1)
#    up8 = concatenate([UpSampling2D(size=(2, 2),data_format=dim_org)(conv7), conv2], axis=1)

    conv8 = Conv2D(64, kernel_size, activation='relu', padding='same',data_format=dim_org)(up8)
    conv8 = Conv2D(64, kernel_size, activation='relu', padding='same',data_format=dim_org)(conv8)
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='valid',data_format=dim_org)(conv8), conv1], axis=1)
#    up9 = concatenate([UpSampling2D(size=(2, 2),data_format=dim_org)(conv8), conv1], axis=1)

    conv9 = Conv2D(32, kernel_size, activation='relu', padding='same',data_format=dim_org)(up9)
    conv9 = Conv2D(32, kernel_size, activation='relu', padding='same',data_format=dim_org)(conv9)
#    conv10 = Conv2D(num_class, (1, 1), activation='relu',data_format=dim_org)(conv9)
    
    conv10 =Flatten()(conv9)
    conv11= Dense(1000)(conv10)
    conv12= LeakyReLU(alpha=params['a'])(conv11)
    conv13= Dropout(0.5)(conv12)     
    conv14=  Dense(num_class, activation='softmax')(conv13)

    model = Model(inputs=[inputs], outputs=[conv14])

#    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    filew.write ('learning rate:'+str(learning_rate)+'\n')
    print ('learning rate:'+str(learning_rate))
    json_string = model.to_json()
    pickle.dump(json_string, open(os.path.join(patch_dir_store,modelname), "wb"),protocol=-1)
    orig_stdout = sys.stdout
    f = open(os.path.join(patch_dir_store,'modelnexunet.txt'), 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()
    print model.layers[-1].output_shape #== (None, 16, 16, 21)
    return model


def get_modelsk5(output_shape,output_label, params,filew,patch_dir_store,namelastc,learning_rate):
    print 'output shape:',output_shape
    print 'output output_label:',output_label
   
    num_class=output_label[-1]
    print 'num_class',num_class
    
    dimpx=output_shape[-1]
    numbits= output_shape[1]

    INP_SHAPE = (numbits, dimpx, dimpx)
    print 'input shape:',INP_SHAPE
    dim_org=keras.backend.image_data_format()
    kernel_size=(3,3)
#    kernel_size=(4,4)

    pool_siz=(2,2)
#    paddingv='valid'#'same'
    paddingv='same'

    maxnormv=3
    startnum=32
    coef={}
    coef[0]=startnum
    print '0',coef[0]
    for i in range(1,4):
        coef[i]=coef[i-1]*2
        print i,coef[i]

    print 'sk5 with num_class :',num_class
    model = Sequential() 

    model.add(Conv2D(coef[0], kernel_size, input_shape=INP_SHAPE, padding=paddingv, 
                      data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))   
    model.add(LeakyReLU(alpha=params['a']))
    model.add(Conv2D(coef[0], kernel_size, input_shape=INP_SHAPE, padding=paddingv, 
                      data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))  
    model.add(LeakyReLU(alpha=params['a']))
    model.add(AveragePooling2D(pool_size=pool_siz,data_format=dim_org))
    model.add(Dropout(0.25)) 
    
    model.add(Conv2D(coef[1], kernel_size, padding=paddingv, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=params['a']))  
    model.add(Conv2D(coef[1], kernel_size, padding=paddingv, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=params['a']))  
    model.add(AveragePooling2D(pool_size=pool_siz,data_format=dim_org))
    model.add(Dropout(0.25)) 

    model.add(Conv2D(coef[2], kernel_size, input_shape=INP_SHAPE, padding=paddingv, 
                      data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))   
    model.add(LeakyReLU(alpha=params['a']))
    model.add(Conv2D(coef[2], kernel_size, input_shape=INP_SHAPE, padding=paddingv, 
                      data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))  
    model.add(LeakyReLU(alpha=params['a']))
    model.add(AveragePooling2D(pool_size=pool_siz,data_format=dim_org))
    model.add(Dropout(0.25)) 
    
    model.add(Conv2D(coef[3], kernel_size, padding=paddingv, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=params['a']))  
    model.add(Conv2D(coef[3], kernel_size, padding=paddingv, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=params['a']))  
    model.add(AveragePooling2D(pool_size=pool_siz,data_format=dim_org))
    model.add(Dropout(0.25)) 
    
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=params['a']))
    model.add(Dropout(0.3))
    
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=params['a']))
    model.add(Dropout(0.3)) 
    
    model.add(Dense(num_class, activation='softmax'))
#    model.add(Dense(num_class, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
#                activity_regularizer=regularizers.l1(0.01)))
#    
    if namelastc !='NAN':
        model.load_weights(namelastc)  
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    filew.write ('sk5 with num_class:'+str(num_class)+'\n')
    filew.write ('learning rate:'+str(learning_rate)+'\n')
    print ('learning rate:'+str(learning_rate))
    json_string = model.to_json()
    pickle.dump(json_string, open(os.path.join(patch_dir_store,modelname), "wb"),protocol=-1)
    orig_stdout = sys.stdout
    f = open(os.path.join(patch_dir_store,'modelsk5.txt'), 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()
    print model.layers[-1].output_shape 
    return model

def get_modelsk6(output_shape,output_label, params,filew,patch_dir_store,namelastc,learning_rate):
    print 'output shape:',output_shape
    print 'output output_label:',output_label
   
    num_class=output_label[-1]
    print 'num_class',num_class
    
    dimpx=output_shape[-1]
    numbits= output_shape[1]

    INP_SHAPE = (numbits, dimpx, dimpx)
    print 'input shape:',INP_SHAPE
    dim_org=keras.backend.image_data_format()
    kernel_size=(3, 3)
#    kernel_size=(4,4)
#    kernel_size2=(2,2)

    pool_siz=(2,2)
    paddingv='valid'#'same'
#    paddingv='same'

    maxnormv=3
    startnum=32
    coef={}
    coef[0]=startnum
    print '0',coef[0]
    for i in range(1,4):
        coef[i]=coef[i-1]*2
        print i,coef[i]

    print 'sk5 with num_class :',num_class
    model = Sequential() 
    #14L
    model.add(Conv2D(coef[0], kernel_size, input_shape=INP_SHAPE, padding=paddingv, 
                      data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))   
    model.add(LeakyReLU(alpha=params['a']))
    #12
    model.add(Conv2D(coef[0], kernel_size, input_shape=INP_SHAPE, padding=paddingv, 
                      data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))  
    model.add(LeakyReLU(alpha=params['a']))
#    model.add(AveragePooling2D(pool_size=pool_siz,data_format=dim_org))
    model.add(Dropout(0.25)) 
    #10
    model.add(Conv2D(coef[1], kernel_size, padding=paddingv, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=params['a']))  
    #8
    model.add(Conv2D(coef[1], kernel_size, padding=paddingv, 
                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
    model.add(LeakyReLU(alpha=params['a']))  
#    model.add(AveragePooling2D(pool_size=pool_siz,data_format=dim_org))
    model.add(Dropout(0.25)) 
    #6

    model.add(Conv2D(coef[2], kernel_size, input_shape=INP_SHAPE, padding=paddingv, 
                      data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))   
    model.add(LeakyReLU(alpha=params['a']))
#    model.add(Conv2D(coef[2], kernel_size, input_shape=INP_SHAPE, padding=paddingv, 
#                      data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))  
#    model.add(LeakyReLU(alpha=params['a']))
    model.add(AveragePooling2D(pool_size=pool_siz,data_format=dim_org))
#    model.add(Dropout(0.25)) 
    #2
#    model.add(Conv2D(coef[3], kernel_size, padding=paddingv, 
#                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
#    model.add(LeakyReLU(alpha=params['a']))  
#    model.add(Conv2D(coef[3], kernel_size, padding=paddingv, 
#                data_format=dim_org, kernel_constraint=maxnorm(maxnormv)))
#    model.add(LeakyReLU(alpha=params['a']))  
#    model.add(AveragePooling2D(pool_size=pool_siz,data_format=dim_org))
#    model.add(Dropout(0.25)) 
    
    model.add(Flatten())
    model.add(Dense(1152))
    model.add(LeakyReLU(alpha=params['a']))
    model.add(Dropout(0.3))
#    
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=params['a']))
#    model.add(Dropout(0.3)) 
    
    model.add(Dense(num_class, activation='softmax'))
#    model.add(Dense(num_class, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
#                activity_regularizer=regularizers.l1(0.01)))
#    
    if namelastc !='NAN':
        model.load_weights(namelastc)  
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    filew.write ('sk6 with num_class:'+str(num_class)+'\n')
    filew.write ('learning rate:'+str(learning_rate)+'\n')
    print ('learning rate:'+str(learning_rate))
    json_string = model.to_json()
    pickle.dump(json_string, open(os.path.join(patch_dir_store,modelname), "wb"),protocol=-1)
    orig_stdout = sys.stdout
    f = open(os.path.join(patch_dir_store,'modelsk6.txt'), 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()
    print model.layers[-1].output_shape 
    return model


def get_model_gen(input_shape, output_shape, params,filew,patch_dir_store,namelastc,learning_rate,parameters_str):

    print('compiling model...')

    kern_size=(params['ke'],params['ke'])
#    kern_size=(2,2)    
    # Dimension of The last Convolutional Feature Map (eg. if input 32x32 and there are 5 conv layers 2x2 fm_size = 27)
    fm_size = input_shape[-1] - (params['cl']*(kern_size[0]-1))

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
    model.add(Conv2D(params['k']*get_FeatureMaps(1, params['fp']), kern_size, 
                     kernel_initializer='orthogonal', input_shape=input_shape[1:],
                     kernel_constraint=maxnorm(3)))
    # model.add(Activation('relu'))
    #SK
    model.add(LeakyReLU(alpha=params['a']))
#    model.add(PReLU(init='zero', weights=None))
    print 'Layer 1 parameters settings:'
    print 'number of filters to be used : ', params['k']*get_FeatureMaps(1, params['fp'])
    print 'kernel size :', kern_size
    print 'input_shape of tensor is : ', input_shape[1:]


    for i in range(2, params['cl']+1):
        print i, params['k']*get_FeatureMaps(i, params['fp'])
        # model.add(Convolution2D(params['k']*get_FeatureMaps(i, params['fp']), 2, 2, init='orthogonal', activation=LeakyReLU(params['a'])))
        model.add(Conv2D(params['k']*get_FeatureMaps(i, params['fp']), 
                         kern_size, kernel_initializer='orthogonal',kernel_constraint=maxnorm(3)))
        # model.add(Activation('relu'))
        model.add(LeakyReLU(alpha=params['a']))
#        model.add(Dropout(params['do']))
        print 'Layer',  i, ' parameters settings:'
        print 'number of filters to be used : ', params['k']*get_FeatureMaps(i, params['fp'])
        print 'kernel size :',kern_size 


    # Add Pooling and Flatten layers to model
    print 'entering 2D Pooling layer'
    print 'Pooling : ', params['pt']
    print 'pool_size : ', pool_siz
#    pool_siz=(2,2) 
    if params['pt'] == 'Avg':
        model.add(AveragePooling2D(pool_size=pool_siz))
    elif params['pt'] == 'Max':
        model.add(MaxPooling2D(pool_size=pool_siz))
    else:
        sys.exit("Wrong type of Pooling layer")
#    print(model.summary())
#    ooo
    model.add(Flatten())

    # dropout is 50% or do=0.5
    model.add(Dropout(params['do']))
    print 'd0',params['do']

    # Add Dense layers and Output to model
    # model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp']))/params['pf']*6, init='he_uniform', activation=LeakyReLU(0)))
    print 'output_dimension : ', int(params['k']*get_FeatureMaps(params['cl'], params['fp']))/params['pf']*6
    model.add(Dense(int((params['k']*get_FeatureMaps(params['cl'], params['fp']))/params['pf']*6), 
                    kernel_initializer='he_uniform',kernel_constraint=maxnorm(3)))
#    model.add(Dense(864, 
#                    kernel_initializer='he_uniform',kernel_constraint=maxnorm(3)))
    
   
    
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=params['a']))
    model.add(Dropout(params['do']))

    
    # model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp']))/params['pf']*2, init='he_uniform', activation=LeakyReLU(0)))
    model.add(Dense(int(params['k']*get_FeatureMaps(params['cl'], params['fp']))/params['pf']*2, 
                    kernel_initializer='he_uniform',
                    kernel_constraint=maxnorm(3)))
    #  model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=params['a']))
    model.add(Dropout(params['do']))
    model.add(Dense(output_shape[1], kernel_initializer='he_uniform', activation='softmax',
                    kernel_constraint=maxnorm(3)))
  



#sk modif for decay, works only with ADAM

#    decay = 1.e-6
#    lr=0.001
#    lr =1.0
#    learning_rate=1e-4
    # Compile model and select optimizer and objective function
    if namelastc !='NAN':
        model.load_weights(namelastc)  
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    filew.write ('geneva with num_class:'+str(output_shape[1])+'\n')
    filew.write ('learning rate:'+str(learning_rate)+'\n')
    filew.write ('param string:'+parameters_str+'\n')
    print ('learning rate:'+str(learning_rate))
    json_string = model.to_json()
    pickle.dump(json_string, open(os.path.join(patch_dir_store,modelname), "wb"),protocol=-1)
    orig_stdout = sys.stdout
    f = open(os.path.join(patch_dir_store,'modelgeneva.txt'), 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()
    print model.layers[-1].output_shape #== (None, 16, 16, 21)
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

def  readclasses(pat,namepat,indexpat,scaleint,multint,rotimg,resiz,shiftv,shifth):
    scanr=namepat[indexpat]
    scanra=geneaug(scanr,scaleint,multint,rotimg,resiz,shiftv,shifth)
    scan=norm(scanra)
    mask=classif[pat]
    return scan, mask  

def gen_random_image(numclass,feature_train,indexpatc,classNumber):
    global numgen  
    numgen+=1
    pat=fidclass(numgen%numclass,classif)
    numberscan=classNumber[pat]
    if  pat in hugeClass:
        indexpat =  random.randint(0, numberscan-1)                       
    else:                                                   
        indexpat =  indexpatc[pat]%numberscan
        
    indexpatc[pat]=indexpat+1
    if pat in toAug:
        keepaenh=1
    else:
        keepaenh=0   
    scaleint,multint,rotimg,resiz,shiftv,shifth=generandom(maxscaleint,
                            maxmultint,maxrot,maxresize,maxshiftv,maxshifth,keepaenh)
    scan,mask=readclasses(pat,feature_train[pat],indexpat,scaleint,multint,rotimg,resiz,shiftv,shifth) 
    return scan,mask,indexpatc

def readclasses2(num_classes,X_testi,y_testi):

    X_test=np.array(X_testi)
    y_test=np.array(y_testi) 
    
    if len(X_test.shape)>3:
        X_test=np.moveaxis(X_test,3,1)
    else:
         X_test = np.expand_dims(X_test,1)  

    y_test = np_utils.to_categorical(y_test, num_classes)

    return  X_test,  y_test  


def batch_generator(batch_size,numclass,feature_train,indexpatc,classNumber):
    while True:
        image_list = []
        mask_list = []
        for i in range(batch_size):
            img, mask,indexpatc = gen_random_image(numclass,feature_train,indexpatc,classNumber)
            image_list.append(img)
            mask_list.append(mask)
        X_test,  ytest  =readclasses2(numclass,image_list,mask_list)
        yield  X_test,  ytest          


def train(x_val, y_val, params,eferror,patch_dir_store,valset,acttrain,modelarch,trainSetSize,
          feature_train,classNumber,paramaug,batch_size):
    ''' TODO: documentation '''
    global numgen,maxshiftv,maxshifth,maxrot,maxshiftv,maxresize,maxscaleint,maxmultint
    maxshiftv=paramaug['maxshiftv'] 
    maxshifth= paramaug['maxshifth']
    maxrot=paramaug['maxrot']
    maxshiftv=paramaug['maxshiftv']
    maxresize=paramaug['maxresize']
    maxscaleint=paramaug['maxscaleint']
    maxmultint=paramaug['maxmultint']
    filew = open(eferror, 'a')
    # Parameters String used for saving the files
    parameters_str = str('_d' + str(params['do']).replace('.', '') +
                         '_a' + str(params['a']).replace('.', '') + 
                         '_k' + str(params['k']) + 
                         '_cl' + str(params['cl']) + 
                         '_s' + str(params['s']).replace('.', '') + 
                         '_ke' + str(params['ke']) + 
                         '_pf' + str(params['pf']).replace('.', '') + 
                         '_pt' + params['pt'] +
                         '_fp' + str(params['fp']).replace('.', '') +
                         '_opt' + params['opt'] +
                         '_obj' + params['obj'])

    # Printing the parameters of the model
    print('[Dropout Param] \t->\t'+str(params['do']))
    print('[Alpha Param] \t\t->\t'+str(params['a']))
    print('[Multiplier] \t\t->\t'+str(params['k']))
    print('[numb conv] \t\t->\t'+str(params['cl']))
    print('[kernel size] \t\t->\t'+str(params['ke']))
    print('[Patience] \t\t->\t'+str(params['patience']))
    print('[Tolerance] \t\t->\t'+str(params['tolerance']))
    print('[Input Scale Factor] \t->\t'+str(params['s']))
    print('[Pooling Type] \t\t->\t'+ params['pt'])
    print('[Pooling Factor] \t->\t'+str(str(params['pf']*100)+'%'))
    print('[Feature Maps Policy] \t->\t'+ params['fp'])
    print('[Optimizer] \t\t->\t'+ params['opt'])
    print('[Objective] \t\t->\t'+ get_Obj(params['obj']))
    print('trainSetSize\t->\t'+str(trainSetSize))
    print('batch_size\t->\t'+str(batch_size))

       
    filew.write('[Dropout Param] \t->\t'+str(params['do'])+'\n')
    filew.write('[Alpha Param] \t\t->\t'+str(params['a'])+'\n')
    filew.write('[Multiplier] \t\t->\t'+str(params['k'])+'\n')
    filew.write('[numb conv] \t\t->\t'+str(params['cl'])+'\n')
    filew.write('[kernel size] \t\t->\t'+str(params['ke'])+'\n')
    filew.write('[Patience] \t\t->\t'+str(params['patience'])+'\n')
    filew.write('[Tolerance] \t\t->\t'+str(params['tolerance'])+'\n')
    filew.write('[Input Scale Factor] \t->\t'+str(params['s'])+'\n')
    filew.write('[Pooling Type] \t\t->\t'+ params['pt']+'\n')
    filew.write('[Pooling Factor] \t->\t'+str(str(params['pf']*100)+'%')+'\n')
    filew.write('[Feature Maps Policy] \t->\t'+ params['fp']+'\n')
    filew.write('[Optimizer] \t\t->\t'+ params['opt']+'\n')
    filew.write('[Objective] \t\t->\t'+ get_Obj(params['obj'])+'\n')
    filew.write('trainSetSize\t->\t'+str(trainSetSize)+'\n')
    filew.write('batch_size\t->\t'+str(batch_size)+'\n')


    print 'x_val is: ', x_val.shape
    
    listmodel=[name for name in os.listdir(patch_dir_store) if name.find('weights')==0] 
    if len(listmodel)>0:
         namelastc=load_model_set(patch_dir_store) 
         print 'load weight found from last training',namelastc
         filew.write('load weight found from last training\n'+namelastc+'\n')
         learning_ratef=learning_rate/10.

    else:
         print 'first training to be run'
         filew.write('first training to be run\n')
         namelastc='NAN'
         learning_ratef=learning_rate
    if modelarch=='sk5':
        model = get_modelsk5(x_val.shape,y_val.shape, params,filew,patch_dir_store,namelastc,learning_ratef)
    elif modelarch=='sk6':
        model = get_modelsk6(x_val.shape,y_val.shape, params,filew,patch_dir_store,namelastc,learning_ratef)
    elif modelarch== 'genova':
         model = get_model_gen(x_val.shape,y_val.shape, params,filew,patch_dir_store,namelastc,learning_ratef,parameters_str)
        
    filew.write ('-----------------\n')
    numclass=y_val.shape[1]
    if acttrain:
        numgen=-1 
        
        indexpatc={}
        for j in classif:
            indexpatc[j]=0
        nb_epoch_i_p=params['patience']

        rese=os.path.join(patch_dir_store,params['res_alias']+parameters_str+'.csv')
    
        print ('starting the loop of training with number of epochs = ', params['patience'])
        t = datetime.datetime.now()
    
        todayn = str('m'+str(t.month)+'_d'+str(t.day)+'_y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'+'\n')
        today = str('m'+str(t.month)+'_d'+str(t.day)+'_y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m')
        filew.write ('starting the loop of training with number of epochs = '+ str(params['patience'])+'\n')
        filew.write('started at :'+todayn)
    
        early_stopping=EarlyStopping(monitor='val_loss', patience=15, verbose=1,min_delta=0.01,mode='auto')                     
        model_checkpoint = ModelCheckpoint(os.path.join(patch_dir_store,'weights_'+today+'.{epoch:02d}-{val_loss:.3f}.hdf5'), 
                                    monitor='val_loss', save_best_only=True,save_weights_only=True,mode='auto')       
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-6,verbose=1)
        csv_logger = CSVLogger(rese,append=True)
        filew.close()

#        number_of_unique_sample= x_train.shape[0]
    
#        steps_per_epoch=number_of_unique_sample/batch_size
        history = model.fit_generator(
          generator=batch_generator(batch_size,numclass,feature_train,indexpatc,classNumber),
                epochs=nb_epoch_i_p,
                steps_per_epoch=trainSetSize//batch_size,
#                sample_weight=class_weights,
                validation_data=(x_val,y_val),
                verbose=2,
                callbacks=[model_checkpoint,reduce_lr,csv_logger,early_stopping] ,
                max_queue_size=2)  

    else:
        filew.write ('no training\n')
        filew.close()
        # Evaluate models
    
    
    filew = open(eferror, 'a')
    
    print 'predict model'
    namelastc=load_model_set(patch_dir_store) 
    print 'load weight found from last training',namelastc
    filew.write('load weight found from last training\n'+namelastc+'\n')
    model.load_weights(namelastc)
       
        #if validation data provided
    y_score = model.predict(x_val, batch_size=500)
    fscore, acc, cm = H.evaluate(np.argmax(y_val, axis=1), np.argmax(y_score, axis=1))
    print('Val F-score: '+str(fscore)+'\tVal acc: '+str(acc))
    print cm
    print '---------------'
    t = datetime.datetime.now()
    todayn = str('m'+str(t.month)+'_d'+str(t.day)+'_y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'+'\n')
    filew.write('  finished at :'+todayn)
    filew.write('  f-score is : '+ str(fscore)+'\n')
    filew.write('  accuray is : '+ str(acc)+'\n')
    filew.write('  confusion matrix\n')
    n= cm.shape[0]
    for cmi in range (0,n): 

#        filew.write('  ')
        for j in range (0,n):
            filew.write(str(cm[cmi][j])+' ')
        filew.write('\n')
    
    filew.write('------------------------------------------\n')
        
#    valset='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool/th0.95_pickle_val_set1_r0'
    print('validation set '+str(valset))
    filew.write('validation set '+str(valset)+'\n')
    (x_val, y_val)= H.load_data_val(valset, numclass)
    y_score = model.predict(x_val, batch_size=500)
    fscore, acc, cm = H.evaluate(np.argmax(y_val, axis=1), np.argmax(y_score, axis=1))
    print('Val F-score: '+str(fscore)+'\tVal acc: '+str(acc))
    print cm
    print '---------------'
    t = datetime.datetime.now()
    todayn = str('m'+str(t.month)+'_d'+str(t.day)+'_y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m'+'\n')
    filew.write('  finished at :'+todayn)
    filew.write('  f-score is : '+ str(fscore)+'\n')
    filew.write('  accuray is : '+ str(acc)+'\n')
    filew.write('  confusion matrix\n')
    n= cm.shape[0]
    for cmi in range (0,n): 

#        filew.write('  ')
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