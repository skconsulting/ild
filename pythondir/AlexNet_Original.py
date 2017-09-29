"""
    Model Name:

        AlexNet - using the Functional Keras API

        Replicated from the Original AlexNet Paper
        http://dandxy89.github.io/ImageModels/alexnet/

    Paper:

         ImageNet classification with deep convolutional neural networks by Krizhevsky et al. in NIPS 2012

    Alternative Example:

        Available at: http://caffe.berkeleyvision.org/model_zoo.html

        https://github.com/uoguelph-mlrg/theano_alexnet/tree/master/pretrained/alexnet

    Original Dataset:

        ILSVRC 2012

"""
from keras.layers import   MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout

from keras.models import Model
from keras import regularizers
from keras.layers import Input, concatenate ,multiply

from keras.layers.convolutional import Conv2D 
#from keras.utils.visualize_util import plot
from Custom_layers import LRN2D
import numpy as np

# global constants


def conv2D_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              activation='relu', batch_norm=True,
              weight_decay=0.0005, dim_ordering='channels_last'):
    '''

        Info:
            Function taken from the Inceptionv3.py script keras github


            Utility function to apply to a tensor a module conv + BN
            with optional weight decay (L2 weight regularization).
    '''
    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    
    
    x = Conv2D(nb_filter, (nb_row, nb_col),
                      strides=subsample,
                      activation=activation,
                      padding=border_mode,
                      kernel_regularizer=W_regularizer,
                      bias_regularizer=b_regularizer,
                      data_format=dim_ordering)(x)
    
    
    
    x = ZeroPadding2D(padding=(1, 1), data_format=dim_ordering)(x)

    if batch_norm:
        x = LRN2D()(x)
        x = ZeroPadding2D(padding=(1, 1), data_format=dim_ordering)(x)

    return x


def create_model(num_class,num_bit,image_rows,image_cols,DIM_ORDERING,WEIGHT_DECAY,USE_BN,DROPOUT):

    # Define image input layer
    if DIM_ORDERING == 'th':
        INP_SHAPE = (num_class, image_rows, image_cols)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 1
        dim_org='channels_first'
    elif DIM_ORDERING == 'tf':
        INP_SHAPE = (image_rows, image_cols, num_class)  # 3 - Number of RGB Colours 224
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 3
        dim_org='channels_last'
    else:
        raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))

    # Channel 1 - Conv Net Layer 1
    x = conv2D_bn(img_input, 3, 11, 11, subsample=(1, 1), border_mode='same',
                  dim_ordering=dim_org,weight_decay=WEIGHT_DECAY,batch_norm=USE_BN)
    x = MaxPooling2D(
        strides=(
            4, 4), pool_size=(
                4, 4), data_format=dim_org)(x)
    x = ZeroPadding2D(padding=(1, 1), data_format=dim_org)(x)

    # Channel 2 - Conv Net Layer 1
    y = conv2D_bn(img_input, 3, 11, 11, subsample=(1, 1), border_mode='same')
    y = MaxPooling2D(
        strides=(
            4, 4), pool_size=(
                4, 4), data_format=dim_org)(y)
    y = ZeroPadding2D(padding=(1, 1), data_format=dim_org)(y)

    # Channel 1 - Conv Net Layer 2
    x = conv2D_bn(x, 48, 55, 55, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), data_format=dim_org)(x)
    x = ZeroPadding2D(padding=(1, 1), data_format=dim_org)(x)

    # Channel 2 - Conv Net Layer 2
    y = conv2D_bn(y, 48, 55, 55, subsample=(1, 1), border_mode='same')
    y = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), data_format=dim_org)(y)
    y = ZeroPadding2D(padding=(1, 1), data_format=dim_org)(y)

    # Channel 1 - Conv Net Layer 3
    x = conv2D_bn(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), data_format=dim_org)(x)
    x = ZeroPadding2D(padding=(1, 1), data_format=dim_org)(x)

    # Channel 2 - Conv Net Layer 3
    y = conv2D_bn(y, 128, 27, 27, subsample=(1, 1), border_mode='same')
    y = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), data_format=dim_org)(y)
    y = ZeroPadding2D(padding=(1, 1), data_format=dim_org)(y)

    # Channel 1 - Conv Net Layer 4
    
#    x1 = merge([x, y], mode='concat', concat_axis=CONCAT_AXIS)
    x1 = concatenate([x, y], axis=CONCAT_AXIS)
    
    x1 = ZeroPadding2D(padding=(1, 1), data_format=dim_org)(x1)
    x1 = conv2D_bn(x1, 192, 13, 13, subsample=(1, 1), border_mode='same')

    # Channel 2 - Conv Net Layer 4
#    y1 = merge([x, y], mode='concat', concat_axis=CONCAT_AXIS)
    y1 = concatenate([x, y], axis=CONCAT_AXIS)
    
    
    y1 = ZeroPadding2D(padding=(1, 1), data_format=dim_org)(y1)
    y1 = conv2D_bn(y1, 192, 13, 13, subsample=(1, 1), border_mode='same')

    # Channel 1 - Conv Net Layer 5
#    x2 = merge([x1, y1], mode='concat', concat_axis=CONCAT_AXIS)
    x2 = concatenate([x1, y1], axis=CONCAT_AXIS)
    
    x2 = ZeroPadding2D(padding=(1, 1), data_format=dim_org)(x2)
    x2 = conv2D_bn(x2, 192, 13, 13, subsample=(1, 1), border_mode='same')

    # Channel 2 - Conv Net Layer 5
#    y2 = merge([x1, y1], mode='concat', concat_axis=CONCAT_AXIS)
    y2 = concatenate([x1, y1], axis=CONCAT_AXIS)

    y2 = ZeroPadding2D(padding=(1, 1), data_format=dim_org)(y2)
    y2 = conv2D_bn(y2, 192, 13, 13, subsample=(1, 1), border_mode='same')

    # Channel 1 - Cov Net Layer 6
    x3 = conv2D_bn(x2, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x3 = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), data_format=dim_org)(x3)
    x3 = ZeroPadding2D(padding=(1, 1), data_format=dim_org)(x3)

    # Channel 2 - Cov Net Layer 6
    y3 = conv2D_bn(y2, 128, 27, 27, subsample=(1, 1), border_mode='same')
    y3 = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), data_format=dim_org)(y3)
    y3 = ZeroPadding2D(padding=(1, 1), data_format=dim_org)(y3)

    # Channel 1 - Cov Net Layer 7
#    x4 = merge([x3, y3], mode='mul', concat_axis=CONCAT_AXIS)
    x4 = multiply([x3, y3])
    
    
    x4 = Flatten()(x4)
    x4 = Dense(2048, activation='relu')(x4)
    x4 = Dropout(DROPOUT)(x4)

    # Channel 2 - Cov Net Layer 7
#    y4 = merge([x3, y3], mode='mul', concat_axis=CONCAT_AXIS)
    y4 = multiply([x3, y3])
    
    y4 = Flatten()(y4)
    y4 = Dense(2048, activation='relu')(y4)
    y4 = Dropout(DROPOUT)(y4)

    # Channel 1 - Cov Net Layer 8
#    x5 = merge([x4, y4], mode='mul')
    x5 = multiply([x4, y4])
    
    x5 = Dense(2048, activation='relu')(x5)
    x5 = Dropout(DROPOUT)(x5)

    # Channel 2 - Cov Net Layer 8
#    y5 = merge([x4, y4], mode='mul')
    y5 = multiply([x4, y4])
    
    y5 = Dense(2048, activation='relu')(y5)
    y5 = Dropout(DROPOUT)(y5)

    # Final Channel - Cov Net 9
#    xy = merge([x5, y5], mode='mul')
    xy = multiply([x5, y5])
    xy = Dense(num_class,
               activation='softmax')(xy)

#    model = Model(input=img_input,
#                  output=[xy])
    model = Model(inputs=[img_input], outputs=[xy])
#    return xy, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING
    return model



def check_print():
    # Create the Model
    num_class = 1         # number of classes
    num_bit=1
#    LEARNING_RATE = 0.01
#    MOMENTUM = 0.9
#    GAMMA = 0.1
    DROPOUT = 0.5
    WEIGHT_DECAY = 0.0005   # L2 regularization factor
    USE_BN = True           # whether to use batch normalization
# Theano - 'th' (channels, width, height)
    # Tensorflow - 'tf' (width, height, channels)
    DIM_ORDERING = 'tf'
#    if DIM_ORDERING == 'th':
#            dim_org='channels_first'
#    elif DIM_ORDERING == 'tf':
#            dim_org='channels_last'

    image_size=100


#    model=get_model(num_class,num_bit,image_size,image_size,False,weights)
    model = create_model(num_class,num_bit,image_size,image_size,DIM_ORDERING,WEIGHT_DECAY,USE_BN,DROPOUT)

    # Create a Keras Model - Functional A
    model.summary()

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')
    print('Model Compiled')
#   model=get_model(num_class,num_bit,image_size,image_size,False,weights)
    imarr = np.ones((image_size,image_size,num_bit))
    imarr = np.expand_dims(imarr, axis=0)
    print 'imarr.shape',imarr.shape
    print 'model.predict(imarr).shape ',model.predict(imarr,verbose=1).shape

    # Save a PNG of the Model Build
#    plot(model, to_file='./Model/AlexNet_Original.png')


    
if __name__ == "__main__":
     check_print()