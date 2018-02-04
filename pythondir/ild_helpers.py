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
import argparse
import cPickle as pickle
import numpy as np
import os
import sys
from keras.utils import np_utils
import sklearn.metrics as metrics
# debug
# from ipdb import set_trace as bp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-do',  help='Dropout param [default: 0.5]')
    parser.add_argument('-a',   help='Conv Layers LeakyReLU alpha param [if alpha set to 0 LeakyReLU is equivalent with ReLU] [default: 0.3]')
    parser.add_argument('-k',   help='Feature maps k multiplier [default: 4]')
    parser.add_argument('-cl',  help='Number of Convolutional Layers [default: 5]')
    parser.add_argument('-s',   help='Input Image rescale factor [default: 1]')
    parser.add_argument('-pf',  help='Percentage of the pooling layer: [0,1] [default: 1]')
    parser.add_argument('-pt',  help='Pooling type: \'Avg\', \'Max\' [default: Avg]')
    parser.add_argument('-fp',  help='Feature maps policy: \'proportional\',\'static\' [default: proportional]')
    parser.add_argument('-opt', help='Optimizer: \'SGD\',\'Adagrad\',\'Adam\' [default: Adam]')
    parser.add_argument('-obj', help='Minimization Objective: \'mse\',\'ce\' [default: ce]')
    parser.add_argument('-pat', help='Patience parameter for early stoping [default: 200]')
    parser.add_argument('-tol', help='Tolerance parameter for early stoping [default: 1.005]')
    parser.add_argument('-csv', help='csv results filename alias [default: res]')
    parser.add_argument('-val', help='val data use[default: TRUE]')
    args = parser.parse_args()

    return args

def load_data(dirp,nb_classes):

    # load the dataset as X_train and as a copy the X_val
    X_train = pickle.load( open( os.path.join(dirp,'X_train.pkl'), "rb" ) )  

    y_train = pickle.load( open(os.path.join(dirp,'y_train.pkl'), "rb" ) )
    X_val = pickle.load( open( os.path.join(dirp,'X_val.pkl'), "rb" ) )
    y_val = pickle.load( open( os.path.join(dirp,'y_val.pkl'), "rb" ) )
    if len(X_train.shape)>3:

        X_train=np.moveaxis(X_train,3,1)
        X_val=np.moveaxis(X_val,3,1)
    else:
         X_train = np.expand_dims(X_train,1)  
         X_val =np.expand_dims(X_val,1)
    print ('Xtrain :',X_train.shape)
    print 'X_train min max :',X_train.min(),X_train.max()

    # labels to categorical vectors
#    uniquelbls = np.unique(y_train)
#    nb_classes = int( uniquelbls.shape[0])
    print ('number of classes :', int(nb_classes)) 
  
#    zbn = np.min(uniquelbls) # zero based numbering
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_val = np_utils.to_categorical(y_val, nb_classes)

    return (X_train, y_train), (X_val, y_val)

def load_data_train(dirp,nb_classes):

    # load the dataset as X_train and as a copy the X_val
    X_train = pickle.load( open( os.path.join(dirp,'X_train.pkl'), "rb" ) )

    y_train = pickle.load( open(os.path.join(dirp,'y_train.pkl'), "rb" ) )
      
    if len(X_train.shape)>3:

        X_train=np.moveaxis(X_train,3,1)
    else:
         X_train = np.expand_dims(X_train,1)  
    
    print ('Xtrain :',X_train.shape)
    print 'X_train min max :',X_train.min(),X_train.max()

    # labels to categorical vectors
#    uniquelbls = np.unique(y_train)
#    nb_classes = int( uniquelbls.shape[0])
    print ('number of classes :', int(nb_classes)) 
#    zbn = np.min(uniquelbls) # zero based numbering
    y_train = np_utils.to_categorical(y_train, nb_classes)
    X_val=[]
    y_val =[]
    return (X_train, y_train), (X_val, y_val)
#    return (X_train, y_train), (X_val, y_val),class_weights
    
def load_data_val(dirp,nb_classes):

    X_val = pickle.load( open( os.path.join(dirp,'X_val.pkl'), "rb" ) )
    y_val = pickle.load( open( os.path.join(dirp,'y_val.pkl'), "rb" ) )
    if len(X_val.shape)>3:
        X_val=np.moveaxis(X_val,3,1)
    else:
         X_val = np.expand_dims(X_val,1)  

    # labels to categorical vectors
#    uniquelbls = np.unique(y_val)
#    nb_classes = int( uniquelbls.shape[0])
    print ('number of classes :', int(nb_classes)) 
  
#    zbn = np.min(uniquelbls) # zero based numbering
    y_val = np_utils.to_categorical(y_val, nb_classes)

    return  (X_val, y_val)

def evaluate(actual,pred):
    fscore = metrics.f1_score(actual, pred, average='macro')
    acc = metrics.accuracy_score(actual, pred)
    cm = metrics.confusion_matrix(actual,pred)

    return fscore, acc, cm

    
def store_model(model):

#    open('../pickle/ILD_CNN_model.json', 'w').write(json_string)
    model.save('../pickle/ILD_CNN_model.h5')
    
