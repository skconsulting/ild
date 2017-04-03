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
'''

import argparse
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils 
import sklearn.metrics as metrics
import cPickle as pickle
import sys
from keras.utils import np_utils 

from keras.models import model_from_json
from keras.models import load_model


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
    args = parser.parse_args()

    return args

def load_data():

    # load the dataset as X_train and as a copy the X_val
    X_train = pickle.load( open( "../pickle/X_train.pkl", "rb" ) )
    y_train = pickle.load( open( "../pickle/y_train.pkl", "rb" ) )
    X_val = pickle.load( open( "../pickle/X_val.pkl", "rb" ) )
    y_val = pickle.load( open( "../pickle/y_val.pkl", "rb" ) )

    

    # adding a singleton dimension and rescale to [0,1]
    X_train = np.asarray(np.expand_dims(X_train,1))/float(255)
    X_val = np.asarray(np.expand_dims(X_val,1))/float(255)

    # labels to categorical vectors
    uniquelbls = np.unique(y_train)
    nb_classes = uniquelbls.shape[0]
    zbn = np.min(uniquelbls) # zero based numbering
    y_train = np_utils.to_categorical(y_train-zbn, nb_classes)
    y_val = np_utils.to_categorical(y_val-zbn, nb_classes)

    return (X_train, y_train), (X_val, y_val)

def load_testdata(pathpickle,tmean,imageDepth):

    # load the dataset as X_train and as a copy the X_val
    X_test = pickle.load( open( pathpickle+"/X_val.pkl", "rb" ) )
    y_test = pickle.load( open(pathpickle+ "/y_val.pkl", "rb" ) )

    # adding a singleton dimension and rescale to [0,1]
    X_test = np.asarray(np.expand_dims(X_test,1))/float(imageDepth)
#    t=np.mean(X_test1)
#    print 'actual mean of Xtest :',t
#    print 'allocated  mean of Xtest :',tmean
#    
#    X_test=X_test1-t

    # labels to categorical vectors
#    uniquelbls = np.unique(y_test)
#    nb_classes = uniquelbls.shape[0]
#    zbn = np.min(uniquelbls) # zero based numbering
##    # only used to make fscore,cm, acc calculation, single dimension required
#    y_test = np_utils.to_categorical(y_test - zbn, nb_classes)
#    

    return (X_test, y_test)

def evaluate(actual,pred):
    fscore = metrics.f1_score(actual, pred, average='macro')
    acc = metrics.accuracy_score(actual, pred)
    cm = metrics.confusion_matrix(actual,pred)

    return fscore, acc, cm

def store_model(model):
    json_string = model.to_json()
    open('../pickle/ILD_CNN_model.json', 'w').write(json_string)
    model.save_weights('../pickle/ILD_CNN_model_weights',overwrite=True)

    return json_string

#def load_model(pfile):
#    '../pickle/ILD_CNN_model.h5'
#    model = model_from_json(open(pfile+'/ILD_CNN_model.json').read())
#    model.load_weights(pfile+'/ILD_CNN_model_weights')
#
#    return model


