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

import ild_helperstest as H
import cnn_modeltest as CNN
import os
import keras
import theano
print keras.__version__
print theano.__version__


import datetime
t = datetime.datetime.now()
today = str('_'+str(t.month)+'-'+str(t.day)+'-'+str(t.year)+'_'+str(t.hour)+'_'+str(t.minute))

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
#print cwd

namedirtop='pickle_ex/pickle_ex77'
namedirtop='pickle'
#namemodel='pickle_ex/pickle_ex63'
namemodel='pickle_ex/pickle_ex77'
namemodel='pickle'
pfile = os.path.join(cwdtop,namedirtop)
pmodel = os.path.join(cwdtop,namemodel)
print pfile, pmodel

# debug
#from ipdb import set_trace as bp

# initialization
args         = H.parse_args()                          # Function for parsing command-line arguments
train_params = {
     'do' : float(args.do) if args.do else 0.5,        # Dropout Parameter
     'a'  : float(args.a) if args.a else 0.3,          # Conv Layers LeakyReLU alpha param [if alpha set to 0 LeakyReLU is equivalent with ReLU]
     'k'  : int(args.k) if args.k else 4,              # Feature maps k multiplier
     's'  : float(args.s) if args.s else 1,            # Input Image rescale factor
     'pf' : float(args.pf) if args.pf else 1,          # Percentage of the pooling layer: [0,1]
     'pt' : args.pt if args.pt else 'Avg',             # Pooling type: Avg, Max
     'fp' : args.fp if args.fp else 'proportional',    # Feature maps policy: proportional, static
     'cl' : int(args.cl) if args.cl else 5,            # Number of Convolutional Layers
     'opt': args.opt if args.opt else 'Adam',          # Optimizer: SGD, Adagrad, Adam
     'obj': args.obj if args.obj else 'ce',            # Minimization Objective: mse, ce
     'patience' : args.pat if args.pat else 100
     ,       # Patience parameter for early stoping
     'tolerance': args.tol if args.tol else 1.005,     # Tolerance parameter for early stoping [default: 1.005, checks if > 0.5%]
     'res_alias': args.csv if args.csv else 'res' + str(today)     # csv results filename alias
}
#
## loading patch data
#(X_train, y_train), (X_val, y_val) = H.load_data()
#
## train a CNN model
#
#model = CNN.train(X_train, y_train, X_val, y_val, train_params)
#
## store the model and weights
#H.store_model(model)
#
#print 'training completed'
print 'loading test set'
#imageDepth=65535 #number of bits used on dicom images (2 **n) 13 bits
imageDepth=8191 #number of bits used on dicom images (2 **n) 13 bits
tmean=0.371
# load test data set 
(X_test, y_test) = H.load_testdata(pmodel,tmean,imageDepth)

# predict with test dataset and record results
pred = CNN.prediction(X_test, y_test, train_params,pfile)

print 'assessment with test set completed'





