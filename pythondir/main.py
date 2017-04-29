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

import ild_helpers as H
import cnn_model as CNN
import h5py
import datetime
t = datetime.datetime.now()
today = str('_'+str(t.month)+'-'+str(t.day)+'-'+str(t.year)+'_'+str(t.hour)+'_'+str(t.minute))



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
     'cl' : int(args.cl) if args.cl else 5,            # Number of Convolutional Layers normal=5
     'opt': args.opt if args.opt else 'Adam',          # Optimizer: SGD, Adagrad, Adam
     'obj': args.obj if args.obj else 'ce',            # Minimization Objective: mse, ce
     'patience' : args.pat if args.pat else 100
     ,       # Patience parameter for early stoping
     'tolerance': args.tol if args.tol else 1.005,     # Tolerance parameter for early stoping [default: 1.005, checks if > 0.5%]
     'res_alias': args.csv if args.csv else 'res' + str(today)     # csv results filename alias
}

#imageDepth=8191 #number of bits used on dicom images (2 **n) 13 bits
imageDepth=28000 #for dct min dct=-12131 maxdct=27282 on HUG  '16_set1_13b2'

#imageDepth=65535 #number of bits used on dicom images (2 **n) 13 bits
#class_weights= (1.0, 26.5, 1.0,
#                 2.27, 1.0,  1.04,  2.27,
#                 180.5, 5.52,  84.7)
# loading patch data
(X_train, y_train), (X_val, y_val),class_weights= H.load_data(imageDepth)

# train a CNN model
print 'class weights',class_weights
model = CNN.train(X_train, y_train, X_val, y_val, train_params,class_weights)

# store the model and weights
H.store_model(model)

print 'training completed'
print 'loading test set'

# load test data set 
(X_test, y_test) = H.load_testdata(imageDepth)

# predict with test dataset and record results
pred = CNN.prediction(X_test, y_test, train_params)

print 'assessment with test set completed'




