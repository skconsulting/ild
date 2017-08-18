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
from param_pix_t import modelname,thrpatch
#from keras.models import load_model
#from keras.models import model_from_json
import ild_helpers as H
import cnn_model as CNN
import os
import cPickle as pickle

import datetime
t = datetime.datetime.now()
today = str('_m'+str(t.month)+'_d'+str(t.day)+'_y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m')
ptrainfile='trainlog'+today+'.txt'
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
     'patience' : args.pat if args.pat else 200
     ,       # Patience parameter for early stoping
     'tolerance': args.tol if args.tol else 1.005,     # Tolerance parameter for early stoping [default: 1.005, checks if > 0.5%]
     'res_alias': args.csv if args.csv else 'res' + str(today)     # csv results filename alias
}


topdir='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool'

#path with data for training
pickel_dirsource_root='pickle'
pickel_dirsource_e='train_set' #path for data fort training
pickel_dirsourcenum='0p' #extensioon for path for data for training
extendir2=''
#########################################################################################

pickleStore='pickle'
if len (extendir2)>0:
    extendir2='_'+extendir2

pickel_dirsource='th'+str(thrpatch)+'_'+pickel_dirsource_root+'_'+pickel_dirsource_e+'_'+pickel_dirsourcenum+extendir2

patch_dir=os.path.join(topdir,pickel_dirsource)
patch_dir_store=os.path.join(patch_dir,pickleStore)

print 'patch dir source : ',patch_dir #path with data for training
print 'weight dir store : ',patch_dir_store #path with weights after training
if not os.path.exists(patch_dir_store):
    os.mkdir(patch_dir_store)


eferror=os.path.join(patch_dir_store,ptrainfile)
errorfile = open(eferror, 'w')
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
errorfile.write('started ' +pickel_dirsource+' at :'+todayn)
errorfile.write('--------------------\n')
errorfile.write('patch dir source : '+patch_dir+'\n') #path with data for training
errorfile.write('weight dir store : '+patch_dir_store+'\n') #path with weights after training
errorfile.write('--------------------\n')
errorfile.close()

(X_train, y_train), (X_val, y_val)= H.load_data(patch_dir)

# train a CNN model

model = CNN.train(X_train, y_train, X_val, y_val, train_params,eferror,patch_dir_store)

errorfile = open(eferror, 'a')
t = datetime.datetime.now()
today = str('m'+str(t.month)+'_d'+str(t.day)+'_y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m')
errorfile.write('training completed at '+today+'\n')
errorfile.close()

json_string = model.to_json()
pickle.dump(json_string, open(os.path.join(patch_dir_store,modelname), "wb"),protocol=-1)

print 'training completed'

