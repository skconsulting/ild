# coding: utf-8
from param_pix_t import thrpatch,classif
#from keras.models import load_model
#from keras.models import model_from_json
import ild_helpers as H
import cnn_model as CNN
import os
import keras
from keras import backend as K
K.set_image_dim_ordering('th')
print 'NEW keras.backend.image_data_format :',keras.backend.image_data_format()
# channel first = theano = samples, channels, rows, cols). 
import sys
import datetime
t = datetime.datetime.now()
today = str('_m'+str(t.month)+'_d'+str(t.day)+'_y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m')
ptrainfile='trainlog'+today+'.txt'
# debug
#from ipdb import set_trace as bp

# initialization
args         = H.parse_args()                          # Function for parsing command-line arguments
train_params = {
     'do' : float(args.do) if args.do else 0.2,        # Dropout Parameter default = 0.2
     'a'  : float(args.a) if args.a else 0.01,          # Conv Layers LeakyReLU default =0.01 alpha param [if alpha set to 0 LeakyReLU is equivalent with ReLU]
     'k'  : int(args.k) if args.k else 4,              # Feature maps k multiplier
     's'  : float(args.s) if args.s else 1,            # Input Image rescale factor
     'pf' : float(args.pf) if args.pf else 1,          # Percentage of the pooling layer: [0,1]1
     'pt' : args.pt if args.pt else 'Avg',             # Pooling type: Avg, Max
     'fp' : args.fp if args.fp else 'proportional',    # Feature maps policy: proportional, static
     'cl' : int(args.cl) if args.cl else 4,            # Number of Convolutional Layers normal=5
     'opt': args.opt if args.opt else 'Adam',          # Optimizer: SGD, Adagrad, Adam
     'obj': args.obj if args.obj else 'ce',            # Minimization Objective: mse, ce
     'patience' : args.pat if args.pat else 200,       # Patience parameter for early stoping 200
     'tolerance': args.tol if args.tol else 1.005,     # Tolerance parameter for early stoping [default: 1.005, checks if > 0.5%]
     'res_alias': args.csv if args.csv else 'res' + str(today),     # csv results filename alias
     'val_data': args.val if args.val else True     # validation data provided  (True) or 10% of training set (False)
}

topdir='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool'

#path with data for training
pickel_dirsource_root='pickle'
pickel_dirsource_e='train' #path for data fort training
pickel_dirsourcenum='set1' #extensioon for path for data for training
#extendir1='2'
extendir1='1'

extendir2=''
nbits=1

valset='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool/th0.95_pickle_val_set1_3b'
print 'validation path'
print valset
if not os.path.exists(valset):
    print 'valset doesnot exist: ',valset
    sys.exit()
actrain=True #put true to actually train, otherwise only predict

#########################################################################################
num_class= len(classif)
print 'number of classes', num_class

pickleStore='pickle'
if len (extendir2)>0:
    extendir2='_'+extendir2

pickel_dirsource='th'+str(thrpatch)+'_'+pickel_dirsource_root+'_'+pickel_dirsource_e+'_'+pickel_dirsourcenum+'_'+extendir1+extendir2
patch_dir=os.path.join(topdir,pickel_dirsource)
if not os.path.exists(patch_dir):
    print 'pickel_dirsource doesnot exist: ',patch_dir
    sys.exit()
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
errorfile.write( 'number of classes'+str(num_class)+'\n')
errorfile.write('patch dir source : '+patch_dir+'\n') #path with data for training
errorfile.write('weight dir store : '+patch_dir_store+'\n') #path with weights after training
errorfile.write('--------------------\n')
errorfile.close()

if train_params['val_data']:
    (X_train, y_train), (X_val, y_val)= H.load_data(patch_dir,num_class,nbits)

else:
     (X_train, y_train), (X_val, y_val)= H.load_data_train(patch_dir,num_class,nbits)

# train a CNN model
model = CNN.train(X_train, y_train, X_val, y_val, train_params,eferror,patch_dir_store,valset,actrain,nbits)

errorfile = open(eferror, 'a')
t = datetime.datetime.now()
today = str('m'+str(t.month)+'_d'+str(t.day)+'_y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m')
errorfile.write('training completed at '+today+'\n')
errorfile.close()

print 'training completed'

