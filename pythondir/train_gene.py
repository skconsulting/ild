# coding: utf-8
from param_pix_t import thrpatch,classif,hugeClass
#from keras.models import load_model
#from keras.models import model_from_json
import ild_helpers as H
import cnn_model_gene as CNN
import os
import cPickle as pickle
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
     'do' : float(args.do) if args.do else 0.5,        # Dropout Parameter default = 0.2
     'a'  : float(args.a) if args.a else 0.01,          # Conv Layers LeakyReLU default =0.01 alpha param [if alpha set to 0 LeakyReLU is equivalent with ReLU]
     'k'  : int(args.k) if args.k else 5,              # Feature maps k multiplier
     'ke'  : int(args.ke) if args.ke else 2,           # kernel size (ke,ke)
     's'  : float(args.s) if args.s else 1,            # Input Image rescale factor
     'pf' : float(args.pf) if args.pf else 1,          # Percentage of the pooling layer: [0,1]1
     'pt' : args.pt if args.pt else 'Avg',             # Pooling type: Avg, Max
     'fp' : args.fp if args.fp else 'proportional',    # Feature maps policy: proportional, static
     'cl' : int(args.cl) if args.cl else 5,            # Number of Convolutional Layers normal=5
     'opt': args.opt if args.opt else 'Adam',          # Optimizer: SGD, Adagrad, Adam
     'obj': args.obj if args.obj else 'ce',            # Minimization Objective: mse, ce
     'patience' : args.pat if args.pat else 200,       # Patience parameter for early stoping 200
     'tolerance': args.tol if args.tol else 1.005,     # Tolerance parameter for early stoping [default: 1.005, checks if > 0.5%]
     'res_alias': args.csv if args.csv else 'res' + str(today)     # csv results filename alias
#     'val_data': args.val if args.val else False     # validation data provided  (True) or 10% of training set (False)
}

topdir='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool'

#path with data for training
pickel_dirsource_root='pickle'
pickel_dirsource_e='train' #path for data fort training
pickel_dirsourcenum='set1' #extensioon for path for data for training
#extendir1='2'
extendir1='3'
#extendir2='3bm5'
extendir2='1b'

#modelarch='sk5'
modelarch='genova'
valset='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool/th0.95_pickle_val_set1_1_1b'
print 'validation path'
print valset
if not os.path.exists(valset):
    print 'valset doesnot exist: ',valset
    sys.exit()
actrain=True #put true to actually train, otherwise only predict
calnum=False # put True to calculate the number of images
#validation_split=0.1 #percentage of val if val_data=False
turn =1 #number of times total images*numclass

batch_size=600

#all in percent
maxshiftv=0
maxshifth=0
maxrot=7
maxresize=0
maxscaleint=0
maxmultint=20
        
dir_train='T'
#########################################################################################
paramaug={}
paramaug['maxshiftv']=maxshiftv
paramaug['maxshifth']=maxshifth
paramaug['maxrot']=maxrot
paramaug['maxshiftv']=maxshiftv
paramaug['maxresize']=maxresize
paramaug['maxscaleint']=maxscaleint
paramaug['maxmultint']=maxmultint


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
patch_dir_train=os.path.join(patch_dir,dir_train)

print 'patch dir source : ',patch_dir #path with data for training
print 'weight dir store : ',patch_dir_store #path with weights after training
print 'train dir  : ',patch_dir_train #path with  training data

if not os.path.exists(patch_dir_store):
    os.mkdir(patch_dir_store)
feature_train={}
classNumber={}
for f in classif:
    classNumber[f]=0
totalimages=0
for category in classif:
        dirpat=os.path.join(patch_dir_train,category)
        feature_train[category]=pickle.load( open( os.path.join(dirpat,"pat.pkl"), "rb" ))
        classNumber[category]=len(feature_train[category])
        if category not in hugeClass: totalimages=totalimages+classNumber[category]
for category in classif:
    print 'number of patches in train: ', category, classNumber[category]
print 'number total of images:',totalimages
trainSetSize=turn*totalimages*num_class
print 'number total of images for trainSetSize:',trainSetSize

print 'ratio with trainSetSize:',1.0*trainSetSize/(totalimages*num_class)
if calnum:
    sys.exit()

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


(X_val, y_val)= H.load_data_val(patch_dir,num_class)

# train a CNN model
model = CNN.train(X_val, y_val, train_params,eferror,patch_dir_store,valset,actrain,
                  modelarch,trainSetSize,feature_train,classNumber,paramaug,batch_size)

errorfile = open(eferror, 'a')
t = datetime.datetime.now()
today = str('m'+str(t.month)+'_d'+str(t.day)+'_y'+str(t.year)+'_'+str(t.hour)+'h_'+str(t.minute)+'m')
errorfile.write('training completed at '+today+'\n')
errorfile.close()

print 'training completed'

