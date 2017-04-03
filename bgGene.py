# coding: utf-8
'''prediction on back-ground only, generate filtered BG for next training
Sylvain kritter 6 february 2017
 '''
import os
import cv2
#import dircache
import sys
import shutil
from scipy import misc
import numpy as np
import keras
import cPickle as pickle

import ild_helpers as H

from Tkinter import *
os.environ['KERAS_BACKEND'] = 'theano'
#from keras.models import model_from_json
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model
import cnn_model as CNN4


def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)

#####################################################################
#define the working directory
HUG='CHU'   
subDir='TOPPATCH'
#subDir='T'
#subDir='patch'
#extent='T'
extent='3d162'
#extent='bg'
subDirc=subDir+'_'+extent
#pickel_dirsource='pickle_ds48'
patch_dest='patch_bg'+'_'+extent+'_1'
label='filter'
thrproba =0.6 #thresholm proba for generation of predicted images
name_dir_patch='patches'
#imageDepth=65535 #number of bits used on dicom images (2 **n) 13 bits
imageDepth=8191 #number of bits used on dicom images (2 **n) 13 bits
#output pickle dir with dataset   
#augf=12
#typei='bmp' #can be jpg
typeid='png' #can be jpg

subdirc=subDir+'_'+extent
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
dirHUG=os.path.join(cwdtop,HUG)

patch_destdir=os.path.join(dirHUG,patch_dest)
if not os.path.exists(patch_destdir):
    os.mkdir(patch_destdir)  
dirHUG=os.path.join(dirHUG,subDirc)

patch_destdir=os.path.join(patch_destdir,name_dir_patch)
if not os.path.exists(patch_destdir):
    os.mkdir(patch_destdir)  
patch_destdir=os.path.join(patch_destdir,'back_ground')
if not os.path.exists(patch_destdir):
    os.mkdir(patch_destdir)  
patch_destdir=os.path.join(patch_destdir,HUG+'_'+label)
if not os.path.exists(patch_destdir):
    os.mkdir(patch_destdir)  
 

print 'dirHUG', dirHUG,patch_destdir


#input pickle dir with dataset to merge
#pickel_dirsourceToMerge='pickle_ds22'


#input patch directory
#patch_dirsource=os.path.join(dirHUG,'patches_norm_16_set0_1')
patch_dirsource=os.path.join(dirHUG,name_dir_patch)
patch_dirsource=os.path.join(patch_dirsource,'back_ground')

print 'patch_dirsource',patch_dirsource
#output database for database generation
#patch_dirSplitsource=   'b'
#augmentation factor

picklein_file = '../pickle_ex/pickle_ex69'
modelname='ILD_CNN_model.h5'

###############################################################
classif ={
        'back_ground':0,
        'consolidation':1,
        'HC':2,
        'ground_glass':3,
        'healthy':4,
        'micronodules':5,
        'reticulation':6,
        'air_trapping':7,
        'cysts':8,
        'bronchiectasis':9,
        'emphysema':10,
        'GGpret':11
            }

clssifcount={
            'back_ground':0,
            'consolidation':0,
            'HC':0,
            'ground_glass':0,
            'healthy':0,
            'micronodules':0,
            'reticulation':0,
            'air_trapping':0,
            'cysts':0,
            'bronchiectasis':0,
            'emphysema':0,
        'GGpret':0
            }
#pickle_dir=os.path.join(cwdtop,pickel_dirsource) 
#pickle_dirToMerge=os.path.join(cwdtop,pickel_dirsourceToMerge)  

#patch_dir=os.path.join(dirHUG,patch_dirsource)
#print patch_dir
print classif

#remove_folder(pickle_dir)
#os.mkdir(pickle_dir)  



def listcl():
    print 'list images'
    dlist=[]
    print 'patch_dirsource',patch_dirsource
    category_dir_list= os.listdir(patch_dirsource)
    print 'category_dir_list',category_dir_list
    for i in category_dir_list:
        print i
        category_dir=os.path.join(patch_dirsource,i)
        image_files = [name for name in os.listdir(category_dir) if  name.find('.'+typeid) > 0]       
        for filei in image_files:
            image=cv2.imread(os.path.join(category_dir,filei),-1)
            dlist.append(image)                 
    return dlist

def maxproba(proba):
    """looks for max probability in result"""
    lenp = len(proba)
    m=0
    for i in range(0,lenp):
        if proba[i]>m:
            m=proba[i]
            im=i
    return im,m
    
def fidclass(numero,classn):
    """return class from number"""
    found=False
#    print numero
    for cle, valeur in classn.items():
        
        if valeur == numero:
            found=True
            return cle     
    if not found:
        return 'unknown'  

    
def ILDCNNpredict(dlist):     
        print 'predict'
        X_predict = np.asarray(np.expand_dims(dlist,1))/float(imageDepth)
        args  = H.parse_args()                          
        train_params = {
     'do' : float(args.do) if args.do else 0.5,        
     'a'  : float(args.a) if args.a else 0.3,          # Conv Layers LeakyReLU alpha param [if alpha set to 0 LeakyReLU is equivalent with ReLU]
     'k'  : int(args.k) if args.k else 4,              # Feature maps k multiplier
     's'  : float(args.s) if args.s else 1,            # Input Image rescale factor
     'pf' : float(args.pf) if args.pf else 1,          # Percentage of the pooling layer: [0,1]
     'pt' : args.pt if args.pt else 'Avg',             # Pooling type: Avg, Max
     'fp' : args.fp if args.fp else 'proportional',    # Feature maps policy: proportional, static
     'cl' : int(args.cl) if args.cl else 5,            # Number of Convolutional Layers
     'opt': args.opt if args.opt else 'Adam',          # Optimizer: SGD, Adagrad, Adam
     'obj': args.obj if args.obj else 'ce',            # Minimization Objective: mse, ce
     'patience' : args.pat if args.pat else 5,         # Patience parameter for early stoping
     'tolerance': args.tol if args.tol else 1.005,     # Tolerance parameter for early stoping [default: 1.005, checks if > 0.5%]
     'res_alias': args.csv if args.csv else 'res'      # csv results filename alias
         }

        modelfile= os.path.join(picklein_file,modelname)
        
        model = load_model(modelfile)
        model.compile(optimizer='Adam', loss=CNN4.get_Obj(train_params['obj']))        

        proba = model.predict_proba(X_predict, batch_size=100)
        print 'generate bg patches'
        nbp=0
        for i in range (0,len(X_predict)):
            prec, mprobai = maxproba(proba[i])
            classlabel=fidclass(prec,classif) 
            clssifcount[classlabel]=clssifcount[classlabel]+1
#            print classlabel,mprobai
            if classlabel =='back_ground' and mprobai>thrproba:
                nampa=os.path.join(patch_destdir,'bg_'+str(nbp)+'.'+typeid)
                nbp+=1                    
                cv2.imwrite (nampa, dlist[i],[int(cv2.IMWRITE_PNG_COMPRESSION),0])
        for i in clssifcount:
            print(i,clssifcount[i])
#        ooo
#        print proba[0]




bglist=listcl()
ILDCNNpredict(bglist)