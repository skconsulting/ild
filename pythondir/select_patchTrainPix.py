# -*- coding: utf-8 -*-
"""
Created on 07 july 2017
class patches against probability
DO NOT applies norming through norm
@author: sylvain
"""
import cPickle as pickle

import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import model_from_json

from param_pix_t import norm,setdata,remove_folder,classif,thrpatch


K.set_image_dim_ordering('th')

print 'NEW keras.backend.image_data_format :',keras.backend.image_data_format()
print '-------------'

#define the working directory for input patches
topdir='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool'
firsto='th'+str(thrpatch)+'_'
firsto=''
toppatch= 'TOPPATCH'
extendir='all'
#extendir0='2'
extendir0='HUG'
#define the working directory for output patches
toppatchres='TOPPATCHOUT'
extendirres=extendir
extendir0res=extendir0

picklepatches='picklepatches' 
namedirtopcpickle=os.path.join(topdir,firsto+toppatch+'_'+extendir+'_'+extendir0)
namedirtopcpickle=os.path.join(namedirtopcpickle,picklepatches)

classifild = classif

#define the directory to collect model
pickel_dirsource_root='pickle'
pickel_dirsource_e='train' #path for data fort training
pickel_dirsourcenum=setdata #extensioon for path for data for training
#extendir2='2'
extendir2='20'

weightildcnn='pickle'

modelArch='CNN.h5'

pickel_dirsource='th'+str(thrpatch)+'_'+pickel_dirsource_root+'_'+pickel_dirsource_e+'_'+pickel_dirsourcenum+'_'+extendir2

pathildcnn =os.path.join(topdir,pickel_dirsource)
pathildcnn =os.path.join(pathildcnn,weightildcnn)
picklein_file =os.path.join(pathildcnn,modelArch)



patchtoppath=os.path.join(topdir,toppatchres+'_'+extendirres+'_'+extendir0res)
patchpicklename='picklepatches.pkl'
#picklepathdir =os.path.join(patchtoppath,picklepath) #path to write classified patches
picklepathdir =os.path.join(patchtoppath,picklepatches) #path to write classified patches



if not os.path.isdir(patchtoppath):
    os.mkdir(patchtoppath)

if not os.path.isdir(picklepathdir):
    os.mkdir(picklepathdir)

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

def genebmppatch(dirName,pat):

    """generate patches from dicom files and sroi"""
    print ('generate patch file')
    (top,tail)=os.path.split(dirName)
#    global constPixelSpacing, dimtabx,dimtaby
    #directory for patches

    patdic=[]
    patdicn=[]

#    dirName='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool/th0.95_TOPPATCH_all_2/picklepatches'

    patdir=os.path.join(dirName,pat)
#    dirName='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool/th0.95_TOPPATCH_all_2/picklepatches'
    subd=os.listdir(patdir)
#    print patdir
    for loca in subd:
        patsubdir=os.path.join(patdir,loca)
        listpickles=os.listdir(patsubdir)        
        for l in listpickles:
                listscan=pickle.load(open(os.path.join(patsubdir,l),"rb"))
#                print patsubdir,l
                for num in range(len(listscan)):
#                    print listscan[num].min(),listscan[num].max()
                    patscan=norm(listscan[num])
#                    print patscan.min(),patscan.max()
#                    ooo
                    patdicn.append(patscan)
                    patdic.append(listscan[num])
    return patdic,patdicn
                



def ILDCNNpredict(patch_list,model):
    print ('Predict started ....')

    X0=len(patch_list)

    if X0 > 0:

        pa = np.expand_dims(patch_list, 1)
        proba = model.predict_proba(pa, batch_size=500,verbose=1)

    else:
        print (' no patch in selected slice')
        proba = ()
    print 'number of patches', len(pa)

    return proba


def modelCompilation(picklein_file,namelastc):
    
    print 'model compilation'   

    json_string=pickle.load( open(picklein_file, "rb"))
    model = model_from_json(json_string)
    model.load_weights(namelastc) 
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#        model.compile()

        
    return model

###############################################################
listpat=[]

#    classifused={
#    'back_ground':0,
#        'healthy':1,    
#        'ground_glass':2}
#classifused=classif
classinsource =[name for name in os.listdir(namedirtopcpickle) if name in classifild]
#classinsource.remove('back_ground')
#print classinsource,namedirtopcpickle
namelastc=load_model_set(pathildcnn) 
#print namelastc

#print picklein_file
patdic={}
model=modelCompilation(picklein_file,namelastc)

thpat={}
for pat in classinsource: 
    thpat[pat]=0.5
"""   
thpat['air_trapping']=0.9
thpat['healthy']=0.9
thpat['bronchiectasis']=0.9
thpat['consolidation']=0.9
thpat['cysts']=0.9
thpat['GGpret']=0.9
thpat['ground_glass']=0.9
thpat['HC']=0.9
thpat['micronodules']=0.9
thpat['reticulation']=0.9
"""
initpatn=0
finalpatn=0
patnumberinit={}
patnumberfinal={}
for pat in classinsource: 
    cp=classifild[pat]
    th=thpat[pat]
    scantab=[]
    propatab=[]
    print 'work on :',pat, 'number',str(cp)
    tabscan,tabscann=genebmppatch(namedirtopcpickle,pat)
    #tabsacn: no norm, tabscann: norm
    patnumberinit[pat]=len(tabscan)
    initpatn=initpatn+len(tabscan)
    print 'number of patches for pattern :',pat,':',len(tabscan)
    
    probapat=ILDCNNpredict(tabscann,model)
#    print probapat.shape
    cpp=np.amax(probapat ,axis=-1)
    cpr=np.argmax(probapat,axis=-1 )
#    print probapat[0]
#    print cpp[0]
#    print cpr[0]
#    ooo
#    probapat=100*probapat
#    probapat=probapat.astype('uint')
##    cpp=np.amax(probapat)
    
    plt.figure(figsize = (4, 3))
#    print 'imamax min max',imamax.min(), imamax.max(),imamax[100][200]
    plt.hist(cpr.flatten(), bins=50, color='c')
    plt.xlabel("class")
    plt.ylabel("Frequency")
    plt.show()
    plt.figure(figsize = (4, 3))
    plt.hist(cpp.flatten(), bins=50, color='c')
    plt.xlabel("proba")
    plt.ylabel("Frequency")
    plt.show()
#    print cpp[10]
#    print cpr[10]
#    print probapat[10]
    
    for image, proba in zip(tabscan, probapat):
        cpr=np.argmax(proba)
        cpp=np.amax(proba)
        if cpr==cp and cpp >th:
            scantab.append(image)
#            print cp,cpr,cpp
#        else:
                     
#    print len(scantab)
    print pat,' init :',len(tabscan),'remainaning with threshodl :',str(th), ':',len(scantab)
    finalpatn=finalpatn+len(scantab)
    patnumberfinal[pat]=len(scantab)
#    cv2.imshow('mask',tabscan[0])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    pdirpat=os.path.join(picklepathdir,pat)
    remove_folder(pdirpat)
    os.mkdir(pdirpat)
    pdirpatc=os.path.join(pdirpat,str(th))
    if not os.path.exists(pdirpatc):
        os.mkdir(pdirpatc)
    pdirpatcf=os.path.join(pdirpatc,'pat.pkl')
    pickle.dump(scantab, open(pdirpatcf, "wb"),protocol=-1)
#    lp=pickle.load( open(pdirpatcf, "rb"))
#    print len(lp)
#    print lp[0].min(),lp[0].max()
#    ooo
print '-------------------'
print 'init number of patches :',initpatn
print 'final number of patches :',finalpatn
print '-------------------'
for pat in classinsource: 
   print pat ,':', thpat[pat],'init:', patnumberinit[pat],'final:',patnumberfinal[pat]
   
  
   