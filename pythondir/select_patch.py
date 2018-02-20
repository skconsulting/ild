# -*- coding: utf-8 -*-
"""
Created on 07 july 2017
class patches against probability
DO NOT applies norming through norm
need to change mean pix in param pix
@author: sylvain
"""
import cPickle as pickle

import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import model_from_json

from param_pix import norm,setdata,remove_folder


K.set_image_dim_ordering('th')

print 'NEW keras.backend.image_data_format :',keras.backend.image_data_format()
print '-------------'
nametophug='SOURCE_IMAGE'

#nameHug='DUMMY' #name of top directory for patches pivkle from dicom
nameHug='DUMMY' #name of top directory for patches pivkle from dicom

subHUG='patchesref'#subdirectory from nameHug input pickle
#subHUG='S3'#subdirectory from nameHug input pickle

toppatch= 'classpatch' #name of top directory for image and label generation

#for set1
classifild ={
        'consolidation':0,
        'HC':1,
        'ground_glass':2,
        'healthy':3,
        'micronodules':4,
        'reticulation':5,
        'bronchiectasis':6,
        'emphysema':7,
        'GGpret':8
        }



cwd=os.getcwd()
#
(cwdtop,tail)=os.path.split(cwd)

picklepatches='picklepatches' 
weightildcnn='weightildcnn'
pcross='set1_c0'
pfront='set1_c0'
modelArch='CNN.h5'
pathmodelarch='modelArch'



path_HUGtop=os.path.join(cwdtop,nametophug)
path_HUG=os.path.join(path_HUGtop,nameHug)
namedirtopc =os.path.join(path_HUG,subHUG)
namedirtopcpickle=os.path.join(namedirtopc,picklepatches)

pathildcnn =os.path.join(path_HUG,weightildcnn)
dirpickleArch=os.path.join(pathildcnn,pathmodelarch)

picklein_file=os.path.join(pathildcnn,pcross)
picklein_file_front=os.path.join(pathildcnn,pfront)



patchesdirnametop = toppatch
patchtoppath=os.path.join(path_HUG,patchesdirnametop)
patchpicklename='picklepatches.pkl'
picklepath = 'picklepatches'
#picklepathdir =os.path.join(patchtoppath,picklepath) #path to write classified patches
picklepathdir =os.path.join(patchtoppath,picklepath) #path to write classified patches



if not os.path.isdir(patchtoppath):
    os.mkdir(patchtoppath)

if not os.path.isdir(picklepathdir):
    os.mkdir(picklepathdir)



def genebmppatch(dirName,pat):

    """generate patches from dicom files and sroi"""
    print ('generate patch file')
    (top,tail)=os.path.split(dirName)
#    global constPixelSpacing, dimtabx,dimtaby
    #directory for patches

    patdic=[]
    patdicn=[]

    dirName='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool/th0.95_TOPPATCH_all_CHU2new_UIP_3_1b/picklepatches'
#    dirName='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool/th0.95_TOPPATCH_all_HUG_ILD_TXT__1b/picklepatches'


    patdir=os.path.join(dirName,pat)
    print patdir

#    dirName='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool/th0.95_TOPPATCH_all_2/picklepatches'

    try:
        subd=os.listdir(patdir)

        for loca in subd:


            patsubdir=os.path.join(patdir,loca)
            listpickles=os.listdir(patsubdir)        
            for l in listpickles:
                    listscan=pickle.load(open(os.path.join(patsubdir,l),"rb"))
#                    print patsubdir,l
                    for num in range(len(listscan)):
#                        print listscan[num].min(),listscan[num].max()

                        patscan=norm(listscan[num])
    #                    print patscan.min(),patscan.max()
                        patdicn.append(patscan)
                        patdic.append(listscan[num])
    except:
        print 'no ',pat   
     
    return patdic,patdicn
                



def ILDCNNpredict(patch_list,model):
    print ('Predict started ....')
    ok=False
    X0=len(patch_list)
    if X0 > 0:

        pa = np.expand_dims(np.array(patch_list), 1)
        proba = model.predict_proba(pa, batch_size=500,verbose=1)
        print 'number of patches', len(pa)
        ok=True
    else:
        print (' no patch in selected slice')
        proba = ()

    return proba,ok

def modelCompilation(picklein_file,namelastc):
    
    print 'model compilation'   

    json_string=pickle.load( open(picklein_file, "rb"))
    model = model_from_json(json_string)
    model.load_weights(namelastc) 
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#        model.compile()
       
    return model



listpat=[]

#    classifused={
#    'back_ground':0,
#        'healthy':1,    
#        'ground_glass':2}
#classifused=classif
classinsource =[name for name in os.listdir(namedirtopcpickle) if name in classifild]
#classinsource.remove('back_ground')
print classinsource,namedirtopcpickle

patdic={}
lismodel=os.listdir(picklein_file)      
namelastc = os.path.join(picklein_file, lismodel[0])
dirpickleArchs=os.path.join(dirpickleArch,setdata)
dirpickleArchsc=os.path.join(dirpickleArchs,modelArch)
print ' architecture path',dirpickleArchsc
print 'weight path:',namelastc



model=modelCompilation(dirpickleArchsc,namelastc)


thpat={}
for pat in classinsource: 
    thpat[pat]=0.8
    
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
initpatn=0
finalpatn=0
patnumberinit={}
patnumberfinal={}
for pat in classinsource: 
#    pat='ground_glass'
    patnumberfinal[pat]=0
   
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
    
    probapat,ok=ILDCNNpredict(tabscann,model)
    
    if ok:

        cpp=np.amax(probapat ,axis=-1)
        cpr=np.argmax(probapat,axis=-1 )
#        print probapat[0]
#        print cpp[0]
#        print cpr[0]
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
        if not os.path.exists(pdirpat):
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
   
  
   