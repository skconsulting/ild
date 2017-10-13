# -*- coding: utf-8 -*-
"""
Created on 07 july 2017
class patches against probability
@author: sylvain
"""

from param_pix import *
import os
import keras
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('tf')
print 'NEW keras.backend.image_data_format :',keras.backend.image_data_format()
print '-------------'
nametophug='SOURCE_IMAGE'

#nameHug='DUMMY' #name of top directory for patches pivkle from dicom
nameHug='DUMMY' #name of top directory for patches pivkle from dicom

subHUG='patchesref'#subdirectory from nameHug input pickle
#subHUG='S3'#subdirectory from nameHug input pickle


toppatch= 'classpatch' #name of top directory for image and label generation


classifild ={
        'consolidation':0,
        'HC':1,
        'ground_glass':2,
        'healthy':3,
        'micronodules':4,
        'reticulation':5,
        'air_trapping':6,
        'cysts':7,
        'bronchiectasis':8,
#        'emphysema':10,
        'GGpret':9
        }



cwd=os.getcwd()
#
(cwdtop,tail)=os.path.split(cwd)

picklepatches='picklepatches' 
weightildcnn='weightildcnn'
pcross='set0_c2'
pfront='set0_c2'
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



def geneaug(image,tt):
    if tt==0:
        imout=image
    elif tt==1:
    # 1 90 deg
        imout = np.rot90(image)
    elif tt==2:
    #2 180 deg
        imout = np.rot90(np.rot90(image))
    elif tt==3:
    #3 270 deg
        imout = np.rot90(np.rot90(np.rot90(image)))
    elif tt==4:
    #4 flip fimage left-right
            imout=np.fliplr(image)
    elif tt==5:
    #5 flip fimage left-right +rot 90
        imout = np.rot90(np.fliplr(image))
    elif tt==6:
    #6 flip fimage left-right +rot 180
        imout = np.rot90(np.rot90(np.fliplr(image)))
    elif tt==7:
    #7 flip fimage left-right +rot 270
        imout = np.rot90(np.rot90(np.rot90(np.fliplr(image))))
    elif tt==8:
    # 8 flip fimage up-down
        imout = imout=np.flipud(image)
    elif tt==9:
    #9 flip fimage up-down +rot90
        imout = np.rot90(np.flipud(image))
    elif tt==10:
    #10 flip fimage up-down +rot180
        imout = np.rot90(np.rot90(np.flipud(image)))
    elif tt==11:
    #11 flip fimage up-down +rot270
        imout = np.rot90(np.rot90(np.rot90(np.flipud(image))))

    return imout




def genebmppatch(dirName,pat):

    """generate patches from dicom files and sroi"""
    print ('generate patch file')
    (top,tail)=os.path.split(dirName)
#    global constPixelSpacing, dimtabx,dimtaby
    #directory for patches

    patdic=[]
    patdicn=[]

    patdir=os.path.join(dirName,pat)
    subd=os.listdir(patdir)
    for loca in subd:
        patsubdir=os.path.join(patdir,loca)
        listpickles=os.listdir(patsubdir)        
        for l in listpickles:
                listscan=pickle.load(open(os.path.join(patsubdir,l),"rb"))
                for num in range(len(listscan)):
                    patscan=normHU(listscan[num])
                    patdicn.append(patscan)
                    patdic.append(listscan[num])
    return patdic,patdicn
                





def preparroi(namedirtopcf,tabscan,tabsroi):
    (top,tail)=os.path.split(namedirtopcf)
    pathpicklepat=os.path.join(picklepathdir,tail)
    if not os.path.exists (pathpicklepat):
                os.mkdir(pathpicklepat)
    
    for num in range(slnt):
        patchpicklenamepatient=str(num)+'_'+patchpicklename   
        pathpicklepatfile=os.path.join(pathpicklepat,patchpicklenamepatient)
        scan_list=[]
        mask_list=[]
        scan_list.append(tabscan[num] ) 
#        print tabscan[0].shape,tabscan[0].min(),tabscan[0].max()
#        maski= tabsroi[num].copy()    
#        np.putmask(maski,maski>0,classif['healthy'])

        mask_list.append(tabsroi[num])
#        print tabsroi[num].min(),tabsroi[num].max(),np.unique(tabsroi[num])
#        oooo
#        if num==3:
##            o=normi(maski)
#            n=normi(tabscan[num] )
#            x=normi(tabsroi[num])
##            f=normi(tabroif)
#            cv2.imshow('maski',x)
#            cv2.imshow('datascan[num] ',n)
##            cv2.imshow('tabroix',x)
##            cv2.imshow('tabroif',f)
##            cv2.imwrite('a.bmp',o)
##            cv2.imwrite('b.bmp',x)
##            cv2.imwrite('c.bmp',tabscan[num])
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            
        patpickle=(scan_list,mask_list)
#        print len(scan_list)
        pickle.dump(patpickle, open(pathpicklepatfile, "wb"),protocol=-1)

def ILDCNNpredict(patch_list,model):
    print ('Predict started ....')

    X0=len(patch_list)
    # adding a singleton dimension and rescale to [0,1]
    
    # look if the predict source is empty
    # predict and store  classification and probabilities if not empty
    if X0 > 0:
        pa = np.asarray(patch_list)
#        print pa.shape
        pa1 = np.expand_dims(patch_list, 1)
#        print pa1.shape
        proba = model.predict_proba(pa1, batch_size=100,verbose=1)

    else:
        print (' no patch in selected slice')
        proba = ()
    print 'number of patches', len(pa)

    return proba


def modelCompilation(t,picklein_file,picklein_file_front,setdata):
    
    print 'model compilation',t
    

    dirpickleArchs=os.path.join(dirpickleArch,setdata)
    dirpickleArchsc=os.path.join(dirpickleArchs,modelArch)

    json_string=pickle.load( open(dirpickleArchsc, "rb"))
    model = model_from_json(json_string)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#        model.compile()

    if t=='cross':
        lismodel=os.listdir(picklein_file)
        
        modelpath = os.path.join(picklein_file, lismodel[0])
#        print modelpath
    if t=='front':
        lismodel=os.listdir(picklein_file_front)
        modelpath = os.path.join(picklein_file_front, lismodel[0])

    if os.path.exists(modelpath):

        model.load_weights(modelpath)  
        
        return model
    else:
        print 'weight dos not exist',modelpath


listpat=[]

#    classifused={
#    'back_ground':0,
#        'healthy':1,    
#        'ground_glass':2}
classifused=classif
classinsource =[name for name in os.listdir(namedirtopcpickle) if name in classifused]
classinsource.remove('back_ground')
print classinsource,namedirtopcpickle

patdic={}
model=modelCompilation('cross',picklein_file,picklein_file_front,setdata)

thpat={}
for pat in classinsource: 
    thpat[pat]=0.8
    
thpat['air_trapping']=0.9
thpat['healthy']=0.95
thpat['bronchiectasis']=0.6
thpat['consolidation']=0.9
thpat['ground_glass']=0.6
thpat['HC']=0.6
thpat['reticulation']=0.6
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
    patnumberinit[pat]=len(tabscan)
    initpatn=initpatn+len(tabscan)
#    print 'number of patches for pattern :',pat,':',len(tabscan)
    probapat=ILDCNNpredict(tabscann,model)
    cpp=np.amax(probapat, axis=-1)
    cpr=np.argmax(probapat,axis=-1)
#    cpp=np.amax(probapat)
    """
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
    """ 
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
print 'init number of patches :',initpatn
print 'final number of patches :',finalpatn
for pat in classinsource: 
   print pat ,':', thpat[pat],'init:', patnumberinit[pat],'final:',patnumberfinal[pat]
   
  
   