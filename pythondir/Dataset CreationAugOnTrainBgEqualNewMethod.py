# coding: utf-8
'''create dataset from patches using split into 3 bmp databases and
 augmentation only on training, using bg, bg number limited to max and with equalization
 for all in training, validation and test
 defined used pattern set, define upsampling
 '''
import os
import cv2
#import dircache
import sys
import shutil
from scipy import misc
import numpy as np

import cPickle as pickle
#from sklearn.cross_validation import train_test_split
import random

#####################################################################
#define the working directory
HUG='CHU'   
subDir='TOPPATCH'
#extent='essai'
extent='3d162'
#extent='0'
#wbg = True # use back-ground or not
hugeClass=['healthy','back_ground']
#hugeClass=['back_ground']
upSampling=10 # define thershold for up sampling in term of ration compared to max class
subDirc=subDir+'_'+extent
pickel_dirsource='pickle_ds62'
name_dir_patch='patches'
#output pickle dir with dataset   
augf=12
#typei='bmp' #can be jpg
typei='png' #can be jpg

#define the pattern set
pset=1 # 1 when with new patters superimposed
#stdMean=True #use mean and normalization for each patch
maxuse=True # True means that all class are upscalled to have same number of 
 #elements, False means that all classes are aligned with the minimum number
useWeight=True #True means that no upscaling are done, but classes will 
#be weighted ( back_ground and healthy clamped to max) 

subdirc=subDir+'_'+extent
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
dirHUG=os.path.join(cwdtop,HUG)
dirHUG=os.path.join(dirHUG,subDirc)
print 'dirHUG', dirHUG
#input pickle dir with dataset to merge
#pickel_dirsourceToMerge='pickle_ds22'


#input patch directory
#patch_dirsource=os.path.join(dirHUG,'patches_norm_16_set0_1')
patch_dirsource=os.path.join(dirHUG,name_dir_patch)
print 'patch_dirsource',patch_dirsource
if not os.path.isdir(patch_dirsource):
    print 'directory ',patch_dirsource,' does not exist'
    sys.exit()
#output database for database generation
patch_dirSplitsource=   'b'
#augmentation factor


###############################################################


pickle_dir=os.path.join(cwdtop,pickel_dirsource) 
#pickle_dirToMerge=os.path.join(cwdtop,pickel_dirsourceToMerge)  


patch_dir=os.path.join(dirHUG,patch_dirsource)
print patch_dir


patch_dirSplit=os.path.join(dirHUG,patch_dirSplitsource)
patch_dir_Tr=os.path.join(patch_dirSplit,'p_Tr')
patch_dir_V=os.path.join(patch_dirSplit,'p_V')
patch_dir_Te=os.path.join(patch_dirSplit,'p_Te')


def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)

remove_folder(patch_dirSplit)
os.mkdir(patch_dirSplit)  
remove_folder(patch_dir_Tr)
os.mkdir(patch_dir_Tr)  
remove_folder(patch_dir_V)
os.mkdir(patch_dir_V)  
remove_folder(patch_dir_Te)
os.mkdir(patch_dir_Te)  
   

remove_folder(pickle_dir)
os.mkdir(pickle_dir)  



#define a list of used labels
#if wbg:
print 'with BG'
if pset ==0:
        usedclassif = [
            'back_ground',
            'consolidation',
            'HC',
            'ground_glass',
            'healthy',
            'micronodules',
            'reticulation',
            'air_trapping',
            'cysts',
            'bronchiectasis'
            ]
            
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
            
             'bronchial_wall_thickening':10,
             'early_fibrosis':11,
             'emphysema':12,
             'increased_attenuation':13,
             'macronodules':14,
             'pcp':15,
             'peripheral_micronodules':16,
             'tuberculosis':17
            }
elif pset==1:
        usedclassif = [
        'back_ground',
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        'emphysema',
        'GGpret'
        ]
            
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

elif pset==2:
            usedclassif = [
            'back_ground',
            'fibrosis',
            'healthy',
            'micronodules'
            ,'reticulation'
            ]
            
            classif ={
        'back_ground':0,
        'fibrosis':1,
        'healthy':2,
        'micronodules':3,
        'reticulation':4,
        }
elif pset==3:
        usedclassif = [
            'back_ground',
            'healthy',
            'air_trapping',
            ]
        classif ={
            'back_ground':0,
            'healthy':1,
            'air_trapping':2,
            }
else:
            print 'eRROR :', pset, 'not allowed'

    
class_weights={}

#define a dictionary with labels
#classif={}
#i=0
print ('classification used:')
for f in usedclassif:
    print (f, classif[f])

print '----------'
#define another dictionaries to calculate the number of label
classNumberInit={}    
classNumberNewTr={}
classNumberNewV={}
classNumberNewTe={}
actualClasses=[]
classAugm={}

for f in usedclassif:
    classNumberInit[f]=0
    classNumberNewTr[f]=0
    classNumberNewV[f]=0
    classNumberNewTe[f]=0
    classAugm[f]=1

# list all directories under patch directory. They are representing the categories

category_list=os.walk( patch_dir).next()[1]


# print what we have as categories
print ('all actual classes:',category_list)
print '----------'

usedclassifFinal=[f for f in usedclassif if f in category_list]

# print what we have as categories and in used one
print ('all actual classes:',usedclassifFinal)
print '----------'
# go through all categories to calculate the number of patches per class
# 
for category in usedclassifFinal:
    category_dir = os.path.join(patch_dir, category)
    print  'the path into the categories is: ', category_dir
    sub_categories_dir_list = (os.listdir(category_dir))
    #print 'the sub categories are : ', sub_categories_dir_list
    for subCategory in sub_categories_dir_list:
        subCategory_dir = os.path.join(category_dir, subCategory)
        print  'the path into the sub categories is: ',subCategory_dir
        #print subCategory_dir
        image_files = [name for name in os.listdir(subCategory_dir) if name.find('.'+typei) > 0 ]
        
       
        for filei in image_files:
                
                classNumberInit[category]=classNumberInit[category]+1


total=0
print('number of patches init')
for f in usedclassifFinal:
    print ('class:',f,classNumberInit[f])
    total=total+classNumberInit[f]
print('total:',total)
print '----------'

#define coeff max
maxl=0
for f in usedclassifFinal:
   if classNumberInit[f]>maxl and f not in hugeClass:
#      print 'fmax', f
      maxl=classNumberInit[f]
print ('max number of patches in : ',maxl)
print '----------'
#define coeff min
minl=1000000
for f in usedclassifFinal:
   if classNumberInit[f]<minl:
      minl=classNumberInit[f]
print ('min number of patches in : ',minl)
print '----------'
#artificially clamp back-ground to maxl
#if maxuse :
#    classNumberInit['back_ground']=maxl
#    classNumberInit['healthy']=maxl
#elif not useWeight:
#    for f in usedclassifFinal:
#      classNumberInit[f]=minl
print ' maximum number of patches ratio'  
classConso={}
for f in usedclassifFinal:
    classConso[f]=float(maxl)/max(1,classNumberInit[f])
#for f in usedclassifFinal:
    print (f,classif[f],' {0:.2f}'.format (classConso[f]))
print '----------'
print ' upsampling if ratio <' ,upSampling
print 'usedclassifFinal', usedclassifFinal
for f in usedclassifFinal:
#    print f, classConso[f], upSampling,maxl, float(maxl)/upSampling/classNumberInit[f]
    if int(classConso[f]) >upSampling:
#         print f
         classAugm[f]=float(maxl)/upSampling/classNumberInit[f]
    print (f,classif[f],' {0:.2f}'.format (classAugm[f]))
print '----------'


def   createStruct(f):
    print('Create patches directories from class : ',f)
    
    patch_dir_Tr_f=os.path.join(patch_dir_Tr,f)
    remove_folder(patch_dir_Tr_f)
    os.mkdir(patch_dir_Tr_f) 

    patch_dir_V_f=os.path.join(patch_dir_V,f)
    remove_folder(patch_dir_V_f)
    os.mkdir(patch_dir_V_f) 

    patch_dir_Te_f=os.path.join(patch_dir_Te,f)
    remove_folder(patch_dir_Te_f)
    os.mkdir(patch_dir_Te_f) 
    
def  copypatch(f):
    if useWeight or maxuse:
        cnif=maxl
    else:
        cnif =minl
    print 'cnif', cnif
    print('copy all in new training  directory for:',f)
    category_dir = os.path.join(patch_dir, f)
    patch_dir_Tr_f=os.path.join(patch_dir_Tr,f)
    patch_dir_tempo=[]
    sub_categories_dir_list = (os.listdir(category_dir))
#    print ('the sub categories are : ', sub_categories_dir_list)
    for subCategory in sub_categories_dir_list:
        subCategory_dir = os.path.join(category_dir, subCategory)
        print  'the path into the sub categories is: ',subCategory_dir
        image_files = [name for name in os.listdir(subCategory_dir) if name.find('.'+typei)>0]
        for filei in image_files:
#             if filei.find('.bmp') > 0:
                 fileSource=os.path.join(subCategory_dir, filei)
                 patch_dir_tempo.append(fileSource)
    print len(patch_dir_tempo)
    cnif = min (cnif,len(patch_dir_tempo))
    print 'new cnif',cnif
#    if maxuse:
    if maxuse or useWeight:
#        if f !='back_ground' and f !='healthy':
        if f not in hugeClass:

                    #copy all the files if not back-ground               
                    for filei in patch_dir_tempo:
                        (top,tail)=os.path.split(filei)
                        fileDest=os.path.join(patch_dir_Tr_f, tail)
                                # load the .bmp file into dest directory
                        shutil.copyfile(filei,fileDest)
        else:
                     #copy only randomly maxl file for back-ground
#                    print 'aa'
                    filenamesort = random.sample(patch_dir_tempo,cnif)
                    for i in range (0,len(filenamesort)):
                        (top,tail)=os.path.split(filenamesort[i])
                        fileDest=os.path.join(patch_dir_Tr_f, tail)
                        shutil.copyfile(filenamesort[i],fileDest)
    else:
                    filenamesort = random.sample(patch_dir_tempo,cnif)
                    for i in range (0,len(filenamesort)):
                        (top,tail)=os.path.split(filenamesort[i])
                        fileDest=os.path.join(patch_dir_Tr_f, tail)
                        shutil.copyfile(filenamesort[i],fileDest)


def  selectpatch(f,n,t,l):
    print(n,' patch selection for:',f, 'for:',t )
    patch_dir_Tr_f=os.path.join(patch_dir_Tr,f)
    listdirf=os.listdir(patch_dir_Tr_f)
#    cni=classNumberInit[f]
#    cni=len(l)
    cnif=int(l*n)
    print cnif
    if t=='T':
        dirdest=os.path.join(patch_dir_Te,f)
    else:
        dirdest=os.path.join(patch_dir_V,f)

    filenamesort = random.sample(listdirf,cnif)
    for i in range (0,len(filenamesort)):
                    (top,tail)=os.path.split(filenamesort[i])
                    fileSource=os.path.join(patch_dir_Tr_f,filenamesort[i])
                    fileDest=os.path.join(dirdest, tail)
                    shutil.move(fileSource,fileDest)    
        
    
#        
  

def listcl(f,p,m):
    patch_dir_Tr_f=os.path.join(patch_dir_Tr,f)
    patch_dir_V_f=os.path.join(patch_dir_V,f)
    patch_dir_Te_f=os.path.join(patch_dir_Te,f)
    dlist = []
    classNumberN=0
    loop=0
    if p=='Tr':
        act='Training Set'
        category_dir=patch_dir_Tr_f
        maxp=m*augf*0.5 #-3 to take into account rounding
    elif p=='V':
        act='Validation Set'
        category_dir=patch_dir_V_f
        maxp=m*0.25
    else:
        act='Test Set'
        category_dir=patch_dir_Te_f
        maxp=m*0.25
    print('list patches from class : ',f, act)                 

    image_files = [name for name in os.listdir(category_dir) if  name.find('.'+typei) > 0]
    maxp1=1
    if useWeight and f not in hugeClass:
         if p=='Tr':
            maxp1=int(classNumberInit[f]*classAugm[f]*0.5*augf)
         else:
            maxp1=int(classNumberInit[f]*classAugm[f]*0.25)
            
    elif maxuse:
        maxp1=maxp
#    print m, maxp1 ,maxp,classNumberInit[f],classAugm[f]  
    while classNumberN<maxp1: 
        loop+=1
#        print('loops;',loop,'for :',f,'classnumber:',classNumberN,'max:',maxp1)
        
        for filei in image_files:

                            image=cv2.imread(os.path.join(category_dir,filei),-1)

        # load the .bmp file into array
                            if p=='Tr':   
#                                print('loopi;',loop,'for :',f,'classnumber:',classNumberN,'max:',maxp1)
                                classNumberN=classNumberN+augf
#                              
                                # 1 append the array to the dataset list    
#                                img=image.astype('float64')
#                                img -=np.mean(img)
#                                img /= np.std(img)
                                dlist.append(image)
                                
        #                        #2 created rotated copies of images
                                image90 = np.rot90(image)  

#                                img=image90.astype('float64')
#                                img -=np.mean(img)
#                                img /= np.std(img)
                                dlist.append(image90)
                                
        #                        #3 created rotated copies of images                        
                                image180 = np.rot90(image90)
#                                img=image180.astype('float64')
#                                img -=np.mean(img)
#                                img /= np.std(img)
                                dlist.append(image180)
                          
        #                        #4 created rotated copies of images                                          
                                image270 = np.rot90(image180)                      
#                                img=image270.astype('float64')
#                                img -=np.mean(img)
#                                img /= np.std(img)
                                dlist.append(image270) 
                                
                                #5 flip fimage left-right
                                imagefliplr=np.fliplr(image)   

#                                img=imagefliplr.astype('float64')
#                                img -=np.mean(img)
#                                img /= np.std(img)
                                dlist.append(imagefliplr) 
                                
                                #6 flip fimage left-right +rot 90
                                image90 = np.rot90(imagefliplr)  
#                                img=image90.astype('float64')
#                                img -=np.mean(img)
#                                img /= np.std(img)
                                dlist.append(image90)
                                
        #                        #7 flip fimage left-right +rot 180                   
                                image180 = np.rot90(image90)
#                                img=image180.astype('float64')
#                                img -=np.mean(img)
#                                img /= np.std(img)
                                dlist.append(image180)
                          
        #                        #8 flip fimage left-right +rot 270                                          
                                image270 = np.rot90(image180)                      
#                                img=image270.astype('float64')
#                                img -=np.mean(img)
#                                img /= np.std(img)
                                dlist.append(image270)                                                                 
                                                            
                                #9 flip fimage up-down
                                imageflipud=np.flipud(image)                                   
#                                img=imageflipud.astype('float64')
#                                img -=np.mean(img)
#                                img /= np.std(img)
                                dlist.append(imageflipud) 
                                
                                 #10 flip fimage up-down +rot90                               
                                image90 = np.rot90(imageflipud)  
#                                img=image90.astype('float64')
#                                img -=np.mean(img)
#                                img /= np.std(img)
                                dlist.append(image90)
                                
        #                         #11 flip fimage up-down +rot180                          
                                image180 = np.rot90(image90)
#                                img=image180.astype('float64')
#                                img -=np.mean(img)
#                                img /= np.std(img)
                                dlist.append(image180)
                          
        #                        #12 flip fimage up-down +rot270                                           
                                image270 = np.rot90(image180)                      
#                                img=image270.astype('float64')
#                                img -=np.mean(img)
#                                img /= np.std(img)
                                dlist.append(image270) 
                                
                                
                            else:
                                classNumberN=classNumberN+1
                                #print image                  
                                # 1 append the array to the dataset list                        
#                                img=image.astype('float64')
#                                img -=np.mean(img)
#                                img /= np.std(img)
#                                dlist.append(image) 
                                dlist.append(image) 
                
#    dlist=np.array(dlist)    
##    print dlist[0]    
#    print len(dlist)  
#    print dlist.shape
    return dlist,classNumberN



# main program 
label_listTr = []
label_listV = []
label_listTe = []
dataset_listTr =[]
dataset_listV =[]
dataset_listTe =[]
for f in usedclassifFinal:
#for f in ('healthy',):

     print('work on :',f)
     dataset_listTri =[]
     dataset_listVi =[]
     dataset_listTei =[]
     #create structure dir for f
     createStruct(f)
     #copy patches in new directory flat
     copypatch(f)
#     ooo
     
     patch_dir_Tr_f=os.path.join(patch_dir_Tr,f)
     listdirf=os.listdir(patch_dir_Tr_f)
     l=len(listdirf)
     # select 1/4 patches for test
     selectpatch(f,0.25,'T',l)

#     select 1/4 patches for validation
     selectpatch(f,0.25,'V',l)
#     bbbb
     
    #fill list with patches

     dataset_listTri,classNumberNewTr[f] = listcl(f,'Tr',maxl)
     dataset_listVi ,classNumberNewV[f]= listcl(f,'V',maxl)
     dataset_listTei,classNumberNewTe[f] = listcl(f,'Te',maxl)
#     resul=equal(maxl,dlf,f)
#     print dataset_listTri.shape
     i=0
     while i <  classNumberNewTr[f] :
        dataset_listTr.append(dataset_listTri[i])

        label_listTr.append(classif[f])
        i+=1

     i=0
     while i <  classNumberNewV[f] :         
        dataset_listV.append(dataset_listVi[i])
        label_listV.append(classif[f])
        i+=1
     i=0
     while i <  classNumberNewTe[f]:
        dataset_listTe.append(dataset_listTei[i])
        label_listTe.append(classif[f])
        i+=1
print '---------------------------'
for f in usedclassifFinal:
    print ('init',f,classNumberInit[f])
    print ('after training',f,classNumberNewTr[f])
    print ('after validation',f,classNumberNewV[f])
    print ('after test',f,classNumberNewTe[f])
    print '---------------------------'
#    print ('final',f,classNumberFinal[f])

print '---------------------------'
print ('training set:',len(dataset_listTr),len(label_listTr))
print ('validation set:',len(dataset_listV),len(label_listV))
print ('test set:',len(dataset_listTe),len(label_listTe))
# transform dataset list into numpy array                   
X_train = np.array(dataset_listTr)
y_train = np.array(label_listTr)
X_val = np.array(dataset_listV)
y_val = np.array(label_listV)
X_test = np.array(dataset_listTe)
y_test = np.array(label_listTe)
#this is already in greyscale
# use only one of the 3 color channels as greyscale info
#X = dataset[:,:, :,1]

#print 'dataset shape is now: ', X.shape
#print('X22 as example:', X[22])
## 
#y = np.array(label_list)
## sampling item 22
#print ('y22 as example:',y[22])
#
#print ('Xshape : ',X.shape)
#print ('yshape : ',y.shape)
#
#
#X_train, X_intermediate, y_train, y_intermediate = train_test_split(X, y, test_size=0.5, random_state=42)
#X_val, X_test, y_val, y_test = train_test_split(X_intermediate, y_intermediate, test_size=0.5, random_state=42)
print '-----------FINAL----------------'
print ('Xtrain :',X_train.shape)
print ('Xval : ',X_val.shape)
print ('Xtest : ',X_test.shape)
print ('ytrain : ',y_train.shape)
print ('yval : ',y_val.shape)
print ('ytest : ',y_test.shape)
    
#min_val=np.min(X_train)
#max_val=np.max(X_train)
#print 'Xtrain', min_val, max_val
#    
for f in usedclassifFinal:
#    print f,classNumberNewTr[f]
#    if wbg:
        class_weights[classif[f]]=round(float(classNumberNewTr['back_ground'])/classNumberNewTr[f],3)
#    else:
#        class_weights[classif[f]]=round(float(classNumberNewTr['healthy'])/classNumberNewTr[f],3)
print '---------------'
#if wbg: 
class_weights[classif['back_ground']]=0.1
print 'weights'
print class_weights

pickle.dump(class_weights, open( os.path.join(pickle_dir,"class_weights.pkl"), "wb" ))
pickle.dump(X_train, open( os.path.join(pickle_dir,"X_train.pkl"), "wb" ))
pickle.dump(X_test, open( os.path.join(pickle_dir,"X_test.pkl"), "wb" ))
pickle.dump(X_val, open(os.path.join(pickle_dir,"X_val.pkl"), "wb" ))
pickle.dump(y_train, open( os.path.join(pickle_dir,"y_train.pkl"), "wb" ))
pickle.dump(y_test, open( os.path.join(pickle_dir,"y_test.pkl"), "wb" ))
pickle.dump(y_val, open( os.path.join(pickle_dir,"y_val.pkl"), "wb" ))
pickle.dump(y_val, open( os.path.join(pickle_dir,"y_val.pkl"), "wb" ))

recuperated_X_train = pickle.load( open( os.path.join(pickle_dir,"X_train.pkl"), "rb" ) )
#min_val=np.min(recuperated_X_train)
#max_val=np.max(recuperated_X_train)
#print 'recuperated_X_train', min_val, max_val

recuperated_class_weights = pickle.load( open(os.path.join(pickle_dir,"class_weights.pkl"), "rb" ) )
print 'recuparated weights'
print recuperated_class_weights