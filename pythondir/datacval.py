# coding: utf-8
'''create dataset from patches for validation only, all patches, no augm
support Hu
version 1.0
second step
S. Kritter
18 august 2017
 '''
 
from param_pix_t import classif
from param_pix_t import thrpatch,setdata
from param_pix_t import norm
#from param_pix_t import normi

from param_pix_t import picklepath,perrorfile
import cv2
import datetime
import os
#import sys
import keras
import numpy as np
#from sklearn.model_selection import train_test_split
import cPickle as pickle
import random


print 'NEW keras.backend.image_data_format :',keras.backend.image_data_format()
print '-------------'

#####################################################################
#define the working directory for input patches
topdir='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool'
#toppatch= 'TOPPATCH'
#extendir='all'
##extendir0='2'
#extendir0='0'
#define the name of directory for patches
#patchesdirnametop = ['th0.95_TOPPATCH_JC_0']
patchesdirnametop = ['th0.95_TOPPATCH_all_CHU2new']

#patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_JC_0']
#print patchesdirnametop

extendir1=''

#define the directory to store data
pickel_dirsource_root='pickle'
#pickel_dirsource_e='train' #path for data fort training
pickel_dirsource_e='val' #path for data fort training
pickel_dirsource_e='essai' #path for data fort training

pickel_dirsourcenum=setdata #extensioon for path for data for training
#extendir2='2'
extendir2='0'
extendir3=extendir1

#augf=5#augmentation factor default 3
#test_size=0.1 #split test training percent
########################################################################
######################  end ############################################
########################################################################

#sepextend2='ROI'
if len (extendir3)>0:
    extendir3='_'+extendir3
    
if len (extendir1)>0:
    extendir1='_'+extendir1

pickel_dirsource='th'+str(thrpatch)+'_'+pickel_dirsource_root+'_'+pickel_dirsource_e+'_'+pickel_dirsourcenum+'_'+extendir2+extendir3
print 'path to directory to store patches:',pickel_dirsource

patch_dir=os.path.join(topdir,pickel_dirsource)

print ('classification used:')
for f in classif:
    print (f, classif[f])



#patchesdirnametop = 'th'+str(round(thrpatch,2))+'_'+toppatch+'_'+extendir+'_'+extendir0+extendir1
#patchesdirnametop = 'th0.95_TOPPATCH_all_HUG'


hugeClass=['healthy','back_ground']
#hugeClass=['']

#input patch directory
#patch_dirsource=os.path.join(topdir,patchesdirnametop)
##patch_dirsource=os.path.join(patch_dirsource,patch_dirsource)
#patch_dirsource=os.path.join(patch_dirsource,picklepath)
print 'path for pickle inputs',str(patchesdirnametop)
print 'path to write data for training',patch_dir

#if not os.path.isdir(patch_dirsource):
#    print 'directory ',patch_dirsource,' does not exist'
#    sys.exit()
#print '--------------------------------'

###############################################################
if not os.path.exists(patch_dir):
#remove_folder(patch_dir)
    os.mkdir(patch_dir)
eferror=os.path.join(patch_dir,perrorfile)
errorfile = open(eferror, 'a')
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
errorfile.write('started at :'+todayn)
errorfile.write('--------------------\n')

#errorfile.write('path for pickle inputs'+patch_dirsource+'\n')
errorfile.write('path for pickle inputs'+str(patchesdirnametop)+'\n')

errorfile.write('path to write data for training'+patch_dir+'\n')
errorfile.write('pattern set :'+str(setdata)+'\n')

errorfile.write('--------------------\n')
errorfile.close()

#define a dictionary with labels
patclass={}

print '----------'
#define another dictionaries to calculate the number of label
classNumberInit={}

for f in classif:
    classNumberInit[f]=0
    patclass[f]=[]

category_list=[]
# list all directories under patch directory. They are representing the categories
for f in  patchesdirnametop:
    patch_dirsource=os.path.join(topdir,f)
    patch_dirsource=os.path.join(patch_dirsource,picklepath)
    category_list=category_list+os.walk( patch_dirsource).next()[1]

# print what we have as categories
#print ('all classes in database:',category_list)
#print '----------'
category_list=list(set(category_list))
print ('all classes in database:',category_list)
print '----------'

usedclassifFinal=[f for f in classif if f in category_list]

# print what we have as categories and in used one
print ('all actual classes:',usedclassifFinal)

numclass= len (usedclassifFinal)
print 'number of classes:', numclass
print '----------'
# go through all categories to calculate the number of patches per class
#
for category in usedclassifFinal:
    for f in patchesdirnametop:
        patch_dirsource=os.path.join(topdir,f)
        patch_dirsource=os.path.join(patch_dirsource,picklepath)
        category_dir = os.path.join(patch_dirsource, category)
    #    print  'the path into the categories is: ', category_dir
        if os.path.exists(category_dir):
            sub_categories_dir_list = (os.listdir(category_dir))
            #print 'the sub categories are : ', sub_categories_dir_list
            for subCategory in sub_categories_dir_list:
                subCategory_dir = os.path.join(category_dir, subCategory)
        #        print  'the path into the sub categories is: ',subCategory_dir
                #print subCategory_dir
                image_files = [name for name in os.listdir(subCategory_dir) if name.find('.pkl') > 0 ]
                for filei in image_files:
#                        try:
                            patclass[category]=patclass[category]+pickle.load(open(os.path.join(subCategory_dir,filei),'rb'))
#                        except:
#                            continue
    classNumberInit[category]=len(patclass[category])


total=0
print('number of patches init')
errorfile = open(eferror, 'a')
errorfile.write('number of patches init \n')
for f in usedclassifFinal:
    print ('class:',f,classNumberInit[f])
    errorfile.write('class ' +f+' '+str(classNumberInit[f])+'\n')
    total=total+classNumberInit[f]
print('total:',total)
errorfile.write('total ' +str(total)+'\n')
errorfile.write('--------------------\n')
errorfile.close()
print '----------'

#define coeff max
maxl=0
lif = [name for name in usedclassifFinal if name  not in hugeClass]
for f in lif:
   if classNumberInit[f]>maxl and f not in hugeClass:
#      print 'fmax', f
      maxl=classNumberInit[f]
      patmax=f
print ('max number of patches : '+str(maxl)+' for: '+ patmax)

errorfile = open(eferror, 'a')
errorfile.write ('max number of patches : '+str(maxl)+ ' for:' +patmax+'\n')
errorfile.write('--------------------\n')
errorfile.close()
for i in hugeClass:
    if i in classif:
        print ('number of patches : ',classNumberInit[i],' for: ', i)
print '----------'
#define coeff min
minl=100000000

for f in usedclassifFinal:
   if classNumberInit[f]<minl:
      minl=classNumberInit[f]
      patmin=f
print ('min number of patches : ',str(minl), ' for:' ,patmin)
errorfile = open(eferror, 'a')
errorfile.write ('min number of patches  : '+str(minl)+ ' for:' +patmin+'\n')
print '----------'
errorfile.write('--------------------\n')
errorfile.close()
print ' maximum number of patches ratio'
classConso={}
for f in usedclassifFinal:
    classConso[f]=float(maxl)/max(1,classNumberInit[f])
#for f in usedclassifFinal:
    print (f,classif[f],' {0:.2f}'.format (classConso[f]))
print '----------'

def genedata(category):
    feature=[]
    label=[]
    lc=classif[category]
    for p in patclass[category]:
#            feature.append(p)
#        kernel=(3,3)
#        imageview=cv2.blur(imageview,kernel)
        
#        p=cv2.medianBlur(p,kernel[0])
        feature.append(norm(p))

        label.append(lc)
    return feature,label



def genf(features_train,labels_train,maxl):
    errorfile = open(eferror, 'a')
    errorfile.write(("%10s" %'pat')+
                    ("%10s" %'min')+
                    ("%10s" %'max')+
                     ("%10s" %'average')+
                      ("%10s" %'std')+'\n')
    feature=[]
    featurepat={}
    label=[]
    numpat={}
    for pat in usedclassifFinal:
        numpat[pat]=0
        featurepat[pat]=[]
    for pat in usedclassifFinal:
        numberscan1=len(features_train[pat])
        if numberscan1>0:
#            print pat,numberscan1,maxl
            numberscan=min(numberscan1,maxl)
            for i in range(0,numberscan):
                numpat[pat]+=1
                if  pat in hugeClass:
                     indexpatc =  random.randint(0, len(features_train[pat])-1)  
                     scan=features_train[pat][indexpatc]
                else:
                     scan=features_train[pat][i]
                mask=labels_train[pat][i]
                feature.append(scan)
                featurepat[pat].append(scan)
                label.append(mask)
            if  pat not in hugeClass: 
                for i in range(0,maxl-numberscan): 
                    numpat[pat]+=1
                    indexaug = random.randint(0, 7)
                    if  pat in hugeClass:
                             indexpatc =  random.randint(0, numberscan-1) 
                             scan=geneaug(features_train[pat][indexpatc],indexaug)
                    else:
                             scan=geneaug(features_train[pat][i%numberscan],indexaug)
                       
        #                featurelab.append(tt)
                    mask=labels_train[pat][i%numberscan]
                    feature.append(scan)
                    featurepat[pat].append(scan)
                    label.append(mask)
            tabpat=np.array(featurepat[pat])
            print pat,tabpat.min(),tabpat.max(),np.mean(tabpat),np.std(tabpat)
            errorfile.write(("%10s" % pat)+
                    ("%10s" % str(tabpat.min()))+ 
                    ("%10s" %str(tabpat.max()))+ 
                    ("%10s" %str(round(np.mean(tabpat),3)))+
                    ("%10s" %str(round(np.std(tabpat),3)))+'\n')
        else:
            print pat, 'no data'
    return feature,label,numpat
    


def genfold(features_train,labels_train,maxl):
    feature=[]
    label=[]
    indexpatc={}
    maxindex=0
    totalpat={}

#    print numclass,maxl
    for j in usedclassifFinal:
        indexpatc[j]=0
        totalpat[j]=0
    
    for numgen in range(maxl*numclass):    
            pat =usedclassifFinal[numgen%numclass]
            numberscan=len(features_train[pat])
#                    print pat,len(features_train[pat])
            if  pat in hugeClass:
                 indexpatc[pat] =  random.randint(0, numberscan-1)                       
            else:                                                   
                indexpatc[pat] =  indexpatc[pat]%numberscan
                if indexpatc[pat]>maxindex:
                    maxindex=indexpatc[pat]
                    patmax=pat
            totalpat[pat]+=1
            indexpat=indexpatc[pat]
            indexpatc[pat]=indexpatc[pat]+1           
            indexaug = random.randint(0, 7)
            scan=geneaug(features_train[pat][indexpat],indexaug)
#                    print pat,indexpat,indexaug
            feature.append(scan)            
            mask=classif[pat]
            label.append(mask)
#                    print pat,mask
#                    tt=(scan,mask)
#                    featurelab.append(tt)
#                    feature.append(scan)
#                    label.append(mask)
    
    print 'final number of patches per roi:'
    for pat in usedclassifFinal:
        print pat,totalpat[pat]
    print '--------'
    print 'max index',patmax,maxindex
    
#    shuffle(featurelab)
#    unzipped = zip(*featurelab )
#    feature=(unzipped[0])
#    label=(unzipped[1])
    return feature,label

def geneaug(image,tt):
    if tt==0:
        imout=image
    elif tt==1:
    # 1 90 deg
        imout = np.rot90(image,1)
    elif tt==2:
    #2 180 deg
        imout = np.rot90( image,2)
    elif tt==3:
    #3 270 deg
        imout = np.rot90(image,3)
    elif tt==4:
    #4 flip fimage left-right
            imout=np.fliplr(image)
    elif tt==5:
    #5 flip fimage left-right +rot 90
        imout = np.rot90(np.fliplr(image))
    elif tt==6:
    #6 flip fimage left-right +rot 180
        imout = np.rot90(np.fliplr(image),2)
    elif tt==7:
    #7 flip fimage left-right +rot 270
        imout = np.rot90(np.fliplr(image),3)

    return imout            
####################################################################################
# main program

feature_d ={}
label_d={}
features_test_f=[]
labels_test_f=[]
print ('----------------')
for f in usedclassifFinal:
     print('genedata from images work on :',f)
     feature_d[f],label_d[f]=genedata(f)
#     features_test_f=features_test_f+feature_d[f]
#     labels_test_f=labels_test_f+label_d[f]
     print ('---')
print 'initial number of patches:'
for f in usedclassifFinal:
    print f,len(feature_d[f]),len(label_d[f])
print '-------'
#features_train_f=[]
#labels_train_f=[]
print 'augmentation training'
features_test_f,labels_test_f,numpat= genf(feature_d,label_d,maxl)
print ('----------------') 
print 'after augmentation:'
for f in usedclassifFinal:
    print f,numpat[f]

print ('----------------') 
print ('Number of data and label ')
print 'data test ',len(features_test_f),len(labels_test_f)
print '----'
errorfile = open(eferror, 'a')
errorfile.write('Number of data and label \n')
errorfile.write('data test '+' '+str(len(features_test_f))+' '+str(len(labels_test_f))+'\n')   
errorfile.write('--------------------\n')    
errorfile.close()

print ('------------')
#errorfile = open(eferror, 'a')


#print 'test',len(features_test_f),len(labels_test_f)
#errorfile.write('test '+' '+str(len(features_test_f))+' '+str(len(labels_test_f))+'\n')
#print ('----')
#errorfile.write('--------------------\n')
#errorfile.close()
# transform dataset list into numpy array

X_test = np.array(features_test_f)
y_test = np.array(labels_test_f)

print '-----------FINAL----------------'

print ('Xtest : ',X_test.shape)

print ('ytest : ',y_test.shape)
errorfile = open(eferror, 'a')

errorfile.write('Xtest : '+' '+str(X_test.shape)+'\n') 

errorfile.write('ytest : '+' '+str(y_test.shape)+'\n')   
errorfile.write('--------------------\n')
errorfile.close()


pickle.dump(X_test, open( os.path.join(patch_dir,"X_val.pkl"), "wb" ),protocol=-1)


pickle.dump(y_test, open( os.path.join(patch_dir,"y_val.pkl"), "wb" ),protocol=-1)



recuperated_X_train = pickle.load( open( os.path.join(patch_dir,"X_val.pkl"), "rb" ) )
min_val=np.min(recuperated_X_train)
max_val=np.max(recuperated_X_train)
mean_val=np.mean(recuperated_X_train)
print 'recuperated_X_val', min_val, max_val,mean_val
