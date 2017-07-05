# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017
Create Xtrain etc data for training 2nd step
@author: sylvain

"""

#from __future__ import print_function
from param_pix import *

#nameHug='HUG'
#nameHug='CHU'
nameHug='DUMMY'
toppatch= 'TOPPATCH'
#extension for output dir
#extendir='ILD_TXT'
#extendir='UIP'
#extendir='ILD6'
#extendir='ILD5'


#extendir='S3'
extendir='lu_f4'
#extendir='small1'
ldummy=False#True for geometric images

extendir2=''
pklnum=1
pickel_dirsource_root='pickle'

##############################################################

sepextend2=''
if len (extendir2)>0:
    sepextend2='_'

pickel_dirsource=pickel_dirsource_root+'_'+extendir+sepextend2+extendir2


#cwd=os.getcwd()
#
#(cwdtop,tail)=os.path.split(cwd)
pickle_dir=os.path.join(cwdtop,pickel_dirsource)
remove_folder(pickle_dir)
os.mkdir(pickle_dir)

def get_class_weights(y):
    counter = collections.Counter(y)
    majority = max(counter.values())
#    for cls, count in counter.items():
#        print cls, count
    return  {cls: float(majority/count) for cls, count in counter.items()}

#remove_folder(pickle_dir)


path_HUG=os.path.join(cwdtop,nameHug)
patchesdirnametop = toppatch+'_'+extendir
patchtoppath=os.path.join(path_HUG,patchesdirnametop)
patchpicklename='picklepatches.pkl'
picklepath = 'picklepatches'
picklepathdir =os.path.join(patchtoppath,picklepath)

def countclasses():
    for category in usedpatient:
        category_dir = os.path.join(picklepathdir, category)
        print  'the patients are:  ', category
        image_files = [name for name in os.listdir(category_dir) if name.find('.pkl') > 0 ]
        for filei in image_files:
            usedpatient[category]=usedpatient[category]+1



def readclasses1(usedpatient,ldummy):
    patch_list=[]
    label_list=[]

    for category in usedpatient:
        category_dir = os.path.join(picklepathdir, category)
        print  'work on: ', category
        image_files = [name for name in os.listdir(category_dir) if name.find('.pkl') > 0 ]
        for filei in image_files:
            pos=filei.find('_')
            numslice=filei[0:pos]
#            usedpatient[category]=usedpatient[category]+1
            usedpatientlist[category].append(numslice)
            readpkl=pickle.load(open(os.path.join(category_dir,filei), "rb"))
            for i in range (len(readpkl[0])):
                                
                
                scan=readpkl[0][i]
#                print scan.min(), scan.max(),scan.shape , type(scan[0][0])
                mask=readpkl[1][i]    
#                print mask.min(), mask.max(),mask.shape , type(mask[0][0])
#                print mask.min(), mask.max(),mask.shape , type(mask[0][0])
                if not ldummy:
                    scanm=norm(scan)
#                    print scan.min(), scan.max(),scan.shape , type(scan[0][0])
#                    print scanm.min(), scanm.max(),scanm.shape , type(scanm[0][0])
                

                else:
                     scanm=preprocess_batch(scan)
#                     print scan.min(), scan.max(),scan.shape , type(scan[0][0])
#                     print scanm.min(), scanm.max(),scanm.shape , type(scanm[0][0])
#                     ooo
#                o=normi(scan)
#                on=normi(scanm)
#                n=normi(mask)
#                cv2.imshow('scan',o)
#                cv2.imshow('scann',on)
#                cv2.imshow('mask',n)
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()
#                ooo
                patch_list.append(scanm)
                label_list.append(mask)
    return patch_list, label_list   

def numbclasses(y):
    y_train = np.array(y)
    uniquelbls = np.unique(y_train)
    for pat in uniquelbls:
        print  fidclass(pat,classif)
    
    nb_classes = int( uniquelbls.shape[0])
    print ('number of classes  in this data:', int(nb_classes)) 
    nb_classes = len( classif)
    print ('number of classes  in this set:', int(nb_classes)) 
    y_flatten=y_train.flatten()
    class_weights= get_class_weights(y_flatten)
#    class_weights[0]=class_weights[0]/200.0
#    class_weights[1]=class_weights[1]/100.0
    
    return int(nb_classes),class_weights

def readclasses2(num_classes,X_trainl,y_trainl,ldummy):

    X_traini, X_testi, y_traini, y_testi = train_test_split(X_trainl,
                                        y_trainl,test_size=0.2, random_state=42)
    
    
    if not ldummy:
        X_train = np.asarray(np.expand_dims(X_traini,3)) 

        X_test = np.asarray(np.expand_dims(X_testi,3))  
        if num_bit==3:
            X_train=np.repeat(X_train,3,axis=3)
            X_test=np.repeat(X_test,3,axis=3)
            
    else:
        if num_bit==3:
            X_train = np.asarray(X_traini) 
            X_test = np.asarray(X_testi)  
        else:
            X_train = np.asarray(X_traini) 
            X_train = np.expand_dims(X_train,3) 
            X_test = np.asarray(X_testi)  
            X_test = np.expand_dims(X_test,3)
    y_train = np.array(y_traini)
    y_test = np.array(y_testi)
#    print X_train.shape
#    print y_train.shape
#    print y_train[3].min(),y_train[3].max()
#    o=normi(X_train[3])
#    x=normi(y_train[3])
#    print x.min(),x.max()
###            f=normi(tabroif)
#    cv2.imshow('X_train',o)
#    cv2.imshow('y_train',x)
###            cv2.imshow('tabroif',f)
#    cv2.imwrite('a.bmp',o)
#    cv2.imwrite('b.bmp',x)
#    cv2.imwrite('c.bmp',y_train[3])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    lytrain=y_train.shape[0]
    lytest=y_test.shape[0]
#    print lytrain,lytest
   
    ytrainr=np.zeros((lytrain,image_rows,image_cols,int(num_classes)),np.uint8)
    ytestr=np.zeros((lytest,image_rows, image_cols,int(num_classes)),np.uint8)
    
    
    for i in range (lytrain):
        for j in range (0,image_rows):
#            print 'y_train[i][j]',y_train[i][j]
#            print y_train[i][j].shape
            
            ytrainr[i][j] = np_utils.to_categorical(y_train[i][j], num_classes)
#            print 'ytrainr[i][j]',ytrainr[i][j]

    for i in range (lytest):
        for j in range (0,image_rows):
            ytestr[i][j] = np_utils.to_categorical(y_test[i][j], num_classes)
  
    """
    for i in range (lytrain):
#        yt=y_train[i].reshape(image_rows*image_cols)
        yt=y_train[i]

#        print yt.shape
        ytrainr[i]=np_utils.to_categorical(yt, num_classes)
    for i in range (lytest):
#        yt=y_train[i].reshape(image_rows*image_cols)
        yt=y_train[i]
        print 'yt.shape',yt.shape
        ytestr[i]=np_utils.to_categorical(yt, num_classes)
        print 'ytestr[i].shape',ytestr[i].shape
    
    print ytrainr.shape
    print ytestr.shape
#    ooo
    """
#    print 'ytestr[i].shape',ytestr[0].shape
#    print 'ytestr[i].shape',ytestr[0][100][100]
        
    
    return X_train, X_test, ytrainr, ytestr  
#    return X_train, X_test, y_train, y_test   


#start main

   
usedpatient={}
usedpatientlist={}

listpatient=[name for name in os.listdir(picklepathdir)]
for c in listpatient:
    usedpatient[c]=0
    usedpatientlist[c]=[]
    
countclasses()
print ('patients found:')
print ( usedpatient)
print('----------------------------')
if ldummy :
    print 'this is dummy'
    print('----------------------------')
list_patient=[]
totalpatient=0
for k,value in usedpatient.items():   
    list_patient.append(k)
    totalpatient=totalpatient+value
print 'total number of patients :',totalpatient
print('----------------------------')
#print list_patient
spl=np.array_split(list_patient,pklnum)

X_trainl={}
y_trainl={}
y_trainlist=[]

for i in range(pklnum):
    totalpatienti=0
    listp=spl[i]
    print 'set number :',i,' ',listp
    print('-' * 30)
    for k in listp:   
        totalpatienti=totalpatienti+usedpatient[k]
    print 'total number of patients in set ',i,' :',totalpatienti
    X_trainl[i],y_trainl[i]=readclasses1(listp,ldummy)     
    y_trainlist=y_trainlist+y_trainl[i]

num_classes,class_weights=numbclasses(y_trainlist)
print "weights"
setvalue=[]
for key,value in class_weights.items():
   print key, fidclass (key,classif), value
   setvalue.append(key)
print('-' * 30)
print 'after adding for non existent :'
for numw in range(num_classes):
    if numw not in setvalue:
        class_weights[numw]=1
for key,value in class_weights.items():
   print key, fidclass (key,classif), value
print('-' * 30)

for i in range(pklnum):
    print 'work on subset :',i
    diri=os.path.join(pickle_dir,str(i))
    remove_folder(diri)
    os.mkdir(diri)
    X_train, X_test, y_train, y_test =readclasses2(num_classes,X_trainl[i],y_trainl[i],ldummy)
    print 'shape X_train :',X_train.shape
    print 'shape X_test :',X_test.shape
    print 'shape y_train :',y_train.shape
    print 'shape y_test :',y_test.shape
    print('-' * 30)
    pickle.dump(X_train, open( os.path.join(diri,"X_train.pkl"), "wb" ),protocol=-1)
    pickle.dump(y_train, open( os.path.join(diri,"y_train.pkl"), "wb" ),protocol=-1)
    pickle.dump(X_test, open( os.path.join(diri,"X_test.pkl"), "wb" ),protocol=-1)
    pickle.dump(y_test, open( os.path.join(diri,"y_test.pkl"), "wb" ),protocol=-1)
pickle.dump(class_weights, open( os.path.join(pickle_dir,"class_weights.pkl"), "wb" ),protocol=-1)
debug=True
if debug:
    xt=  pickle.load(open( os.path.join(diri,"X_train.pkl"), "rb" ))
    yt=pickle.load(open( os.path.join(diri,"y_train.pkl"), "rb" ))
    xcol=30
    ycol=20
    print 'xt', xt.shape
    print 'xt', xt[0][:,:,0].shape
    print 'xt[3][0][0]',xt[0][0][0]
    print 'xt[3][350][160]',xt[0][ycol][xcol]
    print 'yt', yt.shape
    print 'yt[3][0][0]',yt[0][0][0]
    print 'yt[3][350][160]',yt[0][ycol][xcol]
    print 'xt min max', xt[0].min(), xt[3].max()
    print 'yt min max',yt[0].min(), yt[3].max()
    plt.figure(figsize = (5, 5))
    #    plt.subplot(1,3,1)
    #    plt.title('image')
    #    plt.imshow( np.asarray(crpim) )
    plt.subplot(1,2,1)
    plt.title('image')
    plt.imshow( normi(xt[3][:,:,0]*10).astype(np.uint8) )
    plt.subplot(1,2,2)
    plt.title('label')
    plt.imshow( np.argmax(yt[3],axis=2) )
