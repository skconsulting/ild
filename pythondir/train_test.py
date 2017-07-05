# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017
@author: sylvain
"""
from param_pix import *

t = datetime.datetime.now()
today = str('d_'+str(t.month)+'-'+str(t.day)+'-'+str(t.year)+'_'+str(t.hour)+'_'+str(t.minute))


#pickel_dirsource='pickle_lu_f2'
pickel_dirsource='pickle_ILD1'

#pickel_train=pickel_dirsource
#model_dir='pickle'
model_dir=pickel_dirsource


cwd=os.getcwd()

(cwdtop,tail)=os.path.split(cwd)
pickle_dir=os.path.join(cwdtop,pickel_dirsource)
model_dir=os.path.join(cwdtop,model_dir)


pickle_dir_train=os.path.join(cwdtop,model_dir)
print 'pickle_dir_train',pickle_dir_train

def load_train_data(numpidir):

    X_test = pickle.load( open( os.path.join(numpidir,"X_test.pkl"), "rb" ))
    y_test = pickle.load( open( os.path.join(numpidir,"y_test.pkl"), "rb" ))
    num_class= y_test.shape[3]
    img_rows=X_test.shape[1]
    img_cols=X_test.shape[2]
    num_images=X_test.shape[0]

#    numpatl={}
#    for pat in usedclassif:
#        numpatl[pat]=0
##        print fidclass(classif[pat],classif)
##        print pat, numpatl[pat]
#    y_test1=np.moveaxis(y_test,1,3)
#    for i in range (y_test1.shape[0]):
#        for j in range (0,y_test1.shape[1]):
#            for k in range(y_test1.shape[2]):
#                proba=y_test1[i][j][k]
#                numpat=argmax(proba)     
#                pat=fidclass(numpat,classif)
#                numpatl[pat]+=1
#     
#    for pat in usedclassif:
#        print pat,numpatl[pat]
#    tot=numpatl['back_ground']*1.0
#    for pat in usedclassif:
#        print pat,tot/numpatl[pat]
#        

   
  
    return  X_test, y_test,num_class,img_rows,img_cols,num_images


def load_model_set(pickle_dir_train):
    listmodel=[name for name in os.listdir(pickle_dir_train) if name.find('weights')==0]
    print 'load_model',pickle_dir_train
    ordlist=[]
    for name in listmodel:
#        print name
        nfc=os.path.join(pickle_dir_train,name)
        nbs = os.path.getmtime(nfc)
#        print name,nbs,type(nbs)
        
#        posp=name.find('-')+1
#        post=name.find('.h')
#        numepoch=float(name[posp:post])
        tt=(name,nbs)
#        ordlist.append (tt)
        ordlist.append(tt)

    ordlistc=sorted(ordlist,key=lambda col:col[1],reverse=True)

    namelast=ordlistc[0][0]

    namelastc=os.path.join(pickle_dir_train,namelast)
    print 'last weights :',namelast
    
#    model=load_model(namelastc, custom_objects={'myloss': weighted_categorical_crossentropy})


#    model=load_model(namelastc)

    return namelastc
def load_weight(numpidir):
    class_weights=pickle.load(open( os.path.join(numpidir,"class_weights.pkl"), "rb" ))
    num_class= len(class_weights)

    print 'number of classes :',num_class
  
#    for key,value in class_weights.items():
#        print key, value
    
    clas_weigh_l=[]
    for i in range (0,num_class):
#            print i,class_weights[i]
            clas_weigh_l.append(class_weights[i])
    print 'weights for classes:'
    for i in range (0,num_class):
                    print i, clas_weigh_l[i]
    class_weights_r=np.array(clas_weigh_l)
    
    return num_class,class_weights_r

def pred_and_visu(image,model,store,flip,orig):
    print 'predict and visualise'
    im = Image.open(image) 
    im = im.crop((0,0,319,319)) # WARNING : manual square cropping
#    im = im.crop((140,20,630,475)) 
    im = im.resize((image_rows,image_cols))
#    plt.imshow(np.asarray(im))
    crpim = im # WARNING : we deal with cropping in a latter section, this image is already fit
    
    preds = prediction(model, crpim, transform=False)
    print 'preds.shape)',preds.shape
    if flip:
        preds=np.flip(preds,3)
    predf=preds[0,:,:]
    if store:
        pickle.dump(predf, open("predf", "wb" ),protocol=-1)
    num_class=preds.shape[3]
    print 'number of classes :',num_class
    imclass = np.argmax(preds, axis=3)[0,:,:]
    print 'imclass shape :',imclass.shape    
    imcol=cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
    cv2.imwrite('imorig.bmp',imcol)
    if orig:
        cv2.imwrite('imclassorig.bmp',imclass)   
    else:
        cv2.imwrite('imclassnew.bmp',imclass)
    
    plt.figure(figsize = (15, 7))
    plt.subplot(1,3,1)
    plt.title('image')
    plt.imshow( np.asarray(crpim) )
    plt.subplot(1,3,2)
    plt.title('imclass')
    plt.imshow( imclass )
    plt.subplot(1,3,3)
    plt.imshow( np.asarray(crpim) )
    masked_imclass = np.ma.masked_where(imclass == 0, imclass)
    #plt.imshow( imclass, alpha=0.5 )
    plt.title('imclass + image')
    plt.imshow( masked_imclass, alpha=0.5 )
    # List of dominant classes found in the image
    print 'list of dominant classes found:'
    for c in np.unique(imclass):
        print(c, str(description[0,c][0]))
        
    bspreds = bytescale(preds, low=0, high=255)
    
    plt.figure(figsize = (15, 7))
    plt.subplot(2,3,1)
    plt.imshow(np.asarray(crpim))
    plt.subplot(2,3,3+1)
    plt.title('background')
    plt.imshow(bspreds[0,:,:,class2index['background']], cmap='seismic')
    plt.subplot(2,3,3+2)
    plt.title('bicycle')
    plt.imshow(bspreds[0,:,:,class2index['bicycle']], cmap='seismic')
    plt.subplot(2,3,3+3)
    plt.title('person')
    plt.imshow(bspreds[0,:,:,class2index['person']], cmap='seismic')
    plt.show()
    return im
    
def train():
    
    print('Loading test data...')
    print('-'*30)
    
    listplickle=os.walk(pickle_dir).next()[1]
    print 'list of subsets :',listplickle

    num_class,weights = load_weight(pickle_dir)
    print('Creating and compiling model...')
    print('-'*30)
#    num_class=21
#    model = get_model(num_class,image_rows,image_cols,weights)
    model = get_model(num_class,num_bit,image_rows,image_cols,False,weights)
    
#    im=pred_and_visu('rgb.jpg',model,True,True,True)
  
    
    
      
    listmodel=[name for name in os.listdir(model_dir) if name.find('weights')==0]
    if len(listmodel)>0:
         print 'model found'
         namelastc=load_model_set(model_dir)
         
         model.load_weights(namelastc)  
    else:
         print 'no model found'
#             y_score = model.predict(x_val, batch_size=1,verbose=1)
#             yvf= np.argmax(y_val, axis=3).flatten()
#             ysf=  np.argmax(y_score, axis=3).flatten()   
#             fscore, acc, cm = evaluate(yvf,ysf,num_class)
#             print('Val F-score: '+str(fscore)+'\tVal acc
#    print pickle_dir_train
    
#    numpidirb=model_dir
    
#    mloss = weighted_categorical_crossentropy(weights).myloss
    for numpi in listplickle:
    

        numpidir =os.path.join(pickle_dir,numpi)
        print 'sub directory',numpidir
        
        x_val, y_val, num_class,img_rows,img_cols,num_images= load_train_data(numpidir)
      
#        print num_class
        print 'shape x_val :',x_val.shape
        print 'shape y_val :',y_val.shape
        print('-'*30)
        print 'number of images:', num_images
        print 'number of classes:', num_class
        print 'image number of rows :',img_rows
        print 'image number of columns :',img_cols

#        numpidirb=os.path.join(pickle_dir,numpi)
        
        
        print('Predict model...')
        print('-'*30)
        y_score = model.predict(x_val, batch_size=1,verbose=1)
        yvf= np.argmax(y_val, axis=3).flatten()
        ysf=  np.argmax(y_score, axis=3).flatten()   

        f=open (pickle_dir+'/'+numpi+'_res.txt','w')
        fscore, acc, cm = evaluate(yvf,ysf,num_class)

        print('Val F-score: '+str(fscore)+'\tVal acc: '+str(acc))    
        np.set_printoptions(threshold=np.nan)
        print cm
        f.write('f-score is : '+ str(fscore)+'\n')
        f.write( 'accuray is : '+ str(acc)+'\n')
        f.write('confusion matrix\n')
        n= cm.shape[0]
        for i in range (0,n):
            for j in range (0,n):
                f.write(str(cm[i][j])+' ')
            f.write('\n')
        f.close()
       
        
#print 'pickle_dir_train',pickle_dir_train       
train()