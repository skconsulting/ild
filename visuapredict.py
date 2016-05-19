#Sylvain Kritter 18 mai 2016
"""From predicted patches from scan image, visualize results as image overlay """

import os
import numpy as np
import shutil
import scipy.misc
import cPickle as pickle
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw

#font file imported in top directory
font = ImageFont.truetype( 'fonts/arial.ttf', 20)


errorfile = open('errorfile.txt', 'w')
#errorfile.close()

print('hello, world')

#######################################################
#customisation part
# define the dicom image format bmp or jpg
typei='bmp'
#dicom file size in pixels
dimtabx = 512
dimtaby = 512
#patch size in pixels 32 * 32
dimpavx =32
dimpavy = 32
#threshold for probability
thr = 0.8

#pickle with predicted classes
precclaf='predicted_classes.pkl'
 #pickle with predicted probabilities
precprobf='predicted_probabilities.pkl'


cwd=os.getcwd()
#directory name with patient databases, should be in current directory
dirpatientdb = os.path.join(cwd,'ILD')

#directory name with patch db databases, should be in current directory
dirpatientpatchdb = os.path.join(cwd,'predict')

#directory name with predict out dabasase, will be created in current directory
dirpatientoutdb = os.path.join(cwd,'predict_out')

#subdirectory name to put images
imgdirname = 'jpeg'

#subdirectory name to colect pkl files resulting from prediction
predoutdirname = 'predict'

#end customisation part

#########################################################
#color of labels
red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255,255,0)
cyan=(0,255,255)
purple=(255,0,255)
white=(255,255,255)

classif ={'consolidation':0,
 'fibrosis':1,
 'ground_class':2,
 'healthy':3,
 'micronodules':4,
 'reticulation':5}
classifc ={'consolidation':red,
 'fibrosis':blue,
 'ground_class':yellow,
 'healthy':green,
 'micronodules':cyan,
 'reticulation':purple}



def fidclass(numero):
    """return class from number"""
    found=False
    for cle, valeur in classif.items():
        
        if valeur == numero:
            found=True
            return cle
      
    if not found:
        return 'unknown'


def interv(borne_inf, borne_sup):
    """Générateur parcourant la série des entiers entre borne_inf et borne_sup.
    inclus
    Note: borne_inf doit être inférieure à borne_sup"""
    
    while borne_inf <= borne_sup:
        yield borne_inf
        borne_inf += 1

def tagview(fig,text,pro,x,y):
    """write text in image according to label and color"""
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    col=classifc[text]

    deltay=25*(classif[text]%3)
    deltax=175*(classif[text]//3)
    #print text, col
    draw.text((x+deltax, y+deltay),text+' '+pro,col,font=font)
    imgn.save(fig)

def tagviews(fig,text,x,y):
    """write simple text in image """
    imgn=Image.open(fig)
    draw = ImageDraw.Draw(imgn)
    draw.text((x, y),text,white,font=font)
    imgn.save(fig)

def maxproba(proba):
    """looks for max probability in result"""
    lenp = len(proba)
    m=0
    for i in interv(0,lenp-1):
        if proba[i]>m:
            m=proba[i]
    return m



###########end customisation part#####################
#######################################################
def remove_folder(path):
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)
#        print('this direc exist:',path)

    
listpatient=os.listdir(dirpatientdb)    
#listpatient =     121

def loadpkl(f,do):
    """crate image directory and load pkl files"""
    dop =os.path.join(do,predoutdirname)
    #pickle with predicted classes
    preclasspick= os.path.join(dop,precclaf)
    #pickle with predicted probabilities
    preprobpick= os.path.join(dop,precprobf)
    """generate input tables from pickles"""
    dd = open(preclasspick,'rb')
    my_depickler = pickle.Unpickler(dd)
    preclass = my_depickler.load()
#    preclass[0]=0
#    preclass[1]=1
#    preclass[2]=2
#    preclass[3]=3
#    preclass[4]=4
#    preclass[5]=5
    dd.close()
    dd = open(preprobpick,'rb')
    my_depickler = pickle.Unpickler(dd)
    preprob = my_depickler.load()
    dd.close()  
    return (preclass,preprob)


for f in listpatient:
    print('work on: ',f)
    do = os.path.join(dirpatientoutdb,f)
    #os.mkdir(do)
    doj =os.path.join(do,imgdirname)
    remove_folder(doj)
    os.mkdir(doj)
    (preclass,preprob)=loadpkl(f,do)
   
    #we build the list of pach names in patient database
    listnamepatch=[]
    pathbmp=os.path.join(dirpatientpatchdb,f)
    listbmppatch= os.listdir(pathbmp)
    for ff in listbmppatch:
            listnamepatch.append(ff)
    #print listnamepatch
    #print preprob
    #f = 35
    dirpatientf=os.path.join(dirpatientdb,f)
    dirpatientfdb=os.path.join(dirpatientf,typei)
    #print dirpatientfdb
    listbmpscan=os.listdir(dirpatientfdb)
    
    #print listbmpscan
    setname=f
    
    for img in listbmpscan:
        listlabel={}
        listlabelaverage={}
        imgc=os.path.join(dirpatientfdb,img)
#        print img  
        endnumslice=img.find('.'+typei)
        imgcore=img[0:endnumslice]
        #print imgcore
        posend=endnumslice
        while img.find('-',posend)==-1:
            posend-=1
        debnumslice=posend+1
        slicenumber=int((img[debnumslice:endnumslice])) 
        imscan = Image.open(imgc)
        tablscan = np.array(imscan)
        for i in interv(0,25):
           for j in interv (0,511):
              tablscan[i][j]=0
    #initialise index in list of results
        ill = 0
      
        foundp=False
        for ll in listnamepatch:
           
            #we read patches in predict/ setnumber and found localisation    
            debsl=ll.find('_')+1
            endsl=ll.find('_',debsl)
            slicename=int(ll[debsl:endsl])
            debx=ll.find('_',endsl)+1
            endx=ll.find('_',debx)
            xpat=int(ll[debx:endx])
            deby=ll.find('_',endx)+1
            endy=ll.find('.',deby)
            ypat=int(ll[deby:endy])

        #we found label from prediction
            prec=int(preclass[ill])
            classlabel=fidclass(prec)
            classcolor=classifc[classlabel]
        #we found max proba from prediction
            proba=preprob[ill]
            mproba=round(maxproba(proba),2)
#            print(mproba)
            #print(setname, slicename,xpat,ypat,classlabel,classcolor,mproba)
            ill+=1  
            if mproba >thr and slicenumber == slicename:
#                    print(setname, slicename,xpat,ypat,classlabel,classcolor,mproba)
                    
                    foundp=True
                    if classlabel in listlabel:
                        numl=listlabel[classlabel]
                        cur=listlabelaverage[classlabel]
#                        print (numl,cur)
                        listlabel[classlabel]=numl+1
                        averageproba= round((cur*numl+mproba)/(numl+1),2)
                        listlabelaverage[classlabel]=averageproba
                    else:
                        listlabel[classlabel]=1
                        listlabelaverage[classlabel]=mproba
                 
#                        listlabel.append((classlabel,mproba))
                    x=0
                    while x < dimpavx:
                        y=0
                        while y < dimpavy:
                            tablscan[y+ypat][x+xpat]=classcolor
                            if x == 0 or x == dimpavx-1 :
                                y+=1
                            else:
                                y+=dimpavy-1
                        x+=1
                        #tablscans=tablscan[:,:,1]
                        #im = plt.matshow(tablscans)
                        #plt.colorbar(im,label='scan mask')
         #plt.show()
        imgname=os.path.join(doj,imgcore)+'.jpeg'
        scipy.misc.imsave(imgname, tablscan)
        if foundp:
#                print( 'work on patient: ',f,' slice number: ', slicename,slicenumber)
    #              print (imgcore,slicenumber,listlabel,mproba)
            
            tagviews(imgname,'average probability',0,0)
            for ll in listlabel:
                tagview(imgname,ll,str(listlabelaverage[ll]),175,00)
        else:   
            tagviews(imgname,'no recognised label',0,0)
            errorfile.write('no recognised label in: '+str(f)+' '+str (img)+'\n' )
            print('no recognised label in: '+str(f)+' '+str (img) )
errorfile.close()                            
print('completed')
            
