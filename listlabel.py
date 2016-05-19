#Sylvain Kritter 4 mai 2016
import os
import shutil
"""The program listlabel.py go through the patient database and put\
 in a file named “listlabel.txt” the complete list of label and \
 localization per dataset"""


#custome file py
from fillshape import *
from generatetabc import *
print('hello, world')
#customisation part
# define the dicom image format bmp or jpg
typei='bmp'

#directory name with patient databases, should be in current directory
namedirtop = 'ILD'
###########end customisation part#####################
#######################################################
def remove_folder(path):
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)
#        print('this direc exist:',path)

listlabel=[]
cwd=os.getcwd()
#cwd = final
#print (cwd)
mf=open('lislabel.txt',"w")
mf.write('label  _  localisation\n')
mf.write('======================\n')

#create patch and jpeg directory

#jpegpath = final/jpeg
#create patch and jpeg directory
listcwd=os.listdir(cwd)
if 'patch' not in listcwd:
    os.mkdir(cwd+'/patch')
patchpath = cwd+'/patch'
#patchpath = final/jpeg
jpegpath=cwd+'/jpeg'
if 'jpeg' not in listcwd:
    os.mkdir(jpegpath)

namedirtopc =cwd+'/'+namedirtop
#namedirtopc= final/ILD_DB_txtROIs
#print(namedirtopc)
listdirc= (os.listdir(namedirtopc))
#print(listcwd)
#print(listdirc)
npat=0
for f in listdirc:
    mf.write(f+'\n')
#    mf.write('----\n')
    #f = 35
    nbpf=0
    posp=f.find('.',0)
    posu=f.find('_',0)
    namedirtopcf=namedirtopc+'/'+f
    remove_folder(namedirtopcf+'/patchfile')
    #namedirtopcf = final/ILD_DB_txtROIs/35
    if posp==-1 and posu==-1:
        contenudir = os.listdir(namedirtopcf)
#        print(contenudir)
     
        if  'patchfile' not in contenudir:
            os.mkdir(namedirtopcf+'/patchfile')
        fif=False
        for f1 in contenudir:
            posp1=f1.find('.txt',0)
            posc1=f1.find('CT',0)
#            print(posp1,posc1)
            if posp1>0 and posc1==0:
                npat+=1
                fileList =f1
                ##f1 = CT-INSPIRIUM-1186.txt
                pathf1=namedirtopcf+'/'+fileList
                #pathf1=final/ILD_DB_txtROIs/35/CT-INSPIRIUM-1186.txt
                print('work on:',f)
                labell,coefi =fileext(pathf1,namedirtopcf,cwd,patchpath)
                print(f, labell)
                fif=True
                for ff in labell:
                    mf.write(ff+'\n')
                mf.write('--------------------------------\n')
                break
        if not fif:
                       print('ERROR: no content file', f)
       
                

mf.write('================================\n')
mf.write('number of datasets:'+str(npat)+'\n')
mf.close()
print('completed')
