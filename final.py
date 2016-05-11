#Sylvain Kritter 4 mai 2016
import os
import numpy as np
import shutil


#custome file py
from fillshape import *
from generatetabc import *
print('hello, world')
#customisation part
# define the dicom image format bmp or jpg
typei='bmp'
#dicom file size in pixels
dimtabx = 512
dimtaby = 512
#patch size in pixels 32 * 32
dimpavx = 32
dimpavy = 32
#threshold for patch
thr=0.8
#directory name with patient databases, should be in current directory
namedirtop = 'ILD_DB_txtROIs'
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

#create patch and jpeg directory
listcwd=os.listdir(cwd)
if 'patch' not in listcwd:
    os.mkdir(cwd+'/patch')
patchpath = cwd+'/patch'
#patchpath = final/jpeg
jpegpath=cwd+'/jpeg'
if 'jpeg' not in listcwd:
    os.mkdir(jpegpath)

#jpegpath = final/jpeg

namedirtopc =cwd+'/'+namedirtop
#namedirtopc= final/ILD_DB_txtROIs
#print(namedirtopc)
listdirc= (os.listdir(namedirtopc))
#print(listcwd)
#print(listdirc)

for f in listdirc:
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
        for f1 in contenudir:
            posp1=f1.find('.txt',0)
            posc1=f1.find('CT',0)
#            print(posp1,posc1)
            if posp1>0 and posc1==0:
                fileList =f1
                ##f1 = CT-INSPIRIUM-1186.txt
                pathf1=namedirtopcf+'/'+fileList
                #pathf1=final/ILD_DB_txtROIs/35/CT-INSPIRIUM-1186.txt
                print('work on:',f)
                labell,coefi =fileext(pathf1,namedirtopcf,cwd,patchpath)
#                print(label,loca)
        
        listslice= os.listdir(namedirtopcf+'/patchfile') 
#        print('listslice',listslice)
        listcore =[]
        for l in listslice:
#                print(pathl)
            il1=l.find('.',0)
            j=0
            while l.find('_',il1-j)!=-1:
                j-=1
            ilcore=l[0:il1-j-1]
            if ilcore not in listcore:
                listcore.append(ilcore)
    #pathl=final/ILD_DB_txtROIs/35/patchfile/slice_2_micronodulesdiffuse_1.txt
#        print('listcore',listcore)
        for c in listcore:
            
            ftab=True
            tabzc = np.zeros((dimtabx, dimtaby), dtype='i')
            for l in listslice:
#                print('l',l,'c:',c)
                if l.find(c,0)==0:
                    pathl=namedirtopcf+'/patchfile/'+l
                    tabcff = np.loadtxt(pathl,dtype='f')
                    ofile = open(pathl, 'r')
                    t = ofile.read()
                    #print( t)
                    ofile.close()
                    labpos=t.find('label')
                    labposend=t.find('\n',labpos)
                    labposdeb = t.find(' ',labpos)
                    label=t[labposdeb:labposend].strip()
                    locapos=t.find('local')
                    locaposend=t.find('\n',locapos)
                    locaposdeb = t.find(' ',locapos)
                    loca=t[locaposdeb:locaposend].strip()
#                print(label,loca)
                #print(tabcff,coefi)
                    tabccfi=tabcff/coefi
#                print(tabccfi)
                    tabc=tabccfi.astype(int)
                    
#                print(tabc)
#                    print('generate tables from:',l,'in:', f)
                    tabz= reptfull(tabc,dimtabx,dimtaby)
                    tabzc=tabz+tabzc
                    if ftab:
                        reftab=tabc
                        ftab=False
                    
                    reftab=np.concatenate((reftab,tabc),axis=0)
                    
#                    print('end create tables')
                    il=l.find('.',0)
                    iln=l[0:il]
                #coefi=0.79296875
                #iln=slice2micronodulesdiffuse
                #label=micronodules
                #loca=diffuse
#                print(tabc)
#            print('creates patches from:',iln, 'in:', f)
            nbp,tabz1=pavs (reftab,tabzc,dimtabx,dimtaby,dimpavx,dimpavy,namedirtopcf,\
                jpegpath, patchpath,thr,iln,f,label,loca,typei)
#            print('end create patches')
            nbpf=nbpf+nbp
#    print(f,nbpf)
    ofilepw = open(jpegpath+'/nbpat_'+f+'.txt', 'w')
    ofilepw.write('number of patches: '+str(nbpf))
    ofilepw.close()
#   
contenupatcht = os.listdir(jpegpath) 
#        print(contenupatcht)

npatcht=0
for npp in contenupatcht:
#    print('1',npp)
    if npp.find('.txt')>0 and npp.find('nbp')==0:
#        print('2',npp)
        ofilep = open(jpegpath+'/'+npp, 'r')
        tp = ofilep.read()
#        print( tp)
        ofilep.close()
        numpos2=tp.find('number')
        numposend2=len(tp)
        #tp.find('\n',numpos2)
        numposdeb2 = tp.find(':',numpos2)
        nump2=tp[numposdeb2+1:numposend2].strip()
#        print(nump2)
        numpn2=int(nump2)
        npatcht=npatcht+numpn2
#        print(npatch)
ofilepwt = open(jpegpath+'/totalnbpat.txt', 'w')
ofilepwt.write('number of patches: '+str(npatcht))
ofilepwt.close()
                
print('completed')
   
