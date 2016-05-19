#Sylvain Kritter 12 mai 2016
"""From scan image, generates patches bmp files according to patient database\
 and lung mask.\
 A patch is considered valid if the recovering area is above a threshold """

import os
import numpy as np
import shutil
import scipy.misc
from matplotlib import pyplot as plt
from PIL import Image
errorfile = open('errorfile.txt', 'w')



#custome file py

print('hello, world')
#customisation part
# define the dicom image format bmp or jpg
typei='bmp'
#dicom file size in pixels
dimtabx = 512
dimtaby = 512
#patch size in pixels 32 * 32
dimpavx =32
dimpavy = 32
#threshold for patch
thr=0.8
#directory name with patient databases, should be in current directory
namedirtop = 'ILD1'
###########end customisation part#####################
#######################################################
def remove_folder(path):
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)
#        print('this direc exist:',path)



cwd=os.getcwd()
#cwd = final
#print (cwd)

#create patch and jpeg directory
listcwd=os.listdir(cwd)

patchpath = cwd+'/patch_test_mask'
jpegpath = cwd+'/jpeg_test_mask'
remove_folder(patchpath)
remove_folder(jpegpath)
os.mkdir(patchpath)
os.mkdir(jpegpath)
    
    
namedirtopc =cwd+'/'+namedirtop
#namedirtopc= final/ILD_DB_txtROIs
#print(namedirtopc)
listdirc= (os.listdir(namedirtopc))
#print(listcwd)
#print(listdirc)






def pavgene (img,tabim,tablung,dx,dy,px,py,patchpathf,jpegpathf,typei,thr):
    """ generate patches from scan"""
    #img=CT-2170-0002.bmp
    #typei = bmp
#    tabp = np.zeros((dx, dy), dtype='i')
    tabf = np.copy(tabim)
    endnumslice=img.find('.bmp')
    posend=endnumslice
    while img.find('-',posend)==-1:
        posend-=1
    debnumslice=posend+1
    slicenumber=img[debnumslice:endnumslice]
#    print(slicenumber)
#    tabp = np.zeros((dx, dy), dtype='i')
#    tabf = np.copy(tab)
#    pxy=float(px*py)
#
#    i=max(mintabx-px,0)
#    nbp=0
#    strpac=''
    mini=dx-px
    minj=dy-py
    pxy=float(px*py)
#    maxj=max(mintaby-py,0)
    i=0
    while i <= mini:
        j=0
#        j=maxj
        while j<=minj:
#            print(i,j)
            area=0
            x=0
            while x < px:
                y=0
                while y < py:
                   if tablung[y+j][x+i] >0:
                       area = area+1
                   y+=1
                x+=1
#            if i== 96 and j ==224:
#                print(area,pxy)    
            if area/pxy>thr:
                orig = Image.open(img)
                crorig = orig.crop((i, j, i+px, j+py))
                crorig.save(patchpathf+'/p_'+slicenumber+'_'+str(i)+'_'+\
                       str(j)+'.'+typei)
                       #we draw the rectange
                x=0
                while x < px:
                    y=0
                    while y < py:
                        tabf[y+j][x+i]=[255,0,0]
                        if x == 0 or x == px-1 :
                            y+=1
                        else:
                            y+=py-1
                    x+=1
            
            j+=py    
        i+=px
#    im = plt.matshow(tabf)
#    plt.colorbar(im,label='with pavage')
    scipy.misc.imsave(jpegpathf+'/'+'s_'+slicenumber+'.jpg', tabf)


def interv(borne_inf, borne_sup):
    """Générateur parcourant la série des entiers entre borne_inf et borne_sup.
    inclus
    Note: borne_inf doit être inférieure à borne_sup"""
    
  
    while borne_inf <= borne_sup:
        yield borne_inf
        borne_inf += 1

for f in listdirc:
    #f = 35
    print('work on: ',f)
    namedirtopcf=os.path.join(namedirtopc,f)
    namemask=os.path.join(namedirtopcf,'lung_mask/bmp')
    if os.path.isdir(namedirtopcf):
    #namedirtopcf = final/ILD_DB_txtROIs/35
        bmpdir = os.path.join(namedirtopcf,typei)
        patchpathf=os.path.join(patchpath,f)
        jpegpathf=os.path.join(jpegpath,f)
        remove_folder(patchpathf)
        os.mkdir(patchpathf)
        remove_folder(jpegpathf)
        os.mkdir(jpegpathf)
        listbmp= os.listdir(bmpdir)
#        print(listbmp)
        listlungbmp= os.listdir(namemask)
#        print(listlungbmp)
        for img in listbmp:
             endnumslice=img.find('.bmp')
             posend=endnumslice
             while img.find('-',posend)==-1:
                     posend-=1
             debnumslice=posend+1
             slicenumber=int(img[debnumslice:endnumslice])
#             print('sln:',slicenumber,'img:', img,debnumslice,endnumslice)
             
             slns='_'+str(slicenumber)+'.'+typei
#             print(slns)
             for llung in listlungbmp:
                tflung=False
#                print(llung)
#                print(listlungbmp)

                if llung.find(slns) >0:
                    tflung=True
                    lungfile = os.path.join(namemask,llung)
#                    print(lungfile)
                    imlung = Image.open(lungfile)
                    tablung = np.array(imlung)
                    tablungs = tablung[:,:,1]
                    for i in interv(0,25):
                        for j in interv (0,511):
#                            print(i,j)
                            tablungs[i][j]=0
#                    print(tablungs.min(),tablungs.max())
                    for i in interv(0,511):
                        for j in interv (0,511):
#                            print(i,j)
                            if tablungs[i][j]> 0:
                                tablungs[i][j]=1
#                                print(tablungs[i][j])
#                    im = plt.matshow(tablungs)
#                    plt.colorbar(im,label='lung mask')
                    break
             if not tflung:
                    errorfile.write('ERROR lung mask not found '+slns+' in: '+f) 
                    print('ERROR lung mask not found ',slns,' in: ',f)
                     
             bmpfile = os.path.join(bmpdir,img)
             im = Image.open(bmpfile)
             tabim = np.array(im)         
#             tabims = tabim[:,:,1]
#             print('1',tabim[0])
#             print('2',tabims[0])
#             im = plt.matshow(tabims)
#             plt.colorbar(im,label='source')

             pavgene (bmpfile,tabim,tablungs,dimtabx,dimtaby,dimpavx,\
             dimpavy,patchpathf,jpegpathf,typei,thr)
            
errorfile.close()
print('completed')
   
