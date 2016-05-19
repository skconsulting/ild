#Sylvain Kritter 4 mai 2016
import os
import numpy as np
import shutil
import scipy.misc
from matplotlib import pyplot as plt
from PIL import Image
 

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

patchpath = cwd+'/patch_test'
jpegpath = cwd+'/jpeg-test'
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






def      pavgene (img,tabim,dx,dy,px,py,patchpathf,jpegpathf,typei):
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
    print(slicenumber)
#    tabp = np.zeros((dx, dy), dtype='i')
#    tabf = np.copy(tab)
#    pxy=float(px*py)
#
#    i=max(mintabx-px,0)
#    nbp=0
#    strpac=''
    mini=dx-px
    minj=dy-py
#    maxj=max(mintaby-py,0)
    i=0
    while i <= mini:
        j=0
#        j=maxj
        while j<=minj:
#            print(i,j)
#            area=0
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

    
##                crorig.show()
##                         print(label,loca)
#                         crorig.save(patchpath+'/'+label+'/'+loca+'/'+f+\
#                         '_'+iln+'_'+str(nbp)+'.'+typei)
#                         break
#                if not imf:
#                        print('ERROR image not found',namedirtopcf+'/'+typei+'/'+n)
#                strpac=strpac+str(i)+' '+str(j)+'\n'
#                #                print('pavage',i,j)
#                x=0
#                #we draw the rectange
#                while x < px:
#                    y=0
#                    while y < py:
#                        tabp[y+j][x+i]=4
#                        if x == 0 or x == px-1 :
#                            y+=1
#                        else:
#                            y+=py-1
#                    x+=1
#                #we cancel the source
#                x=0
#                while x < px:
#                    y=0
#                    while y < py:
#                        tabf[y+j][x+i]=0
#                        y+=1
#                    x+=1
#            j+=1
#        i+=1
#    tabp =tab+tabp
#    mf=open(jpegpath+'/'+f+'_'+iln+'.txt',"w")
#    mf.write('#number of patches: '+str(nbp)+'\n'+strpac)
#    mf.close()
#    scipy.misc.imsave(jpegpath+'/'+f+'_'+iln+'.jpg', tabp)
##    im = plt.matshow(tabp)
##    plt.colorbar(im,label='with pavage')
###    im = plt.matshow(tabf)
###    plt.colorbar(im,label='ff')
##    plt.show
##    print('fin')
#    return nbp,tabp



for f in listdirc:
    #f = 35
    namedirtopcf=os.path.join(namedirtopc,f)
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
        print(listbmp)
        for img in listbmp:
             bmpfile = os.path.join(bmpdir,img)
             im = Image.open(bmpfile)
             tabim = np.array(im)
             tabim=tabim
#             tabims = tabim[:,:,1]
#             print('1',tabim[0])
#             print('2',tabims[0])
#             im = plt.matshow(tabims)
#             plt.colorbar(im,label='with pavage')

             pavgene (bmpfile,tabim,dimtabx,dimtaby,dimpavx,\
             dimpavy,patchpathf,jpegpathf,typei)
            
    
print('completed')
   
