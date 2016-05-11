#Sylvain Kritter 2 mai 2016
import numpy as np
import os
from matplotlib import pyplot as plt
import scipy.misc
from PIL import Image

def repts(tabc,dx,dy):
    """ we fill tab with summits of tabc"""
    tab = np.zeros((dx, dy), dtype='i')
    i=0
    tabcs0=tabc.shape[0]

    while i < tabcs0:
         x1 =min(511,tabc[i][0])
         y1 =min(511,tabc[i][1]) 
#         print(x1,y1)
         tab[y1][x1]=1
         if i<  tabcs0-1:
             i+=1
             x2 =min(511,tabc[i][0]) 
             y2 =min(511,tabc[i][1]) 
             tab[y2][x2]=1
         i+=1
    return tab

#tabz1 = np.zeros((10, 10), dtype='i')
def repti(tabc,dx,dy):
    """ draw line in between summits from tab"""
    tab = np.zeros((dx, dy), dtype='i')
    tabcs0=tabc.shape[0]
    i=0
    while i < tabcs0:
        x1 =min(511,tabc[i][0])
        y1 =min(511,tabc[i][1]) 
       # print(x1,y1)
        if i< tabcs0-1:
             i+=1
             x2 =min(511,tabc[i][0]) 
             y2 =min(511,tabc[i][1]) 
        else:
             i+=1
             x2 =min(511,tabc[0][0]) 
             y2 =min(511,tabc[0][1]) 
       # print('1',x1,y1,x2,y2)
        if x2-x1 != 0:
            
                 if x1>x2:
                    x2,x1=x1,x2
                    y2,y1=y1,y2
                 xi=x1
                 c=float((y2-y1)/(x2-x1))
                 while xi < x2:
                     yi = min(511,y1 + c * (xi-x1))
                  
                     #print('x1:',x1, 'y1:',y1, 'x2:',x2,'y2:',y2,'xi:',xi,'yi:',yi,'c:',c)             
                     tab[round(yi)][round(xi)]=1
#                     xi+=1
                     if c!=0:
                         xi+=min(abs(c),abs(1/c))
                     else:
                         xi+=1
                            
        else:
                if y1>y2:
                    y2,y1=y1,y2
                yi=y1
                while yi < y2:
         #           print('3',int(x1),int(yi))
                    tab[yi][x1]=1
                    yi+=1
    return tab


"""table
  y I 0
    I 1
    I 2 tab[y][x]
    I .
      0 1 2 3 .
    ---------
         x """

def reptf(tab,tabc,dx,dy):
    """ fill the full area of tab"""
    tabf = np.zeros((dx, dy), dtype='i')
    mintaby= min(510,tabc.take([1],axis=1).min())
    maxtaby= min(510,tabc.take([1],axis=1).max())
    mintabx= min(510,tabc.take([0],axis=1).min())
    maxtabx= min(510,tabc.take([0],axis=1).max())
#    print(mintabx,maxtabx,mintaby,maxtaby)
    x=mintabx

    while x <= maxtabx:
        y=mintaby 
        while y<=maxtaby:
            inst=False
            ycu=y
            xcu=x
            noyh=0
            noyl=0
            noxl=0
            noxr=0
            """ look right horizon"""
            while xcu <=maxtabx:
                if tab[y][xcu] >0 and tab[y][xcu-1]==0:
                    noxr = noxr+1
                xcu+=1
            xcu=x
            """look left horiz"""
            while xcu >=mintabx:
             
                if tab[y][xcu] >0 and tab[y][xcu+1]==0:
                     noxl = noxl+1
                xcu-=1
#                if(x==9 and y==9):
#                    print(x,y,xcu,noxl)
          
            ycu=y
            """look high vertic"""
            while ycu <=maxtaby:
                if tab[ycu][x] >0 and tab[ycu-1][x]==0:
                    noyh = noyh+1
                ycu+=1
            ycu=y
            """look low vertic"""
            while ycu >=mintaby:
                if tab[ycu][x] >0 and tab[ycu+1][x]==0:
                    noyl = noyl+1
                ycu-=1
           
            if noyl ==0 or noyh ==0 or noxl==0 or noxr==0:
               #     a=1
               inst=False 
            else:
#                inst = True
                if (noyl %2 != 0 or noyh%2 != 0 or noxl%2 != 0 or
                noxr%2 != 0):
                    inst=True
            if inst :
                if (tab[y][x]==0):
                    tabf[y][x]=3
            y+=1
        x+=1
    x=1
    return tabf
    
def reptfc(tab,dx,dy):
    """ correct  the full area of tab from artefacts"""
    tabf = np.copy(tab)
    x=1
    while x < dx-1:
        y=1
        while y<dy-1 :
            if (( tabf[y+1][x]==0 or tabf[y][x+1]==0) and \
            tabf[y][x]==3):
                tabf[y][x]=0
               
            y+=1
        x+=1
    y=1
    while y < dy-1:
        x=1
        while x < dx-1 :
            if ((tabf[y][x+1]==0 or tabf[y+1][x]==0) and \
            tabf[y][x]==3):
                tabf[y][x]=0
                
            x+=1
        y+=1
    x=1
    while x < dx-1:
        y=1
        while y<dy-1 :
            if  tabf[y][x]>0 :
                tabf[y][x]=1
                
            y+=1
        x+=1
    return tabf


#tabz2=tabz1+reptf(tabz1,mon_tableauc)  
def reptfull(tabc,dx,dy):
    """ top function to generate ROI table filled from ROI text file"""
    tabz=repts(tabc,dx,dy)
#    scipy.misc.imsave('tabz.jpg', tabz)
#    im1 = plt.matshow(tabz)
#    plt.colorbar(im1,label='with summit')

    tabz1=tabz+repti(tabc,dx,dy)
#    scipy.misc.imsave('tabz1.jpg', tabz1)
#    im2 = plt.matshow(tabz1)
#    plt.colorbar(im2,label='with summit and in between')

    tabz2=tabz1+reptf(tabz1,tabc,dx,dy)
#    scipy.misc.imsave('tabz2.jpg', tabz2)
#    im3 = plt.matshow(tabz2)
#    plt.colorbar(im3,label='with full fill')
     
    tabz3=reptfc (tabz2,dx,dy)
    scipy.misc.imsave('tabz3.jpg', tabz3)
#    im4 = plt.matshow(tabz3)
#    plt.colorbar(im4,label='with correct fill')
#    plt.show 
#    print(tabz3)
#  
    return tabz3

def pavs (tabc,tab,dx,dy,px,py,namedirtopcf,jpegpath,patchpath,thr,iln,f,label,loca,typei):
    """ generate patches from ROI"""
    #namedirtopcf = final/ILD_DB_txtROIs/35
    #jpegpath = final/jpeg
    #patchpath = final/jpeg
    #coefi=0.79296875
   #iln=slice2micronodulesdiffuse
    #label=micronodules
   #loca=diffuse
    #typei = bmp
    mintaby= min(511,tabc.take([1],axis=1).min())
    maxtaby= min(511,tabc.take([1],axis=1).max())
    mintabx= min(511,tabc.take([0],axis=1).min())
    maxtabx= min(511,tabc.take([0],axis=1).max())
#    print(mintabx,maxtabx,mintaby,maxtaby)
    contenujpg = os.listdir(namedirtopcf+'/'+typei)
                #contenujpg in  final/ILD_DB_txtROIs/35/jpg
    debnumslice=iln.find('_')+1
    endnumslice=iln.find('_',debnumslice)
    slicenumber=iln[debnumslice:endnumslice]
    if int(slicenumber)<10:
        slns='000'+slicenumber+'.'+typei
    elif int(slicenumber)<100:
        slns='00'+slicenumber+'.'+typei
    elif int(slicenumber)<1000:
          slns='0'+slicenumber+'.'+typei
    elif int(slicenumber)<10000:
          slns=slicenumber+'.'+typei
    tabp = np.zeros((dx, dy), dtype='i')
    tabf = np.copy(tab)
    pxy=float(px*py)

    i=max(mintabx-px,0)
    nbp=0
    strpac=''
    mini=min(maxtabx-px,dx-px)
    minj=min(maxtaby-py,dy-py)
    maxj=max(mintaby-py,0)
    while i < mini:
        j=maxj
#        print(i,j)
        while j<minj:
           
            ii=0
            area=0
            while ii < px:
                jj=0
                while jj <py:
                    if tabf[j+jj][i+ii] >0:
                        area+=1
                    jj+=1
                ii+=1
            
            if float(area)/pxy >thr:
                #good patch
                nbp+=1
                #print('pavage',i,j)              
                imf=False
                for  n in contenujpg:
#                    print (n)
#                    print(slns)
                   
                    if n.find(slns)>0:
                         imf=True
                         orig = Image.open(namedirtopcf+'/'+typei+'/'+n)
                #crop = ori.crop((left,top,right,bottom))
                         crorig = orig.crop((i, j, i+px, j+py))
#                crorig.show()
#                         print(label,loca)
                         crorig.save(patchpath+'/'+label+'/'+loca+'/'+f+\
                         '_'+iln+'_'+str(nbp)+'.'+typei)
                         break
                if not imf:
                        print('ERROR image not found',namedirtopcf+'/'+typei+'/'+n)
                strpac=strpac+str(i)+' '+str(j)+'\n'
                #                print('pavage',i,j)
                x=0
                #we draw the rectange
                while x < px:
                    y=0
                    while y < py:
                        tabp[y+j][x+i]=4
                        if x == 0 or x == px-1 :
                            y+=1
                        else:
                            y+=py-1
                    x+=1
                #we cancel the source
                x=0
                while x < px:
                    y=0
                    while y < py:
                        tabf[y+j][x+i]=0
                        y+=1
                    x+=1
            j+=1
        i+=1
    tabp =tab+tabp
    mf=open(jpegpath+'/'+f+'_'+iln+'.txt',"w")
    mf.write('#number of patches: '+str(nbp)+'\n'+strpac)
    mf.close()
    scipy.misc.imsave(jpegpath+'/'+f+'_'+iln+'.jpg', tabp)
#    im = plt.matshow(tabp)
#    plt.colorbar(im,label='with pavage')
##    im = plt.matshow(tabf)
##    plt.colorbar(im,label='ff')
#    plt.show
#    print('fin')
    return nbp,tabp
