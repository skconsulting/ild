# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:48:43 2017

@author: sylvain
tool for roi generation
"""

# import the necessary packages
import numpy as np
import cv2
import os

pattern=''
typei='jpg'
dimtabx= 478
dimtaby=478
slicenumber=1
imagesource="b.jpg"

images = np.zeros((dimtabx,dimtaby,3), np.uint8)
quitl=False
tabroi={}
tabroifinal={}
path_patient='path_patient'

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

black=(0,0,0)
red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255,255,0)
cyan=(0,255,255)
purple=(255,0,255)
white=(255,255,255)
darkgreen=(11,123,96)
pink =(255,128,255)
lightgreen=(125,237,125)
orange=(255,153,102)
lowgreen=(0,51,51)
parme=(234,136,222)
chatain=(139,108,66)



classifc ={
    'back_ground':darkgreen,
    'consolidation':cyan,
    'HC':blue,
    'ground_glass':red,
    'healthy':darkgreen,
    'micronodules':green,
    'reticulation':yellow,
    'air_trapping':pink,
    'cysts':lightgreen,
    'bronchiectasis':orange,
    'emphysema':chatain,
    'GGpret': parme,



     'nolung': lowgreen,
     'bronchial_wall_thickening':white,
     'early_fibrosis':white,

     'increased_attenuation':white,
     'macronodules':white,
     'pcp':white,
     'peripheral_micronodules':white,
     'tuberculosis':white
 }


def contour3(im,l,dimtabx,dimtaby):  
    col=classifc[l]
    vis = np.zeros((dimtabx,dimtaby,3), np.uint8)
#    cv2.fillConvexPoly(vis, np.array(im),col)
    imagemax= cv2.countNonZero(np.array(im))
    if imagemax>0:
        cv2.fillPoly(vis, [np.array(im)],col)

    return vis

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global refPt, cropping,pattern,x0,y0,quitl,tabroi,tabroifinal
    
    if event == cv2.EVENT_LBUTTONDOWN:
        posrc=0
        print x,y
        for key,value in classif.items():
            labelfound=False            
            xr=5
            yr=15*posrc
            xrn=xr+10
            yrn=yr+10
            if x>xr and x<xrn and y>yr and y< yrn:
               
                print 'this is',key   
                pattern=key
                cv2.rectangle(images, (200,0), (210,10), classifc[pattern], -1)
                cv2.rectangle(images, (212,0), (340,12), black, -1)
                cv2.putText(images,key,(215,10),cv2.FONT_HERSHEY_PLAIN,0.7,classifc[key],1 )
                labelfound=True
                break  
            posrc+=1 

        if  x> zoneverticalgauche[0][0] and y > zoneverticalgauche[0][1] and x<zoneverticalgauche[1][0] and y<zoneverticalgauche[1][1]:
            print 'this is in menu'
            labelfound=True
            
        if  x> zoneverticaldroite[0][0] and y > zoneverticaldroite[0][1] and x<zoneverticaldroite[1][0] and y<zoneverticaldroite[1][1]:
            print 'this is in menu'
            labelfound=True
            
        if  x> zonehorizontal[0][0] and y > zonehorizontal[0][1] and x<zonehorizontal[1][0] and y<zonehorizontal[1][1]:
            print 'this is in menu'
            labelfound=True
            
        if x>posxdel and x<posxdel+10 and y>posydel and y< posydel+10:
            print 'this is suppress'
            suppress()
            labelfound=True
            
        if x>posxquit and x<posxquit+10 and y>posyquit and y< posyquit+10:
            print 'this is quit'
            quitl=True
            labelfound=True
            
        if x>posxdellast and x<posxdellast+10 and y>posydellast and y< posydellast+10:
            print 'this is delete last'
            labelfound=True
            dellast()

        if x>posxdelall and x<posxdelall+10 and y>posydelall and y< posydelall+10:
            print 'this is delete all'
            labelfound=True
            delall()

        if x>posxcomp and x<posxcomp+10 and y>posycomp and y< posycomp+10:
            print 'this is completed for all'
            labelfound=True  
            completed()
        
        if x>posxreset and x<posxreset+10 and y>posyreset and y< posyreset+10:
            print 'this is reset'
            labelfound=True  
            reseted()

                     
        if not labelfound:
            if len(pattern)>0:
                print 'len pattent',len(tabroi[pattern])
                tabroi[pattern].append((x, y)) 
                cv2.rectangle(images, (x,y), 
                              (x,y), classifc[pattern], 1)
                for l in range(0,len(tabroi[pattern])-1):
                    cv2.line(images, (tabroi[pattern][l][0],tabroi[pattern][l][1]), 
                              (tabroi[pattern][l+1][0],tabroi[pattern][l+1][1]), classifc[pattern], 1)
                    l+=1
            else:
                cv2.rectangle(images, (212,0), (340,12), black, -1)
                cv2.putText(images,'No pattern selected',(215,10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )
                    
def suppress():
    lastp=len(tabroi[pattern])
    if lastp>0:
        cv2.line(images, (tabroi[pattern][lastp-2][0],tabroi[pattern][lastp-2][1]), 
                 (tabroi[pattern][lastp-1][0],tabroi[pattern][lastp-1][1]), black, 1)
        tabroi[pattern].pop()
        for l in range(0,len(tabroi[pattern])-1):
            cv2.line(images, (tabroi[pattern][l][0],tabroi[pattern][l][1]),
                     (tabroi[pattern][l+1][0],tabroi[pattern][l+1][1]), classifc[pattern], 1)
            l+=1

def completed():
    for key,value in classif.items():
                for l in range(0,len(tabroi[key])-1):
    #                tabroifinal[pattern][tabroi[pattern][l][0]][tabroi[pattern][l][1]]=classifc[pattern]
                    cv2.line(images, (tabroi[key][l][0],tabroi[key][l][1]), 
                                  (tabroi[key][l+1][0],tabroi[key][l+1][1]), black, 1)          
               
                vis=contour3(tabroi[key],key,dimtabx,dimtaby)
                tabroifinal[key]=cv2.add(vis, tabroifinal[key])
                tabroi[key]=[]
                imgray = cv2.cvtColor(tabroifinal[key],cv2.COLOR_BGR2GRAY)
                imagemax= cv2.countNonZero(imgray)
                if imagemax>0:
                    posext=imagesource.find('.'+typei)
                    imgcoreScan=imagesource[0:posext]+'_'+str(slicenumber)+'_'+key+'.'+typei                    
                    cv2.imwrite(imgcoreScan,tabroifinal[key])
                
def reseted():
    for key,value in classif.items():
                for l in range(0,len(tabroi[key])-1):
    #                tabroifinal[pattern][tabroi[pattern][l][0]][tabroi[pattern][l][1]]=classifc[pattern]
                    cv2.line(images, (tabroi[key][l][0],tabroi[key][l][1]), 
                                  (tabroi[key][l+1][0],tabroi[key][l+1][1]), black, 1)          
                tabroifinal[key]=np.zeros((dimtabx,dimtaby,3), np.uint8)
                tabroi[key]=[]                
def dellast():
    for l in range(0,len(tabroi[pattern])-1):
        cv2.line(images, (tabroi[pattern][l][0],tabroi[pattern][l][1]), 
                 (tabroi[pattern][l+1][0],tabroi[pattern][l+1][1]), black, 1)          
    tabroi[pattern]=[]   

def delall():
    for key,value in classif.items():
        for l in range(0,len(tabroi[key])-1):
            cv2.line(images, (tabroi[key][l][0],tabroi[key][l][1]), 
                (tabroi[key][l+1][0],tabroi[key][l+1][1]), black, 1)          
        tabroi[key]=[]


# keep looping until the 'q' key is pressed
def loop():

    while True:
    
        imageview=cv2.add(image,images)
        for key,value in classif.items():
                imageview=cv2.add(imageview,tabroifinal[key])
    
        cv2.imshow("image", imageview)
        key = cv2.waitKey(1) & 0xFF
    	# if the 'r' key is pressed, reset the cropping region
        if key == ord("c"):
                print 'completed'
                completed()
                
        elif key == ord("d"):
                print 'delete last'
                suppress()
                
        elif key == ord("l"):
                print 'delete last'
                dellast()
        
        elif key == ord("a"):
                print 'delete all'
                delall()
                
        elif key == ord("r"):
                print 'reset'
                reseted()
    
        elif key == ord("q")  or quitl or cv2.waitKey(20) & 0xFF == 27 :
        #            print 'on quitte', quitl
               cv2.destroyAllWindows()
               break


cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", click_and_crop)
posrc=0
for key,value in classif.items():
    tabroi[key]=[]
    tabroifinal[key]= np.zeros((dimtabx,dimtaby,3), np.uint8)
    xr=5
    yr=15*posrc
    xrn=xr+10
    yrn=yr+10
    cv2.rectangle(images, (xr, yr),(xrn,yrn), classifc[key], -1)
    cv2.putText(images,key,(xr+15,yr+10),cv2.FONT_HERSHEY_PLAIN,0.7,classifc[key],1 )
    posrc+=1

posxdel=dimtabx-20
posydel=15
cv2.rectangle(images, (posxdel,posydel),(posxdel+10,posydel+10), white, -1)
cv2.putText(images,'(d) del',(posxdel-40, posydel+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

posxquit=dimtabx-20
posyquit=0
cv2.rectangle(images, (posxquit,posyquit),(posxquit+10,posyquit+10), white, -1)
cv2.putText(images,'(q) quit',(posxquit-45, posyquit+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

posxdellast=dimtabx-20
posydellast=30
cv2.rectangle(images, (posxdellast,posydellast),(posxdellast+10,posydellast+10), white, -1)
cv2.putText(images,'(l) del last',(posxdellast-68, posydellast+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

posxdelall=dimtabx-20
posydelall=45
cv2.rectangle(images, (posxdelall,posydelall),(posxdelall+10,posydelall+10), white, -1)
cv2.putText(images,'(a) del all',(posxdelall-57, posydelall+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

posxcomp=dimtabx-20
posycomp=60
cv2.rectangle(images, (posxcomp,posycomp),(posxcomp+10,posycomp+10), white, -1)
cv2.putText(images,'(c) completed',(posxcomp-85, posycomp+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

posxreset=dimtabx-20
posyreset=75
cv2.rectangle(images, (posxreset,posyreset),(posxreset+10,posyreset+10), white, -1)
cv2.putText(images,'(r) reset',(posxreset-55, posyreset+10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )

zoneverticalgauche=((0,0),(20,dimtaby))
zonehorizontal=((0,0),(dimtabx,20))
zoneverticaldroite=((dimtabx-20,0),(dimtabx,dimtaby))

cwd=os.getcwd()
(top,tail) =os.path.split(cwd)

patient_path_complet=os.path.join(top,path_patient)
patient_path_complet=os.path.join(patient_path_complet,imagesource)
print patient_path_complet

image = cv2.imread(patient_path_complet)
print type(image)
a= os.walk(patient_path_complet).next()[1]
cv2.imshow('e',image)
loop()