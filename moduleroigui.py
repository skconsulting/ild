# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:21:22 2017

@author: sylvain
version 1.1
28 july 2017
"""
#from param_pix_r import *
from param_pix_r import classif,classifcontour,lung_mask_bmp,lung_mask1,usedclassif
from roigene import openfichierroi,openfichierroilung,checkvolumegeneroi

import os

def lisdirprocess(directorytocheck):
    a= os.walk(directorytocheck).next()[1]
#    print 'listdirprocess',a
    stsdir={}
    for dd in a:
        stpred=[]
        ddd=os.path.join(directorytocheck,dd)
        for key in usedclassif:
            datadir=os.path.join(ddd,key)
            if key in classifcontour:      
                if os.path.exists(datadir):
                    datadir=os.path.join(datadir,lung_mask_bmp)           
                if os.path.exists(datadir):
                    listfile=os.listdir(datadir)
                else:
                    datadir=os.path.join(ddd,lung_mask1)   
                    if os.path.exists(datadir):
                        listfile=os.listdir(datadir)
            else:
                if os.path.exists(datadir):
                        listfile=os.listdir(datadir)
                else:
                        listfile=[]
#                        print dd,key,listfile
            if len(listfile)>0:
                    stpred.append(key)
        stsdir[dd]=stpred
#        print dd,stsdir[dd]

    return a,stsdir


def roirun(indata,path_patient):
    listHug=indata['ll']
    centerHU=indata['centerHU'] 
    limitHU=indata['limitHU'] 

    pos=str(listHug).find(' ROI!:')
    if pos >0:
        listHug=str(listHug)[3:pos]
    else:
        pos=str(listHug).find(' noROI!')
        listHug=str(listHug)[3:pos]
#    print 'listhug',listHug
#    print 'indata',indata
#    print 'path_patient',path_patient
    messageout=openfichierroi(listHug,path_patient,centerHU,limitHU)
    return messageout

def roirunlung(indata,path_patient):
    listHug=indata['ll']
    centerHU=indata['centerHU'] 
    limitHU=indata['limitHU'] 

    pos=str(listHug).find(' ROI!:')
    if pos >0:
        listHug=str(listHug)[3:pos]
    else:
        pos=str(listHug).find(' noROI!')
        listHug=str(listHug)[3:pos]
#    print 'listhug',listHug
#    print 'indata',indata
#    print 'path_patient',path_patient
    messageout=openfichierroilung(listHug,path_patient,centerHU,limitHU)
    return messageout

def checkvolumegene(indata,path_patient):
    listHug=indata['ll']
#    centerHU=indata['centerHU'] 
#    limitHU=indata['limitHU'] 
    pos=str(listHug).find(' ROI!:')
    if pos >0:
        listHug=str(listHug)[3:pos]
    else:
        pos=str(listHug).find(' noROI!')
        listHug=str(listHug)[3:pos]
#    print 'listhug',listHug
#    print 'indata',indata
#    print 'path_patient',path_patient
    messageout=checkvolumegeneroi(listHug,path_patient)
    return messageout