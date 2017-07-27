# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:21:22 2017

@author: sylvain
"""
#from param_pix_r import *
from param_pix_r import classif,classifcontour,lung_mask_bmp
from roigene import openfichierroi,openfichierroilung,checkvolumegeneroi

import os

def lisdirprocess(directorytocheck):
    a= os.walk(directorytocheck).next()[1]
#    print 'listdirprocess',a
    stsdir={}
    for dd in a:
        stpred=[]
        ddd=os.path.join(directorytocheck,dd)
        for key in classif:
            datadir=os.path.join(ddd,key)
            if key in classifcontour:        
                datadir=os.path.join(datadir,lung_mask_bmp)           
            if os.path.exists(datadir):
                listfile=os.listdir(datadir)
                if len(listfile)>0:
                    stpred.append(key)
        stsdir[dd]=stpred

    return a,stsdir


def roirun(indata,path_patient):
    listHug=indata

    pos=str(indata).find(' ROI!:')
    if pos >0:
        listHug=str(indata)[3:pos]
    else:
        pos=str(indata).find(' noROI!')
        listHug=str(indata)[3:pos]
#    print 'listhug',listHug
#    print 'indata',indata
#    print 'path_patient',path_patient
    messageout=openfichierroi(listHug,path_patient)
    return messageout

def roirunlung(indata,path_patient):
    listHug=indata

    pos=str(indata).find(' ROI!:')
    if pos >0:
        listHug=str(indata)[3:pos]
    else:
        pos=str(indata).find(' noROI!')
        listHug=str(indata)[3:pos]
#    print 'listhug',listHug
#    print 'indata',indata
#    print 'path_patient',path_patient
    messageout=openfichierroilung(listHug,path_patient)
    return messageout

def checkvolumegene(indata,path_patient):
    listHug=indata

    pos=str(indata).find(' ROI!:')
    if pos >0:
        listHug=str(indata)[3:pos]
    else:
        pos=str(indata).find(' noROI!')
        listHug=str(indata)[3:pos]
#    print 'listhug',listHug
#    print 'indata',indata
#    print 'path_patient',path_patient
    messageout=checkvolumegeneroi(listHug,path_patient)
    return messageout
