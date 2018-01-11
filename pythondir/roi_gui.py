# -*- coding: utf-8 -*-
"""
V1.0 Created on Sun Apr 05 09:52:27 2017

@author: sylvain Kritter 
Version 1.9

11 January 2018
"""
#from param_pix_r import *
#from moduleroigui import *
from moduleroigui import roirun,lisdirprocess,checkvolumegene

from appJar import gui
import os
import cPickle as pickle
import sys
import webbrowser
#print os.environ['TMP']
#print os.environ['USERPROFILE']
#print os.environ['LOCALAPPDATA']
#print os.environ['APPDATA']
#print os.environ['PROGRAMDATA']

instdirroiGene='Roi_Gene'
instdirMedikey='MedikEye'
instdirroiGeneLocal='Roi_Gene'
instdirMedikeyLocal='MedikEye'
modulepython='modulepython'
moduledoc='doc'
#empofile=os.path.join(os.environ['TMP'],picklefileglobal)
#workingdir= os.path.join(os.environ['USERPROFILE'],workdiruser)
#instdir=os.path.join(os.environ['LOCALAPPDATA'],instdirMHK)
pathMedikEye=os.path.join(os.environ['PROGRAMDATA'],instdirMedikey)
pathRoiGene=os.path.join(pathMedikEye,instdirroiGene)
pathRoiGeneModulepython=os.path.join(pathRoiGene,modulepython)
pathRoiGeneDoc=os.path.join(pathRoiGene,moduledoc)

pathMedikEyelocal=os.path.join(os.environ['LOCALAPPDATA'],instdirMedikeyLocal)
pathRoiGenelocal=os.path.join(pathMedikEyelocal,instdirroiGeneLocal)

print 'pathMedikEye',pathMedikEye
print 'pathRoiGene',pathRoiGene

print 'pathMedikEyelocal',pathMedikEyelocal
print 'pathRoiGenelocal',pathRoiGenelocal


if not os.path.exists(pathMedikEyelocal):
    os.mkdir(pathMedikEyelocal)
if not os.path.exists(pathRoiGenelocal):
    os.mkdir(pathRoiGenelocal)
#workingdir= os.path.join(os.environ['USERPROFILE'],workdiruser)


version ="1.8"
paramsave='data'
source='source'
paramname ='paramname.pkl'
paramdict={}
#cwd=os.getcwd()
#(cwdtop,tail)=os.path.split(pathRoiGene)

paramsaveDir=os.path.join(pathRoiGenelocal,paramsave)
if not os.path.exists(paramsaveDir):
    os.mkdir(paramsaveDir)

paramsaveDirf=os.path.join(paramsaveDir,paramname)

if os.path.exists(paramsaveDirf):
#        paramdict=pickle.load(open( paramsaveDirf, "rb" ))
        lisdirold=pickle.load(open( paramsaveDirf, "rb" ))
        try:
            paramdict=pickle.load(open( paramsaveDirf, "rb" ))
            lisdir= paramdict['path_patient']
            centerHU=paramdict['centerHU']
            limitHU=paramdict['limitHU']
        except:
            paramdict={}
            lisdir=lisdirold           
            paramdict['path_patient']=lisdir
            paramdict['centerHU']=-662
            paramdict['limitHU']=1700
            pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
                
else:
    lisdir=os.environ['USERPROFILE']     
    paramdict['path_patient']=lisdir
    pickle.dump(paramdict,open( paramsaveDirf, "wb" ))       

def press(btn):
    global app
    indata={}
    indata['centerHU']=app.getEntry("centerHU")
    indata['limitHU']=app.getEntry("limitHU")
    indata['ForceGenerate']=app.getCheckBox("ForceGenerate")
#    indata['ImageTreatment']=app.getCheckBox("ImageTreatment")
    indata['ImageTreatment']=False

    
    indata['ll']=app.getListItems("list")
    
    if len(indata['ll'])>0:
        paramdict['centerHU']=indata['centerHU']
        paramdict['limitHU']=indata['limitHU']
        pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
        app.hide()
        roirun(indata,lisdir,False)
        redraw(app)
    else:
        app.errorBox('error', 'no  patient selected')
        redraw(app)

def presslung(btn):
    global app
    indata={}
    indata['ll']=app.getListItems("list")
    indata['centerHU']=app.getEntry("centerHU")
    indata['limitHU']=app.getEntry("limitHU")
    indata['ForceGenerate']=app.getCheckBox("ForceGenerate")
    
    if len( indata['ll'])>0:
        app.hide()
        paramdict['centerHU']=indata['centerHU']
        paramdict['limitHU']=indata['limitHU']
        pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
        roirun(indata,lisdir,True)
        redraw(app)
    else:
        app.errorBox('error', 'no  patient selected')
        redraw(app)

def checkvolume(btn):
    global app
    indata={}
#    print(app.getListItems("list"))
    indata['ll']=app.getListItems("list")
    indata['centerHU']=app.getEntry("centerHU")
    indata['limitHU']=app.getEntry("limitHU")
    pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
#    print ll
    if len(indata['ll'])>0:
        app.hide()
        mes=checkvolumegene(indata,lisdir)
        app.show()
        if mes !=None:
            app.infoBox('volume', mes)
#            app.addMessage('volume', mes)
#            app.addScrolledMessage('volume', mes)

#            app.errorBox('volume', mes)

#        redraw(app)
    else:
        app.errorBox('error', 'no  patient selected')
        redraw(app)


def presshelp(btn):
#    print 'help'
    filehelp=os.path.join(pathRoiGeneDoc,'doc_roi.pdf')
    webbrowser.open_new(r'file://'+filehelp)

def redraw(app):
    app.stop()
    initDraw()

def Stop():
    return app.yesNoBox("Confirm Exit", "Are you sure you want to exit the application?")
#    if ans:
##        app.stop(Stop)
#        return ans
#        sys.exit(1)
#    else:
#        redraw(app)
#    app.stop(Stop)
#    ans= app.yesNoBox("Confirm Exit", "Are you sure you want to exit the application?")
#    sys.exit(1)
    

def boutonStop(btn):
    ans= app.yesNoBox("Confirm Exit", "Are you sure you want to exit the application?")
    if ans:
        app.stop()
        sys.exit(1)
    else:
        app.stop()
        redraw(app)
#    app.stop(Stop)

def selectPatientDir():
    global lisdir,goodir
    lisdirold=lisdir
#    print 'select dir'
    lisdir=os.path.realpath(app.directoryBox(title='path patient',dirName=lisdir))
    pbg=False
#    print lisdir
    if lisdir ==pathRoiGeneModulepython:
#        print 'exit'
        sys.exit(1)
    if os.path.exists(lisdir):
        lisstdirec= os.walk(lisdir).next()[1]
        for i in lisstdirec:
            sourced=os.path.join(os.path.join(lisdir,i),source)
            if os.path.exists(sourced):
                paramdict['path_patient']=lisdir
                pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
                if pbg==False:
                    pbg=True
#                print 'exist',pbg
                break
            else:
               ldcm= [name for name in os.listdir(os.path.join(lisdir,i)) if name.find('.dcm')>0]
               if len(ldcm)>0:
                    paramdict['path_patient']=lisdir
                    pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
#                print 'paramdict',paramdict
                    if pbg==False:
                        pbg=True
                    break
    if pbg:
        app.stop()
        goodir=True
        initDraw()
    else:
        lisdir=lisdirold
        app.errorBox('error', 'path for  patient not correct')
        app.stop()
        goodir=False
        initDraw()


listannotated=[]
goodir=False

def initDraw():
    paramdict=pickle.load(open( paramsaveDirf, "rb" ))
    centerHU=paramdict['centerHU']
    limitHU=paramdict['limitHU']
    global app
    app = gui("ROI form"+version,"1000x500")
#    app.setStopFunction(Stop)
    if not goodir: selectPatientDir()

    if goodir:
        app.addLabel("path_patientt", "path_patient",colspan=2)
        app.addLabel("path_patient", lisdir,colspan=2)
        app.setLabelBg("path_patient", "Blue")
        app.setLabelFg("path_patient", "Yellow")
        app.setLabelBg("path_patientt", "Blue")
        app.setLabelFg("path_patientt", "Yellow")
        
        app.addLabel("top", "Select patient ID:")
        app.setLabelBg("top", "Grey")
        app.setLabelFg("top", "Blue")

        row = app.getRow()
        app.addButton("HELP",  presshelp,row,1)        
        
        some_sg,stsdir=lisdirprocess(lisdir)

        listannotated=[]
        for user in some_sg:
            if len(stsdir[user])>0:
                  lroi=''
                  for r in stsdir[user]:
                      lroi=lroi+' '+r
                  listannotated.append(user+' ROI!: '+lroi)
            else:
                listannotated.append(user+' noROI! ')

        app.addListBox("list",listannotated,row,0)
        app.setListBoxRows("list",10)
        row = app.getRow()
#        app.addHorizontalSeparator( row,1,colour="Red")
#        row = app.getRow()
        app.addLabelNumericEntry("centerHU",row,1)
        app.setEntry("centerHU",centerHU)
        row = app.getRow()
        app.addLabelNumericEntry("limitHU",row,1)
        app.setEntry("limitHU", limitHU)
        row = app.getRow()
        app.addHorizontalSeparator( row,colour="red",colspan=2)
        row = app.getRow()
        app.addLabel("ForceGenerate","Tick to force re-generate all files:",row,0)
        app.addCheckBox("ForceGenerate",row,1)
        app.setCheckBox("ForceGenerate",ticked=False,callFunction=False)
#        app.addCheckBox("ImageTreatment",row,2)
#        app.setCheckBox("ImageTreatment",ticked=False,callFunction=False)
        row = app.getRow()
        app.addHorizontalSeparator( row,colour="red",colspan=2)
        row = app.getRow()
        app.addButton("Generate ROI",  press,row,1)
        row = app.getRow()
        
        app.addButton("Generate Lung_Mask",  presslung,row,1)
        row = app.getRow()
        app.addButton("check volume",  checkvolume,row,1)

#        app.addHorizontalSeparator( colour="red",colspan=2)
    app.addButton("Quit",  boutonStop,row,0)
    app.go()
    sys.exit(1)

#if __name__ == '__main__':
initDraw()
app.stop()
sys.exit(1)
#    app.stop(Stop)