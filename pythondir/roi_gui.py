# -*- coding: utf-8 -*-
"""
Created on Sun Apr 05 09:52:27 2017
Version 1.0
@author: sylvain
"""
import os
import sys
from appJar import gui
from moduleroigui import *
import cPickle as pickle
import webbrowser

#print os.environ['TMP']
#print os.environ['USERPROFILE']
print os.environ['LOCALAPPDATA']
#print os.environ['APPDATA']
print os.environ['PROGRAMDATA']

instdirroiGene='Roi_Gene'
instdirMedikey='MedikEye'
instdirroiGeneLocal='Roi_Gene'
instdirMedikeyLocal='MedikEye'
#empofile=os.path.join(os.environ['TMP'],picklefileglobal)
#workingdir= os.path.join(os.environ['USERPROFILE'],workdiruser)
#instdir=os.path.join(os.environ['LOCALAPPDATA'],instdirMHK)
pathMedikEye=os.path.join(os.environ['PROGRAMDATA'],instdirMedikey)
pathRoiGene=os.path.join(pathMedikEye,instdirroiGene)

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


version ="1.0"
paramsave='data'
source='source'
paramname ='paramname.pkl'
#cwd=os.getcwd()
(cwdtop,tail)=os.path.split(pathRoiGene)

paramsaveDir=os.path.join(pathRoiGenelocal,paramsave)
if not os.path.exists(paramsaveDir):
    os.mkdir(paramsaveDir)

paramsaveDirf=os.path.join(paramsaveDir,paramname)

if os.path.exists(paramsaveDir):
    if os.path.exists(paramsaveDirf):
        lisdir=pickle.load(open( paramsaveDirf, "rb" ))
    else:
        lisdir=cwdtop

def press(btn):
    global app
#    print(app.getListItems("list"))
    ll =app.getListItems("list")
#    print ll
    if len(ll)>0:
        roirun(ll,lisdir)
        redraw(app)
    else:
        app.errorBox('error', 'no  patient selected')
        redraw(app)

def presshelp(btn):
#    print 'help'

    filehelp=os.path.join(pathRoiGene,'doc.pdf')
#    print filehelp
#    webbrowser.open_new(r'file://C:\Users\sylvain\Documents\boulot\startup\radiology\roittool\modulepython\doc.pdf')
    webbrowser.open_new(r'file://'+filehelp)

#    app.infoBox( 'help', 'aa')


def redraw(app):

    app.stop(Stop)
    initDraw()

#def checkStop():
#    return app.yesNoBox("Confirm Exit", "Are you sure you want to exit the application?")

def Stop():

    return True

def boutonStop(btn):
    ans= app.yesNoBox("Confirm Exit", "Are you sure you want to exit the application?")
    if ans:
#        app.stop(Stop)

        sys.exit(1)
    else:
        redraw(app)

#    app.stop(Stop)

def selectPatientDir():
    global lisdir,goodir
    lisdirold=lisdir
#    print 'select dir'
    lisdir=os.path.realpath(app.directoryBox(title='path patient',dirName=lisdir))
    pbg=False
#    print lisdir
    if lisdir ==pathRoiGene:
#        print 'exit'
        sys.exit(1)
    if os.path.exists(lisdir):
        lisstdirec= os.walk(lisdir).next()[1]
        for i in lisstdirec:
            sourced=os.path.join(os.path.join(lisdir,i),source)
            if os.path.exists(sourced):
                pickle.dump(lisdir,open( paramsaveDirf, "wb" ))
                if pbg==False:
                    pbg=True
#                print 'exist',pbg
                break
    if pbg:
        app.stop(Stop)
        goodir=True
        initDraw()
    else:
        lisdir=lisdirold
        app.errorBox('error', 'path for  patient not correct')
        app.stop(Stop)
        goodir=False
        initDraw()


listannotated=[]
goodir=False

def initDraw():

    global app
    app = gui("ROI form"+version,"1000x400")
    app.setStopFunction(Stop)
    app.addLabel("top", "Select patient directory:", 0, 0)
    if not goodir: selectPatientDir()
#    aa=app.directoryBox(title='path patient',dirName=lisdir)
#    print aa
#    app.addEntry("path_patient")
#    app.setEntry("path_patient", lisdir)
#    app.setFocus("path_patient")
#    app.addButton("select dir",  selectPatientDir)
#    app.addHorizontalSeparator( colour="red")
    app.addButton("HELP",  presshelp)
    app.setLabelBg("top", "green")
#    print goodir
    if goodir:

#        app.addLabel("top1", "Select one patient:")
#        app.setLabelBg("top1", "blue")
#        app.setLabelFg("top1", "yellow")
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

        app.addListBox("list",listannotated)
        app.setListBoxRows("list",10)
#        app.setLabelBg("list", "blue")
        app.addButton("Generate ROI",  press)
        app.addHorizontalSeparator( colour="red")
    app.addButton("Quit",  boutonStop)
    app.go()

if __name__ == '__main__':
    initDraw()