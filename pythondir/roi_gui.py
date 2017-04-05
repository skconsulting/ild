# -*- coding: utf-8 -*-
"""
Created on Sun Apr 05 09:52:27 2017

@author: sylvain
"""
import os
from appJar import gui
from moduleroigui import *
import cPickle as pickle

paramsave='data'
source='source'
paramname ='paramname.pkl'
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)

paramsaveDir=os.path.join(cwdtop,paramsave)
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
    print(app.getListItems("list"))
    roirun(app.getListItems("list"),lisdir)
    redraw(app)
       
def redraw(app):
   
    app.stop(Stop)
    initDraw()
    
def checkStop():
    return app.yesNoBox("Confirm Exit", "Are you sure you want to exit the application?")

def Stop():
    return True

def boutonStop(btn):
     app.stop(Stop)
    
def selectPatientDir(btn):
    global lisdir,goodir
    lisdirold=lisdir
    lisdir=app.getEntry("path_patient")
    pbg=False
    if os.path.exists(lisdir):
        lisstdirec= os.walk(lisdir).next()[1]
        for i in lisstdirec:
            sourced=os.path.join(os.path.join(lisdir,i),source)
            if os.path.exists(sourced):
                pickle.dump(lisdir,open( paramsaveDirf, "wb" ))
                if pbg==False:
                    pbg=True
                print 'exist',pbg
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
    app = gui("ROI form","1000x400")

    app.addLabel("top", "Select patient directory:", 0, 0)
    app.addEntry("path_patient")    
    app.setEntry("path_patient", lisdir)   
    app.setFocus("path_patient")
    app.addButton("select dir",  selectPatientDir)
    app.addHorizontalSeparator( colour="red")

    app.setLabelBg("top", "green")
#    print goodir
    if goodir:
      
        app.addLabel("top1", "Select one patient:")
        app.setLabelBg("top1", "blue")
        app.setLabelFg("top1", "yellow")
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

initDraw()