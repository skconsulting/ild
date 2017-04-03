# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 09:52:27 2017

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

#path_pickle='CNNparameters'
paramsaveDir=os.path.join(cwdtop,paramsave)
if not os.path.exists(paramsaveDir):
    os.mkdir(paramsaveDir)

paramsaveDirf=os.path.join(paramsaveDir,paramname)

if os.path.exists(paramsaveDir):
    if os.path.exists(paramsaveDirf):
        lisdir=pickle.load(open( paramsaveDirf, "rb" ))
    else:
        lisdir=cwdtop        

#path_patient=''
#lisdir=os.path.join(cwdtop,path_patient)
#
#some_sg,stsdir=lisdirprocess(lisdir)



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
    print app.getEntry("path_patient")
    lisdirold=lisdir
    lisdir=app.getEntry("path_patient")
    pbg=False
    if os.path.exists(lisdir):
        a= os.walk(lisdir).next()[1]
        for i in a:
#            print os.path.join(lisdir,i)
            sourced=os.path.join(os.path.join(lisdir,i),source)
#            print sourced
            if os.path.exists(sourced):
                pickle.dump(lisdir,open( paramsaveDirf, "wb" ))
                if pbg==False:
                    pbg=True
                print 'exist',pbg
                break
    if pbg:
        print 'good dir'
        app.stop(Stop)       
        goodir=True
        initDraw()
    else: 
        lisdir=lisdirold
        print 'bad dir'
        app.stop(Stop)    
#        app.addMessage("mess", """Not correct""")
        goodir=False
        initDraw()

    
listannotated=[]
goodir=False
def initDraw():
    global app
    app = gui("ROI form","1000x400")
#    app.setInPadding([40,20]) # padding inside the widget
#    app.setPadding([10,10]) # padding outside the widget
    app.addLabel("top", "Select patient directory:", 0, 0)
    app.addEntry("path_patient")    
    app.setEntry("path_patient", lisdir)   
    app.setFocus("path_patient")
    app.addButton("select dir",  selectPatientDir)
    app.addHorizontalSeparator( colour="red")
#    app.addLabel("sepa", "")
    app.setLabelBg("top", "green")
    print goodir
    if goodir:                       # Row 1,Column 1
      
        app.addLabel("top1", "Select one patient:")
        app.setLabelBg("top1", "blue")
        app.setLabelFg("top1", "yellow")
        some_sg,stsdir=lisdirprocess(lisdir)
#        print some_sg
#        print stsdir
        listannotated=[]
        for user in some_sg:
            if len(stsdir[user])>0:
                  lroi=''
                  for r in stsdir[user]:
                      lroi=lroi+' '+r
                  listannotated.append(user+' ROI!: '+lroi)
            else:
                listannotated.append(user+' noROI! ')
        print listannotated
        app.addListBox("list",listannotated)
        app.setListBoxRows("list",10)
#        app.setLabelBg("list", "blue")
        app.addButton("Generate ROI",  press)
        app.addHorizontalSeparator( colour="red")
    app.addButton("Quit",  boutonStop)

    app.go()


initDraw()