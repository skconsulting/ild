# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 09:52:27 2017

@author: sylvain
"""
import os
from appJar import gui
from modulepredictgui import *
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



def predict(btn):
    global app
    indata={}
#    print(app.getListItems("list"))
#    print(app.getEntry("Percentage of pad Overlapp"))
    indata['thrpatch']=app.getEntry("Percentage of pad Overlapp")
    indata['thrproba']=app.getEntry("Treshold proba for predicted image generation")
    indata['thrprobaMerge']=app.getEntry("Treshold proba for merge cross and front view")
    indata['thrprobaUIP']=app.getEntry("Treshold proba for volume calculation")
    indata['threedpredictrequest']=app.getRadioButton("predict_style")
    indata['picklein_file']=app.getEntry("cross view weight")
    indata['picklein_file_front']= app.getEntry("front view weight")
    indata['lispatientselect']=app.getListItems("list")
#    roirun(app.getListItems("list"),lisdir)
    if len(indata['lispatientselect']) >0:
#    roirun(app.getListItems("list"),lisdir)
        predictrun(indata,lisdir)
    else:
        app.errorBox('error', 'no patient selected for predict')  
    app.stop(Stop)
    initDraw()

def visualisation(btn):
    global app,continuevisu
#    print(app.getListItems("list"))
    indata={}
    if frontpredict:
        indata['3dasked']=True
        indata['3d']=app.getRadioButton("3d")
        

    else:
        indata['3dasked']=False
        indata['viewstyle']=app.getRadioButton("planar")
    indata['lispatient']=selectpatient
#    if len(indata['lispatient']) >0:
#    roirun(app.getListItems("list"),lisdir)
    visuarun(indata,lisdir)
#    else:
#        app.errorBox('error', 'no patient selected for visu')
    app.stop(Stop)
    continuevisu=False
    visuDraw()
       
def redraw(app):   
    app.stop(Stop)
    initDraw()

def visuDrawl(btn):   
    app.stop(Stop)
    visuDraw()
    
    
def checkStop():
    return app.yesNoBox("Confirm Exit", "Are you sure you want to exit the application?")

def Stop():
    return True

def boutonStop(btn):
     app.stop(Stop)


def selection(btn):
    global frontpredict,continuevisu,selectpatient
    selectvisu=app.getListItems("list")
    selectpatient=selectvisu[0]
    frontexist=selectvisu[0].find('Cross & Front')
    selectpatient=selectvisu[0]
    if frontexist>0:
        frontpredict=True
        
    else:
        frontpredict=False
    continuevisu=True
    app.stop(Stop)
    visuDraw()
    
    
def gobackselection(btn):
    global continuevisu
    continuevisu=False
    app.stop(Stop)
    visuDraw()
 
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
#                print 'exist',pbg
                break
    if pbg:
#        print 'good dir'
        app.stop(Stop)       
        goodir=True
        initDraw()
    else: 
        lisdir=lisdirold

        app.errorBox('error', 'path for  patient not correct')
        app.stop(Stop)    
#        app.addMessage("mess", """Not correct""")
        goodir=False
        continuevisu=False
        initDraw()

    
listannotated=[]
goodir=False

def initDraw():
    global app
   
    app = gui("ROI form","1000x600")
    app.setBg("lightBlue")
    app.setFont(10)
#    app.setInPadding([40,20]) # padding inside the widget
#    app.setPadding([10,10]) # padding outside the widget
    app.addLabel("top", "Select patient directory:", 0, 0)
    app.setLabelBg("top","blue")
    app.setLabelFg("top","yellow")
    app.addEntry("path_patient")    
    app.setEntry("path_patient", lisdir)   
    app.setFocus("path_patient")
    app.addButton("select dir",  selectPatientDir)
    app.addHorizontalSeparator( colour="red",colspan=2)
#    app.addLabel("sepa", "")
#    app.setLabelBg("top", "green")
    print goodir
    if goodir:                       # Row 1,Column 1
      
        app.addLabel("top1", "Select patient:")
        app.setLabelBg("top1", "blue")
        app.setLabelFg("top1", "yellow")
        some_sg,stsdir=lisdirprocess(lisdir)
#        print some_sg
#        print stsdir
        listannotated=[]
        for user in some_sg:
            if stsdir[user]['front']==True:
                lroi=' Cross & Front'
                listannotated.append(user+' PREDICT!: '+lroi)
            elif stsdir[user]['cross']==True:
                lroi=' Cross'
                listannotated.append(user+' PREDICT!: '+lroi)
            else:
                listannotated.append(user+' noPREDICT! ')
#        print listannotated
        app.addListBox("list",listannotated)
        app.setListBoxMulti("list", multi=True)
        app.setListBoxRows("list",10)
        app.addHorizontalSeparator( colour="red",colspan=2)
        app.setFont(8)
        app.setSticky("w")
        app.addLabel("l1", "Prediction parameters:")
        app.setLabelBg("l1","blue")
        app.setLabelFg("l1","yellow")
        row = app.getRow()
        app.addLabelNumericEntry("Percentage of pad Overlapp",row,0)
        app.setEntry("Percentage of pad Overlapp", 0.8)        
        
        app.addLabelNumericEntry("Treshold proba for predicted image generation",row,1)
        app.setEntry("Treshold proba for predicted image generation", 0.6)
        
        row = app.getRow()
        app.addLabelNumericEntry("Treshold proba for volume calculation",row,0)
        app.setEntry("Treshold proba for volume calculation", 0.6)
        
        app.addLabelNumericEntry("Treshold proba for merge cross and front view",row,1)
        app.setEntry("Treshold proba for merge cross and front view", 0.6)
        row = app.getRow()
        app.addHorizontalSeparator(row,0,2, colour="red")
        app.addHorizontalSeparator(row,1,2, colour="red")
        row = app.getRow()
        app.addLabel("l2", "CNN weights",row,0)
        app.setLabelBg("l2","blue")
        app.setLabelFg("l2","yellow")
        app.addLabel("l3", "predict type: cross only or cross + front",row,1)
        app.setLabelBg("l3","blue")
        app.setLabelFg("l3","yellow")
        row = app.getRow()        
        app.addLabelEntry("cross view weight",row,0)
#        app.addVerticalSeparator(row,0 ,colour="red")
        app.setEntry("cross view weight","pickle_ex74")
        app.addRadioButton("predict_style", "Cross Only",row,1)
        row = app.getRow()
        app.addLabelEntry("front view weight",row,0)
        app.setEntry("front view weight","pickle_ex711")
        app.addRadioButton("predict_style", "Cross + Front",row,1)
        
        app.setFont(10)
        app.setSticky("n")
        app.addHorizontalSeparator( colour="red")
        app.addButton("Predict",  predict)
        app.addHorizontalSeparator( colour="red")
        app.addButton("Visualisation",  visuDrawl)
        app.addHorizontalSeparator( colour="red")        
    app.addButton("Quit",  boutonStop)
    app.go()

def visuDraw():
    global app
    print "visudraw"
    app = gui("ROI form","1000x600")
    app.setBg("lightBlue")

    app.setFont(10)
#    app.setInPadding([40,20]) # padding inside the widget
#    app.setPadding([10,10]) # padding outside the widget
    app.addLabel("top", "Select patient directory:", 0, 0)
    app.setLabelBg("top","blue")
    app.setLabelFg("top","yellow")
    app.addEntry("path_patient")    
    app.setEntry("path_patient", lisdir)   
    app.setFocus("path_patient")
    app.addButton("select dir",  selectPatientDir)
    app.addHorizontalSeparator( colour="red")
#    app.addLabel("sepa", "")
#    app.setLabelBg("top", "green")
    print goodir
    if goodir:                       # Row 1,Column 1
      
        
#        print some_sg
#        print stsdir
        if not continuevisu:
            app.addLabel("top1", "Select patient:")
            app.setLabelBg("top1", "blue")
            app.setLabelFg("top1", "yellow")
            some_sg,stsdir=lisdirprocess(lisdir)
            listannotated=[]
            app.startLabelFrame("patient List:")
            app.setSticky("ew")
            for user in some_sg:
#                userf=''
                if stsdir[user]['front']==True:
                    lroi=' Cross & Front'
                    listannotated.append(user+' PREDICT!: '+lroi)
#                    userf=user+' PREDICT!: '+lroi
                elif stsdir[user]['cross']==True:
                    lroi=' Cross'
                    listannotated.append(user+' PREDICT!: '+lroi)
#                    userf=user+' PREDICT!: '+lroi
                else:
                    listannotated.append(user+' noPREDICT! ')
#                    userf=user+' noPREDICT! '
#                app.addRadioButton("patienttovisu",userf)
    #        print listannotated
            app.addListBox("list",listannotated)
            app.setListBoxMulti("list", multi=False)            
            app.setListBoxRows("list",10)
            app.stopLabelFrame()
            app.addButton("Selection",  selection)
            
        else:
            app.addLabel("l11", "Patient selected: "+selectpatient)  
            app.addButton("Go back to Selection",  gobackselection)
            app.setLabelBg("l11","blue")
            app.setLabelFg("l11","yellow") 
            app.addHorizontalSeparator( colour="red")
            app.setFont(8)
            app.setSticky("w")
            if not frontpredict:
                app.addRadioButton("planar","cross view")                            
            else:
                row = app.getRow() # get current row
                app.addLabel("l1", "Type of planar view,select one",row,0) 
                app.addLabel("l2", "Type of orbit 3d view,select one",row,1)
                app.setLabelBg("l1","blue")
                app.setLabelFg("l1","yellow") 
                app.setLabelBg("l2","blue")
                app.setLabelFg("l2","yellow")
                row = app.getRow() # get current row
                app.addRadioButton("planar","none",row,0) 
                app.addRadioButton("3d","none",row,1)  
                row = app.getRow() # get current row
                app.addRadioButton("planar","cross view",row,0) 
                app.addRadioButton("3d","from cross predict",row,1)  
                row = app.getRow() # get current row
                app.addRadioButton("planar","front view",row,0)   
                app.addRadioButton("3d","from front predict",row,1)  
                row = app.getRow() # get current row
                app.addRadioButton("planar","merge view",row,0)        
                app.addRadioButton("3d","from cross + front merge",row,1)    
            app.addHorizontalSeparator( colour="red")
            app.setSticky("n")
#            app.setStrech("row")
            app.addHorizontalSeparator( colour="red",colspan=1)
            app.setFont(10)
            app.addButton("Visualisation",  visualisation)
            app.addHorizontalSeparator( colour="red")
    app.addButton("Quit",  boutonStop)
    app.go()
selectpatient=''
frontpredict=False
continuevisu=False
initDraw()