# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 09:52:27 2017

@author: sylvain
"""
import os
import sys
from appJar import gui
from modulepredictgui import *
import cPickle as pickle
import webbrowser

#print os.environ['TMP']
print os.environ['USERPROFILE']
print os.environ['LOCALAPPDATA']
#print os.environ['APPDATA']
print os.environ['PROGRAMDATA']

instdirPredict='Predict'
instdirMedikey='MedikEye'
instdirpredictLocal='Predict'
instdirMedikeyLocal='MedikEye'
#empofile=os.path.join(os.environ['TMP'],picklefileglobal)
#workingdir= os.path.join(os.environ['USERPROFILE'],workdiruser)
#instdir=os.path.join(os.environ['LOCALAPPDATA'],instdirMHK)
pathMedikEye=os.path.join(os.environ['PROGRAMDATA'],instdirMedikey)
pathPredict=os.path.join(pathMedikEye,instdirPredict)

pathMedikEyelocal=os.path.join(os.environ['LOCALAPPDATA'],instdirMedikeyLocal)
pathPredictlocal=os.path.join(pathMedikEyelocal,instdirpredictLocal)
#
#print 'pathMedikEye',pathMedikEye
#print 'pathPredict',pathPredict
#
#print 'pathMedikEyelocal',pathMedikEyelocal
#print 'pathPredictlocal',pathPredictlocal

if not os.path.exists(pathMedikEyelocal):
    os.mkdir(pathMedikEyelocal)
if not os.path.exists(pathPredictlocal):
    os.mkdir(pathPredictlocal)
#workingdir= os.path.join(os.environ['USERPROFILE'],workdiruser)


version ="1.0"
paramsave='data'
source='source'
paramname ='paramname.pkl'
cwd=os.getcwd()

pathPredict=cwd
paramsaveDir=os.path.join(pathPredictlocal,paramsave)
if not os.path.exists(paramsaveDir):
    os.mkdir(paramsaveDir)

paramsaveDirf=os.path.join(paramsaveDir,paramname)

paramdict={}

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)

#path_pickle='CNNparameters'



if os.path.exists(paramsaveDir):
    if os.path.exists(paramsaveDirf):
        paramdict=pickle.load(open( paramsaveDirf, "rb" ))
        if len(paramdict)==1:
            lisdir= paramdict['path_patient']

        else:
            lisdir= paramdict['path_patient']
            thrpatch= paramdict['thrpatch']
            thrproba= paramdict['thrproba']
            thrprobaMerge= paramdict['thrprobaMerge']
            thrprobaUIP= paramdict['thrprobaUIP']
            picklein_file= paramdict['picklein_file']
            picklein_file_front= paramdict['picklein_file_front']

    else:
        lisdir=os.environ['USERPROFILE']
        thrpatch= 0.8
        thrproba=0.6
        thrprobaMerge=0.6
        thrprobaUIP= 0.6
        picklein_file= "pickle_ex74"
        picklein_file_front= "pickle_ex711"

def predict(btn):
    global continuevisu,goodir,app
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
        paramdict['thrpatch']=indata['thrpatch']
        paramdict['thrproba']=indata['thrproba']
        paramdict['thrprobaMerge']= indata['thrprobaMerge']
        paramdict['thrprobaUIP']=indata['thrprobaUIP']
        paramdict['picklein_file']=indata['picklein_file']
        paramdict['picklein_file_front']=indata['picklein_file_front']
        pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
#    roirun(app.getListItems("list"),lisdir)
        predictrun(indata,lisdir)
        app.stop(Stop)
        visuDraw()
    else:
        app.errorBox('error', 'no patient selected for predict')
        app.stop(Stop)
        initDraw()
    goodir=False
    continuevisu=False


def visualisation(btn):
    global app,continuevisu
#    print(app.getListItems("list"))
    indata={}
    indata['lispatient']=selectpatient
#    print 'frontpredict',frontpredict
    if frontpredict:
        indata['3dasked']=True
    else:
        indata['3dasked']=False
    indata['viewstyle']=app.getRadioButton("planar")
#    if len(indata['lispatient']) >0:
#    roirun(app.getListItems("list"),lisdir)
    visuarun(indata,lisdir)
#    else:
#        app.errorBox('error', 'no patient selected for visu')
    app.stop(Stop)
    continuevisu=True
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
    ans= app.yesNoBox("Confirm Exit", "Are you sure you want to exit the application?")
    if ans:
#        app.stop(Stop)

        sys.exit(1)

    else:
        redraw(app)


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



def selectPatientDir():
    global lisdir,goodir
    lisdirold=lisdir
#    print 'select dir'
    lisdir=os.path.realpath(app.directoryBox(title='path patient',dirName=lisdir))
    pbg=False
    print lisdir
    if lisdir ==pathPredict:
        print 'exit'
        app.stop(Stop)
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

def selectPatientDirB(btn):
    selectPatientDir()


def presshelp(btn):
#    print 'help'

    filehelp=os.path.join(pathPredict,'doc.pdf')
#    print filehelp
#    webbrowser.open_new(r'file://C:\Users\sylvain\Documents\boulot\startup\radiology\roittool\modulepython\doc.pdf')
    webbrowser.open_new(r'file://'+filehelp)





listannotated=[]
goodir=False

def initDraw():
    global app

    app = gui("Predict form","1000x600")
    app.setBg("lightBlue")
    app.setFont(10)
    app.setStopFunction(Stop)
    app.addButton("HELP",  presshelp)

#    app.addLabel("top", "Select patient directory:", 0, 0)
    if not goodir: selectPatientDir()
#    app.setLabelBg("top","blue")
#    app.setLabelFg("top","yellow")

    app.addHorizontalSeparator( colour="red",colspan=2)
#    app.addLabel("sepa", "")
#    app.setLabelBg("top", "green")
#    print goodir
    if goodir:                       # Row 1,Column 1
        app.addLabel("path_patient", lisdir,colspan=2)
        app.addButton("Change Patient Dir",  selectPatientDirB)


        app.addHorizontalSeparator( colour="red")
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
        app.addListBox("list",listannotated,colspan=2)
        app.setListBoxMulti("list", multi=True)
        app.setListBoxRows("list",10)
        app.addHorizontalSeparator( colour="red",colspan=1)
        app.setFont(8)
        app.setSticky("w")
        app.addLabel("l1", "Prediction parameters:")
        app.setLabelBg("l1","blue")
        app.setLabelFg("l1","yellow")
        row = app.getRow()

        app.addLabelNumericEntry("Percentage of pad Overlapp",row,0)
        app.setEntry("Percentage of pad Overlapp", thrpatch)

        app.addLabelNumericEntry("Treshold proba for predicted image generation",row,1)
        app.setEntry("Treshold proba for predicted image generation", thrproba)

        row = app.getRow()
        app.addLabelNumericEntry("Treshold proba for volume calculation",row,0)
        app.setEntry("Treshold proba for volume calculation",thrprobaUIP)

        app.addLabelNumericEntry("Treshold proba for merge cross and front view",row,1)
        app.setEntry("Treshold proba for merge cross and front view", thrprobaMerge)
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
        app.setEntry("cross view weight",picklein_file)
        app.addRadioButton("predict_style", "Cross Only",row,1)
        row = app.getRow()
        app.addLabelEntry("front view weight",row,0)
        app.setEntry("front view weight",picklein_file_front)
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
#    print "visudraw"
    app = gui("Predict form","1000x600")
    app.setBg("lightBlue")

    app.setFont(10)
    app.addButton("HELP",  presshelp)
    app.addHorizontalSeparator( colour="red")
    app.addLabel("path_patient", lisdir)
    app.addButton("Change Patient Dir",  selectPatientDirB)

    app.addHorizontalSeparator( colour="red")

    if goodir:

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
                row = app.getRow() # get current row
                app.addLabel("l1", "Type of planar view,select one",row,0)
                app.addLabel("l2", "Type of orbit 3d view,select one",row,1)
                app.setLabelBg("l1","blue")
                app.setLabelFg("l1","yellow")
                app.setLabelBg("l2","blue")
                app.setLabelFg("l2","yellow")
                row = app.getRow() # get current row
#                app.addRadioButton("planar","none",row,0)
#                app.addRadioButton("planar","none",row,1)
                row = app.getRow() # get current row
                app.addRadioButton("planar","cross view",row,0)
                app.addRadioButton("planar","from cross predict",row,1)
            else:
                row = app.getRow() # get current row
                app.addLabel("l1", "Type of planar view,select one",row,0)
                app.addLabel("l2", "Type of orbit 3d view,select one",row,1)
                app.setLabelBg("l1","blue")
                app.setLabelFg("l1","yellow")
                app.setLabelBg("l2","blue")
                app.setLabelFg("l2","yellow")
                row = app.getRow() # get current row
#                app.addRadioButton("planar","none",row,0)
#                app.addRadioButton("3d","none",row,1)
                row = app.getRow() # get current row
                app.addRadioButton("planar","cross view",row,0)
                app.addRadioButton("planar","from cross predict",row,1)
                row = app.getRow() # get current row
                app.addRadioButton("planar","front view",row,0)
                app.addRadioButton("planar","from front predict",row,1)
                row = app.getRow() # get current row
                app.addRadioButton("planar","merge view",row,0)
                app.addRadioButton("planar","from cross + front merge",row,1)
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