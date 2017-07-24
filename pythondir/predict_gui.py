# -*- coding: utf-8 -*-
"""
V1.0 Created on Sun Apr 05 09:52:27 2017

@author: sylvain Kritter 

Version 1.1 15-June-2017
bug fixed:
    1
"""
from param_pix_p import *
from tdGenePredictGui import *

from modulepredictgui import *


print os.environ['USERPROFILE']
print os.environ['LOCALAPPDATA']
#print os.environ['APPDATA']
print os.environ['PROGRAMDATA']

instdirPredict='Predict'
instdirMedikey='MedikEye'
instdirpredictLocal='Predict'
instdirMedikeyLocal='MedikEye'
modulepython='modulepython'
moduledoc='doc'


pathMedikEye=os.path.join(os.environ['PROGRAMDATA'],instdirMedikey)
pathPredict=os.path.join(pathMedikEye,instdirPredict)
pathPredictModulepython=os.path.join(pathPredict,modulepython)
pathPredictDoc=os.path.join(pathPredict,moduledoc)

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
#cwd=os.getcwd()

#pathPredict=cwd
paramsaveDir=os.path.join(pathPredictlocal,paramsave)
if not os.path.exists(paramsaveDir):
    os.mkdir(paramsaveDir)

paramsaveDirf=os.path.join(paramsaveDir,paramname)

paramdict={}

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
            subErosion = paramdict['subErosion in mm']

    else:
        lisdir=os.environ['USERPROFILE']
        thrpatch= 0.95
        thrproba=0.7
        thrprobaMerge=0.7
        thrprobaUIP= 0.7
        picklein_file= "pickle_ex80"
        picklein_file_front= "pickle_ex81"
        subErosion = 15  # erosion factor for subpleura in mm

def predict(btn):
    global continuevisu,goodir,app
    indata={}
#    print(app.getListItems("list"))
#    print(app.getEntry("Percentage of pad Overlapp"))
    indata['thrpatch']=app.getEntry("Percentage of pad Overlapp")
    indata['thrproba']=app.getEntry("Threshold proba for predicted image generation")
    indata['thrprobaMerge']=app.getEntry("Threshold proba for merge cross and front view")
    indata['thrprobaUIP']=app.getEntry("Threshold proba for volume calculation")
    indata['threedpredictrequest']=app.getRadioButton("predict_style")
    indata['picklein_file']=app.getEntry("cross view weight")
    indata['picklein_file_front']= app.getEntry("front view weight")
    indata['subErosion']= app.getEntry("subErosion in mm")
    indata['lispatientselect']=app.getListItems("list")
    indata['Fast']=app.getCheckBox("Fast")

#    roirun(app.getListItems("list"),lisdir)
    if len(indata['lispatientselect']) >0:
        paramdict['thrpatch']=indata['thrpatch']
        paramdict['thrproba']=indata['thrproba']
        paramdict['thrprobaMerge']= indata['thrprobaMerge']
        paramdict['thrprobaUIP']=indata['thrprobaUIP']
        paramdict['picklein_file']=indata['picklein_file']
        paramdict['subErosion in mm']=indata['subErosion']
        paramdict['picklein_file_front']=indata['picklein_file_front']
        pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
#    roirun(app.getListItems("list"),lisdir)
#        print indata
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
#    indata['subErosion']= app.getEntry("subErosion in mm")
    indata['thrprobaUIP']=app.getEntry("Threshold proba")
    indata['lispatientselect']=selectpatient

#    indata['thrpatch']=app.getEntry("Percentage of pad Overlapp")
#    print 'frontpredict',frontpredict
    if frontpredict:
        indata['3dasked']=True
    else:
        indata['3dasked']=False
    indata['viewstyle']=app.getRadioButton("planar")

    visuarun(indata,lisdir)

    app.stop(Stop)
    continuevisu=True
    visuDraw()

def redraw(app):
    app.stop(Stop)
    initDraw()

def visuDrawl(btn):
    global frontpredict,continuevisu,selectpatient,app

    selectvisu=app.getListItems("list")
    if len(selectvisu) >0:
         selectpatient=selectvisu[0]
         frontexist=selectvisu[0].find('Cross & Front')
         selectpatient=selectvisu[0]
         if frontexist>0:
             frontpredict=True
         else:
            frontpredict=False
         continuevisu=True
    else: 
        continuevisu=False
    app.stop(Stop)
    visuDraw()


def checkStop():
    return app.yesNoBox("Confirm Exit", "Are you sure you want to exit the application?")

def Stop():
    return True

def boutonStop(btn):
    global app
    ans= app.yesNoBox("Confirm Exit", "Are you sure you want to exit the application?")
    if ans:
#        app.stop(Stop)
        sys.exit(1)
    else:
        redraw(app)

def selection(btn):
    global frontpredict,continuevisu,selectpatient,app
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
    global continuevisu,app
    continuevisu=False
    app.stop(Stop)
    visuDraw()


def selectPatientDir():
    global lisdir,goodir
    lisdirold=lisdir
#    print 'select dir'
    lisdir=os.path.realpath(app.directoryBox(title='path patient',dirName=lisdir))
    pbg=False
#    print 'lisdir,pathPredict',lisdir,pathPredict
    if lisdir ==pathPredictModulepython:
        print 'exit'
        app.stop(Stop)
        sys.exit(1)

    if os.path.exists(lisdir):
        lisstdirec= os.walk(lisdir).next()[1]
#        print 'lisstdirec',lisstdirec
        for i in lisstdirec:
            sourced=os.path.join(os.path.join(lisdir,i),source)
            print 'sourced',sourced
            if os.path.exists(sourced):
                paramdict['path_patient']=lisdir
#                print 'paramdict',paramdict
                pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
                if pbg==False:
                    pbg=True
                break
            else:
               ldcm= [name for name in os.listdir(os.path.join(lisdir,i)) if name.find('.dcm')>0]
               if len(ldcm)>0:
                    paramdict['path_patient']=lisdir
#                print 'paramdict',paramdict
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
    filehelp=os.path.join(pathPredictDoc,'doc_predict.pdf')
#    print filehelp
#    webbrowser.open_new(r'file://C:\Users\sylvain\Documents\boulot\startup\radiology\roittool\modulepython\doc.pdf')
    webbrowser.open_new(r'file://'+filehelp)

def initDrawB(btn):
    global goodir,app
    app.stop(Stop)
#    print 'initDrawB'
    goodir=True
    initDraw()

listannotated=[]
goodir=False
def initDraw():
    global app
#    print goodir
    app = gui("Predict form","1000x700")
    app.setBg("lightBlue")
    app.setFont(10)
    app.setStopFunction(Stop)
    app.addButton("HELP",  presshelp)

#    app.addLabel("top", "Select patient directory:", 0, 0)
    if not goodir: selectPatientDir()

    if goodir:
        app.addLabel("path_patientt", "path_patient",colspan=2)
        app.addLabel("path_patient", lisdir,colspan=2)
        app.setLabelBg("path_patient", "Blue")
        app.setLabelFg("path_patient", "Yellow")
        app.setLabelBg("path_patientt", "Blue")
        app.setLabelFg("path_patientt", "Yellow")


        app.addButton("Change Patient Dir",  selectPatientDirB,colspan=2)
        app.addHorizontalSeparator( colour="red",colspan=2)


#        app.addHorizontalSeparator( colour="red")
        app.addLabel("top1", "Select patient for prediction:")
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
#        app.setFont(8)
        app.setSticky("w")
        app.addLabel("l1", "Prediction parameters:")
        app.setLabelBg("l1","blue")
        app.setLabelFg("l1","yellow")
        row = app.getRow()

        app.addLabelNumericEntry("Percentage of pad Overlapp",row,0)
        app.setEntry("Percentage of pad Overlapp", thrpatch)

        app.addLabelNumericEntry("Threshold proba for predicted image generation",row,1)
        app.setEntry("Threshold proba for predicted image generation", thrproba)
        row = app.getRow()
        app.addLabelNumericEntry("Threshold proba for volume calculation",row,0)
        app.setEntry("Threshold proba for volume calculation",thrprobaUIP)

        app.addLabelNumericEntry("Threshold proba for merge cross and front view",row,1)
        app.setEntry("Threshold proba for merge cross and front view", thrprobaMerge)
        row = app.getRow()

        app.addLabelNumericEntry("subErosion in mm",row,1)
        app.setEntry("subErosion in mm", subErosion)
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
        row = app.getRow()
        app.addLabel("Fast","Tick  for fast run (without store of results):",row,0)
        app.addCheckBox("Fast",row,1)
        app.setCheckBox("Fast",ticked=True,callFunction=False)

#        app.setFont(10)
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
#    app.decreaseButtonFont(2)
    app.setFont(10,font=None)
    app.addButton("HELP",  presshelp)
    app.addLabel("path_patientt", "path_patient",colspan=2)
    app.addLabel("path_patient", lisdir,colspan=2)
    app.setLabelBg("path_patient", "Blue")
    app.setLabelFg("path_patient", "Yellow")
    app.setLabelBg("path_patientt", "Blue")
    app.setLabelFg("path_patientt", "Yellow")
    app.addButton("Change Patient Dir",  selectPatientDirB,colspan=2)
    app.addButton("Go back to prediction form",  initDrawB,colspan=2)
    app.addHorizontalSeparator( colour="red",colspan=2)
#    app.setFont(10)
    if goodir:

        if not continuevisu:
            app.addLabel("top1", "Select patient to visualize:")
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
            row = app.getRow() # get current row
#            app.addLabelNumericEntry("subErosion in mm",row,1)
#            app.setEntry("subErosion in mm", subErosion)
            app.addLabelNumericEntry("Threshold proba",row,0)
            app.setEntry("Threshold proba",thrprobaUIP)
           
#            app.addLabelNumericEntry("Percentage of pad Overlapp")
#            app.setEntry("Percentage of pad Overlapp", thrpatch)
            app.addLabel("l11", "Patient selected: "+selectpatient)
            app.addButton("Go back to Selection",  gobackselection)
            app.setLabelBg("l11","blue")
            app.setLabelFg("l11","yellow")
            app.addHorizontalSeparator( colour="red")
#            app.setFont(8)
            app.setSticky("w")
            if not frontpredict:
                row = app.getRow() # get current row
                app.addLabel("l1", "Type of planar view,select one",row,0)
                app.addLabel("l2", "Type of orbit 3d view,select one",row,1)
                app.setLabelBg("l1","blue")
                app.setLabelFg("l1","yellow")
                app.setLabelBg("l2","blue")
                app.setLabelFg("l2","yellow")
#                row = app.getRow() # get current row
#                app.addRadioButton("planar","none",row,0)
#                app.addRadioButton("planar","none",row,1)
                row = app.getRow() # get current row
                app.addRadioButton("planar","cross view",row,0)

                app.addRadioButton("planar","from cross predict",row,1)
                row = app.getRow() # get current row
                app.addRadioButton("planar","volume view from cross",row,0)
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

                app.addRadioButton("planar","cross view",row,0)
                app.addRadioButton("planar","from cross predict",row,1)
                row = app.getRow() # get current row
                app.addRadioButton("planar","volume view from cross",row,0)
                app.addRadioButton("planar","from front predict",row,1)
                row = app.getRow() # get current row
                app.addRadioButton("planar","volume view from front",row,0)
                app.addRadioButton("planar","from cross + front merge",row,1)

                app.addRadioButton("planar","front view")
                app.addRadioButton("planar","merge view")
                app.addRadioButton("planar","front projected view")
               
            app.addHorizontalSeparator( colour="red")
            app.setSticky("n")
#            app.setStrech("row")
            app.addHorizontalSeparator( colour="red",colspan=1)
#            app.setFont(10)
            app.addButton("Visualisation",  visualisation)
            app.addHorizontalSeparator( colour="red")
    app.addButton("Quit",  boutonStop)
    app.go()
selectpatient=''
frontpredict=False
continuevisu=False
initDraw()