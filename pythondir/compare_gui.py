# -*- coding: utf-8 -*-
"""
V1.0 Created on Sun Apr 05 09:52:27 2017

@author: sylvain Kritter 

version 1.5
6 September 2017
"""
#from param_pix_p import *
#from tdGenePredictGui import predictrun
from comparevisu import visuarun,lisdirprocess,predictmodule
from param_pix_c import source

from appJar import gui
import os
import sys
import webbrowser

import cPickle as pickle

print os.environ['USERPROFILE']
print os.environ['LOCALAPPDATA']
#print os.environ['APPDATA']
print os.environ['PROGRAMDATA']

instdirPredict='Compare'
instdirMedikey='MedikEye'
instdirpredictLocal='Score'
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
print 'pathMedikEye',pathMedikEye
print 'pathPredict',pathPredict

print 'pathMedikEyelocal',pathMedikEyelocal
print 'pathPredictlocal',pathPredictlocal


if not os.path.exists(pathMedikEyelocal):
    os.mkdir(pathMedikEyelocal)
if not os.path.exists(pathPredictlocal):
    os.mkdir(pathPredictlocal)
#workingdir= os.path.join(os.environ['USERPROFILE'],workdiruser)
#print pathMedikEyelocal
#print pathPredictlocal


version ="1.4"
paramsave='data'
paramname ='paramname.pkl'

paramsaveDir=os.path.join(pathPredictlocal,paramsave)
if not os.path.exists(paramsaveDir):
    os.mkdir(paramsaveDir)
    
paramsaveDirf=os.path.join(paramsaveDir,paramname)
paramdict={}

lisdirref=os.environ['USERPROFILE']
lisdircomp=os.environ['USERPROFILE']
paramdict['path_patient_ref']=os.environ['USERPROFILE']
paramdict['path_patient_comp']=os.environ['USERPROFILE']

if os.path.exists(paramsaveDirf):
    paramdictold=pickle.load(open( paramsaveDirf, "rb" ))
    try:
        paramdict=pickle.load(open( paramsaveDirf, "rb" ))
        lisdirref= paramdict['path_patient_ref']
        lisdircomp= paramdict['path_patient_comp']
    except:
        paramdict['path_patient_ref']=os.environ['USERPROFILE']
        paramdict['path_patient_comp']=os.environ['USERPROFILE']
        pickle.dump(paramdict,open( paramsaveDirf, "wb" )) 
else:
    print 'first time'
    

    pickle.dump(paramdict,open( paramsaveDirf, "wb" )) 

def calculate(btn):
    global app
    indata={}
    message=''
    lisdirref=paramdict['path_patient_ref']
    lisdircomp= paramdict['path_patient_comp']

    indata['lispatientselectref'],b,c=lisdirprocess(lisdirref)
    
    paramdict['lispatientselectref']= indata['lispatientselectref']

    indata['lispatientselectcomp'],b,c=lisdirprocess(lisdircomp)
    
    paramdict['lispatientselectcomp']= indata['lispatientselectcomp']
    pickle.dump(paramdict,open( paramsaveDirf, "wb" )) 
    
    if len(indata['lispatientselectcomp']) >0 and len(indata['lispatientselectref']) >0:
        app.hide()
        listdir,message=predictmodule(indata,lisdirref,lisdircomp)  
        app.show()        
        if len(message)>0:
            app.errorBox('error', message)
            app.stop(Stop)
            goodir=False
            initDraw(goodir)
        else:
            app.stop(Stop)
            visuDraw()
    else:
        app.errorBox('error', 'no patient selected for comparison')
        app.stop(Stop)
        goodir=False
        initDraw(goodir)
    


def predictScore (btn):
    global continuevisu,goodir,app
   
    indata={}
    message=''
#    print(app.getListItems("list"))
#    print(app.getEntry("Percentage of pad Overlapp"))
    indata['Select All']=app.getCheckBox("Select All")
    indata['thrpatch']=app.getEntry("Percentage of pad Overlapp")
    indata['thrproba']=app.getEntry("Threshold proba")
#    indata['thrprobaUIP']=app.getEntry("Threshold proba for volume calculation")
    indata['threedpredictrequest']=app.getRadioButton("predict_style")
    indata['picklein_file']=app.getEntry("cross view weight")
    indata['picklein_file_front']= app.getEntry("front view weight")
    indata['lispatientselect']=app.getListItems("list")
    indata['ForceGenerate']=app.getCheckBox("ForceGenerate")

#    indata['subErosion']= app.getEntry("subErosion in mm")
    
    indata['Fast']=app.getCheckBox("Fast")
    indata['centerHU']=app.getEntry("centerHU")
    indata['limitHU']=app.getEntry("limitHU")
    paramdict['Select All']=indata['Select All']
    paramdict['thrpatch']=indata['thrpatch']
    paramdict['thrproba']=indata['thrproba']

    paramdict['picklein_file_front']=indata['picklein_file_front']
    paramdict['picklein_file']=indata['picklein_file']
    paramdict['centerHU']=indata['centerHU']
    paramdict['limitHU']=indata['limitHU']
#    paramdict['lispatientselect']= indata['lispatientselect']
    paramdict['threedpredictrequest']= indata['threedpredictrequest']

    allAsked=False
    if len(app.getListItems("list"))==0 and indata['Select All']:
        allAsked=True
#        print lisdir
        indata['lispatientselect'],b,c=lisdirprocess(lisdir)
    paramdict['lispatientselect']= indata['lispatientselect']

    pickle.dump(paramdict,open( paramsaveDirf, "wb" ))       
    if len(indata['lispatientselect']) >0:
        app.hide()
        listdir,message=predictmodule(indata,lisdir) 
        
    
        indata['viewstyle']='reportAll'
        indata['lispatientselect']= indata['lispatientselect'][0]
         
        indata['viewstylet']='Cross'
        visuarun(indata,lisdir)
        if indata['threedpredictrequest']!='Cross Only':
            indata['viewstylet']='FrontProjected'
            visuarun(indata,lisdir)
            indata['viewstylet']='Merge'
            visuarun(indata,lisdir)

        
        app.show()        
        if len(message)>0:
            app.errorBox('error', message)
            app.stop(Stop)
            initDraw()
        else:
            app.stop(Stop)
            if allAsked:
                continuevisu=False
                initDraw()
            else:
                continuevisu=True
                visuDraw()
    else:
        app.errorBox('error', 'no patient selected for predict')
        app.stop(Stop)
        initDraw()
#        continuevisu=False
    goodir=False
    continuevisu=False


def visualisation(btn):
    global app,continuevisu
#    print(app.getListItems("list"))
    indata={}
#    selectvisu=app.getListItems("list")
    
    
#    selectpatient=selectvisu[0]
#    paramdict['lispatientselect']=app.getListItems("list")
    lisdirref=paramdict['path_patient_ref'] 
    lisdircomp=paramdict['path_patient_comp']  
    indata['lispatientselectref']=paramdict['lispatientselectref'] 
    indata['lispatientselect']=app.getListItems("list")
    if len(indata['lispatientselect'])==0:
        app.errorBox('error', 'no patient selected')
        app.stop(Stop)
        continuevisu=False
        visuDraw()
        
    indata['viewstyle']=app.getRadioButton("planar")
    app.hide()
    visuarun(indata,lisdirref,lisdircomp)
    app.stop(Stop)
    continuevisu=True
    visuDraw()

def redraw(app):
    app.stop(Stop)
    initDraw()

def visuDrawl(btn):
    global continuevisu,app

#    selectvisu=app.getListItems("list")
#    paramdict['lispatientselect']=selectvisu
#    pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
#    if len(selectvisu) >0:
##         selectpatient=selectvisu[0]
##         frontexist=selectvisu[0].find('Cross & Front')
###         selectpatient=selectvisu[0]
##         if frontexist>0:
##             frontpredict=True
##         else:
##            frontpredict=False
#         continuevisu=True
#    else: 
#        continuevisu=False
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
        app.stop(Stop)
        sys.exit(1)
    else:
        redraw(app)

def gscore(indata):
    global app,continuevisu
    indata={}
    lisdirref=paramdict['path_patient_ref'] 
    lisdircomp=paramdict['path_patient_comp']  
    indata['lispatientselect']=' '
        
    indata['viewstyle']="reportAll"
    app.hide()
    visuarun(indata,lisdirref,lisdircomp)
    app.stop(Stop)
    continuevisu=False
    initDraw(True)
    
        
def gscorec(btn):
    indata={}
    indata['viewstylet']='Cross'
    gscore(indata)
    
def gscoref(btn):
    indata={}
    indata['viewstylet']='FrontProjected'
    gscore(indata)

def gscorem(btn):
    indata={}
    indata['viewstylet']='Merge'
    gscore(indata)
   

def selection(btn):
    global frontpredict,continuevisu,app
    selectvisu=app.getListItems("list")
    
    
#    selectpatient=selectvisu[0]
    paramdict['lispatientselect']=selectvisu
    pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
    frontexist=selectvisu[0].find('Cross & Front')
#    selectpatient=selectvisu[0]
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
#    global lisdirref,goodir,lisdircomp
    lisdirref=paramdict['path_patient_ref']
    lisdircomp=paramdict['path_patient_comp']
    lisdirrefold=lisdirref
    lisdircompold=lisdircomp
    pbgref=False
    pbgcomp=False
#    print 'select dir'
#    print lisdir
    lisdirref=os.path.realpath(app.directoryBox(title='path patients ref',dirName=lisdirref))
    if lisdirref ==pathPredictModulepython:
        print 'exit'
        app.stop(Stop)
        sys.exit(1)
        
    if os.path.exists(lisdirref):
        lisstdirec= os.walk(lisdirref).next()[1]
#        print 'lisstdirec ref',lisstdirec
        for i in lisstdirec:
            sourced=os.path.join(os.path.join(lisdirref,i),source)
            print 'sourced',sourced
            if os.path.exists(sourced):
                paramdict['path_patient_ref']=lisdirref
#                print 'paramdict',paramdict
                pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
                if pbgref==False:
                    pbgref=True
                break
            else:
               ldcm= [name for name in os.listdir(os.path.join(lisdirref,i)) if name.find('.dcm')>0]
               if len(ldcm)>0:
                    paramdict['path_patient_ref']=lisdirref
#                print 'paramdict',paramdict
                    pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
                    if pbgref==False:
                        pbgref=True
                    break    
        
        
    lisdircomp=os.path.realpath(app.directoryBox(title='path patients comp',dirName=lisdircomp))
    if  lisdircomp==pathPredictModulepython:
        print 'exit'
        app.stop(Stop)
        sys.exit(1)
  
#    print 'lisdir,pathPredict',lisdir,pathPredict
  
    if os.path.exists(lisdircomp):
        lisstdirec= os.walk(lisdircomp).next()[1]
#        print 'lisstdirec comp',lisstdirec
        for i in lisstdirec:
            sourced=os.path.join(os.path.join(lisdircomp,i),source)
            print 'sourced',sourced
            if os.path.exists(sourced):
                paramdict['path_patient_comp']=lisdircomp
#                print 'paramdict',paramdict
                pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
                if pbgcomp==False:
                    pbgcomp=True
                break
            else:
               ldcm= [name for name in os.listdir(os.path.join(lisdircomp,i)) if name.find('.dcm')>0]
               if len(ldcm)>0:
                    paramdict['path_patient_comp']=lisdircomp
#                print 'paramdict',paramdict
                    pickle.dump(paramdict,open( paramsaveDirf, "wb" ))
                    if pbgcomp==False:
                        pbgcomp=True
                    break
                   
    if pbgref or pbgcomp :
        app.stop(Stop)
        goodir=True
        initDraw(goodir)
    else:
        lisdirref=lisdirrefold
        lisdircomp=lisdircompold
        app.errorBox('error', 'path for  patient not correct')
        app.stop(Stop)
        goodir=False
        initDraw(goodir)
    
    return goodir

def selectPatientDirB(btn):
    return selectPatientDir()


def presshelp(btn):
    filehelp=os.path.join(pathPredictDoc,'doc_score.pdf')
#    print filehelp
#    webbrowser.open_new(r'file://C:\Users\sylvain\Documents\boulot\startup\radiology\roittool\modulepython\doc.pdf')
    webbrowser.open_new(r'file://'+filehelp)

def initDrawB(btn):
    global goodir,app
    app.stop(Stop)
#    print 'initDrawB'
    goodir=True
    initDraw(goodir)


def initDraw(goodir):
    global app
   
    app = gui("Compare form","1000x800")
    app.setResizable(canResize=True)
    app.setBg("lightBlue")
    app.setFont(10)
    app.setStopFunction(Stop)
    app.addButton("HELP",  presshelp)

#    app.addLabel("top", "Select patient directory:", 0, 0)
    if not (goodir):
        goodir=selectPatientDir()

    if goodir:
        lisdircomp=paramdict['path_patient_comp']
        lisdirref=paramdict['path_patient_ref']
        app.addLabel("path_patientt", "path_patients" )
        app.setLabelBg("path_patientt", "Blue")
        app.setLabelFg("path_patientt", "Yellow")
        app.addHorizontalSeparator( colour="red" )
        app.addLabel("path_patient_ref" ,"path ref :"+lisdirref )
        app.setLabelBg("path_patient_ref", "Blue")
        app.setLabelFg("path_patient_ref", "Yellow")
        app.addHorizontalSeparator( colour="red" )
        app.addLabel("path_patient_comp", "path comp :"+lisdircomp )
        app.setLabelBg("path_patient_comp", "Blue")
        app.setLabelFg("path_patient_comp", "Yellow")

        app.addButton("Change Patient Dir",  selectPatientDirB,colspan=2)
        app.addHorizontalSeparator( colour="red" )

#        app.addHorizontalSeparator( colour="red")
        some_sg_ref,stsdir_ref,setrefdict_ref=lisdirprocess(lisdirref)
        some_sg_comp,stsdir_comp,setrefdict_comp=lisdirprocess(lisdircomp)
#        print some_sg
        print 'setrefdict_ref',setrefdict_ref
        print 'setrefdict_comp',setrefdict_comp
        listannotatedref=[]
        for user in some_sg_ref:
            if stsdir_ref[user]['front']==True:
                lroi=' Cross & Front'
                listannotatedref.append(user+' PREDICT!: ' +setrefdict_ref[user]+lroi)
            elif stsdir_ref[user]['cross']==True:
                lroi=' Cross'
                listannotatedref.append(user+' PREDICT!: '+setrefdict_ref[user]+lroi)
            else:
                listannotatedref.append(user+' noPREDICT! ')
        print 'listannotatedref',listannotatedref
#        row = app.getRow()
        app.addListBox("listref",listannotatedref,colspan=2)
        app.setListBoxMulti("listref", multi=True)
        app.setListBoxRows("listref",10)
        app.addHorizontalSeparator( colour="red",colspan=1)
        
        listannotatedcomp=[]
        for user in some_sg_comp:
            if stsdir_comp[user]['front']==True:
                lroi=' Cross & Front'
                listannotatedcomp.append(user+' PREDICT!: ' +setrefdict_comp[user]+lroi)
            elif stsdir_comp[user]['cross']==True:
                lroi=' Cross'
                listannotatedcomp.append(user+' PREDICT!: '+setrefdict_comp[user]+lroi)
            else:
                listannotatedcomp.append(user+' noPREDICT! ')
#        print listannotated
        print 'listannotatedcomp',listannotatedcomp
        app.addListBox("listcomp",listannotatedcomp,colspan=2)
        app.setListBoxMulti("listcomp", multi=True)
        app.setListBoxRows("listcomp",10)
        app.addHorizontalSeparator( colour="red",colspan=1)

        app.addButton("Calculate",  calculate)
        app.addButton("Visualisation",  visuDrawl)
        app.addButton("Report All",  gscore)

    app.addButton("Quit",  boutonStop)
    app.go()

def visuDraw():
    global app
#    print "visudraw"
    paramdict=pickle.load(open( paramsaveDirf, "rb" ))

    app = gui("Visualization compare form","1000x600")
    app.setResizable(canResize=True)
    app.setBg("lightBlue")
#    app.decreaseButtonFont(2)
    app.setFont(10,font=None)
    app.addButton("HELP",  presshelp)
    lisdircomp=paramdict['path_patient_comp']
    lisdirref=paramdict['path_patient_ref']
    app.addLabel("path_patientt", "path_patients" )
    app.setLabelBg("path_patientt", "Blue")
    app.setLabelFg("path_patientt", "Yellow")
    app.addHorizontalSeparator( colour="red" )
    app.addLabel("path_patient_ref" ,"path ref :"+lisdirref )
    app.setLabelBg("path_patient_ref", "Blue")
    app.setLabelFg("path_patient_ref", "Yellow")
    app.addHorizontalSeparator( colour="red" )
    app.addLabel("path_patient_comp", "path comp :"+lisdircomp )
    app.setLabelBg("path_patient_comp", "Blue")
    app.setLabelFg("path_patient_comp", "Yellow")
    app.addButton("Change Patient Dir",  selectPatientDirB,colspan=2)
    app.addButton("Go back to prediction form",  initDrawB,colspan=2)
    app.addHorizontalSeparator( colour="red",colspan=2)
    app.addLabel("top1", "Select patient to visualize:")

    some_sg,stsdir,setrefdict=lisdirprocess(lisdirref)
    listannotated=[]
    app.startLabelFrame("patient List:")
    app.setSticky("ew")
    for user in some_sg:
        if stsdir[user]['front']==True:
            lroi=' Cross & Front'
            listannotated.append(user+' PREDICT!: ' +setrefdict[user]+lroi)
        elif stsdir[user]['cross']==True:
            lroi=' Cross'
            listannotated.append(user+' PREDICT!: '+setrefdict[user]+lroi)
        else:
            listannotated.append(user+' noPREDICT! ')

    app.addListBox("list",listannotated)
    app.setListBoxMulti("list", multi=False)
    app.setListBoxRows("list",10)
    app.stopLabelFrame()
#    app.addButton("Selection",  selection)
    
    app.addRadioButton("planar","view")
    app.addRadioButton("planar","report")
    app.addHorizontalSeparator( colour="red",colspan=2)
    app.addButton("Visualisation",  visualisation)
#   
    app.addButton("Quit",  boutonStop)
    app.go()

############################################################################
#selectpatient=''
#frontpredict=False
continuevisu=False
listannotated=[]
goodir=False
initDraw(goodir)