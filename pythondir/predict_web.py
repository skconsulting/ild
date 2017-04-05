# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 09:27:11 2017
@author: sylvain
"""

import random
import string
import cherrypy
import webbrowser
import os
import simplejson
import sys
#'127.0.0.1'
import os, os.path
#import module
from module import *
#
#reservedparm=[ 'thrpatch','thrproba','thrprobaUIP','thrprobaMerge','picklein_file',
#                      'picklein_file_front','tdornot','threedpredictrequest',
#                      'onlyvisuaasked','cross','front','merge']
htmldir='html'

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
#path_pickle='CNNparameters'
path_patient=''
path_html=os.path.join(cwdtop,'static/html')
#dirpickle=os.path.join(cwdtop,path_pickle)

top = os.path.join(path_html, 'top.html')
patienttop = os.path.join(path_html, 'patienttop.html')
patientbottom = os.path.join(path_html, 'patientbottom.html')

visuatop = os.path.join(path_html, 'visuatop.html')
visuabottomend = os.path.join(path_html, 'visuabottomend.html')
visuabottomcross = os.path.join(path_html, 'visuabottomcross.html')
visuabottom = os.path.join(path_html, 'visuabottom.html')
visuatopfollow = os.path.join(path_html, 'visuatopfollow.html')

visuarestop = os.path.join(path_html, 'visuarestop.html')

visuaresbottom = os.path.join(path_html, 'visuaresbottom.html')


class PredictTool(object):
    def header(self):
        return """
        <!DOCTYPE html>
        <html>
         <head>
	<title> Predict Config </title>
    </head>"""

    def footer(self):
        return "</body></html>"
    @cherrypy.expose
    
    def index(self):
        return open(top)
    
    @cherrypy.expose
    def submit(self, name):
        cherrypy.response.headers['Content-Type'] = 'application/json'
        return simplejson.dumps(dict(title="Hello, %s" % name))
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def getData(self):
        return {
      'foo' : 'bar',
      'baz' : 'another one'
              }
        
    @cherrypy.expose
    def generate(self,**kwargs):
        global path_patient,listdir
        print "generate"
        print "generate", kwargs
        print "generate", path_patient
#        runPredict=False
        nota=True
        try:
            a= kwargs['thrpatch']
        except KeyError:
            print 'Not predict, back to visu'
            nota=False
            
        if nota:
            print 'run predict'
            listdir=predict(kwargs,path_patient)
#            runPredict=True
#        else:
#            for key, value in kwargs.items():
#                if key not in reservedparm:
#                    listdir.append(key)
#        listdirdummy,stsdir=lisdirprocess(path_patient)        
#        if not runPredict:        
        listdir,stsdir=lisdirprocess(path_patient)
        oneatleast=0
#        print 'generate lisdir',listdir
#        print 'generate stsdir',stsdir
        for key, value in stsdir.items():
            for key1, value1 in value.items():
                if  value1 == 'True':
                    oneatleast+=1
        print 'oneatleast',oneatleast
        a=open(visuatop,'r')
        app=a.read()
        a.close()
        yield app   
        if oneatleast>0:
            yield "<h2 class = 'warningnopatient' > No patient has been predicted </h2>"
       
        isfront=0
        for user in listdir:                
            if stsdir[user]['front']==True:                
              isfront+=1
        if isfront>0:     
                a=open(visuabottom,'r')
                app=a.read()
                a.close()
                yield app
        else:
                a=open(visuabottomcross,'r')
                app=a.read()
                a.close()
                yield app
        a=open(visuatopfollow,'r')
        app=a.read()
        a.close()
        yield app   
        i=0
        for user in listdir:                
            predictdone=False
            if stsdir[user]['cross']==True:                
                statustext=' CROSS'
                predictdone=True
            else:
                    statustext=''
            if stsdir[user]['front']==True:
                statustext=statustext+ '& FRONT'
            else:
                statustext=statustext+''
            if predictdone:
                    predictdone='classpredictdone'
                    if i==0:
                        yield "<input type='radio' checked name = 'lispatient' value='"+user+"' id='"+user+"'/>"
                    else:
                        yield "<input type='radio'  name = 'lispatient' value='"+user+"' id='"+user+"'/>"
                    i+=1
                    yield "<label id='"+predictdone+"'for '"+user+"'>"+user+statustext+" </label> <br>"
                    
        a=open(visuabottomend,'r')
        app=a.read()
        a.close()
        yield app

#        return kwargs

    @cherrypy.expose
    def visualisation(self,**kwargs):
        print 'visualisation'
        print kwargs
        global path_patient
        nota=True
        messageout=''
        try:
            a= kwargs['typeofview']
        except KeyError:
            print 'Not cross view'
            nota=False
        nota3d=True
        try:
            a= kwargs['typeofo3dview']
        except KeyError:
            print 'Not 3d view'
            nota3d=False
        
            
        if nota:   
#        print kwargs['thrpatch']
            messageout=visuarun(kwargs,path_patient)
#            yield self.header()
        print 'messageout',messageout
        listHug=kwargs['lispatient']
        a=open(visuarestop,'r')
        app=a.read()
        a.close()
        yield app
        yield "<h2>      visualization patient "+listHug+ "</h2>"
        if nota==False and nota3d==False:
             yield "<h2 class = 'warningnopatient' > no view type selected</h2>"
        if len(messageout)>0:
            yield "<h2 class = 'warningnopatient' >"+messageout+"</h2>"
        a=open(visuaresbottom,'r')
        app=a.read()
        a.close()
        yield app
     
        
        if nota3d:                
            type3dview=kwargs['typeofo3dview']                     
            patient_path_complet=os.path.join(path_patient,listHug)
            patient_path_complet=os.path.join(patient_path_complet,htmldir)
            if type3dview=='3dcross':
                namehtml=listHug+'_uip.html'
            elif type3dview=='3dfront':
                namehtml=listHug+'_uip3d.html'
            else:
                namehtml=listHug+'_uipmerge.html'
                
            b=open(os.path.join(patient_path_complet,namehtml),'r')
            app2=b.read()
            b.close()
            yield app2
            
    @cherrypy.expose
    def stop(self):
        cherrypy.engine.exit()
        sys.exit()
#        os.kill()
        
    @cherrypy.expose
    def generatedir(self,lisdir=path_patient):
        print 'generatedir'

        global path_patient
        path_patient=lisdir
        print 'path_patient', path_patient
        some_sg,stsdir=lisdirprocess(lisdir)
        a=open(patienttop,'r')
        app=a.read()
        a.close()
        yield app
 
        for user in some_sg:    
            predictdoneF=False
            yield "<input type='checkbox' checked name = 'lispatientselect' value='"+user+"' id='"+user+"'/>"
            if stsdir[user]['cross']==True:
                statustext=' CROSS predict already done'
                predictdoneF=True
            else:
                    statustext=''
            if stsdir[user]['front']==True:
                statustext=statustext+ ', FRONT predict already done'
            else:
                statustext=statustext+''
            if predictdoneF:
                    predictdone='classpredictdone'
            else:
                    predictdone='classpredictnotdone'                    
            yield "<label id='"+predictdone+"'for '"+user+"'>"+user+statustext+" </label> <br>"
        a=open(patientbottom,'r')
        app=a.read()
        a.close()
        yield app
        
        
    index.exposed = True             
    @cherrypy.expose
    def display(self):
        return cherrypy.session['mystring']
    
def open_page():
    webbrowser.open("http://127.0.0.1:8082/")
#cherrypy.tree.mount(AjaxApp(), '/', config=conf)
#tutconf = os.path.join(os.path.dirname(__file__), 'predict.conf')
if __name__ == '__main__':
    conf = {
        'global':{'server.socket_host': '127.0.0.1',
                        'server.socket_port': 8082,
                        'server.thread_pool' : 10,
                        'tools.sessions.on' : True,
                         'tools.encode.encoding' : "Utf-8",
                         'log.error_file': "myapp.log",
                           'log.screen':True ,  
                           'tools.sessions.timeout': 100000
                        },
        '/': {
            'tools.sessions.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd()),
            'tools.sessions.storage_class' : cherrypy.lib.sessions.FileSession,
            'tools.sessions.storage_path' : "/some/directory",
              'log.screen':True   

    
        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': '../static'                                     
        }
       
    }

#        
cherrypy.engine.subscribe('start', open_page)
cherrypy.quickstart(PredictTool(), '/', conf)
 