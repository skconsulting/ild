# coding: utf-8
#Sylvain Kritter 21 septembre 2016
""" merge chu Grenoble and HUg patchdatabases"""
import os
import shutil

#global environment

source='CHU'
patchsource='TOPPATCH_16_set0_13b2'
patsonorm='patches'
dest='HUG'
patchdestT='TOPPATCH_16_set0_13b2'
patchdest='patches'


#########################################################################
cwd=os.getcwd()
(top,tail)=os.path.split(cwd)
sourcedir = os.path.join(top,source)
sourcedir = os.path.join(sourcedir,patchsource)
sourcedir = os.path.join(sourcedir,patsonorm)

destdir=os.path.join(top,dest)
destdir1=os.path.join(destdir,patchdestT)
destdir=os.path.join(destdir1,patchdest)

usedclassif = [
        'back_ground',
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
        'emphysema',
        'HCpret',
        'HCpbro',
        'GGpbro',
        'GGpret',
        'bropret'
        ]

def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)
#usedclassif = [
#            'back_ground',
#            'consolidation',
#            'HC',
#            'ground_glass',
#            'healthy',
#            'micronodules',
#            'reticulation',
#            'air_trapping',
#            'cysts',
#            'bronchiectasis'
#            ]
print sourcedir,destdir

category_list=os.listdir(sourcedir)
print 'category_list',category_list
listsource=[f for f in usedclassif if f in category_list]
print 'listsource',listsource
for l in listsource:
    print l
    sourcedirl = os.path.join(sourcedir,l)
    listsourcel=os.listdir(sourcedirl)
    for k in listsourcel:
      print k
      sourcedirk = os.path.join(sourcedirl,k)
      desdirl = os.path.join(destdir,l)
      destdirk = os.path.join(desdirl,k)
      remove_folder(destdirk)
#      remove_folder(desdirl)
#      print l, k,sourcedirk,destdirk
      if not os.path.exists(desdirl):          
          os.mkdir(desdirl)  
      if not os.path.exists(destdirk):          
          os.mkdir(destdirk)     
      listsourcem=os.listdir(sourcedirk)
      for f in listsourcem:
        sourcedirf = os.path.join(sourcedirk,f)
        sourcedestf = os.path.join(destdirk,f)
#        print sourcedirf,sourcedestf

        shutil.copyfile(sourcedirf,sourcedestf)
    

#    