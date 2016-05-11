import os
import shutil

typei='bmp'

namedirtop = 'ILD_DB_txtROIs'
#create bmp dir
cwd=os.getcwd()
nametopc=os.path.join(cwd,namedirtop)
dd=os.listdir(nametopc)
#print(dd)
for dirnam in dd:
    subdir = os.path.join(nametopc,dirnam)
#    print(subdir)
    if os.path.isdir(subdir):
        listcwd=os.listdir(subdir)
#        print(listcwd)
        if typei not in listcwd:
            os.mkdir(os.path.join(subdir,typei))

#print('1:',dd)
#copy file.jpg in typei file.jpg 
# cwd=os.getcwd()
# dd=os.listdir(cwd)
for dirnam in dd:
    #dirnam = 35
    if os.path.isdir(os.path.join(nametopc, dirnam)):
        subdir = os.path.join(nametopc,dirnam)
#    print(subdir)
        #subdir = top/35
        dd1=os.listdir(subdir)
        for ff in dd1:     
#            print(ff)
            if ff.find(typei) >0 :
#                    print(ff)
             difbmp=os.path.join(subdir,typei)
             ncff=os.path.join(subdir,ff)
             shutil.copyfile(ncff,os.path.join(difbmp,ff) )
             os.remove(os.path.join(ncff))
# print('comp part 1')




#        os.mkdir(os.path.join(cwd, dirnam)+'/bmp')

#for dirname, dirnames, filenames in os.walk('.'):
#
#    print('1:',dirname, dirnames, filenames)
#    
#    # print path to all subdirectories first.
#    for subdirname in dirnames:
#        print('2:',os.path.join(dirname, subdirname))
#        os.mkdir(os.path.join(dirname, subdirname)+'/dd')
#
#    # print path to all filenames.
#    for filename in filenames:
#        print(os.path.join(dirname, filename))
#
#    # Advanced usage:
#    # editing the 'dirnames' list will stop os.walk() from recursing into there.
#    if '.git' in dirnames:
#        # don't go into any .git directories.
#        dirnames.remove('.git')