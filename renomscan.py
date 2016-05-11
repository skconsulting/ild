import os
import shutil



""" to rename scan in file 171 to 185 with scan number"""
cwd=os.getcwd()
dd=os.listdir(cwd)

for dirnam in dd:
    #dirnam = 35
    if os.path.isdir(os.path.join(cwd, dirnam)):
        subdir = os.path.join(cwd,dirnam)
#    print(subdir)
        #subdir = top/35
        dd1=os.listdir(subdir)
        num=0
        for ff in dd1:     
    
#            print(ff)
          if ff.find('dcm') >0 :
#                    print(ff)
            num+=1

            corfpos=ff.find('.dcm')
            cor=ff[0:corfpos]
            ncff=os.path.join(subdir,ff)
            if num<10:
                    nums='000'+str(num)
            else:
                nums='00'+str(num)
            newff=cor+'-'+nums+'.dcm'
            print(newff)
            shutil.copyfile(ncff,os.path.join(subdir,newff) )
            os.remove(ncff)
print('comp part 1')




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