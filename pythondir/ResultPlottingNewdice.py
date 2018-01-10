# coding: utf-8
"""
# # Plot Results of training
# allows interpretation of impact of various parameters setting 
# 
# ### needs correct <file>.csv file

"""
import csv
import os

import matplotlib.pyplot as plt


topdir='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool'

#path with data for training
pickel_dirsource_th='0.95'
pickel_dirsource_root='pickle'
pickel_dirsource_e='train' #path for data fort training
pickel_dirsourcenum='set1' #extensioon for path for data for training
extendir1='1'
extendir2=''
##########################################################################

pickleStore='pickle'
if len (extendir2)>0:
    extendir2='_'+extendir2

pickel_dirsource='th'+pickel_dirsource_th+'_'+pickel_dirsource_root+'_'+ pickel_dirsource_e+'_'+pickel_dirsourcenum+'_'+extendir1+extendir2

patch_dir=os.path.join(topdir,pickel_dirsource)
patch_dir_store=os.path.join(patch_dir,pickleStore)

print 'patch dir source : ',patch_dir #path with data for training
print 'weight dir store : ',patch_dir_store #path with weights after training
#
#oooo
#
#cwd=os.getcwd()
#(cwdtop,tail)=os.path.split(cwd)
##print cwd
##namedirtop='pickle_ex/pickle_ex71'
#namedirtop='pickle'
pfile = patch_dir_store
print 'path to get csv with train data',pfile
# which file is the source
fileis = [name for name in os.listdir(pfile) if ".csv" in name.lower()]
#print filei
ordfb=[]
ordf=[]
for f in fileis:
   nbs = os.path.getmtime(os.path.join(pfile,f))
   tt=(f,nbs)
#   if f.find("Best")>0  :              
#        ordfb.append(tt)               
#   else:
   ordf.append(tt)
#ordlistfb=sorted(ordfb,key=lambda col:col[1],reverse=True)
#print ordlistfb
#fb=ordlistfb[0][0]
ordlistft=sorted(ordf,key=lambda col:col[1],reverse=True)
#print ordlistft
ft=ordlistft[0][0]
 
print 'all:', ft
#print 'best :',fb
#ft='training.log'
##total file
filei = os.path.join(pfile,ft)
print filei
with open(filei, 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    categorical_accuracy = []
    val_accuracy = []
    train_loss = []
    lr=[]
    train_loss = []
    val_loss = []
    x = []
#    print reader
    for row in reader:
#        print row
        categorical_accuracy.append([float(row['dice_coef'])])
        val_accuracy.append([float(row['val_dice_coef'])])
        lr.append([float(row['lr'])])
        train_loss.append([float(row['loss'])])
        val_loss.append([float(row['val_loss'])])
        x.append(row['epoch'])
#        print row[' Val_loss']
lrf= float(row['lr'])
print '%0.2e'% lrf

print '--------------'
print 'Current Last Epoch: ',row['epoch'][0:4]
print 'loss      ','val_loss  ','lr      ', 'train_acc','val_acc'
print ( row['loss'][0:6],row['val_loss'][0:6],'%0.2e'% lrf,
        row['dice_coef'][0:4],row['val_dice_coef'][0:4])
print '--------------'

# plotting

fig = plt.figure()

#fig.set_size_inches(9, 7)

ax = fig.add_subplot(1,1,1)
ax.set_title('lr Results',fontsize=10)
ax.set_xlabel('# Epochs')
ax.set_ylabel('value')
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.yaxis.set_ticks_position('both')

ax.plot(x,lr, label='lr');
#ax.plot(x,train_loss, label='train_loss');
#ax.plot(x,val_loss, label='val_loss');

legend = ax.legend(loc='upper right', shadow=True,fontsize=10)
plt.show()
fig = plt.figure()

#fig.set_size_inches(9, 7)

ax = fig.add_subplot(1,1,1)
ax.set_title('Training Results',fontsize=10)
ax.set_xlabel('# Epochs')
ax.set_ylabel('value')
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.yaxis.set_ticks_position('both')


#plt.ylim(0.6,0.8)
plt.ylim(0,1)
ax.plot(x,categorical_accuracy, label='categorical_accuracy');
ax.plot(x,val_accuracy, label='val_categorical_accuracy');
#ax.plot(x,train_loss, label='train_loss');
#ax.plot(x,val_loss, label='val_loss');

legend = ax.legend(loc='lower right', shadow=True,fontsize=10)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x,train_loss, label='loss');
ax.plot(x,val_loss, label='val_loss');
legend = ax.legend(loc='upper right', shadow=True,fontsize=10)
plt.show()

print '--------'*10

