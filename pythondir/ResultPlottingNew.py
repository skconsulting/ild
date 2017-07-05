# coding: utf-8

# # Plot Results of training
# allows interpretation of impact of various parameters setting 
# 
# ### needs correct <file>.csv file


import csv
import os

import matplotlib.pyplot as plt

cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
#print cwd
#namedirtop='pickle_ex/pickle_ex71'
namedirtop='pickle'
pfile = os.path.join(cwdtop,namedirtop)
print pfile
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
    val_loss = []
    x = []
#    print reader
    for row in reader:
#        print row
        categorical_accuracy.append([float(row['categorical_accuracy'])])
        val_accuracy.append([float(row['val_categorical_accuracy'])])
        train_loss.append([float(row['loss'])])
        val_loss.append([float(row['val_loss'])])
        x.append(row['epoch'])
#        print row[' Val_loss']
print '--------------'
print 'Current Last Epoch: ',row['epoch'][0:2]
print ' categorical_accuracy',' val_categorical_accuracy',' loss',' val_loss'
print (row['categorical_accuracy'][0:6],row['val_categorical_accuracy'][0:6],\
       row['loss'][0:6],row['val_loss'][0:6])
print '--------------'

# plotting

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

legend = ax.legend(loc='lower left', shadow=True,fontsize=10)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x,train_loss, label='loss');
ax.plot(x,val_loss, label='val_loss');
legend = ax.legend(loc='lower left', shadow=True,fontsize=10)
plt.show()

print '--------'*10

