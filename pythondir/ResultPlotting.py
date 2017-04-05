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
namedirtop='pickle_ex/pickle_ex71'
namedirtop='pickle'
pfile = os.path.join(cwdtop,namedirtop)
print pfile
# which file is the source
fileis = [name for name in os.listdir(pfile) if ".csv" in name.lower()]
#print filei
for f in fileis:
       
#       if ".csv" in f.lower():
#           print f
           if f.find("Best")>0  and f.find ("res")==0:
#               print 'best:', f
               fb= f
           elif f.find("res")==0:
#               print 'all:',f

               ft=f
print 'all:', ft
print 'best :',fb
##total file
filei = os.path.join(pfile,ft)
print filei
with open(filei, 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    fscore = []
    val_accuracy = []
    train_loss = []
    val_loss = []
    x = []
#    print reader
    for row in reader:
#        print row
        fscore.append([float(row[' Val_fscore'])])
        val_accuracy.append([float(row[' Val_acc'])])
        train_loss.append([float(row[' Train_loss'])])
        val_loss.append([float(row[' Val_loss'])])
        x.append(row['Epoch'])
#        print row[' Val_loss']
print '--------------'
print 'Current Last Epoch: ',row['Epoch'][0:2]
print ' Val_fscore',' Val_acc',' Train_loss',' Val_loss'
print (row[' Val_fscore'][0:6],row[' Val_acc'][0:6],\
       row[' Train_loss'][0:6],row[' Val_loss'][0:6])
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


plt.ylim(0.6,0.8)
plt.ylim(0,1)
ax.plot(x,fscore, label='fscore');
ax.plot(x,val_accuracy, label='val_acc');
ax.plot(x,train_loss, label='train_loss');
ax.plot(x,val_loss, label='val_loss');

legend = ax.legend(loc='lower left', shadow=True,fontsize=10)
#plt.show()


filei = os.path.join(pfile,fb)
with open(filei, 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    fscore = []
    val_accuracy = []
    train_loss = []
    val_loss = []
    x = []
    for row in reader:
        
        fscore.append([float(row[' Val_fscore'])])
        val_accuracy.append([float(row[' Val_acc'])])
        train_loss.append([float(row[' Train_loss'])])
        val_loss.append([float(row[' Val_loss'])])
        x.append(row['Epoch'])
        
fig = plt.figure()
print 'Best Last Epoch: ',row['Epoch'][0:2]
print ' Val_fscore',' Val_acc',' Train_loss',' Val_loss'
print (row[' Val_fscore'][0:6],row[' Val_acc'][0:6],\
       row[' Train_loss'][0:6],row[' Val_loss'][0:6])
 
#fig.set_size_inches(9, 7)

ax = fig.add_subplot(1,1,1)
ax.set_title('Training Results Best',fontsize=10)
ax.set_xlabel('# Epochs')
ax.set_ylabel('value')
ax.yaxis.tick_right()
ax.yaxis.set_ticks_position('both')
ax.yaxis.set_label_position("right")
plt.ylim(0,1)
#plt.plot(x,fscore)
ax.plot(x,fscore, label='fscore');
ax.plot(x,val_accuracy, label='val_acc');
ax.plot(x,train_loss, label='train_loss');
ax.plot(x,val_loss, label='val_loss');
legend = ax.legend(loc='lower left', shadow=True,fontsize=10)


plt.show()
#(X_test, y_test) = H.load_testdata()

# predict with test dataset and record results
#pred = CNN.prediction(X_test, y_test, train_params)
