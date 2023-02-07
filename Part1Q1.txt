# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 21:07:05 2023

@author: SS TECH
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 21:02:06 2023

@author: SS TECH
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat

#Partie 1 Visualisation des données :
#question 1
#Charger la base de données sub3_comp.mat
annots = loadmat('C:/Users/SS TECH/Downloads/sub3_comp.mat')
#affecter chaque partie base de données a une variable mdata,mdata2,mdata3
mdata = annots['test_data']
mdata2 = annots['train_data']
mdata3 = annots['train_dg']
#Convertir les sous-base de données en des Dataframe a travers la commande predifinie du 
test_data = pd.DataFrame(mdata)
train_data = pd.DataFrame(mdata2)
train_dg = pd.DataFrame(mdata3)
#Affichage des sous base données
print(test_data)
plt.plot(test_data)
plt.title('Test Data')
plt.show()
print(train_data)
plt.plot(train_data)
plt.title('Train_Data')
plt.show()
print(train_dg)
plt.plot(train_dg)
plt.title('Train_dg')
plt.show()
#Conversion des sous bases de données de Dataframe a NumPy array pour on puisse affecter des calculs
traindg = train_dg.to_numpy()
print(train_dg)
traind = train_data.to_numpy()
print(train_data)
end_shape=traindg.shape
i= range(1,end_shape[0])

begin=34500; end=38800; step=1;
#train_dg c'est une nouvelle variable qui contient une portion du traindg pour l'affichage
train_dgs=traindg[begin:end:step,:]
#amin function
#On'a appliquer cette commande pour afin d'avoir dans notre matrice que des valeurs 
#positives .
train_dgn=train_dgs-np.amin(train_dgs,axis=0)
it=i[begin:end:step]
#Affichage des differentes flexions des doigts
plt.plot(it,train_dgn[:,0],'tab:purple')
plt.plot(it,train_dgn[:,1],'tab:blue')
plt.plot(it,train_dgn[:,2],'tab:green')
plt.plot(it,train_dgn[:,3],'tab:orange')
plt.plot(it,train_dgn[:,4],'tab:red')
plt.suptitle('Time traces of Finger Flexions')
plt.show()