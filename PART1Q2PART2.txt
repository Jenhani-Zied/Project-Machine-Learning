# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 21:11:46 2023

@author: SS TECH
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:53:22 2023

@author: SS TECH
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
#Partie 1 Visualisation des données :
#question 1
annots = loadmat('C:/Users/SS TECH/Downloads/sub3_comp.mat')
mdata = annots['test_data']
mdata2 = annots['train_data']
mdata3 = annots['train_dg']
test_data = pd.DataFrame(mdata)
train_data = pd.DataFrame(mdata2)
train_dg = pd.DataFrame(mdata3)
train_data = train_data.to_numpy()
Xall = train_data[:5999,:]
Xall2 = train_data[6000:12000,:]
Xall3 = train_data[6000:100000,:]
Xall3 = pd.DataFrame(Xall3)
Xall2 = pd.DataFrame(Xall2)
Xall = pd.DataFrame(Xall)
train_dg = train_dg.to_numpy()
Yall= train_dg[0:5999,0]
Yall2 = train_dg[6000:12000,0]
Yall3 = train_dg[6000:100000,0]
Yall2 = pd.DataFrame(Yall2)
Yall3 = pd.DataFrame(Yall3)
Yall = pd.DataFrame(Yall)
Fe = [50]
Fe = np.array(Fe)
print("-------------------Xall3 = train_dg[6000:100000,0]------------  ")
print("La frequence Fe")
print(Fe)
print("La taille de Xall")
print(Xall.shape)
print("La taille de Yall")
print(Yall.shape)
#question 2 
#Subdiviser les deux matrices Xall et Yall en X_train , X_test et Y_train , Y_test
X_train , X_test , Y_train , Y_test = train_test_split(Xall3,Yall3,test_size=0.3,random_state=10)


print("La matrice X_train")
print(X_train)
print("La matrice X_test")
print(X_test)
print("La matrice Y_train")
print(Y_train)
print("La matrice Y_test")
print(Y_test)
#part 2 Regression a moindre carré
# z est une matrice unitaire de lignes = len(X_train) et d'une colonne
z = np.ones((X_train.shape[0],1))
#La nouvelle X_train est concatiner avec une autre celle z +1 Colonne avec la commande hstack
#reason for adding a new column is to add a bias term, also known as an intercept term, 
#which can help the model fit the data better
X_train = np.hstack((X_train,z))
#affichage du X_train
print ("La matrice du nouvelle X_train")
print (X_train)
#reshaping can be used to prepare the data for certain types of computation
z= z.reshape(z.shape[0],1)
print ("La taille du  z")
print(z.shape)
print ("La nouvelle taille du  X_train")
print(X_train.shape)
#alpha is step to use between the iterations
alpha =  0.01
#nombre d'iterations = 200
iterations = 2000
#m la longeur de X_train
m = len(X_train)
#La normalisation des données est une étape de prétraitement courante en 
#apprentissage automatique et consiste à s'assurer que toutes
#les caractéristiques ont la même échelle. Cela est important car certains algorithmes
X_train = (X_train - X_train.mean()) / X_train.std()
Y_train = (Y_train - Y_train.mean()) / Y_train.std()
theta = np.random.rand(X_train.shape[1])
print ("La valeur du theta initial")
print(theta)
theta = theta.reshape(theta.shape[0],1)
print ("La nouvelle valeur de thetal")
print(theta)
#GRADIENT DESCENT
print("----------Methode de Gradient Descent----------")
def gradient_descent(x, y, theta, iterations, alpha):
    past_costs = np.zeros(iterations)
    past_thetas = [theta]
    for i in range(iterations):
        prediction = x.dot(theta)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)
        past_costs[i]= cost
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))
        past_thetas.append(theta)
        
    return past_thetas, past_costs
past_thetas, past_costs = gradient_descent(X_train, Y_train, theta, iterations, alpha)
theta = past_thetas[-1]
plt.plot(past_costs)

plt.show()
past_thetas1, past_costs1 = gradient_descent(X_train, Y_train, theta, iterations, 0.001)
past_thetas2, past_costs2 = gradient_descent(X_train, Y_train, theta, iterations, 1)
past_thetas3, past_costs3 = gradient_descent(X_train, Y_train, theta, iterations, 0.03)
past_thetas, past_costs = gradient_descent(X_train, Y_train, theta, iterations, alpha)
theta = past_thetas[-1]

plt.subplot(1, 3, 1)
plt.plot(past_costs1)
plt.subplot(1, 3, 2)
plt.plot(past_costs2)
plt.subplot(1, 3, 3)
plt.plot(past_costs3)
predictions = X_train.dot(theta)
predictions = pd.DataFrame(predictions)
print ("c'est une matrice qui contient les valeurs des flexions predictés")
print(predictions)
print(predictions.shape)
R_square = r2_score(Y_train, predictions) 
print("La valeur de R_square")
print(R_square)

#----------------------------------------------------------------------------
plt.scatter(predictions, Y_train)
# Add labels to the axes
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.plot([Yall.min(), Yall.max()], [Yall.min(), Yall.max()], 'k--', lw=2)
# Show the plot
plt.show()
#------------------------------------------------------------------------------
# Create a Ridge model with a regularization parameter of 0.1
print("--------------Methode Ridge-------------")
ridge = Ridge(alpha=0.1)

# Fit the model to the data
ridge.fit(X_train, Y_train)

# Predict the values of Y using the model
Y_pred = ridge.predict(X_train)
#Y_predm = Y_pred[0:1800]
print("Mean Squared Error:", mean_squared_error(Y_train,Y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_train,Y_pred)))
print("R Squared Error:", np.sqrt(r2_score(Y_train, Y_pred)))

# Print the model coefficients
print("Affichge du matrice Y_pred")
print(Y_pred)
print(Y_pred.shape)
#affichagee ridge method
print("--------------Affichge PLOT Methode Ridge-------------")
plt.scatter(Y_pred, Y_train)
plt.plot([Yall.min(), Yall.max()], [Yall.min(), Yall.max()], 'k--', lw=2)
# Show the plot
plt.show()
#affichage du moindre carré
print("--------------Affichge PLOT Methode Gradien Descent-------------")
plt.scatter(predictions, Y_train)
# Add labels to the axes
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.plot([Yall.min(), Yall.max()], [Yall.min(), Yall.max()], 'k--', lw=2)
# Show the plot
plt.show()
print(Y_test.shape)
print(Y_pred.shape)




