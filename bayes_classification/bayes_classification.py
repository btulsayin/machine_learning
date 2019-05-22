# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:42:30 2019

@author: betul
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

def dataset_detail(dataset):
    #Veri kümesinin ilk 5 değerine bakma
    dataset.head()

def img_show():
    img=mpimg.imread('iris_types.jpg')
    plt.figure(figsize=(20,40))
    plt.axis('off')
    plt.imshow(img)

def find_accuracy(cm):
    #confusion matrisinden doğruluk bulma
    a = cm.shape
    corrPred = 0
    falsePred = 0
    
    for row in range(a[0]):
        for c in range(a[1]):
            if row == c:
                corrPred +=cm[row,c]
            else:
                falsePred += cm[row,c]
    print('\nDoğru tahminler: ', corrPred)
    print('\nYanlış tahminler', falsePred)
    print ('\nNaive Bayes Sınıflandırmasının Doğruluğu: ', corrPred/(cm.sum()))
    
def main(): 
    #Veri kümesini bağımsız ve bağımlı değişkenlere bölme
    X = dataset.iloc[:,:4].values
    y = dataset['species'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 82)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    nvclassifier = GaussianNB()
    nvclassifier.fit(X_train, y_train)
    y_pred = nvclassifier.predict(X_test)
    print("\n**** Test set ****")
    print('\n',y_pred)
    print("\n*** top 5 values ***")
    #gercek ve tahmin edilen deger karsılastırma
    y_compare = np.vstack((y_test,y_pred)).T
    #sol=gercek deger , sag=tahmin edilen deger
    #ilk 5 deger
    y_compare[:5,:]
    print("\n",y_compare[:5,:])
    cm = confusion_matrix(y_test, y_pred)
    print("\n**** confusion_matrix ****")
    print("\n",cm)
    find_accuracy(cm)
    
if __name__ == '__main__':
    # Dataset okuma
    dataset = pd.read_csv('iris_dataset.csv')
    dataset_detail(dataset)
    img_show()
    main()   