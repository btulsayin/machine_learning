# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 08:13:31 2019

@author: betul
"""
#Lineer Regresyon İle Ofis Alanına Göre Kira Hesaplama

import numpy as np
import matplotlib.pyplot as plt

def tahmin_katsayilari(x,y):
    n=np.size(x)
    ortalama_x, ortalama_y=np.mean(x),np.mean(y)
    
    SS_xy=np.sum(y*x -n * ortalama_y * ortalama_x)
    SS_xx=np.sum(x*x -n * ortalama_x * ortalama_x)
    
    b_1=SS_xy/SS_xx
    b_0=ortalama_y-b_1*ortalama_x
    
    return(b_0,b_1)
    
def grafik(x,y,b):
    plt.scatter(x,y,color="m",marker="o",s=30)
    y_pred = b[0]+b[1]*x
    plt.plot(x, y_pred,color="g")
    plt.xlabel("Boyut")
    plt.ylabel("Maliyet")
    plt.show()

x=np.array([10,20,30,40,50,60,70,80,90,100])
y=np.array([300,350,500,700,800,850,900,900,1000,1200])    

b=tahmin_katsayilari(x,y)
print("tahmin: \nb_0= {} \nb_1= {}".format(b[0],b[1]))

grafik(x,y,b)
    
    
