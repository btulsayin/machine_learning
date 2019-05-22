# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:16:05 2019

@author: betul 
"""

import numpy as np
import matplotlib.pyplot as plt

def veriseti():
    #Rastgele bir veri kümesi hazırlama
    X = 2 * np.random.rand(100,1)
    y = 4 +3 * X+np.random.randn(100,1)
    return X,y

#eğitim verisinin görselleştirilmesi
def verisetigrapics(X,y):
    plt.plot(X,y,'b.')
    plt.xlabel("$x$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    _ =plt.axis([0,2,0,15])
    
def cal_cost(theta,X,y):
    ### Maliyet hesaplama
    ## Verilen X ve Y için maliyeti hesaplar. 
    ## theta = thetas vektörü
    ## X = X'in np.zeros satırı ((2, j))
    ## y = Gerçek y'nin np.zeros ((2,1))
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost    


def gradient_descent(X,y,theta,learning_rate=0.01,iterations=100):
    #### gradient_descent ile theta guncelle ####
    # X = X Matrisi
    # y = Y'nin Vektörü
    # theta = thetas np.random.randn vektörü (j, 1) 
    # iterations = yineleme sayısı
    # Son theta vektörünü ve yineleme sayıları üzerindeki maliyet geçmişi dizisini döndürür
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    for it in range(iterations):
        
        prediction = np.dot(X,theta)
        
        theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
        theta_history[it,:] =theta.T
        cost_history[it]  = cal_cost(theta,X,y)
        
    return theta, cost_history, theta_history



def stocashtic_gradient_descent(X,y,theta,learning_rate=0.01,iterations=10):
    ##### stocashtic_gradient_descent (SGD) ile theta degerlerinin guncellenmesi #####
    # X = X Matrisi
    # y = Y'nin Vektörü
    # theta = thetas np.random.randn vektörü (j, 1) 
    # iterations = yineleme sayısı
    # Son theta vektörünü ve yineleme sayıları üzerindeki maliyet geçmişi dizisini döndürür
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    
    for it in range(iterations):
        cost =0.0
        for i in range(m):
            rand_ind = np.random.randint(0,m)
            X_i = X[rand_ind,:].reshape(1,X.shape[1])
            y_i = y[rand_ind].reshape(1,1)
            prediction = np.dot(X_i,theta)

            theta = theta -(1/m)*learning_rate*( X_i.T.dot((prediction - y_i)))
            cost += cal_cost(theta,X_i,y_i)
        cost_history[it]  = cost
        theta_history[it,:] =theta.T
        
    return theta, cost_history, theta_history

# X = X Matrisi
# y = Y'nin Vektörü
# theta = thetas np.random.randn vektörü (j, 1) 
# iterations = yineleme sayısı
# Son theta vektörünü ve yineleme sayıları üzerindeki maliyet geçmişi dizisini döndürür

def momentum(X,y,theta,learning_rate=0.01,iterations=10, momentum=0.9):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    v = np.zeros_like(theta)
    
    for it in range(iterations):
        cost =0.0
        for i in range(m):
            rand_ind = np.random.randint(0,m)
            X_i = X[rand_ind,:].reshape(1,X.shape[1])
            y_i = y[rand_ind].reshape(1,1)
            prediction = np.dot(X_i,theta)
            
            v = momentum*v - (1/m)*learning_rate*( X_i.T.dot((prediction - y_i)))
            theta = theta + v
            cost += cal_cost(theta,X_i,y_i)
        cost_history[it]  = cost
        theta_history[it,:] =theta.T
        
    return theta, cost_history, theta_history


# n_iter = yineleme sayısı
# lr = Öğrenme Oranı
# ax = Gradyan Descent çizmek için eksen

def plot_GD_SGD_MOMENT(n_iter,lr,ax,ax1=None, mod="gd"):
  
      _ = ax.plot(X, y,'b.')
      theta = np.random.randn(2,1)
      
      tr =0.1
      cost_history = np.zeros(n_iter)
      
      #print(X_b,y,theta,lr,1)
      #gelen mod değerlerine göre çizilecek eğri rengi ayarlama
      color = 'r-'
      if mod == "sgd":
        color = 'g-'
      elif mod == "moment":
        color = 'o-'
        
      for i in range(n_iter):
        pred_prev = X_b.dot(theta)
        #gelen modlara göre metodları çağırarak işlem yapıldı.
        if mod == "gd":
          theta,h,_ = gradient_descent(X_b,y,theta,lr,1)
        elif mod == "sgd":
          #X,y,theta,learning_rate=0.01,iterations=10
          theta,h,_ = stocashtic_gradient_descent(X_b,y,theta,lr,1)      
        elif mod == "moment":
          theta,h,_ = momentum(X_b,y,theta,lr,1)
    
        pred = X_b.dot(theta)
    
        cost_history[i] = h[0]
    
        if ((i % 25 == 0) ):
            _ = ax.plot(X,pred,color,alpha=tr, label=mod)
            ax.legend(loc='best')
            if tr < 0.8:
                tr = tr+0.2
      if not ax1== None:
        _ = ax1.plot(range(n_iter),cost_history,'b.')
        
def stocashtic_gradient_descent_cost():
    # stocashtic gradient descent için
    lr =0.5
    n_iter = 50
    
    theta = np.random.randn(2,1)
    
    X_b = np.c_[np.ones((len(X),1)),X]
    theta,cost_history = stocashtic_gradient_descent(X_b,y,theta,lr,n_iter)
    
    
    print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
    print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))

def momentum_cost():
    #momentum için
    lr =0.5
    n_iter = 50
    
    theta = np.random.randn(2,1)
    theta_m,cost_history_m = momentum(X_b,y,theta,lr,n_iter)
    
    
    print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta_m[0][0],theta_m[1][0]))
    print('Final cost/MSE:  {:0.3f}'.format(cost_history_m[-1]))        
    
def draw_grafics():
    _,ax = plt.subplots(figsize=(14,10))
    plot_GD_SGD_MOMENT(100,0.1,ax, mod="gd") # iterasyon = 100 - lr = 0.1 - red
    plot_GD_SGD_MOMENT(100,0.1,ax, mod="sgd") # iterasyon = 100 - lr = 0.1 - green
    plot_GD_SGD_MOMENT(100,0.1,ax, mod="moment") # iterasyon = 100 - lr = 0.1 - orange  
    
    
if __name__== "__main__":
    plt.style.use(['ggplot'])
    X,y=veriseti()
    verisetigrapics(X,y)
    cal_cost(X,y)
    theta, cost_history, theta_history=gradient_descent(X,y)
    theta, cost_history, theta_history=stocashtic_gradient_descent(X,y)
    momentum(X,y,theta)
    plot_GD_SGD_MOMENT()
    stocashtic_gradient_descent_cost()
    momentum_cost()
    draw_grafics()