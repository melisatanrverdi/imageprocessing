#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
class Perceptron(object):
    def __init__(self, input_size, lr = 1, epochs = 10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                y = self.predict(X[i])
                e = d[i] - y
                self.W = self.W + self.lr * e * np.insert(X[i], 0, 1)
    


# In[52]:


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


# In[53]:


mp = Perceptron(5)
x = np.asarray([-10,-2,-30,4,-50])
mp.predict(x)


# In[54]:


d = np.array([0, 0, 0, 1])


# In[55]:


mp.activation_fn(-10)


# In[56]:


X


# In[57]:


d


# In[58]:


perceptron = Perceptron(input_size = 2)


# In[59]:


perceptron.W


# In[60]:


perceptron.fit(X, d)


# In[63]:


perceptron.W


# In[64]:


print(perceptron.W)


# In[69]:


perceptron.predict(np.asarray([1, 1]))


# In[61]:


mp.W


# In[62]:


np.insert(mp.W, 0, 100)


# In[ ]:


##1.activation_fn; x değeri 0'dan büyük eşitse 1, 0'dan küçükse 0 değeri döndüren bir fonksiyondur.Predict fonksiyonu gelen verinin(x) başına bir tane 1 koyar b0 olarak.Sonra z yi üretir, a'da z'den activation_fn'la ürettiği 1 veya 0 değerleridir.Fit fonksiyonu x ve d'yi alarak W'yi ve error'u belirler yani W'leri günceller.
##2.XOR'un böyle bir yeteneği yoktur.
##3.X yani input mxnx3 bir veri olur.40 öğrenci için imza her bir sütununda bir kişiye ait imza resmi olmak üzere (m*n*3,40)'lık bir matris bizim verimiz olacaktır.Koddaki d'ye karşılık yani çıktı için 40 farklı değer olacaktır.
##4.error = y' - y şeklindedir.Yani o anki değer ile gerçek değer arasındaki fark bulunarak hesaplanır.

