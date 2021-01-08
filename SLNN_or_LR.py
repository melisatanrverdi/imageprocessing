#!/usr/bin/env python
# coding: utf-8

# In[32]:


from sklearn.datasets import fetch_openml
dataset = fetch_openml("mnist_784")


# In[33]:


type(dataset), type(dataset["data"]), type(dataset["data"])


# In[34]:


X, y = dataset["data"], dataset["target"]


# In[35]:


X = X / 255.0


# In[36]:


X.shape


# In[37]:


y.shape


# In[38]:


img_1 = X[1,:].reshape(28, 28)


# In[39]:


import matplotlib.pyplot as plt


# In[40]:


plt.imshow(img_1, cmap = 'gray')
plt.show()


# In[41]:


y[1]


# In[42]:


i = 10010
img_1 = X[i,:].reshape(28, 28)
plt.imshow(img_1, cmap = 'gray')
plt.title(y[i])
plt.show()


# In[43]:


import numpy as np
y_new = np.zeros(y.shape)
y_new[np.where(y == 0.0)[0]] = 1
y = y_new


# In[46]:


m = 60000
m_test = X.shape[0] - m
    
X_train, X_test = X[:m].T, X[m:].T
y_train, y_test = y[:m].reshape(1,m), y[m:].reshape(1,m_test)


# In[48]:


np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib

i = 3
plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
print(y_train[:,i])


# In[51]:


epsilon = 1e-5
def sigmoid(z):
    s = 1 / (1 + np.exp(-z)
    return s


# In[55]:


def compute_loss(Y, Y_hat):
    
    m = Y.shape[1]
    L = -(1./m) * ( np.sum( np.multiply(np.log(Y_hat + epsilon), Y) ) + np.sum( np.multiply(np.log(1-Y_hat + epsilon),(1-Y)) ) )
    return L


# In[66]:


learning_rate = 1
X = X_train
Y = y_train

n_x = X.shape[0]
m = X.shape[1]

W = np.random.rand(n_x, 1) * 0.01
b = np.zeros((1, 1))

for i in range(2000):
    Z = np.matmul(W.T, X) + b
    A = sigmoid(Z)
    
    cost = compute_loss(Y, A)
    
    dW = (1/m) * np.matmul(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y, axis=1, keepdims=True)
    
    W = W - learning_rate * dW
    b = b - learning_rate *db
    
    if (i % 100 == 0):
        print("Epoch", i, "cost: ", cost)
    print("Final cost:", cost)


# In[69]:


y_test.ndim, y_test.shape


# In[70]:


y_test[0,100] = 1.0


# In[71]:


from sklearn.metrics import classification_report, confusion_matrix

Z = np.matmul(W.T, X_test) + b
A = sigmoid(Z)

predictions = (A>.5)[0,:]
labels = (y_test == 1)[0,:]

print(confusion_matrix(predictions, labels))


# In[72]:


print(classification_report(predictions, labels))


# In[ ]:




