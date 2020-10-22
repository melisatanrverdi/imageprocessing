#!/usr/bin/env python
# coding: utf-8

# In[139]:


path = os.getcwd()
jpg_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
jpg_files


# In[140]:


import os
import numpy as np
import matplotlib.pyplot as plt


# In[141]:


im_1 = plt.imread("OneDrive\Masaüstü\canakkale-web-tasarim.jpg")
im_1.shape
im_1_gray = convert_rgb_to_gray(im_1)
im_1_bw = convert_rgb_to_bw(im_1)


# In[142]:


im_1.shape


# In[143]:


im_1[0,0,0] #im_1[row,col,rgb] im_1[10,30] = 45 


# In[144]:


temp_1 = im_1[0,0,:]
int(temp_1[0]/3+temp_1[1]/3+temp_1[2]/3) #(temp_1[0]+temp_1[1]+temp_1[2])/3


# In[145]:


def get_value_from_triple(temp_1):
   # temp_1 = im_1[0,0,:]
    return int(temp_1[0]/3+temp_1[1]/3+temp_1[2]/3) #(temp_1[0]+temp_1[1]+temp_1[2])/3

def get_0_1_from_triple(temp_1):
   # temp_1 = im_1[0,0,:]
   temp = int(temp_1[0]/3 +temp_1[1]/3+ temp_1[2]/3)
   if temp<110:
        return 0
   else:
        return 1
get_value_from_triple(im_1[10,10,:])


# In[146]:


def convert_rgb_to_bw(im_1):
    m,n,k = im_1.shape
    new_image=np.zeros((m,n), dtype='uint8')
    for i in range(m):
        for j in range(n):
            s = get_0_1_from_triple(im_1[i,j,:])
            new_image[i,j]=s
    return new_image              


# In[147]:


def convert_rgb_to_gray(im_1): #RGB(R 0-255,G,B),gray(0,.......255), black white(0,1)
    m, n, k = im_1.shape
    new_image=np.zeros((m,n), dtype='uint8')
    for i in range(m):
        for j in range(n):
            s = get_value_from_triple(im_1[i,j,:])
            new_image[i,j]=s
    return new_image        


# In[148]:


im_1_gray = convert_rgb_to_gray(im_1)
im_1_bw = convert_rgb_to_bw(im_1)


# In[149]:


plt.subplot(1,2,1)
plt.imshow(im_1_gray, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(im_1_bw, cmap='gray')
plt.show()


# In[150]:


plt.imsave('canakkale_gray.jpg',im_2,cmap='gray')


# In[151]:


plt.imsave('canakkale_bw.jpg',im_1,cmap='gray')

