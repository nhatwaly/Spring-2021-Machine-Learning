#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('C:\\Users\\nhatw\\Google Drive\\UTD\\Jupyter')
path = os.getcwd()


# In[2]:


iris = pd.read_csv('iris.csv')


# In[3]:


iris.head()


# In[4]:


iris.info()


# In[5]:


iris.describe()


# In[6]:


plt.figure(figsize=(16,9))
sns.countplot(data=iris,x='sepal_length',hue='species')


# In[7]:


plt.figure(figsize=(16,9))
sns.countplot(data=iris,x='sepal_width',hue='species')


# In[8]:


plt.figure(figsize=(16,9))
sns.countplot(data=iris,x='petal_length',hue='species')


# In[9]:


plt.figure(figsize=(16,9))
sns.countplot(data=iris,x='petal_width',hue='species')


# In[10]:


plt.figure(figsize=(16,9))
sns.jointplot(data=iris, x="sepal_length", y="sepal_width", hue="species", kind="kde")


# In[11]:


plt.figure(figsize=(16,9))
sns.jointplot(data=iris, x="petal_length", y="petal_width", hue="species", kind="kde")


# In[12]:


X = iris.drop('species', axis =1)
y = iris['species']


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)


# In[15]:


from sklearn.ensemble import RandomForestClassifier


# In[16]:


forest = RandomForestClassifier(n_estimators=10, max_features='auto',random_state=101)


# In[17]:


forest.fit(X_train, y_train)


# In[18]:


preds = forest.predict(X_test)


# In[19]:


from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score


# In[20]:


preds


# In[21]:


confusion_matrix(y_test,preds)


# In[22]:


classification_report(y_test,preds)


# In[23]:


plot_confusion_matrix(forest, X_test,y_test)


# In[24]:


accuracy_score(y_test,preds)


# In[25]:


print('This is the only data set which Random Forest has worse accuracy than Decision Tree')


# In[ ]:




