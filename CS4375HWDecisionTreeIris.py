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


# In[9]:


plt.figure(figsize=(16,9))
sns.displot(data=iris, x='sepal_length',y='sepal_width',hue='species')


# In[10]:


plt.figure(figsize=(16,9))
sns.displot(data=iris, x='petal_length',y='petal_width',hue='species')


# In[11]:


plt.figure(figsize=(16,9))
sns.countplot(data=iris,x='sepal_length',hue='species')


# In[12]:


plt.figure(figsize=(16,9))
sns.countplot(data=iris,x='sepal_width',hue='species')


# In[13]:


plt.figure(figsize=(16,9))
sns.countplot(data=iris,x='petal_length',hue='species')


# In[14]:


plt.figure(figsize=(16,9))
sns.countplot(data=iris,x='petal_width',hue='species')


# In[15]:


plt.figure(figsize=(16,9))
sns.jointplot(data=iris, x="sepal_length", y="sepal_width", hue="species", kind="kde")


# In[16]:


plt.figure(figsize=(16,9))
sns.jointplot(data=iris, x="petal_length", y="petal_width", hue="species", kind="kde")


# In[17]:


X = iris.drop('species', axis =1)
y = iris['species']


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)


# In[20]:


from sklearn.tree import DecisionTreeClassifier


# In[21]:


tree = DecisionTreeClassifier()


# In[22]:


tree_model = tree.fit(X_train, y_train)


# In[23]:


preds = tree.predict(X_test)


# In[24]:


X.head()


# In[25]:


y.head()


# In[26]:


from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score


# In[27]:


preds


# In[28]:


confusion_matrix(y_test,preds)


# In[29]:


classification_report(y_test,preds)


# In[30]:


plot_confusion_matrix(tree, X_test,y_test)


# In[31]:


accuracy_score(y_test,preds)


# In[32]:


from sklearn.tree import export_text

exp = export_text(tree,feature_names=list(X.columns.values))


# In[33]:


print(exp)


# In[ ]:




