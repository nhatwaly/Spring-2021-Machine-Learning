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


loan= pd.read_csv('loan_data.csv')


# In[3]:


loan.info()


# In[4]:


loan = loan.dropna()


# In[5]:


loan.info()


# In[6]:


loan.describe()


# In[7]:


plt.figure(figsize = (16,9))
sns.displot(x=loan['fico'],y=loan['days.with.cr.line'],hue=loan['credit.policy'])


# In[8]:


plt.figure(figsize=(16,9))
sns.countplot(x='credit.policy',hue='not.fully.paid',data =loan)


# In[9]:


plt.figure(figsize=(16,9))
sns.countplot(x='purpose',hue='not.fully.paid',data =loan)


# In[10]:


plt.figure(figsize=(16,9))
sns.jointplot(x='fico',y='int.rate',data=loan)


# In[11]:


plt.figure(figsize=(16,9))
sns.lmplot(x = 'fico', y = 'int.rate', col = 'not.fully.paid', hue = 'credit.policy', data = loan)


# In[12]:


loan_final = pd.get_dummies(loan, columns = ['purpose'], drop_first =True)


# In[13]:


loan_final.head()


# In[14]:


X = loan_final.drop('not.fully.paid', axis =1)
y = loan_final['not.fully.paid']


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)


# In[17]:


from sklearn.tree import DecisionTreeClassifier


# In[18]:


tree = DecisionTreeClassifier()


# In[19]:


tree.fit(X_train, y_train)


# In[20]:


preds = tree.predict(X_test)


# In[21]:


X.head()


# In[22]:


y.head()


# In[23]:


from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score


# In[24]:


preds


# In[25]:


confusion_matrix(y_test,preds)


# In[26]:


classification_report(y_test,preds)


# In[27]:


plot_confusion_matrix(tree, X_test,y_test)


# In[28]:


accuracy_score(y_test,preds)


# In[42]:


from sklearn.tree import export_text

exp = export_text(tree,feature_names=list(X.columns.values))


# In[43]:


print(exp)


# In[ ]:




