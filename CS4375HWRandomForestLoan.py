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
sns.lmplot(x = 'fico', y = 'int.rate', col = 'not.fully.paid', hue = 'credit.policy', data = loan)


# In[10]:


loan_final = pd.get_dummies(loan, columns = ['purpose'], drop_first =True)


# In[11]:


X = loan_final.drop('not.fully.paid', axis =1)
y = loan_final['not.fully.paid']


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)


# In[14]:


from sklearn.ensemble import RandomForestClassifier


# In[15]:


forest = RandomForestClassifier(n_estimators=10, max_features='auto',random_state=101)


# In[16]:


forest.fit(X_train, y_train)


# In[17]:


preds = forest.predict(X_test)


# In[18]:


from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score


# In[19]:


preds


# In[20]:


confusion_matrix(y_test,preds)


# In[21]:


classification_report(y_test,preds)


# In[22]:


plot_confusion_matrix(forest, X_test,y_test)


# In[23]:


accuracy_score(y_test,preds)


# In[ ]:




