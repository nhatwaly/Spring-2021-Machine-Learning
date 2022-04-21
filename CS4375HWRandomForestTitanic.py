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


titanic = pd.read_csv('titanic.csv')


# In[3]:


titanic.head()


# In[4]:


titanic.info()


# In[5]:


titanic.describe()


# In[6]:


plt.figure(figsize=(16,9))
sns.displot(x='Age',hue='Sex',data =titanic,kde=True)


# In[7]:


plt.figure(figsize=(20,9))
sns.displot(x='Fare',hue='Pclass',data =titanic,palette='bright',hue_order=[1,2,3])


# In[8]:


plt.figure(figsize=(16,9))
sns.countplot(data=titanic, x='Pclass')


# In[9]:


plt.figure(figsize=(16,9))
sns.displot(data=titanic, x='SibSp',y='Parch',binwidth=(1, 1))


# In[10]:


plt.figure(figsize=(16,9))
sns.displot(data=titanic, x='Age',y='Fare')


# In[11]:


titanic_final = titanic.drop(['Name','Ticket','Cabin'], axis =1)


# In[12]:


titanic_final = pd.get_dummies(titanic_final, columns = ['Sex','Embarked'], drop_first =False)


# In[13]:


titanic_final.fillna(titanic_final.mean(), inplace=True)


# In[14]:


titanic_final.describe()


# In[15]:


X = titanic_final.drop('Survived', axis =1)
y = titanic_final['Survived']


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)


# In[18]:


from sklearn.ensemble import RandomForestClassifier


# In[19]:


forest = RandomForestClassifier(n_estimators=10, max_features='auto',random_state=101)


# In[20]:


forest.fit(X_train, y_train)


# In[21]:


preds = forest.predict(X_test)


# In[22]:


from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score


# In[23]:


preds


# In[24]:


confusion_matrix(y_test,preds)


# In[25]:


classification_report(y_test,preds)


# In[26]:


plot_confusion_matrix(forest, X_test,y_test)


# In[27]:


accuracy_score(y_test,preds)


# In[ ]:




