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


titanic.info()


# In[6]:


titanic.head()


# In[7]:


titanic.describe()


# In[8]:


plt.figure(figsize=(16,9))
sns.displot(x='Age',hue='Sex',data =titanic,kde=True)


# In[9]:


plt.figure(figsize=(20,9))
sns.displot(x='Fare',hue='Pclass',data =titanic,palette='bright',hue_order=[1,2,3])


# In[10]:


plt.figure(figsize=(16,9))
sns.countplot(data=titanic, x='Pclass')


# In[11]:


plt.figure(figsize=(16,9))
sns.displot(data=titanic, x='SibSp',y='Parch',binwidth=(1, 1))


# In[12]:


plt.figure(figsize=(16,9))
sns.displot(data=titanic, x='Age',y='Fare')


# In[13]:


titanic_final = titanic.drop(['Name','Ticket','Cabin'], axis =1)


# In[14]:


titanic_final = pd.get_dummies(titanic_final, columns = ['Sex','Embarked'], drop_first =False)


# In[15]:


titanic_final.head()


# In[16]:


titanic_final.describe()


# In[17]:


titanic_final.fillna(titanic_final.mean(), inplace=True)


# In[24]:


titanic_final.describe()


# In[18]:


X = titanic_final.drop('Survived', axis =1)
y = titanic_final['Survived']


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)


# In[21]:


from sklearn.tree import DecisionTreeClassifier


# In[22]:


tree = DecisionTreeClassifier()


# In[25]:


tree_model = tree.fit(X_train, y_train)


# In[26]:


preds = tree.predict(X_test)


# In[27]:


X.head()


# In[28]:


y.head()


# In[29]:


from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score


# In[30]:


preds


# In[31]:


confusion_matrix(y_test,preds)


# In[32]:


classification_report(y_test,preds)


# In[33]:


plot_confusion_matrix(tree, X_test,y_test)


# In[34]:


accuracy_score(y_test,preds)


# In[36]:


from sklearn.tree import export_text

exp = export_text(tree,feature_names=list(X.columns.values))


# In[37]:


print(exp)


# In[ ]:




