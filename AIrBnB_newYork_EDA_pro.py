#!/usr/bin/env python
# coding: utf-8

# # Task 1. Importing all dependencies

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Task 2. Loading Datasets

# In[10]:


import os
print(os.getcwd())


# In[14]:


data = pd.read_csv('datasets.csv', encoding_errors= 'ignore')


# # Task 3. Initial Exploration

# In[15]:


data.head()


# In[16]:


data.tail()


# In[17]:


data.shape


# In[18]:


data.info()


# In[19]:


# Statistical summary
data.describe()


# # Task 4. Data Cleaning

# In[20]:


data.isnull().sum()


# In[24]:


# dropping all missing values rows
data.dropna(inplace=True)
# data.fillna()
data.isnull().sum()


# In[26]:


data.shape


# In[28]:


# dealing with duplicates rows
data.duplicated().sum()


# In[30]:


#deleting all duplicated rows
data[data.duplicated()]
data.drop_duplicates(inplace= True)


# In[31]:


data.duplicated().sum()


# In[35]:


# type casting
# changing data types
data.dtypes
data['id']=data['id'].astype(object)
data['host_id']=data['host_id'].astype(object)
data.dtypes


# # EDA
# # Task 5. Data Analysis
# # Univariate Analysis

# In[36]:


#price distribution
data['price']


# In[37]:


sns.histplot(data=data,x='price')


# In[39]:


#identifying outliers in price
df=data[data['price']<1500]
sns.boxplot(data=df,x='price')


# In[44]:


#price distribution using histogram
plt.figure(figsize=(8,5))
sns.histplot(data=df,x='price',bins= 100)
plt.title("Price Distribution")
plt.ylabel("Frequency")
plt.show()


# In[45]:


df.columns


# In[46]:


#availability_365 distribution using histogram
plt.figure(figsize=(6,3))
sns.histplot(data=df,x='availability_365')
plt.title("availability_365 Distribution")
plt.ylabel("Frequency")
plt.show()


# In[47]:


data.dtypes


# In[48]:


df.groupby(by='neighbourhood_group')['price'].mean()


# # Feature Engineering

# In[51]:


# ['price per bed']
df['price per bed']=df['price']/df['beds']
df.head()


# In[52]:


#average price per bed
df.groupby(by='neighbourhood_group')['price per bed'].mean()


# # Bi variate analysis: one variable dependent on another variable

# In[53]:


df.columns


# In[54]:


#price dependency on neighbourhood
sns.barplot(data=df, x='neighbourhood_group',y='price',hue='room_type')


# In[58]:


#number of reviews and price rel
plt.figure(figsize=(8,5))
plt.title("Locality and Review Dependency")
sns.scatterplot(data=df, x='number_of_reviews',y='price',hue='neighbourhood_group')
plt.show()


# In[59]:


df.dtypes


# In[61]:


#pairplot
sns.pairplot(data=df,vars=['price','minimum_nights','number_of_reviews','availability_365'],hue='room_type')


# In[65]:


#Geographical representation of airbnb listing
plt.figure(figsize=(10,7))
sns.scatterplot(data=df,x='longitude',y='latitude',hue='room_type')
plt.title('Geographical distribution of airbnb listing')
plt.show()


# In[66]:


df.dtypes


# In[83]:


#heat map- correlation of one variable with others for numerical column
corr=df[['latitude','longitude','price','minimum_nights','number_of_reviews','reviews_per_month','availability_365','beds']].corr()
corr


# In[84]:


plt.figure(figsize=(8,6))
sns.heatmap(data=corr, annot= True)


# In[ ]:




