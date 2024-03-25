#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')



# In[16]:


df=pd.read_csv('/Users/Yuvraj1/Desktop/Automobile.csv')


# In[17]:


df


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.describe(include='all')


# In[9]:


df.describe(include='all').T


# In[10]:


sns.histplot(data=df,x='price')


# In[11]:


plt.xlabel('Frequency')
plt.ylabel('Price of the cars')
plt.xlim(3000,50000)
plt.ylim(0,70)
plt.title('Histogram:Price')
sns.histplot(data=df,x='price', color='orange');


# In[12]:


sns.histplot(data=df,x='price',bins=5)


# In[13]:


sns.histplot(data=df,x='price',kde='true')


# In[14]:


sns.histplot(data=df,x='price',bins=100,kde='true')


# In[15]:


sns.histplot(data=df,x='price',hue='body_style',kde='true')


# In[16]:


compare= sns.FacetGrid(df,col='body_style')
compare.map (sns.histplot,'price')


# In[17]:


g = sns.FacetGrid(df,col='body_style')
g.map(sns.histplot,'price')


# In[18]:


sns.boxplot(data=df,x='curb_weight')


# In[19]:


plt.title('Boxplot:Curb_Weight')
plt.xlabel('Frequency')
plt.xlim(30,300)
sns.axes_style('whitegrid')
sns.boxplot(data=df,x='horsepower',color='green')


# In[20]:


sns.boxplot(data=df,x='body_style',y='price')


# In[21]:


plt.figure(figsize=(10,7))
sns.boxplot(data=df,x='body_style',y='price')


# In[22]:


sns.countplot(data=df,x='body_style')


# In[23]:


sns.countplot(data=df,x='body_style', hue='fuel_type')


# In[24]:


sns.countplot(data=df,x='make')


# In[25]:


plt.figure(figsize=(30,15))
sns.countplot(data=df,x='make')


# In[26]:


flights= sns.load_dataset('flights')

sns.lineplot(data=flights, x='month', y='passengers');


# In[27]:


sns.lineplot(data=flights, x='month', y='passengers', ci=False);


# In[28]:


sns.lineplot(data=flights, x='month', y='passengers', ci=False, hue='year');


# In[30]:


sns.scatterplot(data=df,x='engine_size',y='horsepower')


# In[31]:


sns.scatterplot(data=df,x='engine_size',y='horsepower', hue='fuel_type')


# In[32]:


sns.scatterplot(data=df,x='engine_size',y='horsepower', hue='fuel_type',style='fuel_type')


# In[33]:


sns.lmplot(data=df,x='curb_weight',y='horsepower')


# In[34]:


sns.lmplot(data=df,x='curb_weight',y='horsepower',hue='fuel_type')


# In[37]:


sns.lmplot(data=df,x='curb_weight',y='horsepower',hue='fuel_type', ci=False)


# In[38]:


sns.lmplot(data=df,x='curb_weight',y='horsepower',col='fuel_type')


# In[39]:


sns.jointplot(data=df,x='engine_size',y='horsepower')


# In[43]:


sns.jointplot(data=df,x='engine_size',y='horsepower',kind='hex')
plt.colorbar()


# In[45]:


sns.jointplot(data=df,x='engine_size',y='horsepower',kind='kde',fill=True)


# In[46]:


sns.jointplot(data=df,x='engine_size',y='horsepower',kind='reg')


# In[47]:


sns.violinplot(data=df,x='horsepower')


# In[49]:


sns.violinplot(data=df,x='fuel_type', y='horsepower',orient='v')


# In[50]:


sns.stripplot(data=df,x='engine_size')


# In[51]:


sns.stripplot(data=df,x='body_style',y='engine_size')


# In[54]:


sns.stripplot(data=df,x='body_style',y='engine_size',jitter=True)
plt.figure(figsize=(20,7))


# In[55]:


sns.stripplot(data=df,x='body_style',y='engine_size',hue='number_of_doors',jitter=True)


# In[57]:


sns.swarmplot(data=df,x='number_of_doors',y='price')


# In[59]:


sns.swarmplot(data=df,x='fuel_type',y='price',hue='number_of_doors')


# In[60]:


sns.swarmplot(data=df,x='fuel_type',y='price',hue='number_of_doors',dodge=True)


# In[7]:


sns.catplot(data=df,x='fuel_type',y='horsepower')


# In[8]:


sns.catplot(data=df,x='fuel_type',y='horsepower', hue='fuel_type',kind='point')


# In[9]:


sns.catplot(data=df,x='body_style',y='horsepower', hue='fuel_type',kind='point')


# In[10]:


sns.catplot(data=df,x='body_style',y='horsepower',col='drive_wheels',kind='bar',palette='pastel')


# In[11]:


sns.pairplot(data=df [['normalized_losses','wheel_base','curb_weight','engine_size','price','peak_rpm']])


# In[12]:


sns.pairplot (data=df,vars=['wheel_base','curb_weight','engine_size','price'],hue='number_of_doors')


# In[13]:


sns.pairplot (data=df,vars=['wheel_base','curb_weight','engine_size','price'],corner=True)


# In[20]:


sns.heatmap(data=df[['engine_size','wheel_base','curb_weight','price']].corr())


# In[23]:


sns.heatmap(data=df[['engine_size','wheel_base','curb_weight','price']].corr(), annot=True)


# In[24]:


sns.heatmap(data=df[['engine_size','wheel_base','curb_weight','price']].corr(), annot=True, cbar=False)


# In[27]:


sns.heatmap(data=df[['engine_size','wheel_base','curb_weight','price']].corr(), annot=True, cmap='YlGnBu')


# In[28]:


import plotly.express as px


# In[29]:


his = px.histogram(df,x='price')


# In[30]:


his.show()


# In[34]:


bar = px.bar (df,x='peak_rpm',y='horsepower')


# In[35]:


bar.show()


# In[36]:


scat = px.scatter(df,x='price',y='horsepower')
scat.show()


# In[37]:


fig_3d = px.scatter_3d (df, x='fuel_type',y='horsepower',z='price',color='horsepower')
fig_3d.show()


# In[ ]:




