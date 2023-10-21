#!/usr/bin/env python
# coding: utf-8

# # CASE STUDY : PREDICTING AVOCADO PRICES USING FACEBOOK PROPHET 
# 
# 

# ## PROBLEM STATEMENT

# - Data represents weekly 2018 retail scan data for National retail volume (units) and price. 
# - Retail scan data comes directly from retailers’ cash registers based on actual retail sales of Hass avocados. 
# - The Average Price (of avocados) in the table reflects a per unit (per avocado) cost, even when multiple units (avocados) are sold in bags. 
# - The Product Lookup codes (PLU’s) in the table are only for Hass avocados. Other varieties of avocados (e.g. greenskins) are not included in this table.
# 
# Some relevant columns in the dataset:
# 
# - Date - The date of the observation
# - AveragePrice - the average price of a single avocado
# - type - conventional or organic
# - year - the year
# - Region - the city or region of the observation
# - Total Volume - Total number of avocados sold
# - 4046 - Total number of avocados with PLU 4046 sold
# - 4225 - Total number of avocados with PLU 4225 sold
# - 4770 - Total number of avocados with PLU 4770 sold
# 
# 

# ![image.png](attachment:image.png)
# Image Source: https://www.flickr.com/photos/30478819@N08/33063122713

# ## IMPORTING DATA AND LIBRARIES

# In[31]:


import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import seaborn as sns

import random
from prophet import Prophet

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


avocado_df = pd.read_csv('avocado.csv')


# ## EXPLORING THE DATASET  

# In[33]:


avocado_df.head()


# In[34]:


avocado_df = avocado_df.sort_values("Date")


# In[35]:


plt.figure(figsize=(10,10))
plt.plot(avocado_df['Date'], avocado_df['AveragePrice'])


# In[36]:


avocado_df


# In[37]:


# Bar Chart to indicate the number of regions 
plt.figure(figsize=[20,12])
sns.countplot(x = 'region', data = avocado_df)

plt.title('Number Of Regions', fontsize=20)
plt.xlabel('REGION', fontsize=20)
plt.ylabel('COUNT', fontsize=20)
plt.xticks(rotation = 90)
plt.show()


# In[38]:


# Bar Chart to indicate the year
plt.figure(figsize=[25,12])
sns.countplot(x = 'year', data = avocado_df)
plt.xlabel('YEAR', fontsize=20)
plt.ylabel('COUNT', fontsize=20)
plt.xticks(rotation = 0)
plt.show()


# In[39]:


avocado_prophet_df = avocado_df[['Date', 'AveragePrice']] 


# In[40]:


avocado_prophet_df


# ## DATA PREP FOR PROPHET

# In[41]:


avocado_prophet_df = avocado_prophet_df.rename(columns={'Date':'ds', 'AveragePrice':'y'})


# In[42]:


avocado_prophet_df


# ## CREATING PROPHET

# In[43]:


m = Prophet()
m.fit(avocado_prophet_df)


# ## FORECASTING INTO FUTURE

# In[44]:


future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[45]:


forecast


# In[46]:


figure = m.plot(forecast, xlabel='Date', ylabel='Price')


# In[47]:


figure3 = m.plot_components(forecast)


# ## IN A REGION

# In[48]:


avocado_df = pd.read_csv('avocado.csv')


# In[49]:


avocado_df


# In[50]:


avocado_df_sample = avocado_df[avocado_df['region']=='West']


# In[51]:


avocado_df_sample


# In[52]:


avocado_df_sample


# In[53]:


avocado_df_sample = avocado_df_sample.sort_values("Date")


# In[54]:


avocado_df_sample


# In[55]:


plt.figure(figsize=(10,10))
plt.plot(avocado_df_sample['Date'], avocado_df_sample['AveragePrice'])


# In[56]:


avocado_df_sample = avocado_df_sample.rename(columns={'Date':'ds', 'AveragePrice':'y'})


# In[57]:


m = Prophet()
m.fit(avocado_df_sample)
# Forcasting into the future
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[58]:


figure = m.plot(forecast, xlabel='Date', ylabel='Price')


# In[59]:


figure3 = m.plot_components(forecast)

