#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[2]:


data = pd.read_csv("C:/Users/ROSHINI S/Downloads/Rent.csv")


# In[3]:


data


# In[4]:


data.sample()


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.shape


# In[8]:


data.isnull().sum()


# In[9]:


data.columns


# In[10]:


data['area'].unique()


# In[11]:


data['rent'].unique()


# In[12]:


data.info()


# In[13]:


x = data.drop("rent",axis=1)
y=data["rent"]


# In[14]:


x


# In[15]:


y.head()


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[17]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[18]:


# Linear regression Model
lr_model = LinearRegression()
lr_model.fit(x_train_scaled, y_train)


# In[19]:


lr_predictions = lr_model.predict(x_test_scaled)
lr_mse = mean_squared_error(y_test, lr_predictions)
print ("Linear Regression MSE:", lr_mse )


# In[21]:


# Neural Network Model
model = Sequential([
    Dense(64, activation = 'relu', input_shape = (x_train_scaled.shape[1],)),
    Dense(64, activation = 'relu'),
    Dense(1)
])


# In[23]:


model.compile(optimizer='adam', loss='mse')


# In[24]:


model.fit(x_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)


# In[25]:


#Evaluate Neural Network Model
nn_mse = model. evaluate(x_test_scaled, y_test, verbose=0)
print('Neural Network MSE:', nn_mse)


# In[32]:


# Example prediction
example_input = scaler.transform([[3,2,1200,1]])
prediction = model.predict(example_input)
print('Predicted rent price:',prediction)


# In[ ]:




