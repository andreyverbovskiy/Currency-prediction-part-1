#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd


# In[10]:


money = pd.read_excel("DOLLAR.xlsx")


# In[11]:


money.describe()


# In[12]:


future = 7  # currecy on 7 future days
past = 28  # currency on previous 28 days


# In[13]:


import sklearn #for building models


# In[14]:


money


# In[16]:


values = money.curs


# In[17]:


start = past
end = len(values) - future


# In[ ]:





# In[22]:


raw_data = []
for i in range(start, end): #take values tat we can work with
    past_and_future_values = values[(i-past):(i+future)]#35 eleents: 28 past and 7 future for every row
    raw_data.append(list( past_and_future_values))


# In[23]:


raw_data


# In[24]:


#need to make a better loking table/ uniting past and future datasets


# In[26]:


past_columns = []
for i in range(past):
    past_columns.append("past_{}".format(i))

print(past_columns)


# In[27]:


future_columns = []
for i in range(future):
    future_columns.append("future_{}".format(i))

print(future_columns)


# In[28]:


#making a dataframe, setting the raws and columns


# In[29]:


df = pd.DataFrame(raw_data, columns=(past_columns + future_columns))


# In[33]:


df.head(15)


# In[34]:


df.shape #numer of rows and columns


# # Practice to make the model learn (studentbook for the model)
# we show this data for odel to learn it

# In[35]:


X = df[past_columns] [:-1]#This is what we use for predictions :-1 means taking everyting except the last data set


# In[36]:


Y = df[future_columns][:-1] #This is what we want to get     :-1 means taking everyting except the last data set


# In[37]:


Y


# # Testing (exam for the model)  
# here we ceck ow well it learned y showing new data tat it did not know

# In[39]:


X_test = df[past_columns][-1:]     #-1: means taking only the last data set
Y_test = df[future_columns][-1:]   #-1: means taking only the last data set


# In[40]:


from sklearn.linear_model import LinearRegression


# In[41]:


LinReg = LinearRegression()


# In[42]:


LinReg


# In[43]:


LinReg.fit(X,Y) #this command make model learn


# In[46]:


prediction = LinReg.predict(X_test)[0]


# In[49]:


prediction #prediction for 7 future days


# In[51]:


Y_test


# In[52]:


import matplotlib.pyplot as plt #now compare the result with actual data


# In[54]:


plt.plot(prediction, label = "Prediction")
plt.plot(Y_test.iloc[0], label = "Real") #as we need values, not headings from "o" we use "iloc"
plt.legend()


# In[57]:


from sklearn.metrics import mean_absolute_error


# In[59]:


mean_absolute_error(Y_test, [prediction]) #the avarage error


# In[64]:


from sklearn.neighbors import KNeighborsRegressor #states the type of object by its neighbors 


# In[85]:


KNN = KNeighborsRegressor(n_neighbors = 2) #choose te numer of neighors


# In[86]:


KNN.fit(X,Y)


# In[87]:


prediction = KNN.predict(X_test)[0]


# In[88]:


prediction


# In[89]:


mean_absolute_error(Y_test, [prediction])


# In[90]:


plt.plot(prediction, label = "Prediction")
plt.plot(Y_test.iloc[0], label = "Real") #as we need values, not headings from "o" we use "iloc"
plt.legend()


# In[104]:


from sklearn.neural_network import MLPRegressor #Algorithm on the basis of Artificial neural network 


# In[221]:


MLP = MLPRegressor(max_iter=800, hidden_layer_sizes=(100,100,100), random_state=42)


# In[222]:


MLP.fit(X,Y) #program tells us tat result is not sufficient enough and asks to change soe parameters


# In[223]:


prediction = MLP.predict(X_test)[0]


# In[224]:


mean_absolute_error(Y_test, [prediction])


# In[225]:


plt.plot(prediction, label = "Prediction")
plt.plot(Y_test.iloc[0], label = "Real") #as we need values, not headings from "o" we use "iloc"
plt.legend()


# In[ ]:





# In[ ]:




