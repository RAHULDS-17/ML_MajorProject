#!/usr/bin/env python
# coding: utf-8

# IPL SCORE PREDICTION

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('ipl2017.csv') 


# In[3]:


df.head()


# In[4]:


df.isnull().values.any()


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


df.shape


# 

# 

# In[8]:


df=df.drop(["mid" , "date" , "venue" , "batsman" , "bowler" , "striker" , "non-striker"] , axis=1 )


# In[9]:


df.columns


# 

# In[10]:


df['bat_team'].unique()


# In[11]:


consistent_teams = ['Kolkata Knight Riders' , 'Chennai Super Kings' , 'Rajasthan Royals' , 
                    'Mumbai Indians' , 'Kings XI Punjab' , 'Royal Challengers Bangalore' ,
                    'Delhi Daredevils' , 'Sunrisers Hyderabad']


# In[12]:



print('Before removing inconsistent teams: {}'.format(df.shape))
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
print('After removing inconsistent teams: {}'.format(df.shape))


# In[13]:


df.head()


# In[ ]:





# In[ ]:





# In[16]:


df.replace(['Kolkata Knight Riders' , 'Chennai Super Kings', 'Rajasthan Royals', 'Mumbai Indians', 
            'Kings XI Punjab', 'Royal Challengers Bangalore', 'Delhi Daredevils', 'Sunrisers Hyderabad']
            ,['KKR','CSK','RR','MI','KXIP','RCB','DD','SRH'],inplace=True)


# In[17]:


encode = {'bat_team': {'KKR':1,'CSK':2,'RR':3,'MI':4,'KXIP':5,'RCB':6,'DD':7,'SRH':8},
          'bowl_team': {'KKR':1,'CSK':2,'RR':3,'MI':4,'KXIP':5,'RCB':6,'DD':7,'SRH':8}}
df.replace(encode, inplace=True)


# In[18]:


df.head()


# 

# In[19]:


y=df["total"]


# In[20]:


X=df.drop("total" , axis=1)


# In[21]:


X.head()


# In[22]:


type(X)


# In[23]:


X.shape


# 

# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)


# 

# In[26]:


from sklearn.preprocessing import StandardScaler


# In[27]:


scaler = StandardScaler()


# In[28]:


X_train = scaler.fit_transform(X_train)


# In[29]:


X_test = scaler.transform(X_test)


# 

# In[30]:


from sklearn.ensemble import RandomForestRegressor


# In[31]:


model = RandomForestRegressor(n_estimators=100, max_features=None)


# In[32]:


model.fit(X_train,y_train)


# In[33]:


score = model.score(X_test,y_test)


# In[34]:


score


# 

# In[35]:


df_new = {"bat_team":[6,1,4,2], "bowl_team":[3,5,2,3], "runs":[90,56,110,129], "wickets":[2,2,4,8],
          "overs":[15.2,11.1,18.1,19.6], "runs_last_5":[19,12,36,48], "wickets_last_5":[0,1,0,1]}
df_new = pd.DataFrame(df_new)


# In[36]:


df_new.head()


# In[37]:


model.predict(df_new)


# In[ ]:




