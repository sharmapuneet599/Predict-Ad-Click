
# coding: utf-8

# **Problem Statement**
# 
# A leading affiliate network company from Europe wants to leverage machine learning to improve (optimise) their conversion rates and eventually their topline. Their network is spread across multiple countries in europe such as Portugal, Germany, France, Austria, Switzerland etc.
# 
# Affiliate network is a form of online marketing channel where an intermediary promotes products / services and earns commission based on conversions (click or sign up). The benefit companies sees in using such affiliate channels is that, they are able to reach to audience which doesn’t exist in their marketing reach.
# 
# The company wants to improve their CPC (cost per click) performance. A future insight about an ad performance will give them enough headstart to make changes (if necessary) in their upcoming CPC campaigns.
# 
# In this challenge, you have to predict the probability whether an ad will get clicked or not.

# **Data Description**
# 
# You are given three files to download: train.csv, test.csv and sample_submission.csv Variables in this data set are anonymized due to privacy. 
# The training data is given for 10 days . The test data is given for next 3 days.
# 
# **Variable - Description**
# - ID - Unique ID
# - datetime - timestamp
# - siteid - website id
# - offerid - offer id (commission based offers)
# - category - offer category
# - merchant - seller ID
# - countrycode - country where affiliates reach is present
# - browserid - browser used
# - devid - device used
# - click - target variable

# **Evaluation Metric**
# 
# Submission will be evaluated based on AUC-ROC score. Higher the better.

# In[1]:


# Import libraries necessary for this project
import pandas as pd
import numpy as np

# Load the training data from train.csv 
data = pd.read_csv(r"C:\Users\sharm\Desktop\wtf\train.csv")

# Convert feature countrycode from string to float 
data["countrycode"] = data["countrycode"].astype('category')
data["countrycodenumber"] = data["countrycode"].cat.codes

# Convert feature browserid from string to float
data["browserid"] = data["browserid"].astype('category')
data["browserid"] = data["browserid"].cat.codes

# Convert feature devid from string to float
data["devid"] = data["devid"].astype('category')
data["devid"] = data["devid"].cat.codes

# Inserting the features for traning
features_train = data.drop(['click','datetime','ID','countrycode','browserid','devid','siteid'], axis = 1)

# Inserting the labels for training 
labels = data['click']


# In[2]:


# Load the test data from test.csv
test = pd.read_csv(r"C:\Users\sharm\Desktop\wtf\test.csv")

# Convert feature countrycode from string to float 
test["countrycode"] = test["countrycode"].astype('category')
test["countrycodenumber"] = test["countrycode"].cat.codes

# Convert feature browserid from string to float
test["browserid"] = test["browserid"].astype('category')
test["browserid"] = test["browserid"].cat.codes

# Convert feature devid from string to float
test["devid"] = test["devid"].astype('category')
test["devid"] = test["devid"].cat.codes

# Inserting the features for testing
features_test = test.drop(['datetime','ID','countrycode','browserid','devid','siteid'], axis = 1)


# In[3]:


# Import 'train_test_split'
from sklearn.model_selection import train_test_split

# Shuffle and split the data into training and testing subsets 
X_train, X_test, y_train, y_test = train_test_split(features_train , labels , test_size=0.25 , random_state=0)

# Success
print ('Training and testing split was successful')


# In[4]:


# Import the Regressor
from sklearn import linear_model

# Create a Linear Regressor object
regressor = linear_model.LinearRegression()

# Fit the training data to the model 
regressor.fit(X_train,y_train)

# Fitting is successful
print ('The data is successful fitted')


# In[9]:


# Predict on the Validation set 
predict = regressor.predict(X_test)

# Model successfully predicted on Validation set
print('Prediction is done on Validation set')


# In[6]:


# Import roc_auc_score' for evaluation on training data
from sklearn.metrics import roc_auc_score

# Calculate the performance score between 'y_test' and 'predict' via roc_auc_score
roc_score = roc_auc_score(y_test,predict)

# Print the Validation Score
print ('Validation score : ',roc_score, sep = ' ')


# In[7]:


# Predict on the test data
prediction_test = regressor.predict(features_test)

# Model successfully predicted on Test data
print('Prediction is done on Test data')


# In[8]:


# Taking the column ID in a variable
ID = test['ID']

# Exporting result in .csv format 
pd.DataFrame({'ID':ID,'click':prediction_test}).to_csv(r'C:\Users\sharm\Desktop\Submission.csv',index=False)

# Exporting is successful
print('Exporting is successful')

