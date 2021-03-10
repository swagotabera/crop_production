#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from numpy import loadtxt
from numpy import unique
from numpy import arange
from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pickle


# In[5]:


X_train_Final=pd.read_csv("X_train_Final.csv")
X_test_Final=pd.read_csv("X_test_Final.csv")
y_train_Final=pd.read_csv("y_train_Final.csv")
y_test_Final=pd.read_csv("y_test_Final.csv")


# In[6]:


X_train_Final.shape,y_train_Final.shape,X_test_Final.shape,y_test_Final.shape


# In[7]:


categorical = [var for var in X_train_Final.columns if X_train_Final[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))


# In[8]:


continous = [var for var in X_train_Final.columns if X_train_Final[var].dtype!='O']
print('There are {} continous variables'.format(len(continous)))


# In[9]:


# let's visualise the values of the discrete variables
discrete = []

for var in continous:
    if len(X_train_Final[var].unique()) < 20 :
        print(var, ' values: ', X_train_Final[var].unique())
        discrete.append(var)
print()
print('There are {} discrete variables'.format(len(discrete)))


# In[10]:


X_train_Final[discrete] = X_train_Final[discrete].astype('O')


# In[11]:


# Modified categorical variables
categorical = [var for var in X_train_Final.columns if X_train_Final[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))


# In[12]:


# let's order the labels according to the mean target value
X_train_Final.groupby(['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop'])[ 'Production'].mean().sort_values()


# In[13]:


def find_category_mappings(df, variable, target):

    # first  we generate an ordered list with the labels
    ordered_labels = X_train_Final.groupby([variable])[target].mean().sort_values().index
    # return the dictionary with mappings
    return {k: i for i, k in enumerate(ordered_labels, 0)}
def integer_encode(train, test, variable, ordinal_mapping):
    X_train_Final[variable] = X_train_Final[variable].map(ordinal_mapping)
    X_test_Final[variable] = X_test_Final[variable].map(ordinal_mapping)


# In[14]:


# and now we run a loop over the remaining categorical variables
for variable in ['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 
       'Production']:
    mappings = find_category_mappings(X_train_Final, variable, 'Production')
    integer_encode(X_train_Final, X_test_Final, variable, mappings)


# In[15]:


X_train_Final.drop(columns='Production', axis=1, inplace=True )
X_test_Final.drop(columns='Production', axis=1, inplace=True )


# In[16]:


X_train_Final.head()


# In[17]:


#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler()
#X_train_Final = sc.fit_transform(X_train_Final)
#X_test_Final = sc.transform(X_test_Final)


# In[18]:


X_train_Final.shape,X_test_Final.shape,y_train_Final.shape,y_test_Final.shape


# In[19]:


# Missing numerical data

X_train_Final=pd.DataFrame(X_train_Final)
for col in X_train_Final:
    if X_train_Final[col].isnull().mean() > 0:
        print(col, X_train_Final[col].isnull().mean())
        


# In[20]:


X_test_Final=pd.DataFrame(X_test_Final)
for col in X_test_Final:
    if X_test_Final[col].isnull().mean() > 0:
        print(col, X_test_Final[col].isnull().mean())


# In[21]:


X_train_Final.shape,X_test_Final.shape,y_train_Final.shape,y_test_Final.shape


# In[22]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(X_train_Final[["District_Name" , "Crop"]])
X_train_Final[["District_Name" , "Crop"]] = imputer.transform(X_train_Final[["District_Name" , "Crop"]])
X_test_Final[["District_Name" , "Crop"]] = imputer.transform(X_test_Final[["District_Name" , "Crop"]])


# In[23]:


X_train_Final.shape,X_test_Final.shape,y_train_Final.shape,y_test_Final.shape


# In[24]:


y_train_Final=y_train_Final.values.ravel()
y_test_Final=y_test_Final.values.ravel()


# In[25]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
RFRegression=Pipeline([
('Decision',RandomForestRegressor(max_depth=2000, max_features= 'sqrt', n_estimators=200,bootstrap='True',
                                  random_state =1,min_samples_split=2))
])
RFRegression.fit(X_train_Final,y_train_Final)
#prediction
X_train_pred=RFRegression.predict(X_train_Final)
X_test_pred=RFRegression.predict(X_test_Final)
print('train r2: {}'.format(r2_score(y_train_Final,X_train_pred)))

print('test r2: {}'.format(r2_score(y_test_Final,X_test_pred)))


# In[26]:


pickle.dump(RFRegression,open('model.pkl','wb'))


# In[ ]:


model=pickle.load(open('model.pkl'))

