#!/usr/bin/env python
# coding: utf-8

# ## SUPERVISED LEARNING ALGORITHMS

# ### SIMPLE LINEAR REGRESSION

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("D:\DataSalary.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[10]:


data = data.drop("Index",axis=1) #del data["Index"]


# In[11]:


data.shape


# In[12]:


data.columns


# In[13]:


data.isna().sum()


# In[14]:


data.describe()


# In[15]:


data.dtypes


# In[16]:


x = data.iloc[:,0:1].values
y = data.iloc[:,-1:].values


# In[17]:


plt.title("Experience vs salary")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.scatter(x,y,color = "r")
plt.show()


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[19]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[20]:


y_pred = model.predict(x_test)


# In[21]:


y_pred


# In[22]:


plt.title("Experience vs salary")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.scatter(x,y,color = "r")
plt.plot(x,model.predict(x),"b")
plt.show()


# In[23]:


model.predict(x)


# In[24]:


model.predict([[10]])


# In[25]:


m = model.coef_


# In[26]:


c = model.intercept_


# In[27]:


print(m,c)


# In[28]:


y = m * 10 + c


# In[29]:


y


# In[30]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[31]:


print("Mean absolute error = ",mean_absolute_error(y_test,y_pred))
print("Mean squared error = ",mean_squared_error(y_test,y_pred))
print("Root mean squared error = ",np.sqrt(mean_squared_error(y_test,y_pred)))
print("Root squared error = ",r2_score(y_test,y_pred))


# In[33]:


import joblib


# In[34]:


a= joblib.dump(model,"salary_predict")


# In[35]:


a


# In[39]:


n = joblib.load("salary_predict")


# In[40]:


prediction = n.predict([[4]])


# In[41]:


prediction


# ### Logistic Regression on Iris Flower DataSet

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[42]:


data = pd.read_csv("D:\IRIS.csv")


# In[43]:


data.head()


# In[44]:


data.shape


# In[45]:


data.isna().sum()


# In[46]:


data.dtypes


# In[48]:


data.describe()


# In[50]:


plt.title("Sepal length vs sepal width")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.scatter(data[data["species"]=="Iris-setosa"].iloc[:,0],data[data["species"]=="Iris-setosa"].iloc[:,1],label="Iris setosa")
plt.scatter(data[data["species"]=="Iris-versicolor"].iloc[:,0],data[data["species"]=="Iris-versicolor"].iloc[:,1],label="Iris versicolor")
plt.scatter(data[data["species"]=="Iris-virginica"].iloc[:,0],data[data["species"]=="Iris-virginica"].iloc[:,1],label = "Iris virginica")
plt.legend()
plt.show()


# In[51]:


plt.title("Petal length vs Petal width")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.scatter(data[data["species"]=="Iris-setosa"].iloc[:,2],data[data["species"]=="Iris-setosa"].iloc[:,3],label="Iris setosa")
plt.scatter(data[data["species"]=="Iris-versicolor"].iloc[:,2],data[data["species"]=="Iris-versicolor"].iloc[:,3],label="Iris versicolor")
plt.scatter(data[data["species"]=="Iris-virginica"].iloc[:,2],data[data["species"]=="Iris-virginica"].iloc[:,3],label = "Iris virginica")
plt.legend()
plt.show()


# In[52]:


x = data.iloc[:,0:4].values
y = data.iloc[:,-1:].values


# In[54]:


plt.hist(data["sepal_length"])


# In[56]:


plt.hist(data["sepal_width"])


# In[57]:


plt.hist(data["petal_length"])


# In[58]:


plt.hist(data["petal_width"])


# In[59]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 7)


# In[60]:


from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings(action="ignore")
model = LogisticRegression()
model.fit(x_train,y_train)


# In[61]:


y_pred = model.predict(x_test)
y_pred


# In[62]:


from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
print(accuracy_score(y_test,y_pred))


# In[63]:


print(confusion_matrix(y_test,y_pred))


# In[64]:


print(classification_report(y_test,y_pred))


# In[65]:


import joblib


# In[66]:


a = joblib.dump(model,"logistic_regression")
a


# In[68]:


n = joblib.load("logistic_regression")


# ## Algerian Forest Fire Dataset - Temperature Prediction
# 
# 
# 
# * Data Collection
# * Exploratory data analysis
# * Data Cleaning
# * Linear Regression Model Training
# * Ridge Regression Model Training
# * Lasso Regression Model Training
# * Elastincet Regression Model Training
# 
# 

# # Importing the libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Reading and Cleaning

# In[2]:


df = pd.read_csv("D:\\Algerian_forest_fires_dataset.csv")
df.head()


# ## Drop an  row

# In[3]:


df.loc[122:125]


# In[4]:


df.drop([122,123,124],inplace=True)
df.reset_index(inplace=True)
df.drop('index',axis=1,inplace=True)


# ## Shape of the data after dropping the column 

# In[5]:


df.shape


# ## Creating a column region  

# In[6]:


df.loc[:122, 'region'] = 'bejaia'
df.loc[122:, 'region'] = 'Sidi-Bel Abbes'


# ## Columns of the Dataset 

# In[7]:


df.columns


# ## Stripping the names of the columns as it has unwanted spaces

# In[8]:


df.columns=[i.strip() for i in df.columns]
df.columns


# ## Dropping the Classes Features  

# In[9]:


df.drop('Classes',axis=1,inplace=True)
df.head()


# ## Replacing the day,month,year feature with date feature

# In[10]:


df['date']=pd.to_datetime(df[['day','month','year']])
df.drop(['day','month','year'],axis=1,inplace=True)


# In[11]:


df.head()


# ## Checking the datatypes of features 

# In[12]:


df.dtypes


# ## Changing the datatypes of features 

# In[13]:


df['Temperature']=df['Temperature'].astype(int)
df['RH']=df['RH'].astype(int)
df['Ws']=df['Ws'].astype(int)
df['Rain']=df['Rain'].astype(float)
df['FFMC']=df['FFMC'].astype(float)
df['DMC']=df['DMC'].astype(float)
df['ISI']=df['ISI'].astype(float)
df['BUI']=df['BUI'].astype(float)


# In[14]:


df.dtypes


# ## Applying  Label encoding in DC,FWI,region features 

# In[15]:


from sklearn.preprocessing import LabelEncoder
LabelEncoder=LabelEncoder()


# In[16]:


df['DC']=LabelEncoder.fit_transform(df['DC'])
df['FWI']=LabelEncoder.fit_transform(df['FWI'])
df['region']=LabelEncoder.fit_transform(df['region'])


# In[17]:


df.dtypes


# In[18]:


df.head()


# ## Checking the null values 

# In[19]:


df.isnull().sum()


# ## Observation
# 
# Zero null value in the dataset 

# ## Univariate Analysis 

# In[20]:


numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']


# In[21]:


numeric_features


# ## Features Information
# 
# *Date : (DD/MM/YYYY) Day, month ('june' to 'september'), year (2012) Weather data observations
# 
# *Temp : temperature noon (temperature max) in Celsius degrees: 22 to 42
# 
# *RH : Relative Humidity in %: 21 to 90
# 
# *Ws :Wind speed in km/h: 6 to 29
# 
# *Rain: total day in mm: 0 to 16.8 FWI Components
# 
# *Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5
# 
# *Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9
# 
# *Drought Code (DC) index from the FWI system: 7 to 220.4
# 
# *Initial Spread Index (ISI) index from the FWI system: 0 to 18.5
# 
# *Buildup Index (BUI) index from the FWI system: 1.1 to 68
# 
# *Fire Weather Index (FWI) Index: 0 to 31.1
# 

# In[22]:


plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)

for i in range(0, len(numeric_features)):
    plt.subplot(5, 3, i+1)
    sns.kdeplot(x=df[numeric_features[i]],shade=True, color='r')
    plt.xlabel(numeric_features[i])
    plt.tight_layout()


# ## Visualization of Target Feature 

# In[23]:


plt.subplots(figsize=(14,7))
sns.histplot(x=df.Temperature, ec = "black", color='blue', kde=True)
plt.title("Temperature Distribution", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=15)
plt.xlabel("Temperatures", weight="bold", fontsize=12)
plt.show()


# ## Observation
# 
# Temperature occur most of the time in range 32.5 to 35.0

# ## Temperature Vs Rain 

# In[24]:


import matplotlib
matplotlib.rcParams['figure.figsize']=(20,10)

sns.barplot(x="Temperature",y="Rain",data=df)


# ## Observation
# 
# When the temperature is around 22 , most of the time rain occurr

# ## Which region has most temperature? 

# In[25]:


import matplotlib
matplotlib.rcParams['figure.figsize']=(20,10)

sns.barplot(x="region",y="Temperature",data=df)


# ## Observation 
# 
# Region represented by 0 i.e. 'Sidi-Bel Abbes' has highest temperature

# ## Correlation of the features 

# In[26]:


df.corr()


# ## Multivariate analysis 

# In[27]:


import seaborn as sns
plt.figure(figsize=(10,8))
sns.heatmap(df.corr())


# ## Observation
# 
# The target feature Temperature is highly positively correlated with FFMC,ISI

# ## Temperature Vs date feature 

# In[28]:


plt.figure(figsize=(8,8))
sns.lineplot(x='Temperature',y='date',data=df,color='r')


# ## Temperature Vs FFMC 

# In[29]:


plt.figure(figsize=(10,10))
sns.jointplot(x='Temperature',y='FFMC',data=df,color='g')


# ## Temperature Vs ISI 

# In[30]:


plt.figure(figsize=(10,10))
sns.regplot(x='Temperature',y='ISI',data=df)


# ## Checking the outliers of the target 'Temperature' feature 

# In[31]:


sns.boxplot(df['Temperature'])


# ## Boxplot of Rain Vs Temperature 

# In[32]:


sns.boxplot(x ='Temperature', y ='Rain', data = df)


# ## Boxplot of 'FFMC' Vs Temperature

# In[33]:


sns.boxplot(x ='Temperature', y ='FFMC', data = df)


# ## Boxplot of ISI Vs Temperature 

# In[34]:


sns.boxplot(x ='Temperature', y ='ISI', data = df)


# ## Boxplot of region Vs Temperature 

# In[35]:


sns.boxplot(x ='region', y ='Temperature', data = df)


# ## Boxplot of BUI Vs Temperature 

# In[36]:


sns.boxplot(x ='Temperature', y ='BUI', data = df)


# ## Boxplot DMC Vs Temperature 

# In[37]:


sns.boxplot(x ='Temperature', y ='DMC', data = df)


# ## Creating Dependent and Independent features 

# In[38]:


df.columns


# In[39]:


## Independent Features

x=pd.DataFrame(df, columns=['RH','Ws','Rain','FFMC','DMC','DC','ISI','BUI','FWI','region'])  

## Dependent Features

y=pd.DataFrame(df,columns=['Temperature'])


# ## Independent Features 

# In[40]:


x


# ## Dependent Features 

# In[41]:


y


# ## TrainTest Split

# In[42]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(
x,y,test_size=0.33,random_state=10)


# In[43]:


x_train.shape


# In[44]:


x_test.shape


# In[45]:


y_train.shape


# In[46]:


y_test.shape


# ## Independent training dataset 

# In[47]:


x_train


# ## Independent Test Dataset 

# In[48]:


x_test


# ## Dependent Training Dataset 

# In[49]:


y_train


# ## Dependent Test Dataset 

# In[50]:


y_test


# ## Standardizing or Feature Scaling

# In[51]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()  ## Initialising


# In[52]:


scaler


# In[53]:


x_train=scaler.fit_transform(x_train)


# In[54]:


x_test=scaler.transform(x_test)


# In[55]:


x_train


# In[56]:


x_test


# ## Model Training 

# In[57]:


from sklearn.linear_model import LinearRegression


# In[58]:


regression=LinearRegression()


# In[59]:


regression
regression.fit(x_train,y_train)


# ## Coefficient 

# In[60]:


print(regression.coef_)


# ## Intercept 

# In[61]:


print(regression.intercept_)


# ## Prediction for Test Data 

# In[62]:


reg_pred=regression.predict(x_test)
reg_pred


# In[63]:


import seaborn as sns
sns.distplot(reg_pred-y_test)


# ## Assumption of Linear Regression

# In[64]:


plt.scatter(y_test,reg_pred)
plt.xlabel("Test truth data")
plt.ylabel('Test predicted data')


# ## Residuals 

# In[65]:


residual=y_test-reg_pred


# In[66]:


residual


# In[67]:


sns.displot(residual,kind='kde')


# ## Scatterplot with prediction and residual 

# In[68]:


plt.scatter(reg_pred,residual)


# ## Performance Metrics 

# In[69]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(y_test,reg_pred))
print(mean_absolute_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# ## R square and adjusted R square 

# In[70]:


from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)


# ## Adjusted R square 

# In[71]:


1-(1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)


# ## Ridge Regression Algorithm 

# In[72]:


from sklearn.linear_model import Ridge


# In[73]:


ridge=Ridge()


# In[74]:


ridge


# In[75]:


ridge.fit(x_train,y_train)


# In[76]:


## Coefficient

print(ridge.coef_)


# In[77]:


## intercept

print(ridge.intercept_)


# In[78]:


## prediction for test data

ridge_pred=ridge.predict(x_test)
ridge_pred


# In[79]:


import seaborn as sns
sns.distplot(reg_pred-y_test)


# In[80]:


## Assumption of ridge regression

plt.scatter(y_test,ridge_pred)
plt.xlabel("Test truth data")
plt.ylabel('Test predicted data')


# In[81]:


## residuals

residual=y_test-ridge_pred
residual


# In[82]:


sns.displot(residual,kind='kde')


# ## Scatter plot with residual and prediction 

# In[83]:


plt.scatter(ridge_pred,residual)


# ## Performance Matrics 

# In[84]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(y_test,ridge_pred))
print(mean_absolute_error(y_test,ridge_pred))
print(np.sqrt(mean_squared_error(y_test,ridge_pred)))


# ## R square 

# In[85]:


from sklearn.metrics import r2_score
ridge_score=r2_score(y_test,ridge_pred)
print(ridge_score)


# ## Adjusted R square 

# In[86]:


1-(1-ridge_score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)


# ## Lasso Regression 

# In[87]:


from sklearn.linear_model import Lasso


# In[88]:


lasso=Lasso()


# In[89]:


lasso


# In[90]:


lasso.fit(x_train,y_train)


# ## Coefficients and Intercepts

# In[91]:


print(lasso.coef_)


# In[92]:


print(lasso.intercept_)


# In[93]:


## Prediction for test data

lasso_pred = lasso.predict(x_test)


# In[94]:


lasso_pred


# ## Performace Matrics 

# In[95]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(y_test,lasso_pred))
print(mean_absolute_error(y_test,lasso_pred))
print(np.sqrt(mean_squared_error(y_test,lasso_pred)))


# ## R Square 

# In[96]:


from sklearn.metrics import r2_score
lasso_score=r2_score(y_test,lasso_pred)
print(lasso_score)


# ## Adjusted R square 

# In[97]:


1-(1-lasso_score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)

