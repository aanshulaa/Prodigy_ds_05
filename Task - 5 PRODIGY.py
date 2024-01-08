#!/usr/bin/env python
# coding: utf-8

# TASK 5

# Analyze traffic accident data to identify patterns related to road conditions, weather, and time of day. Visualize accident hotspots and contributing factors.

# Loading Libraries and Data

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')


# In[3]:


#load and read the file
df = pd.read_csv("C:\\Users\\kansh\\OneDrive\\Desktop\\PRODIGY\\RTA Dataset.csv")

df.head()


# In[4]:


#shape/ size of the data
df.shape


# In[5]:


#checking the numerical statistics of the data
df.describe()


# In[6]:


df.describe(include="all")


# In[7]:


#checking data types of each columns
df.info()


# Exploratory Data Analysis

# In[8]:


#finding duplicate values
df.duplicated().sum()


# In[9]:


#Distribution of Accident severity
df['Accident_severity'].value_counts()


# In[10]:


#plotting the final class
sns.countplot(x = df['Accident_severity'])
plt.title('Distribution of Accident severity')


# Handling missing values

# In[11]:


#checking missing values
df.isna().sum()


# In[12]:


#dropping columns which has more than 2500 missing values and Time column
df.drop(['Service_year_of_vehicle','Defect_of_vehicle','Work_of_casuality', 'Fitness_of_casuality','Time'],
        axis = 1, inplace = True)
df.head()


# In[13]:


#storing categorical column names to a new variable
categorical=[i for i in df.columns if df[i].dtype=='O']
print('The categorical variables are',categorical)


# In[14]:


#for categorical values we can replace the null values with the Mode of it
for i in categorical:
    df[i].fillna(df[i].mode()[0],inplace=True)


# In[15]:


#checking the current null values
df.isna().sum()


# Data Visualization

# In[17]:


#plotting relationship between Number_of_casualties and Number_of_vehicles_involved
sns.scatterplot(x=df['Number_of_casualties'], y=df['Number_of_vehicles_involved'], hue=df['Accident_severity'])


# There is no visible correlation between Number_of_casualties and Number_of_vehicles_involved columns

# In[18]:


#joint Plot
sns.jointplot(x='Number_of_casualties',y='Number_of_vehicles_involved',data=df)


# In[19]:


#checking the correlation between numerical columns
df.corr()


# In[20]:


#plotting the correlation using heatmap
sns.heatmap(df.corr())


# In[21]:


#storing numerical column names to a variable
numerical=[i for i in df.columns if df[i].dtype!='O']
print('The numerica variables are',numerical)


# In[22]:


#distribution for numerical columns
plt.figure(figsize=(10,10))
plotnumber = 1
for i in numerical:
    if plotnumber <= df.shape[1]:
        ax1 = plt.subplot(2,2,plotnumber)
        plt.hist(df[i],color='red')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('frequency of '+i, fontsize=10)
    plotnumber +=1


# Most accidents are occured when 2 vehicles are involved and 1 casuality is happend mostly in the accidents.

# In[23]:


#count plot for categorical values
plt.figure(figsize=(10,200))
plotnumber = 1

for col in categorical:
    if plotnumber <= df.shape[1] and col!='Pedestrian_movement':
        ax1 = plt.subplot(28,1,plotnumber)
        sns.countplot(data=df, y=col, palette='muted')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(col.title(), fontsize=14)
        plt.xlabel('')
        plt.ylabel('')
    plotnumber +=1


# Handling Categorical Values

# In[24]:


df.dtypes


# Since there are so many categorical values, we need to use feature selection We need to perform label encoding before applying chi square analysis

# In[25]:


#importing label encoing module
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

#creating a new data frame from performing the chi2 analysis
df1=pd.DataFrame()

#adding all the categorical columns except the output to new data frame
for i in categorical:
    if i!= 'Accident_severity':
        df1[i]=le.fit_transform(df[i])


# In[27]:


#confirming the data type
df1.info()


# In[28]:


plt.figure(figsize=(22,17))
sns.set(font_scale=1)
sns.heatmap(df1.corr(), annot=True)


# In[30]:


#label encoded data set
df1.head()


# In[32]:


#import chi2 test
from sklearn.feature_selection import chi2
f_p_values = chi2(df1, df['Accident_severity'])


# In[33]:



#f_p_values will return Fscore and pvalues
f_p_values


# In[34]:


#for better understanding and ease of access adding them to a new dataframe
f_p_values1=pd.DataFrame({'features':df1.columns, 'Fscore': f_p_values[0], 'Pvalues':f_p_values[1]})
f_p_values1


# In[35]:


#since we want lower Pvalues we are sorting the features
f_p_values1.sort_values(by='Pvalues',ascending=True)


# we need higher Fscore and lower the Pvalues, so by evaluating, we can remove Owner_of_vehicle, Type_of_vehicle, Road_surface_conditions, Pedestrian_movement,Casualty_severity,Educational_level,Day_of_week,Sex_of_driver,Road_allignment, Sex_of_casualty

# In[36]:


#after evaluating we are removing lesser important columns and storing to a new data frame
df2=df.drop(['Owner_of_vehicle', 'Type_of_vehicle', 'Road_surface_conditions', 'Pedestrian_movement',
         'Casualty_severity','Educational_level','Day_of_week','Sex_of_driver','Road_allignment',
         'Sex_of_casualty'],axis=1)
df2.head()


# In[37]:


df2.shape


# In[38]:


df2.info()


# In[39]:


#to check distinct values in each categorical columns we are storing them to a new variable
categorical_new=[i for i in df2.columns if df2[i].dtype=='O']
print(categorical_new)


# In[40]:


for i in categorical_new:
    print(df2[i].value_counts())


# In[41]:


#get_dummies
dummy=pd.get_dummies(df2[['Age_band_of_driver', 'Vehicle_driver_relation', 'Driving_experience',
                          'Area_accident_occured', 'Lanes_or_Medians', 'Types_of_Junction', 'Road_surface_type', 
                          'Light_conditions', 'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 
                          'Casualty_class', 'Age_band_of_casualty', 'Cause_of_accident']],drop_first=True)
dummy.head()


# In[42]:


#concatinate dummy and old data frame
df3=pd.concat([df2,dummy],axis=1)
df3.head()


# In[43]:


#dropping dummied columns
df3.drop(['Age_band_of_driver', 'Vehicle_driver_relation', 'Driving_experience', 'Area_accident_occured', 'Lanes_or_Medians',
          'Types_of_Junction', 'Road_surface_type', 'Light_conditions', 'Weather_conditions', 'Type_of_collision',
          'Vehicle_movement','Casualty_class', 'Age_band_of_casualty', 'Cause_of_accident'],axis=1,inplace=True)
df3.head()


# Seperating Independent and Dependent

# In[44]:


x=df3.drop(['Accident_severity'],axis=1)
x.shape


# In[45]:


x.head()


# In[46]:


y=df3.iloc[:,2]
y.head()


# In[47]:


#checking the count of each item in the output column
y.value_counts()


# In[48]:


#plotting count plot using seaborn
sns.countplot(x = y, palette='muted')


# Oversampling

# In[6]:


get_ipython().system('pip install imbalanced-learn')


# In[9]:


import numpy as np
from imblearn.over_sampling import SMOTE

# Replace the Ellipsis with your actual data
# Example feature matrix with two features
x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])

# Example labels
y = np.array([0, 1, 0, 1])

# Initialize SMOTE
oversample = SMOTE()

# Apply SMOTE to generate synthetic samples
xo, yo = oversample.fit_resample(x, y)



# In[10]:


#importing SMOTE 
from imblearn.over_sampling import SMOTE
oversample=SMOTE()
xo,yo=oversample.fit_resample(x,y)


# In[12]:


import numpy as np
import pandas as pd  # Import pandas

# Your previous code here

# Checking the oversampling output
y1 = pd.DataFrame(yo)
y1.value_counts()


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

# Your previous code here

# Plotting the countplot
sns.countplot(x=yo, palette='muted')
plt.show()


# This task involves analyzing traffic accident data to discern patterns associated with road conditions, weather, and time of day. The objective is to identify accident hotspots and contributing factors. Through data analysis and visualization, the goal is to gain insights into the correlation between specific conditions (such as slippery roads or adverse weather) and the occurrence of accidents. By pinpointing hotspots and understanding the factors influencing accidents, this analysis aims to contribute to improving road safety measures and traffic management strategies.

# 
# THANK YOU
# Project by: Anshula Killamsetty
