#In this code we have preprocessed the epilepsy dataset before buidling the prediciton pipeline. We select the
#desired features, and ascertain that contionus variables are numeric. Categorical varialbes have been binary encoded
#or one-hot encoded. We have define a 80 - 20% train- test split for the model. #Imputation is done using sklearn
#preproccessing using mean value for continous data and most-frequent value for categorical data.


import pandas as pd
import numpy as np
import random
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

#Read csv into pandas data frame


e = pd.read_csv("epilepsy.csv")

el = pd.DataFrame(list(e))

#Remove eid column and second year variable columns except Number of Emergency visits in Year 2

l = list(range(1,7)) + list(range(10,20)) + [27,29] + list(range(31,39)) + list(range(47,89))

e2 = e.iloc[:,l]

#checking data types

e2df = pd.DataFrame(e2.dtypes)

#any age like '>90' has been taken as 90
e2['AgeAtFirstVisit'][e2['AgeAtFirstVisit'].str.match('\d+\D') == True] = "90"

e2['AgeAtFirstVisit'] = pd.to_numeric(e2['AgeAtFirstVisit'])

#Categorical gender and geo
print(e2['Gender'].value_counts())
print(e2['Gender'].value_counts())

#Remove unknown Gender

e2 = e2[e2['Gender'] != 'U']


#Binary encode Gender
e2.Gender[e2.Gender == 'M'] = 0
e2.Gender[e2.Gender == 'F'] = 1
e2['Gender'] = e2['Gender'].astype('category')


#One-hot encode geo

lb = LabelBinarizer()

lb_geo = lb.fit_transform(e2['geo'])
lb_geo_df = pd.DataFrame(lb_results, columns=lb.classes_)

#print(lb_results_df.head())

e3 = pd.concat([e2, lb_results_df], axis=1)


#removing redundant columns
e4 = e3.drop(['geo'], axis =1)
#e4 = e4.drop(['ERNum_ptyr2'], axis =1)
e4 = e4[pd.notnull(e4['ERNum_ptyr2'])]

#x,y
y = e4['ERNum_ptyr2']
x = e4.drop(['ERNum_ptyr2'], axis = 1)

#test-train split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)

#Missing data impute using mean values

#Continous train set
xtrdf = pd.DataFrame(xtrain.dtypes)
xtrdf['counter'] = range(len(xtrdf))
lcon = [0] + list(range(5,32)) + [65]
lcat = list(range(1,5)) + list(range(32,65)) + list(range(66,77))
xtrcon = xtrain.iloc[:,lcon]
xtrcat = xtrain.iloc[:,lcat]

null_data = x[x.isnull().any(axis=1)]

#Imputations mean value for continous data
mean_imputer_con = Imputer(missing_values='NaN', strategy='mean', axis=0)
mean_imputer = mean_imputer_con.fit(xtrcon)
imputed_xtrcon = mean_imputer.transform(xtrcon)
imputed_xtrcon_df = pd.DataFrame(imputed_xtrcon, columns = xtrcon.columns)

#Imputing most frequent value for categorical data
freq_imputer_cat = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
freq_imputer = freq_imputer_cat.fit(xtrcat)
imputed_xtrcat = freq_imputer.transform(xtrcat)
imputed_xtrcat_df = pd.DataFrame(imputed_xtrcat, columns = xtrcat.columns)

col_cat_names = list(imputed_xtrcat_df)

for col in col_cat_names:
    imputed_xtrcat_df[col] = imputed_xtrcat_df[col].astype('category',copy=False)
#xtris = pd.DataFrame(StandardScaler().fit_transform(xtri))

#Normalizing the continous data

imputed_xtrcon_norm = Normalizer().fit_transform(imputed_xtrcon_df)
imputed_xtrcon_norm_df = pd.DataFrame(imputed_xtrcon_norm, columns = imputed_xtrcon_df.columns)


xtrin = pd.concat([imputed_xtrcon_norm_df, imputed_xtrcat_df], axis=1)
xtrin.to_csv("epilepsy_preprocess_imputed_scaledcontinouse_0405.csv")


xtest_nm = xtest.drop([7130, 8038])

#Normalizing testing data
xtest_nmn = Normalizer().fit_transform(xtest_nm)
xtest_nmn = pd.DataFrame(xtest_nmn, columns = xtest_nm.columns)

#pandas index 8038, 7130 is null
ytest_nm = ytest.drop([7130, 8038])

xtrin.to_csv("xtrin_final.csv")
ytrain.to_csv("ytrain_final.csv")
xtest_nm.to_csv("xtest_final.csv")
ytest_nm.to_csv("ytest_final.csv")

#Unnorlmalized data
xtri = pd.concat([imputed_xtrcon_df, imputed_xtrcat_df],axis =1)
xtri.to_csv("epilepsy_preprocess_imputed_0406.csv")