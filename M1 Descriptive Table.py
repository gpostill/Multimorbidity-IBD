#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:43:00 2024

@author: gepostill
"""


#Note - for privacy all file names are written as FILE_NAME to conceal the file path of data and figures on the laptop used to create the code

################################################
#Importing Packages   
################################################


import pandas as pd
import numpy as np
from tableone import TableOne


###############################################
#Importing Data   
################################################

#Importing the data -- all conditionals all ages 
data = pd.read_csv(FILE_NAME)


#Copying the original dataframe and selecting the columns included in the prediction 
df = data.copy()


#################################
#TABLE 1 DATA 
################################

variables = ['AGE_IBD', 'sex','Age_death', 'Asthma', 'Age_Asthma', 
             'CHF', 'Age_CHF', 'COPD', 'Age_COPD','Myocardial_infarction', 'Age_Myocardial_infarction', 
             'Hypertension', 'Age_Hypertension', 'Arrhythmia', 'Age_Arrhythmia', 'CCS', 'Age_CCS',
             'Stroke', 'Age_Stroke', 'Cancer', 'Age_Cancer', 'Dementia',
             'Age_Dementia', 'Renal_Disease', 'Age_Renal_Disease', 'Diabetes',
             'Age_Diabetes', 'Osteoporosis', 'Age_Osteoporosis','Rheumatoid_Arthritis', 'Age_Rheumatoid_Arthritis', 
             'Osteoarthritis','Age_OsteoArthritis', 'Mood_disorder', 'Age_Mood_disorder',
             'Other_Mental_disorder', 'Age_Other_Mental_disorder', 'Degree', 'Degree_Cat'] 

categorical = ['sex', 'Asthma', 'CHF', 'COPD','Myocardial_infarction', 'Hypertension', 'Arrhythmia', 
               'CCS', 'Stroke', 'Cancer', 'Dementia','Renal_Disease', 'Diabetes',
               'Osteoporosis', 'Rheumatoid_Arthritis', 'Ostearthritis', 'Mood_disorder',
               'Other_Mental_disorder', 'Degree_Cat'] 

#Create TableOne Object 
table1 = TableOne(df, columns=variables, categorical=categorical, groupby='Cohort', pval=True, smd=True)
table1.to_csv(FILE_NAME)

exit()



########################################
#TABLE 1 EXPLORATORY ANALYSIS - IBD ONLY
########################################



#`Subset to only IBD patients
df = df[df["Cohort"]=="Case"]

#Remove deaths from 2019 and 2020 - these do not have cause of death

#MISSING DATA  
df = df.replace('.',np.nan)
missing_counts = df.isna().sum()
missing_counts_stratified = df.groupby('Cohort').apply(lambda x: x.isna().sum())


##Age 
age_stats = df.groupby('Cohort')['Age_death'].agg(['mean','std'])

def median(x): 
    return np.median(x)
def q25(x): 
    return np.percentile(x, 25)
def q75(x): 
    return np.percentile(x, 75)

age_median = df.groupby('Cohort')['Age_death'].agg([('Median', median), ('q25', q25), ('q75',q75)])


#Sex 
sex_stats = pd.crosstab(df['sex'], df['Cohort'], dropna=False)
sex_stats['case_per'] = sex_stats['Case']/sex_stats['Case'].sum()*100


#RIO
#Creating a function to dichotomize the numerical values into '>=40' and '<40'
def dichotomize(value):
    if value >= 40: 
        return 'Rural'
    elif value < 40: 
        return 'Urban'
    else: 
        return 'Not classified'

df['rio2008'] = df['rio2008'].astype(float)
df['RIO_cat'] = df['rio2008'].apply(dichotomize)

rio_stats = pd.crosstab(df['RIO_cat'], df['Cohort'], dropna=False)
rio_stats['case_per'] = rio_stats['Case']/rio_stats['Case'].sum()*100


#Material_deprivation 
dep_stats = pd.crosstab(df['Material_deprivation'], df['Cohort'], dropna=False)
dep_stats['case_per'] = dep_stats['Case']/dep_stats['Case'].sum()*100

#Incomes
inc_stats = pd.crosstab(df['incquint'], df['Cohort'], dropna=False)
inc_stats['case_per'] = inc_stats['Case']/inc_stats['Case'].sum()*100

#Education 
edu_stats = pd.crosstab(df['Education_Quintile'], df['Cohort'], dropna=False)
edu_stats['case_per'] = edu_stats['Case']/edu_stats['Case'].sum()*100

#Number of Chronic Conditions 
degree_stats = pd.crosstab(df['Degree_category'], df['Cohort'], dropna=False)
degree_stats['case_per'] = degree_stats['Case']/degree_stats['Case'].sum()*100





###############################################
#TRAINING DATA - DESCRIPTIVE ANALYSIS
################################################

#Importing the data -- need to get ID value then subset original df to get all columns
data_training = pd.read_csv(FILE_NAME)

#Create a list of training and testing IDs
training_IDs = data_training['ID'].to_list()
df_training = df[df['ID'].isin(training_IDs)] #re-writing this with the pre-defined training value 


#MISSING DATA  
missing_counts_train = df_training.isna().sum()


#Age 
age_stats_training = df_training.groupby('Cohort')['Age_death'].agg(['mean','std'])
age_median_training = df_training.groupby('Cohort')['Age_death'].agg([('Median', median), ('q25', q25), ('q75',q75)])

#Premature 
prem_training = df_training['Premature'].value_counts().reset_index() #Creating a count for the variable 
prem_training['Percent'] = prem_training['count'] / len(df_training) * 100 #creating the percent for the variable

#Sex 
sex_training = df_training['sex'].value_counts().reset_index() #Creating a count for the variable 
sex_training['Percent'] = sex_training['count'] / len(df_training) * 100 #creating the percent for the variable

#RIO
#Creating a function to dichotomize the numerical values into '>=40' and '<40'
df_training['rio2008'] = df_training['rio2008'].astype(float)
df_training['RIO_cat'] = df_training['rio2008'].apply(dichotomize)

rio_training = df_training['RIO_cat'].value_counts().reset_index() #Creating a count for the variable 
rio_training['Percent'] = rio_training['count'] / len(df_training) * 100 #creating the percent for the variable

#Material_deprivation 
dep_training = df_training['Material_deprivation'].value_counts().reset_index() #Creating a count for the variable 
dep_training['Percent'] = dep_training['count'] / len(df_training) * 100 #creating the percent for the variable


#Number of Chronic Conditions 
degree_training = df_training['Degree_category'].value_counts().reset_index() #Creating a count for the variable 
degree_training['Percent'] = degree_training['count'] / len(df_training) * 100 #creating the percent for the variable

#Prevalence of each of Chronic Conditions 
conditions = ['Asthma','CHF','COPD','Myocardial_infarction', 'Hypertension','Arrhythmia','CCS','Stroke','Cancer','Dementia',
              'Rental_Disease','Diabetes','Osteoporosis','Rheumatoid_Arthritis','Osteoarthritis','Mood_disorder','Other_Mental_disorder']

for condition in conditions: 
    condition_training = df_training[condition].value_counts().reset_index() #Creating a count for the variable 
    condition_training['Percent'] = condition_training['count'] / len(df_training) * 100 #creating the percent for the variable
    print(condition_training)


#Age of each chronic condition
#Subsetting to the relevant columns (only binary yes/no of conditions and sex)
ages = ['Age_Asthma','Age_CHF','Age_COPD','Age_Myocardial_infarction','Age_Hypertension','Age_Arrhythmia','Age_CCS','Age_Stroke','Age_Cancer','Age_Dementia',
         'Age_Renal_Disease','Age_Diabetes','Age_Osteoporosis','Age_Rheumatoid_Arthritis','Age_OsteoArthritis','Age_Mood_disorder','Age_Other_Mental_disorder']

df_age_train = pd.DataFrame(columns=['variable', 'mean', 'sd', 'median', 'q1', 'q3'])

for age in ages:
    mean = df_training[age].mean()
    sd = df_training[age].std()
    median = df_training[age].median()
    q1 = df_training[age].quantile(0.25)
    q3 = df_training[age].quantile(0.75)

    print(age)
    print(f"mead: {mean},   sd: {sd},   median {median}, IQR: {q1} - {q3}")

    #Add row to list
    #df_age_train = df_age_train.append({'variable':age, 'mean':mean, 'sd':sd, 'median':median, 'q1':q1, 'q3':q3}, ignore_index=True)


#Importing the data -- all conditionals all ages 
df_U60 = pd.read_csv(FILE_NAME)
df_U60 = df_U60[df_U60["Cohort"]=="Case"] #`Subset to only IBD patients
df_train_U60 = df_U60[df_U60['ID'].isin(training_IDs)] #re-writing this with the pre-defined training value 

#Prevalence of each of Chronic Conditions 
conditions = ['Asthma','CHF','COPD','Myocardial_infarction', 'Hypertension','Arrhythmia','CCS','Stroke','Cancer','Dementia',
              'Renal_Disease','Diabetes','Osteoporosis','Rheumatoid_Arthritis','Osteoarthritis','Mood_disorder','Other_Mental_disorder']

for condition in conditions: 
    condition_training = df_train_U60[condition].value_counts().reset_index() #Creating a count for the variable 
    condition_training['Percent'] = condition_training['count'] / len(df_train_U60) * 100 #creating the percent for the variable
    print('Under 60 Training', condition_training)




###############################################
#TESTING DATA  - DESCRIPTIVE ANALYSIS
################################################

#Importing the data -- need to get ID value then subset original df to get all columns
data_testing = pd.read_csv(FILE_NAME)

#Create a list of trianing and testing IDs
testing_IDs = data_testing['ID'].to_list()
df_testing = df[df['ID'].isin(testing_IDs)] #re-writing this with the pre-defined training value 


#MISSING DATA  
missing_counts_test = df_testing.isna().sum()


#Age 
age_test = df_testing.groupby('Cohort')['Age_death'].agg(['mean','std'])
#age_median_test = df_testing.groupby('Cohort')['Age_death'].agg([('Median', median), ('q25', q25), ('q75',q75)])

#Premature 
prem_testing = df_testing['Premature'].value_counts().reset_index() #Creating a count for the variable 
prem_testing['Percent'] = prem_testing['count'] / len(df_testing) * 100 #creating the percent for the variable

#Sex 
sex_test = df_testing['sex'].value_counts().reset_index() #Creating a count for the variable 
sex_test['Percent'] = sex_test['count'] / len(df_testing) * 100 #creating the percent for the variable

#RIO
#Creating a function to dichotomize the numerical values into '>=40' and '<40'
df_testing['rio2008'] = df_testing['rio2008'].astype(float)
df_testing['RIO_cat'] = df_testing['rio2008'].apply(dichotomize)

rio_test = df_testing['RIO_cat'].value_counts().reset_index() #Creating a count for the variable 
rio_test['Percent'] = rio_test['count'] / len(df_testing) * 100 #creating the percent for the variable

#Material_deprivation 
dep_test = df_testing['Material_deprivation'].value_counts().reset_index() #Creating a count for the variable 
dep_test['Percent'] = dep_test['count'] / len(df_testing) * 100 #creating the percent for the variable


#Number of Chronic Conditions 
degree_test = df_testing['Degree_category'].value_counts().reset_index() #Creating a count for the variable 
degree_test['Percent'] = degree_test['count'] / len(df_testing) * 100 #creating the percent for the variable

#Prevalence of each of Chronic Conditions 
conditions = ['Asthma','CHF','COPD','Myocardial_infarction', 'Hypertension','Arrhythmia','CCS','Stroke','Cancer','Dementia',
              'Rental_Disease','Diabetes','Osteoporosis','Rheumatoid_Arthritis','Osteoarthritis','Mood_disorder','Other_Mental_disorder']

for condition in conditions: 
    condition_training = df_testing[condition].value_counts().reset_index() #Creating a count for the variable 
    condition_training['Percent'] = condition_training['count'] / len(df_testing) * 100 #creating the percent for the variable
    print(condition_training)


#Age of each chronic condition
#Subsetting to the relevant columns (only binary yes/no of conditions and sex)
ages = ['Age_Asthma','Age_CHF','Age_COPD','Age_Myocardial_infarction','Age_Hypertension','Age_Arrhythmia','Age_CCS','Age_Stroke','Age_Cancer','Age_Dementia',
         'Age_Renal_Disease','Age_Diabetes','Age_Osteoporosis','Age_Rheumatoid_Arthritis','Age_OsteoArthritis','Age_Mood_disorder','Age_Other_Mental_disorder']

df_age_train = pd.DataFrame(columns=['variable', 'mean', 'sd', 'median', 'q1', 'q3'])

for age in ages:
    mean = df_testing[age].mean()
    sd = df_testing[age].std()
    median = df_testing[age].median()
    q1 = df_testing[age].quantile(0.25)
    q3 = df_testing[age].quantile(0.75)

    print(age)
    print(f"mead: {mean},   sd: {sd},   median {median}, IQR: {q1} - {q3}")
    #Add row to list
    #df_age_train = df_age_train.append({'variable':age, 'mean':mean, 'sd':sd, 'median':median, 'q1':q1, 'q3':q3}, ignore_index=True)


#Importing the data -- all conditionals all ages 
df_test_U60 = df_U60[df_U60['ID'].isin(testing_IDs)] #re-writing this with the pre-defined training value 

#Prevalence of each of Chronic Conditions 
conditions = ['Asthma','CHF','COPD','Myocardial_infarction', 'Hypertension','Arrhythmia','CCS','Stroke','Cancer','Dementia',
              'Renal_Disease','Diabetes','Osteoporosis','Rheumatoid_Arthritis','Osteoarthritis','Mood_disorder','Other_Mental_disorder']

for condition in conditions: 
    condition_training = df_test_U60[condition].value_counts().reset_index() #Creating a count for the variable 
    condition_training['Percent'] = condition_training['count'] / len(df_test_U60) * 100 #creating the percent for the variable
    print('Under 60 Testing', condition_training)


