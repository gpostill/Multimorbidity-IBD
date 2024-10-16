#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 07:41:59 2024

@author: gepostill
"""

#Note - for privacy all file names are written as FILE_NAME to conceal the file path of data and figures on the laptop used to create the code

################################################
#Importing Packages   
################################################

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


################################################
#Importing Data   
################################################

#Importing the data -- all conditionals all ages 
df = pd.read_csv(FILE_NAME)

#`Separeate Cases and Controls
cases = df[df["Cohort"]=="Case"]
controls = df[df["Cohort"]=="Control"]

#List of conditions
condition_columns = ['AGE_IBD','Age_Asthma','Age_CHF','Age_COPD','Age_Myocardial_infarction','Age_Hypertension','Age_Arrhythmia','Age_CCS','Age_Stroke','Age_Cancer','Age_Dementia',
                     'Age_Renal_Disease','Age_Diabetes','Age_Osteoporosis','Age_Rheumatoid_Arthritis','Age_OsteoArthritis','Age_Mood_disorder','Age_Other_Mental_disorder']

#convert age columns to integers 
df[condition_columns] = df[condition_columns].astype(float).astype(pd.Int32Dtype())

#Initialize dictionaries to hold age-specific condition counts
age_range = range(df[condition_columns].min().min(),df[condition_columns].max().max())
case_counts = {age: [] for age in age_range}
control_counts = {age: [] for age in age_range}

#Fill the dictionaries with condition counts
for age in age_range: 
    case_counts[age] = (cases[condition_columns]<=age).sum(axis=1)
    control_counts[age] = (controls[condition_columns] <= age).sum(axis=1)
    
#Convert to DataFrames and calculated the averages 
case_counts_df = pd.DataFrame(case_counts)
control_counts_df = pd.DataFrame(control_counts)

case_averages = case_counts_df.mean()
control_averages = control_counts_df.mean()
#control_averages = control_counts_df.mean()


#Plotting the figure 
plt.figure(figsize=(12,6))
plt.plot(case_averages, label='IBD Population', color='dodgerblue')
plt.fill_between(case_averages.index, case_averages, color='dodgerblue', alpha=0.4)
plt.plot(control_averages, label='Matched Controls', color='navy')
plt.fill_between(control_averages.index, control_averages, color='navy', alpha=0.4)

plt.xlabel('Age', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.ylabel('Average Number of Conditions', fontsize=14, fontweight='bold')
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.savefig(FILE_NAME, dpi=700)
plt.show()