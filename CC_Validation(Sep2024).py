#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 07:47:08 2024

@author: gepostill
"""


#Note - for privacy all file names are written as FILE_NAME to conceal the file path of data and figures on the laptop used to create the code

##############################
#Importing Packages   
##############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px -- we would need to get permission to install the package plotly

import xgboost as xgb

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, roc_curve, auc, log_loss, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, label_binarize, OneHotEncoder, TargetEncoder

#Calibration plots: 
from sklearn.calibration import calibration_curve

import joblib #used for saving the model 
import shap #used for the interpretation of model performance (SHapely Additive exPlanations)

#Removing warnings for clearner output
import warnings
warnings.filterwarnings('ignore')


#########################
#CREATE FUNCTIONS 
#########################



def eval_model(model_name, y_test, Y_pred):
    """
    Parameters
    --------------
    model_name: string of the model 
    
    Returns
    ------
    conf_matrix : Confusion MAtric 
    class_report : Classification Report 
    
    """
    #Confusion Matrix
    print(model_name+" Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, Y_pred)
    print(conf_matrix)
    
    #Classification Report
    print(" Classification Report:")
    class_report = classification_report(y_test, Y_pred)
    print(class_report)

    return conf_matrix, class_report



def auc_model(retrained_model, X_train, y_train, X_test, y_test, plot_name):
    """
    model_name: string of the model 
    
    This function creates a graph of AUC

    """
    
    #Predicting on the training dataset and computing the accuracy 
    Y_pred_train = retrained_model.predict(X_train)
    accuracy_score_retrain = accuracy_score(y_train, Y_pred_train) #Gets the accuracy score
    fpr_train, tpr_train, thresholds_train = roc_curve(Y_pred_train, y_train) #Gets the true positive and false positive rates of the model compared to label 
    roc_auc_train = auc(fpr_train, tpr_train)

    print("The accuracy of the optimized model on the training data: ")
    print(accuracy_score_retrain)

    #Predicting on the test 
    Y_pred_test = retrained_model.predict(X_test)
    accuracy_score_retest = accuracy_score(y_test, Y_pred_test)
    fpr_test, tpr_test, thresholds_test = roc_curve(Y_pred_test, y_test) #Gets the true positive and false positive rates of the model compared to label 
    roc_auc_test = auc(fpr_test, tpr_test)

    print("The accuracy of the optimized model on the testing data: ")
    print(accuracy_score_retest)
    
    #Ploting ROC Curve
    plt.figure(figsize=(6,4))
    plt.plot(fpr_train, tpr_train, color='dodgerblue', lw=2, label='Train AUC = {:.2f}'.format(roc_auc_train))
    plt.plot(fpr_test, tpr_test, color='navy', lw=2, label='Test AUC = {:.2f}'.format(roc_auc_test))
    #plt.plot([0,1], [0,1], color='gray', lw=, label="Random Guess") 
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    #plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right', fontsize=10)
    plt.savefig(plot_name, dpi=700)
    plt.show()

    return accuracy_score_retrain, accuracy_score_retest

#Create a function to bootstrap samples 
def bootstrap_sample(X, y): 
    n = len(X)
    idx = np.random.choice(n, size=n, replace=True)
    return X[idx], y.iloc[idx]  #using different formats because X is an array and y is a series 

#Create a funciton to calculate the confidence intervals 
def calculate_confidence_interval(metrics): 
    lower_bound = np.percentile(metrics, 2.5)
    upper_bound = np.percentile(metrics, 97.5)
    return lower_bound, upper_bound



#########################
#IMPORT DATAFRAMES
#########################

#IMPORTING THE DATAFRAMES THAT HAVE BEEN LABELLED WITH THE ASSIGNED CLUSTERS 
df_train_age = pd.read_csv(FILE_NAME)
df_train_age_M = pd.read_csv(FILE_NAME)
df_train_age_F = pd.read_csv(FILE_NAME)

#df_test_age = pd.read_csv(FILE_NAME)
#df_test_age_M = pd.read_csv(FILE_NAME)
#df_test_age_F = pd.read_csv(FILE_NAME)


#I MADE A MISTAKE ABOVE
#I need to split the sample size of the above because they need to learn the relationship in the same way 


training_full_data = pd.read_csv(FILE_NAME)
training_full_data_MALE = pd.read_csv(FILE_NAME)
training_full_data_FEMALE = pd.read_csv(FILE_NAME)


#splitting the data into testing and training data
df_train_age, df_test_age = train_test_split(training_full_data, test_size=0.2, random_state=30) 
df_train_age_M, df_test_age_M = train_test_split(training_full_data_MALE, test_size=0.2, random_state=30) 
df_train_age_F, df_test_age_F = train_test_split(training_full_data_FEMALE, test_size=0.2, random_state=30) 




########################################
#MULTICLASS PROBLEM
########################################

#############
#BOTH SEXES
##############

#Subsetting to the relevant columns (only binary yes/no of conditions and sex)
X_train = df_train_age.drop('Cluster',axis=1) #Remove the label from dataset;
X_train = X_train.drop('Unnamed: 0',axis=1) #Remove the label from dataset;
y_train = df_train_age['Cluster'] #Creating the label is: 'Premature'

X_test = df_test_age.drop('Cluster',axis=1) #Remove the label from dataset;
X_test = X_test.drop('Unnamed: 0',axis=1) #Remove the label from dataset;
y_test = df_test_age['Cluster'] #Creating the label is: 'Premature'

#Create a list of feature naems
feature_names_list = X_train.columns

#Creating a StandardScaler to normalize the training data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  #Scaling the training data 

#Creating a StandardScaler to normalize the testing data 
X_test_scaled = scaler.transform(X_test)    #Uing the same scaler as the trianing data  

#define the XGBoost classifier 
xgb_clf = xgb.XGBClassifier(objective='multi:softprob', num_class=3, use_label_encoder=False, eval_metric='mlogloss')
#xgb_clf = xgb.XGBClassifier(objective='binary:logisitic', num_class=3, use_label_encoder=False, eval_metric='mlogloss')

#set up the Randomized dSearchCV to find the best parameters 
param_grid = {
    'max_depth': [1, 3, 5],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 200, 300], 
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0], 
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], 
    "min_child_weight" : [1, 3, 5],           #Controls the Minimum sum of instance weight (hessian) needed in a child. It helps control overfitting 
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    } 

#set up the parameter grid for RandomizedSeacrhCV
random_search = RandomizedSearchCV(xgb_clf, param_distributions=param_grid, n_iter=50, scoring ='accuracy', n_jobs=1, cv=5, verbose=1, random_state=42)

#run the random search 
random_search.fit(X_train_scaled, y_train)

#Create dataframe of the parameters in the final model  
best_param = random_search.best_params_
df_params = pd.DataFrame(list(best_param.items()), columns=['Key','XGB'])
df_params.to_csv('/linux_home/gepostill/Files/u/gepostill/EXPORT/AGEcc_multiclass_params_XGB_BOTH.csv') #Export dataframe

#TRAIN THE MODEL WITH THE BEST PARAMETER 
best_model = random_search.best_estimator_


##  PERFORMANCE METRICS ## 

#Binarize the labels for AUC calculation (one-vs-rest approach)
y_test_binarized = label_binarize(y_test, classes=[0,1,2])

#Create predictions with model on testing data
Y_pred_XGB2 = best_model.predict(X_test_scaled) #Prediction with retrained model 
Y_pred_XGB2_proba = best_model.predict_proba(X_test_scaled)

#Confusion Matrix and Classification Report
XGB2_conf_matrix, XGB2_class_report = eval_model("XGBoost Model", y_test, Y_pred_XGB2)

#AUC - One-vs-Rest AUC for multi-class
auc_metric = roc_auc_score(y_test_binarized, Y_pred_XGB2_proba, multi_class='ovr')
print(f'The AUC of the model is {auc_metric}')

#Accuracy + AUC plot with the new retrained model
#acc_train_XGB2, acc_test_XGB2 = auc_model(best_model, X_train, y_train, X_test, y_test, 
#                                      FILE_NAME)

## BOOTSTRAP PERFORMANCE METRICS ## 

n_iterations = 1000 #Number of iterations 

#Lists to store the metrics 
bootstrap_accuracies = []
bootstrap_aucs = []
bootstrap_precisions = []
bootstrap_recalls = []
bootstrap_f1s = []

#Create a dictionary for the various classes to store metrics 
num_class=3 ## Setting it up like this so that the code can be re-used
class_metrics = {class_label: [] for class_label in range(num_class)}

#Iterate over the number of bootstrap samples 
for i in range(n_iterations): 
    #Bootstrap the data 
    X_test_bootstrap, y_test_bootstrap = bootstrap_sample(X_test_scaled, y_test)
        
    #Predict on the test set 
    y_pred_bootstrap = best_model.predict(X_test_bootstrap)
    y_test_bootstrap_proba = best_model.predict_proba(X_test_bootstrap) #Needed for AUC
    
    #Calculate performance metrics 
    accuracy = accuracy_score(y_test_bootstrap, y_pred_bootstrap)
    precision = precision_score(y_test_bootstrap, y_pred_bootstrap, average='weighted') #Weighting for the multiclass nature
    recall = recall_score(y_test_bootstrap, y_pred_bootstrap, average='weighted') #Weighting for the multiclass nature
    f1 = f1_score(y_test_bootstrap, y_pred_bootstrap, average='weighted') #Weighting for the multiclass nature

    #One-vs-Rest AUC for multi-class
    y_test_binarized_bootstrap = label_binarize(y_test_bootstrap, classes=[0,1,2])
    auc_bootstrap = roc_auc_score(y_test_binarized_bootstrap, y_test_bootstrap_proba, multi_class='ovr')

    #Store the metrics 
    bootstrap_accuracies.append(accuracy)
    bootstrap_aucs.append(auc_bootstrap)
    bootstrap_precisions.append(precision)
    bootstrap_recalls.append(recall)
    bootstrap_f1s.append(f1)
        
    #Generate the classification report for the bootstrapped test set --- alternative approach 
    report = classification_report(y_test_bootstrap, y_pred_bootstrap, output_dict=True)
    
    #Store the metrics for each class 
    for class_label in range(num_class):
        class_metrics[class_label].append({
            'precision': report[str(class_label)]['precision'],
            'recall': report[str(class_label)]['recall'],
            'f1_score': report[str(class_label)]['f1-score']
            })
            
 
###### Confidence intervals overall 
#Compute 95% CI for each metric & print results 
accuracy_ci = calculate_confidence_interval(bootstrap_accuracies)
print(f"95% CI for Accuracy: {accuracy_ci}")
precision_ci = calculate_confidence_interval(bootstrap_precisions)
print(f"95% CI for Precision: {precision_ci}")
recall_ci = calculate_confidence_interval(bootstrap_recalls)
print(f"95% CI for Recall: {recall_ci}")
f1_ci = calculate_confidence_interval(bootstrap_f1s)
print(f"95% CI for F1 Score: {f1_ci}")
auc_ci = calculate_confidence_interval(bootstrap_aucs)
print(f"95% CI for AUC: {auc_ci}")

###### Class Confidence Intervals  
class_confidence_intervals = {}
for class_label in range(num_class):
    precision_list = [metrics['precision'] for metrics in class_metrics[class_label]]
    recall_list = [metrics['recall'] for metrics in class_metrics[class_label]]
    f1_list = [metrics['f1_score'] for metrics in class_metrics[class_label]]
    
    precision_ci_class = calculate_confidence_interval(precision_list)
    recall_ci_class = calculate_confidence_interval(recall_list)
    f1_ci_class = calculate_confidence_interval(f1_list)

    #Store the confidence intervals for this class 
    class_confidence_intervals[class_label] = {
        'precision_ci_class': precision_ci_class, 
        'recall_ci_class': recall_ci_class, 
        'f1_ci_class': f1_ci_class, 
        }
    
    #Print the results for this class 
    print(f"Class {class_label} - 95% CI for Precision: {precision_ci_class}")
    print(f"Class {class_label} - 95% CI for Recall: {recall_ci_class}")
    print(f"Class {class_label} - 95% CI for F1 Score: {f1_ci_class}")
    
    

## EXPLAINABILITY ##

#Reverse scaling 
X_test_original = scaler.inverse_transform(X_test)
#Generate SHAP values 
class_names = ['Cluster 0 ', 'Cluster 1', 'Cluster 2']
CONDITIONS = ['Asthma', 'CHF', 'COPD', 'Myocardial_infarction', 'Hypertension', 'Arrhythmia', 'CCS', 'Stroke', 'Cancer', 'Dementia',
              'Renal_Disease', 'Diabetes', 'Osteoporosis', 'Rheumatoid_Arthritis', 'Oste_Arthritis', 'Mood_disorder', 'Other_Mental_disorder']

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test) #using the not-rescaled version

#Create an overall barplot 
shap.summary_plot(shap_values, X_test.values, plot_type='bar', class_names=class_names, feature_names=X_test.columns)
plt.savefig(FILE_NAME)
plt.show()

for i in [0,1,2]: 
    #Create an overall barplot 
    shap.summary_plot(shap_values[:, :, i], X_test.values, feature_names=X_test.columns)
    i = str(i)
    plt.savefig(f'FILE_NAME_{I}')
    plt.show()

    





#############
#MALE SEXES
##############

#Subsetting to the relevant columns (only binary yes/no of conditions and sex)
X_train_M = df_train_age_M.drop('Cluster',axis=1) #Remove the label from dataset;
X_train_M = X_train_M.drop('Unnamed: 0',axis=1) #Remove the label from dataset;
y_train_M = df_train_age_M['Cluster'] #Creating the label is: 'Premature'

X_test_M = df_test_age_M.drop('Cluster',axis=1) #Remove the label from dataset;
X_test_M = X_test_M.drop('Unnamed: 0',axis=1) #Remove the label from dataset;
y_test_M = df_test_age_M['Cluster'] #Creating the label is: 'Premature'

#Creating a StandardScaler to normalize the training data 
scaler_M = StandardScaler()
X_train_M_scaled = scaler_M.fit_transform(X_train_M)  #Scaling the training data 

#Creating a StandardScaler to normalize the testing data 
X_test_M_scaled = scaler_M.transform(X_test_M)    #Scaling the testing data 

#define the XGBoost classifier 
xgb_clf = xgb.XGBClassifier(objective='multi:softprob', num_class=3, use_label_encoder=False, eval_metric='mlogloss')

#set up the Randomized dSearchCV to find the best parameters 
param_grid = {
    'max_depth': [1, 3, 5],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 200, 300], 
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0], 
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], 
    "min_child_weight" : [1, 3, 5],           #Controls the Minimum sum of instance weight (hessian) needed in a child. It helps control overfitting 
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    } 

#set up the parameter grid for RandomizedSeacrhCV
random_search_M = RandomizedSearchCV(xgb_clf, param_distributions=param_grid, n_iter=50, scoring ='accuracy', n_jobs=1, cv=5, verbose=1, random_state=42)

#run the random search 
random_search_M.fit(X_train_M_scaled, y_train_M)

#Create dataframe of the parameters in the final model  
best_param_M = random_search_M.best_params_
df_params_M = pd.DataFrame(list(best_param_M.items()), columns=['Key','XGB'])
df_params_M.to_csv(FILE_NAME) #Export dataframe

#TRAIN THE MODEL WITH THE BEST PARAMETER 
best_model_M = random_search_M.best_estimator_

#Assessing the number of boosting rounds (trees) in the best model 
#boosting_rounds_M = best_model_M.best_iteration
#print(f"The XGBoost Model has {boosting_rounds_M} boosting rounds (trees)")


##  MODEL PERFORMANCE - MALES

#Model prediction and prediction probability 
Y_pred_XGB2_M = best_model_M.predict(X_test_M_scaled) #Prediction with retrained model 
Y_pred_XGB2_proba_M = best_model_M.predict_proba(X_test_M_scaled)

#Confusion Matrix and Classification Rport
XGB2_conf_matrix_M, XGB2_class_report_M = eval_model("XGBoost Model", y_test_M, Y_pred_XGB2_M)

#Binarize the labels for AUC calculation (one-vs-rest approach)
y_test_binarized_M = label_binarize(y_test_M, classes=[0,1,2])

#AUC - One-vs-Rest AUC for multi-class
auc_metric_M = roc_auc_score(y_test_binarized_M, Y_pred_XGB2_proba_M, multi_class='ovr')
print(f'The AUC of the male model is {auc_metric_M}')

## BOOTSTRAP PERFORMACNE METRICS - MALES ## 

n_iterations = 1000 #Number of iterations 

#Lists to store the metrics 
bootstrap_accuracies_M = []
bootstrap_aucs_M = []
bootstrap_precisions_M = []
bootstrap_recalls_M = []
bootstrap_f1s_M = []

#Create a dictionary for the various classes to store metrics 
num_class=3 ## Setting it up like this so that the code can be re-used
class_metrics_M = {class_label: [] for class_label in range(num_class)}

#Iterate over the number of bootstrap samples 
for i in range(n_iterations): 
    #Bootstrap the data 
    X_test_bootstrap_M, y_test_bootstrap_M = bootstrap_sample(X_test_M_scaled, y_test_M)
        
    #Predict on the test set 
    y_pred_bootstrap = best_model_M.predict(X_test_bootstrap_M)
    y_test_bootstrap_proba = best_model_M.predict_proba(X_test_bootstrap_M) #Needed for AUC
    
    #Calculate performance metrics 
    accuracy_M = accuracy_score(y_test_bootstrap_M, y_pred_bootstrap)
    precision_M = precision_score(y_test_bootstrap_M, y_pred_bootstrap, average='weighted') #Weighting for the multiclass nature
    recall_M = recall_score(y_test_bootstrap_M, y_pred_bootstrap, average='weighted') #Weighting for the multiclass nature
    f1_M = f1_score(y_test_bootstrap_M, y_pred_bootstrap, average='weighted') #Weighting for the multiclass nature

    #One-vs-Rest AUC for multi-class
    y_test_binarized_bootstrap = label_binarize(y_test_bootstrap_M, classes=[0,1,2])
    auc_bootstrap_M = roc_auc_score(y_test_binarized_bootstrap, y_test_bootstrap_proba, multi_class='ovr')

    #Store the metrics 
    bootstrap_accuracies_M.append(accuracy_M)
    bootstrap_aucs_M.append(auc_bootstrap_M)
    bootstrap_precisions_M.append(precision_M)
    bootstrap_recalls_M.append(recall_M)
    bootstrap_f1s_M.append(f1_M)
   
    #Generate the classification report for the bootstrapped test set --- alternative approach 
    report = classification_report(y_test_bootstrap_M, y_pred_bootstrap, output_dict=True)
    
    #Store the metrics for each class 
    for class_label in range(num_class):
        class_metrics_M[class_label].append({
            'precision': report[str(class_label)]['precision'],
            'recall': report[str(class_label)]['recall'],
            'f1_score': report[str(class_label)]['f1-score']
            })

#Compute 95% CI for each metric & print results 
accuracy_ci_M = calculate_confidence_interval(bootstrap_accuracies_M)
print(f"Male 95% CI for Accuracy: {accuracy_ci_M}")
precision_ci_M = calculate_confidence_interval(bootstrap_precisions_M)
print(f"Male 95% CI for Precision: {precision_ci_M}")
recall_ci_M = calculate_confidence_interval(bootstrap_recalls_M)
print(f"Male 95% CI for Recall: {recall_ci_M}")
f1_ci_M = calculate_confidence_interval(bootstrap_f1s_M)
print(f"Male 95% CI for F1 Score: {f1_ci_M}")
auc_ci_M = calculate_confidence_interval(bootstrap_aucs_M)
print(f"Male 95% CI for AUC: {auc_ci_M}")


###### Class Confidence Intervals  - MALES
class_confidence_intervals_M = {}
for class_label in range(num_class):
    precision_list = [metrics['precision'] for metrics in class_metrics_M[class_label]]
    recall_list = [metrics['recall'] for metrics in class_metrics_M[class_label]]
    f1_list = [metrics['f1_score'] for metrics in class_metrics_M[class_label]]
    
    precision_ci_class = calculate_confidence_interval(precision_list)
    recall_ci_class = calculate_confidence_interval(recall_list)
    f1_ci_class = calculate_confidence_interval(f1_list)

    #Store the confidence intervals for this class 
    class_confidence_intervals_M[class_label] = {
        'precision_ci_class': precision_ci_class, 
        'recall_ci_class': recall_ci_class, 
        'f1_ci_class': f1_ci_class, 
        }
    
    #Print the results for this class 
    print(f"Males Class {class_label} - 95% CI for Precision: {precision_ci_class}")
    print(f"Males Class {class_label} - 95% CI for Recall: {recall_ci_class}")
    print(f"Males Class {class_label} - 95% CI for F1 Score: {f1_ci_class}")
    
    
### EXPLAINABILTIY - MALES

#Reverse scaling 
X_test_original_M = scaler_M.inverse_transform(X_test_M_scaled)

#Generate SHAP values 
CONDITIONS = ['Asthma', 'CHF', 'COPD', 'Myocardial_infarction', 'Hypertension', 'Arrhythmia', 'CCS', 'Stroke', 'Cancer', 'Dementia',
              'Renal_Disease', 'Diabetes', 'Osteoporosis', 'Rheumatoid_Arthritis', 'Oste_Arthritis', 'Mood_disorder', 'Other_Mental_disorder']

explainer_M = shap.TreeExplainer(best_model_M)
shap_values_M = explainer.shap_values(X_test_M)

#Create an overall barplot 
shap.summary_plot(shap_values_M, X_test_M.values, plot_type='bar', class_names=class_names, feature_names=X_test_M.columns)
plt.savefig(FILE_NAME)
plt.show()

for i in [0,1,2]: 
    #Create an overall barplot 
    shap.summary_plot(shap_values_M[:, :, i], X_test_M.values, feature_names=X_test_M.columns)
    i = str(i)
    plt.savefig(FILE_NAME)
    plt.show()

    

##########################################################################################
#FEMALE SEXES - Not retraining model, just subsetting performance measures on females  
#########################################################################################

#Subsetting to the relevant columns (only binary yes/no of conditions and sex
X_train_F = df_train_age_F.drop('Cluster',axis=1) #Remove the label from dataset;
X_train_F = X_train_F.drop('Unnamed: 0',axis=1) #Remove the label from dataset;
y_train_F = df_train_age_F['Cluster'] #Creating the label 

X_test_F = df_test_age_F.drop('Cluster',axis=1) #Remove the label from dataset;
X_test_F = X_test_F.drop('Unnamed: 0',axis=1) #Remove the label from dataset;
y_test_F = df_test_age_F['Cluster'] #Model predicting cluster label 


#Creating a StandardScaler to normalize the training data 
scaler_F = StandardScaler()
X_train_F_scaled = scaler_F.fit_transform(X_train_F)  #Scaling the training data 

#Creating a StandardScaler to normalize the testing data 
X_test_F_scaled = scaler_F.transform(X_test_F)    #Scaling the testing data 

#define the XGBoost classifier 
xgb_clf = xgb.XGBClassifier(objective='multi:softprob', num_class=3, use_label_encoder=False, eval_metric='mlogloss')

#set up the Randomized dSearchCV to find the best parameters 
param_grid = {
    'max_depth': [1, 3, 5],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 200, 300], 
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0], 
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], 
    "min_child_weight" : [1, 3, 5],           #Controls the Minimum sum of instance weight (hessian) needed in a child. It helps control overfitting 
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    } 

#set up the parameter grid for RandomizedSeacrhCV
random_search_F = RandomizedSearchCV(xgb_clf, param_distributions=param_grid, n_iter=50, scoring ='accuracy', n_jobs=1, cv=5, verbose=1, random_state=42)

#run the random search 
random_search_F.fit(X_train_F_scaled, y_train_F)

#Create dataframe of the parameters in the final model  
best_param_F = random_search_F.best_params_
df_params_F = pd.DataFrame(list(best_param_F.items()), columns=['Key','XGB'])
df_params_F.to_csv(FILE_NAME) #Export dataframe

#TRAIN THE MODEL WITH THE BEST PARAMETER 
best_model_F = random_search_F.best_estimator_


##  MODEL PERFORMANCE - FEMALES

#Model prediction and prediction probability 
Y_pred_XGB2_F = best_model_F.predict(X_test_F_scaled) #Prediction with retrained model 
Y_pred_XGB2_proba_F = best_model_F.predict_proba(X_test_F_scaled)

#Confusion Matrix and Classification Report
XGB2_conf_matrix_F, XGB2_class_report_F = eval_model("XGBoost Model", y_test_F, Y_pred_XGB2_F)

#Binarize the labels for AUC calculation (one-vs-rest approach)
y_test_binarized_F = label_binarize(y_test_F, classes=[0,1,2])

#AUC - One-vs-Rest AUC for multi-class
auc_metric_F = roc_auc_score(y_test_binarized_F, Y_pred_XGB2_proba_F, multi_class='ovr')
print(f'The AUC of the female model is {auc_metric_F}')

## BOOTSTRAP PERFORMACNE METRICS - FEMALE ## 

n_iterations = 1000 #Number of iterations 

#Lists to store the metrics 
bootstrap_accuracies_F = []
bootstrap_aucs_F = []
bootstrap_precisions_F = []
bootstrap_recalls_F = []
bootstrap_f1s_F = []

#Create a dictionary for the various classes to store metrics 
num_class=3 ## Setting it up like this so that the code can be re-used
class_metrics_F = {class_label: [] for class_label in range(num_class)}

#Iterate over the number of bootstrap samples 
for i in range(n_iterations): 
    #Bootstrap the data 
    X_test_bootstrap_F, y_test_bootstrap_F = bootstrap_sample(X_test_F_scaled, y_test_F)
        
    #Predict on the test set 
    y_pred_bootstrap = best_model_F.predict(X_test_bootstrap_F)
    y_test_bootstrap_proba = best_model_F.predict_proba(X_test_bootstrap_F) #Needed for AUC
    
    #Calculate performance metrics 
    accuracy_F = accuracy_score(y_test_bootstrap_F, y_pred_bootstrap)
    precision_F = precision_score(y_test_bootstrap_F, y_pred_bootstrap, average='weighted') #Weighting for the multiclass nature
    recall_F = recall_score(y_test_bootstrap_F, y_pred_bootstrap, average='weighted') #Weighting for the multiclass nature
    f1_F = f1_score(y_test_bootstrap_F, y_pred_bootstrap, average='weighted') #Weighting for the multiclass nature

    #One-vs-Rest AUC for multi-class
    y_test_binarized_bootstrap = label_binarize(y_test_bootstrap_F, classes=[0,1,2])
    auc_bootstrap_F = roc_auc_score(y_test_binarized_bootstrap, y_test_bootstrap_proba, multi_class='ovr')

    #Store the metrics 
    bootstrap_accuracies_F.append(accuracy_F)
    bootstrap_aucs_F.append(auc_bootstrap_F)
    bootstrap_precisions_F.append(precision_F)
    bootstrap_recalls_F.append(recall_F)
    bootstrap_f1s_F.append(f1_F)
 
    #Generate the classification report for the bootstrapped test set --- alternative approach 
    report = classification_report(y_test_bootstrap_F, y_pred_bootstrap, output_dict=True)
    
    #Store the metrics for each class 
    for class_label in range(num_class):
        class_metrics_F[class_label].append({
            'precision': report[str(class_label)]['precision'],
            'recall': report[str(class_label)]['recall'],
            'f1_score': report[str(class_label)]['f1-score']
            })

#Compute 95% CI for each metric & print results 
accuracy_ci_F = calculate_confidence_interval(bootstrap_accuracies_F)
print(f"Female 95% CI for Accuracy: {accuracy_ci_F}")
precision_ci_F = calculate_confidence_interval(bootstrap_precisions_F)
print(f"Female 95% CI for Precision: {precision_ci_F}")
recall_ci_F = calculate_confidence_interval(bootstrap_recalls_F)
print(f"Female 95% CI for Recall: {recall_ci_F}")
f1_ci_F = calculate_confidence_interval(bootstrap_f1s_F)
print(f"Female 95% CI for F1 Score: {f1_ci_F}")
auc_ci_F = calculate_confidence_interval(bootstrap_aucs_F)
print(f"Female 95% CI for AUC Score: {auc_ci_F}")

###### Class Confidernce Intervals  
class_confidence_intervals_F = {}
for class_label in range(num_class):
    precision_list = [metrics['precision'] for metrics in class_metrics_F[class_label]]
    recall_list = [metrics['recall'] for metrics in class_metrics_F[class_label]]
    f1_list = [metrics['f1_score'] for metrics in class_metrics_F[class_label]]
    
    precision_ci_class = calculate_confidence_interval(precision_list)
    recall_ci_class = calculate_confidence_interval(recall_list)
    f1_ci_class = calculate_confidence_interval(f1_list)

    #Store the confidence intervals for this class 
    class_confidence_intervals_F[class_label] = {
        'precision_ci_class': precision_ci_class, 
        'recall_ci_class': recall_ci_class, 
        'f1_ci_class': f1_ci_class, 
        }
    
    #Print the results for this class 
    print(f"Females Class {class_label} - 95% CI for Precision: {precision_ci_class}")
    print(f"Females Class {class_label} - 95% CI for Recall: {recall_ci_class}")
    print(f"Females Class {class_label} - 95% CI for F1 Score: {f1_ci_class}")
    

### EXPLAINABILITY

#Reverse scaling 
X_test_original_F = scaler_F.inverse_transform(X_test_F_scaled)

#Generate SHAP values 
CONDITIONS = ['Asthma', 'CHF', 'COPD', 'Myocardial_infarction', 'Hypertension', 'Arrhythmia', 'CCS', 'Stroke', 'Cancer', 'Dementia',
              'Renal_Disease', 'Diabetes', 'Osteoporosis', 'Rheumatoid_Arthritis', 'Oste_Arthritis', 'Mood_disorder', 'Other_Mental_disorder']

explainer_F = shap.TreeExplainer(best_model_F)
shap_values_F = explainer_F.shap_values(X_test_original_F)

#Create an overall barplot 
shap.summary_plot(shap_values_F, X_test_F.values, plot_type='bar', class_names=class_names, feature_names=X_test_F.columns)
plt.savefig(FILE_NAME)
plt.show()

for i in [0,1,2]: 
    #Create an overall barplot 
    shap.summary_plot(shap_values_F[:, :, i], X_test_F.values, feature_names=X_test_F.columns)
    i = str(i)
    plt.savefig(f'FILE_NAME_{I}')
    plt.show()

    