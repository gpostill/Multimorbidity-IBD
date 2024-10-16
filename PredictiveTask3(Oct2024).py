#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 17:02:50 2023

@author: gepostill
"""


#Note - for privacy all file names are written as FILE_NAME to conceal the file path of data and figures on the laptop used to create the code

#################################
#Importing the necessary packages
#################################
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px -- we would need to get permission to install the package plotly

import xgboost as xgb

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, roc_curve, auc, log_loss, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, label_binarize, OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#Calibration plots: 
from sklearn.calibration import calibration_curve

import joblib #used for saving the model 
import shap #used for the interpretation of model performance (SHapely Additive exPlanations)

#Removing warnings for clearner output
import warnings
warnings.filterwarnings('ignore')


################################################
#Defining Functions    
################################################


def model_pipeline(categorical_columns, high_level_columns, initialized_model): 
    """
    Parameters
    ----------
    categorical_columns: LIST of categorical columns
    high_level_columns: LIST of columns with many categories
    initialized_model : MODEL (already initialized)

    Returns
    -------
    model : returns a Pipeline 
        Includes the transformation of columns and the model.
        
    
    NOTE: In this task, We forego the pipeline because we are not pre-processing the data(already pre-processed
                                                                                          
    """
    
    #Initialize a column transformer that will handle categorical data encoding
        #One-hot encoding is applied to the categorical columns specified in the list 'categorical_columns'
        #Target encoding is applied specifically to [XXX] columns 
    ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [c for c in categorical_columns]),
                            ('target_encoder', TargetEncoder(), high_level_columns)],
                           remainder='passthrough')
        
    #Creating a Pipeline: 
            #First, the data goes through the specified column transformations (ct)
            #Then, the transformed data is used to train or predict using the gradient Boosting model 
    model = Pipeline([('pre_process', ct),           # Pre-processing the data
                       ('model', initialized_model)])     #Training/predicting 
        
    #Displaying the pipeline architecture
    model  
    
    return model



def hyp_tuning(MODEL_NAME, model, param_dict, skf, X_train, y_train): 
    """
    This model tunes the hyperparameters with GridSearch(). GridSearchCV() needs to be implemented. 
    
    
    Parameters
    ----------
    model : Sk learn model - pretrained 
        DESCRIPTION.
    param_dist : DIC - distionary of the parameter values to be tested 
        Needs to be specific to the model being used.
    skf : Stratified KFold
    
    Returns
    -------
    retrained_model : TYPE
        DESCRIPTION.
        
    """
    
    #Setting up the GridSearchCV to find the best hyperparameters for the gradient Boosting model 
    grid_search = GridSearchCV(model, param_grid=param_dict, cv=skf)

    #Fitting the GridSearchcv on the training data
    grid_search.fit(X_train, y_train)
    
    #Updating the model's parameters with the best ones found from gridSearchCV
    best_model = grid_search.best_estimator_

    #Retraining the model with the best parameters on the training dataset 
    retrained_model = best_model.fit(X_train, y_train)
    
    #Print the parameters of the best model: 
    best_params = retrained_model.get_params()
    print(MODEL_NAME + " Best Model Parameters")
    print(best_params)


    return retrained_model, best_params



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
    
    #Plotting ROC Curve
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


def feature_importance(model, X, plt_name):
    """
    Parameters
    ----------
    model : SKLearn MODEL, retrained
    plt_name : STR of location where png will be saved

    Returns
    -------
    feature_imp : TYPE
        DESCRIPTION.

    """
    feature_imp = model.feature_importances_
    
    #Get the names of the features 
    feature_names = X.columns # the format is not as nice so i will create improved format below
    
    #sort the features based on their importance 
    sorted_indices = np.argsort(feature_imp)
    
    #Plot the feature importances 
    plt.figure(figsize=(10,6))
    bars = plt.bar(range(len(feature_imp)), feature_imp[sorted_indices], color='navy')
    plt.xticks(range(len(feature_imp)), np.array(feature_names)[sorted_indices], rotation=70)
    plt.ylabel('Feature Importance')
    
    for bar, importance in zip(bars, feature_imp[sorted_indices]):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.005, '{:.2f}'.format(importance), ha='center', va='bottom')
    
    plt.savefig(plt_name)
    plt.show()
    
    return feature_imp



#Create a function to bootstrap samples 
def bootstrap_sample(X, y): 
    n = len(X)
    idx = np.random.choice(n, size=n, replace=True)
    return X[idx], y.iloc[idx]  #using different formats because X is an array and y is a series 



#Create a function to calculate the confidence intervals 
def calculate_confidence_interval(metrics): 
    lower_bound = np.percentile(metrics, 2.5)
    upper_bound = np.percentile(metrics, 97.5)
    return lower_bound, upper_bound




## BOOTSTRAP PERFORMANCE METRICS ## 
def bootstrapCI(best_model, X2_test, y2_test, num_class=2 , n_iterations=1000): 
    
    #Lists to store the metrics 
    bootstrap_accuracies = []
    bootstrap_aucs = []
    bootstrap_precisions = []
    bootstrap_recalls = []
    bootstrap_f1s = []
    
    #Create a dictionary for the various classes to store metrics 
    class_metrics = {class_label: [] for class_label in range(num_class)}
    
    #Iterate over the number of bootstrap samples 
    for i in range(n_iterations): 
        #Bootstrap the data 
        X_test_bootstrap, y_test_bootstrap = bootstrap_sample(X2_test, y2_test)
            
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
        
    return accuracy_ci, precision_ci, recall_ci, f1_ci, auc_ci, class_confidence_intervals





################################################
#Importing Data   
################################################

#Importing the data -- all conditionals all ages 
data = pd.read_csv(FILE_NAME)


#Copying the original dataframe and selecting the columns included in the prediction 
df = data.copy()

#`Subset to only IBD patients
df = df[df["Cohort"]=="Case"]

#Subsetting to the relevant columns (only binary yes/no of conditions and sex)
df = df[['ID','Premature','sex','Asthma','CHF','COPD','Myocardial_infarction', 'Hypertension','Arrhythmia','CCS','Stroke','Cancer','Dementia',
         'Renal_Disease','Diabetes','Osteoporosis','Rheumatoid_Arthritis','Osteoarthritis','Mood_disorder','Other_Mental_disorder',
         'Age_Asthma','Age_CHF','Age_COPD','Age_Myocardial_infarction','Age_Hypertension','Age_Arrhythmia','Age_CCS','Age_Stroke','Age_Cancer','Age_Dementia',
         'Age_Renal_Disease','Age_Diabetes','Age_Osteoporosis','Age_Rheumatoid_Arthritis','Age_OsteoArthritis','Age_Mood_disorder','Age_Other_Mental_disorder']]

#Encoding the categorical variables 
df = df.replace({"No" : 0, "Yes" : 1})
df = df.replace({"M" : 0, "F" : 1})

print("The number of patients included in the model")
print(len(df))


################################################
#Importing and splitting data Data Splitting 80:20  
################################################
#Using the same testing data as the other tasks

#Import the training and testing data to get the ID values
df_train = pd.read_csv(FILE_NAME)
df_test = pd.read_csv(FILE_NAME)

#Create a list of trianing and testing IDs
training_IDs = df_train['ID'].to_list()
testing_IDs = df_test['ID'].to_list()


#Subsetting to the relevant columns (only binary yes/no of conditions and sex)
df2 = df[['ID','Premature','sex',
         'Age_Asthma','Age_CHF','Age_COPD','Age_Myocardial_infarction','Age_Hypertension','Age_Arrhythmia','Age_CCS','Age_Stroke','Age_Cancer','Age_Dementia',
         'Age_Renal_Disease','Age_Diabetes','Age_Osteoporosis','Age_Rheumatoid_Arthritis','Age_OsteoArthritis','Age_Mood_disorder','Age_Other_Mental_disorder']]

X2 = df2.drop('Premature',axis=1) #Remove the label from dataset;
y2 = df2['Premature'] #Creating the label is: 'Premature'

#splitting the data into testing and training data
#X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=30) 

df2_train = df2[df2['ID'].isin(training_IDs)] #re-writing this with the pre-defined training value 
df2_test = df2[df2['ID'].isin(testing_IDs)] #re-writing this with the the pre-defined testing value 

#Remove the label from dataset 
X2_train = df2_train.drop('ID',axis=1) #The label is: 'Premature'
X2_train = X2_train.drop('Premature',axis=1) #The label is: 'Premature'
y2_train = df2_train['Premature']

#Remove the label from dataset 
X2_test = df2_test.drop('ID',axis=1) #The label is: 'Premature'
X2_test = X2_test.drop('Premature',axis=1) #The label is: 'Premature'
y2_test = df2_test['Premature']

#Remove ID from X2
X2 = X2.drop('ID',axis=1) #Remove the label from dataset;


##################
#Creating a list of male and female indices - to be used later when evaluating model performance 
male_indices = list(X2_test[X2_test['sex'] == 0].index)
female_indices = list(X2_test[X2_test['sex'] == 1].index)

#Creating a StandardScaler to normalize the data 
scaler = StandardScaler()
X2_train = scaler.fit_transform(X2_train)  #Scaling the training data 
X2_test = scaler.transform(X2_test)    #Scaling the testing data 

#Initializing stratification of dataset for testing the models 
#Using StratifiedKFold for cross-validation, ensuring each fold has the same proportion of observations with each target value 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)


      
################################################
#XGBoost 
################################################

#Importing saved model if applicable:
#retrained_XGB2 = joblib.load(FILE_NAME)

#Initializing the XGBoost Classifier with base parameters  
xgb_classifier2 = xgb.XGBClassifier()

#Training the Gradient Boosting model on the training 
model_xgb2 = xgb_classifier2.fit(X2_train, y2_train) 

#Predicting on the training dataset and computing the accuracy 
Y_pred_xgb2 = model_xgb2.predict(X2_test)
accuracy_XGB2 = accuracy_score(y2_test, Y_pred_xgb2)
print("The accuracy score of the original XGBoost model: " + str(accuracy_XGB2))

#Defining the hyperparameters to be tuned using GridSearchCV
param_dict_XGB = {"learning_rate" : [0.01, 0.1, 0.2],       #Controls the contribution of each weak learner 
                  "n_estimators" : [50, 100, 200],           #Number of boosting stages to be run
                  "max_depth" : [1, 5, 10],                    #Controls the depth of each tree in the ensemble 
                  "min_child_weight" : [1, 3, 5],           #Controls the Minimum sum of instance weight (hessian) needed in a child. It helps control overfitting 
                  #"gamma" : [0, 0.1, 0.2],                   #Minimum loss reduction required to make a further partition on a leaf node (encourages pruning of the tree)
                  #"subsample" : [0.6, 0.8, 1.0],            #Fraction of samples used for training each tree (prevents overfitting by introducing randomness)
                  #"reg_lambda" : [0, 3, 5],                  #L2 regularization (helps prevent overfitting)
                  #"reg_alpha" : [0, 0.1, 0.5]              #L1 regularization (helps prevent overfitting)
                     } 
        #Note I chose to omit "col_sample_bytree" - similar effect as subsample but applies to the features at each level of the tree - due to the computational cost / time 
        #Omitted scale pos weights because the classes are not imbalanced 

exit() 

retrained_XGB2, XGB2_params = hyp_tuning("XGBoost", model_xgb2, param_dict_XGB, skf, X2_train, y2_train)

#Create dataframe of the parameters in the final model  
df_XGB2_params = pd.DataFrame(list(XGB2_params.items()), columns=['Key','XGB'])
#Export dataframe of the improved parameters
df_XGB2_params.to_csv(FILE_NAME)

exit()


################################################
#Importing the model  
################################################

retrained_XGB2 = joblib.load(FILE_NAME)

exit()

################################################
#ALTERNATIVELY - Retraining with the best paramters  
################################################

#Importing the optimal parameters
df_params = pd.read_csv(FILE_NAME, encoding='utf-8-sig')
#Converting the DataFrame to a dictionary
xgb_params = dict(zip(df_params['Key'], df_params['XGB']))

#Alternatively - Defining the hyperparameters to be used
xgb_params = {"enable_categorical" : False,
                  "gamma" : 0.2, 
                  "learning_rate" : 0.1,
                  "max_depth" : 10,
                  "min_child_weight" : 3,
                  "n_estimators" : 50, 
                  "reg_alpha" : 0,
                  "reg_lambda" : 0,
                  "subsample" : 0.6}
        #Note I chose to omit "col_sample_bytree" - similar effect as subsample but applies to the features at each level of the tree - due to the computational cost / time 
        #Omitted scale pos weights because the classes are not imbalanced 


#Specifying the model 
model = xgb.XGBClassifier(**xgb_params)

#Training the model with the optimal parameters
retrained_XGB2 = model.fit(X2_train, y2_train) 

################################################
#Evaluating the model 
################################################


#Confusion Matrix and Classification Rport
Y_pred_XGB2 = retrained_XGB2.predict(X2_test) #Prediction with retrained model 
XGB2_conf_matrix, XGB2_class_report = eval_model("XGBoost Model", y2_test, Y_pred_XGB2)

#Accuracy + AUC plot with the new retrained model
acc_train_XGB2, acc_test_XGB2 = auc_model(retrained_XGB2, X2_train, y2_train, X2_test, y2_test, FILE_NAME)

#Assessing the accuracy across the folds - prior to optimization of parameters
cv_XGB2_results = cross_val_score(retrained_XGB2, X2, y2, cv=skf, scoring='accuracy')
print("Cross Validation Accuracy Scores of Optimized XGBoost Model: ")
print(cv_XGB2_results)

#Bootstrap confidence intervals 
accuracy_ci, precision_ci, recall_ci, f1_ci, auc_ci, class_confidence_intervals = bootstrapCI(retrained_XGB2, X2_test, y2_test, num_class=2 , n_iterations=1000)



################################################
#Evaluating the model - by SEX
################################################

#Creating a dataframe for the analysis - this allows use of the actual column names 
X2_test_df = pd.DataFrame(X2_test, index=y2_test.index, columns=list(X2.columns))

#Split the dataset by sex 
    ##Males
mask = X2_test_df.index.isin(male_indices)
X2_test_male = X2_test_df[mask] #For features
X2_test_male = X2_test_male.to_numpy()
y2_test_male = y2_test[mask]  #For labels 
print('Number of males in testing data: ')
print(len(male_indices))

    ##Females
mask = X2_test_df.index.isin(female_indices)
X2_test_female = X2_test_df[mask] #For features
X2_test_female = X2_test_female.to_numpy()
y2_test_female = y2_test[mask]  #For labels 
print('Number of females in testing data: ')
print(len(female_indices))


#Evaluate the model performance for male group - Confusion Matrix and Classification Rport 
Y_pred_test_M = retrained_XGB2.predict(X2_test_male) #predicting outcomes for males only 
XGB2_conf_matrix_MALE, XGB2_class_report_MALE = eval_model("XGBoost Model", y2_test_male, Y_pred_test_M)

print("The accuracy of the optimized model on male testing data: ")
accuracy_score_male = accuracy_score(y2_test_male, Y_pred_test_M)
print(accuracy_score_male)

print("Confusions matrix for male group:")
print(XGB2_conf_matrix_MALE)

print("Classification report for male group:")
print(XGB2_conf_matrix_MALE)


#Bootstrap confidence intervals 
accuracy_ci_M, precision_ci_M, recall_ci_M, f1_ci_M, auc_ci_M, class_confidence_intervals_M = bootstrapCI(retrained_XGB2, X2_test_male, y2_test_male, num_class=2 , n_iterations=1000)

#########

#Evaluate the model performance for FEMALE group - Confusion Matrix and Classification Rport 
Y_pred_test_F = retrained_XGB2.predict(X2_test_female) #predicting outcomes for males only 
XGB2_conf_matrix_FEMALE, XGB2_class_report_FEMALE = eval_model("XGBoost Model", y2_test_female, Y_pred_test_F)

print("The accuracy of the optimized model on female testing data: ")
accuracy_score_female = accuracy_score(y2_test_female, Y_pred_test_F)
print(accuracy_score_female)

print("Confusions matrix for female group:")
print(XGB2_conf_matrix_FEMALE)

print("Classification report for female group:")
print(XGB2_conf_matrix_FEMALE)

#Bootstrap confidence intervals 
accuracy_ci_F, precision_ci_F, recall_ci_F, f1_ci_F, auc_ci_F, class_confidence_intervals_F = bootstrapCI(retrained_XGB2, X2_test_female, y2_test_female, num_class=2 , n_iterations=1000)





##Plotting ROC Curve - for males and females on one plot
fpr_male, tpr_male, thresholds_male = roc_curve(Y_pred_test_M, y2_test_male) #Gets the true positive and false positive rates of the model compared to label 
roc_auc_male = auc(fpr_male, tpr_male)

fpr_female, tpr_female, thresholds_female = roc_curve(Y_pred_test_F, y2_test_female) #Gets the true positive and false positive rates of the model compared to label 
roc_auc_female = auc(fpr_female, tpr_female)


plt.figure(figsize=(6,4))
plt.plot(fpr_male, tpr_male, color='navy', lw=2, label='Male Testing Data AUC = {:.2f}'.format(roc_auc_male))
plt.plot(fpr_female, tpr_female, color='dodgerblue', lw=2, label='Female Testing Data AUC = {:.2f}'.format(roc_auc_female))
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
#plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right',fontsize=10)
plt.savefig('/linux_home/gepostill/Files/u/gepostill/EXPORT/Task3B_ROC_MALE_FEMALE.png', dpi=700)
plt.show()



#Model Calibration Plots  - across male and female

#Calculate the probability of positive class 
y_proba_male = retrained_XGB2.predict_proba(X2_test_male)[:, 1] #Note this df uses the original indices
y_proba_female = retrained_XGB2.predict_proba(X2_test_female)[:, 1] #Note this df uses the original indices

#Create a calibration curve 
prob_true_M, prob_pred_M = calibration_curve(y2_test_male, y_proba_male, n_bins=10)
prob_true_F, prob_pred_F = calibration_curve(y2_test_female, y_proba_female, n_bins=10)

#Plot calibration curve
plt.plot(prob_pred_M, prob_true_M, marker = 'd', color='navy', linestyle = '-', label="Males")
plt.plot(prob_pred_F, prob_true_F, marker = 'o', color='dodgerblue', linestyle = '-', label="Females")
plt.plot([0,1], [0,1], linestyle='--', color='gray', label="Hypothetical perfect calibration") #Diagonal line representing perfect calibration 
plt.xlabel('Predicted probability')
plt.ylabel('Empirical probability')
plt.title('Calibration Plot')
plt.legend(loc='lower right')
plt.savefig(FILE_NAME)
plt.show()


################################################
#Feature Importance
################################################

#GINI IMPORTANCE 

#Feature Importance of Final RF Model 
feature_imp_XGB2 = feature_importance(retrained_XGB2, X2, FILE_NAME)
df_feature_imp_XGB2 = pd.DataFrame(feature_imp_XGB2)
df_feature_imp_XGB2['Condition'] = X2.columns
df_feature_imp_XGB2.to_csv(FILE_NAME)

#SHAP ANALYSIS

#Note on the SHAP analysis - the features that are still scaled. This should not affect 

#Create explainer
explainer = shap.Explainer(retrained_XGB2)
shap_values = explainer.shap_values(X2_test_df)

#Visualize the SHAP Values
shap.summary_plot(shap_values, X2_test_df)
plt.savefig(FILE_NAME)
plt.show()

#Could consider feature importance of individual prediction instances, particularly errors... 


################################################
#Model Calibration Plots  
################################################

#Calculate the probability of positive class 
y_pred_proba = retrained_XGB2.predict_proba(X2_test_df)[:, 1] #Note this df uses the original indices

#Create a calibration curve 
prob_true, prob_pred = calibration_curve(y2_test, y_pred_proba, n_bins=10)

#Plot calibration curve
plt.plot(prob_pred, prob_true, marker = 'o', linestyle = '-')
plt.plot([0,1], [0,1], linestyle='--', color='gray') #Diagonal line representing perfect calibration 
plt.xlabel('Predicted probability')
plt.ylabel('Empirical probability')
plt.title('Calibration Plot')
plt.savefig(FILE_NAME)
plt.show()


################################################
#ERROR ANALYSIS  
################################################

#Print the confusion Matrix 
print(XGB2_conf_matrix)

#Re-scale the data back to its original scale
X2_test_original_scale = scaler.inverse_transform(X2_test) ## re-scaling the array 
X2_test_original_scale = pd.DataFrame(X2_test_original_scale, index=y2_test.index, columns=list(X2.columns)) # turning back to a dataframe

#Add a column with predicition and true labels
X2_test_original_scale["Prediction"] = Y_pred_XGB2
X2_test_original_scale["true_label"] = y2_test

#Create a new column indicating false positives and false negatives 
X2_test_original_scale['Prediction_Type'] = 'Correct' #Start with all prediction as correct
X2_test_original_scale.loc[(Y_pred_XGB2 == 1) & (y2_test == 0), 'Prediction_Type'] = 'False Positive'
X2_test_original_scale.loc[(Y_pred_XGB2 == 0) & (y2_test == 1), 'Prediction_Type'] = 'False Negative'

#Crete a new column that indicates prediction error 
correct_prediction = (Y_pred_XGB2 == y2_test)
X2_test_original_scale['Correct_Prediction'] = 'Incorrect'
X2_test_original_scale.loc[correct_prediction, 'Correct_Prediction'] = 'Correct'


#Subset to only errors
X2_test_errors = X2_test_original_scale[X2_test_original_scale['Correct_Prediction'] == 'Incorrect']
X2_test_FP = X2_test_original_scale[X2_test_original_scale['Prediction_Type'] == 'False Positive']
X2_test_FN = X2_test_original_scale[X2_test_original_scale['Prediction_Type'] == 'False Negative']


#Prevalence of each of Chronic Conditions 
#Note: if age is missing - not diagnosed
missing_counts_errors = X2_test_errors.isna().sum()
missing_counts_FP = X2_test_FP.isna().sum()
missing_counts_FN = X2_test_FN.isna().sum()

#Premature - all
prem_error = X2_test_errors['true_label'].value_counts().reset_index() #Creating a count for the variable 
prem_error['Percent'] = prem_error['count'] / len(X2_test_errors) * 100 #creating the percent for the variable

#Premature - FP 
prem_FN = X2_test_FN['true_label'].value_counts().reset_index() #Creating a count for the variable 
prem_FN['Percent'] = prem_FN['count'] / len(X2_test_FN) * 100 #creating the percent for the variable

#Premature - FP 
prem_FP = X2_test_FP['true_label'].value_counts().reset_index() #Creating a count for the variable 
prem_FP['Percent'] = prem_FP['count'] / len(X2_test_FP) * 100 #creating the percent for the variable

#Note for the sex analysis 0: Male and 1:Female
#Sex - all
sex_error = X2_test_errors['sex'].value_counts().reset_index() #Creating a count for the variable 
sex_error['Percent'] = sex_error['count'] / len(X2_test_errors) * 100 #creating the percent for the variable

#Sex - FP 
sex_FN = X2_test_FN['sex'].value_counts().reset_index() #Creating a count for the variable 
sex_FN['Percent'] = sex_FN['count'] / len(X2_test_FN) * 100 #creating the percent for the variable

#Sex - FP 
sex_FP = X2_test_FP['sex'].value_counts().reset_index() #Creating a count for the variable 
sex_FP['Percent'] = sex_FP['count'] / len(X2_test_FP) * 100 #creating the percent for the variable


#Age of each chronic condition
#Subsetting to the relevant columns (only binary yes/no of conditions and sex)
ages = ['Age_Asthma','Age_CHF','Age_COPD','Age_Myocardial_infarction','Age_Hypertension','Age_Arrhythmia','Age_CCS','Age_Stroke','Age_Cancer','Age_Dementia',
         'Age_Renal_Disease','Age_Diabetes','Age_Osteoporosis','Age_Rheumatoid_Arthritis','Age_OsteoArthritis','Age_Mood_disorder','Age_Other_Mental_disorder']

df_age_train = pd.DataFrame(columns=['variable', 'mean', 'sd', 'median', 'q1', 'q3'])

for age in ages:
    mean = X2_test_errors[age].mean()
    sd = X2_test_errors[age].std()
    median = X2_test_errors[age].median()
    q1 = X2_test_errors[age].quantile(0.25)
    q3 = X2_test_errors[age].quantile(0.75)

    #Add row to list
    print(age)
    print(f"mead: {mean},   sd: {sd},   median {median}, IQR: {q1} - {q3}")


#Plotting the features of outliers 
feature_list = ['sex', 'Age_Asthma', 'Age_CHF', 'Age_COPD', 'Age_Myocardial_infarction',
                'Age_Hypertension', 'Age_Arrhythmia', 'Age_CCS', 'Age_Stroke',
                'Age_Cancer',  'Age_Renal_Disease', 'Age_Diabetes',
                'Age_Osteoporosis', 'Age_Rheumatoid_Arthritis', 'Age_OsteoArthritis',
                'Age_Mood_disorder', 'Age_Other_Mental_disorder', 'Age_Dementia']

#Define custom bin edges 
bin_edge = np.arange(0, 65, 5)

for feature in feature_list: 
    plt.figure(figsize=(8,6))
    plt.xlim(0,60)
    sns.histplot(X2_test_errors[feature], kde=True, color='red', bins=bin_edge, label='Outliers')
    sns.histplot(X2_test_original_scale[feature], kde=True, bins=bin_edge, color='blue', label='All Data')
    #plt.title(f'Distribution of {feature} for Outliers vs. All Data')
    plt.xlabel(feature)
    
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'FILE_NAME_{feature}.png', dpi=700)
    plt.show()


#Consider analysis of false negatives vs. false positives.... 


################################################
#Exporting the model
################################################

joblib.dump(retrained_XGB2, FILE_NAME)
