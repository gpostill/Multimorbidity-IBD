#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:32:11 2023

@author: gepostill
"""


#Importing the necessary packages 
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
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

import joblib #used for saving the model 
import shap #used for the interpretation of model performance (SHapely Additive exPlanations)

#Removing warnings for clearner output
import warnings
warnings.filterwarnings('ignore')


################################################
#Defining Functions    
################################################


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
    

#Create a function to bootstrap samples 
def bootstrap_sample(X, y): 
    n = len(X)
    idx = np.random.choice(n, size=n, replace=True)
    return X.iloc[idx], y.iloc[idx]  #using different formats because X is an array and y is a series 


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
    
    #Create a doctionary for the various classes to store metrics 
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
    
        #AUC 
        auc_bootstrap = roc_auc_score(y_test_bootstrap, y_test_bootstrap_proba[:, 1]) #passing only the positive class
    
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
                
     
    ###### Confidernce intervals overall 
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
    
    
    ###### Class Confidernce Intervals  
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
data = pd.read_csv('/sasroot/projects/life/p0904.161.000/level3/gepostill/cleanedfile.csv')


#Copying the original dataframe ans selecting the columns included in the prediction 
df = data.copy()

#`Subset to only IBD patients
df = df[df["Cohort"]=="Case"]

#Subsetting to the relevant columns (only binary yes/no of conditions and sex)
df = df[['ID','Premature','sex','Asthma','CHF','COPD','Myocardial_infarction', 'Hypertension','Arrythmia','CCS','Stroke','Cancer','Dementia',
         'Rental_Disease','Diabetes','Osteoporosis','Rheumatoid_Arthritis','Oste_Arthritis','Mood_disorder','Other_Mental_disorder']]

#Encoding the categorical variables 
df = df.replace({"No" : 0, "Yes" : 1})
df = df.replace({"M" : 0, "F" : 1})

print("The number of patients included in the model")
print(len(df))


################################################
#Data Splitting 80:20  
################################################

#Remove the label from dataset 
X = df.drop('Premature',axis=1) #The label is: 'Premature'
y = df['Premature']

#splitting the data into testing and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30) 


#export train and testing data for use in other predictive tasks
df_train = X_train.join(y_train)
df_train.to_csv('/sasroot/projects/life/p0904.161.000/level3/gepostill/training_data.csv', index = False)

df_test = X_test.join(y_test)
df_test.to_csv('/sasroot/projects/life/p0904.161.000/level3/gepostill/testing_data.csv', index = False)

#Create a list of trianing and testing IDs
training_IDs = X_train['ID'].to_list()
testing_IDs = X_test['ID'].to_list()

 
#Remove the ID column from the dataframe
X_train.drop(columns=['ID'], inplace=True)
X_test.drop(columns=['ID'], inplace=True)

#Creating a StandardScaler to normalize the data -- not needed because the whole dataset is binary (0/1) -- this is a placeholder to show consideration
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#Initializing stratification of dataset for testing the models 
#Using StratifiedKFold for cross-validation, ensuring each fold has the same proportion of observations with each target value 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)

################################################
#Logistic Regression 
################################################

#Initializing the LR model 
LR_model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
#NOTE: In this task, We forego the pipeline because we are not pre-processing the data(already pre-processed

#Training the pipeline model on the training data
trained_LR_model = LR_model.fit(X_train, y_train)

#Predicting on the training dataset and computing the accuracy 
Y_pred_LR = trained_LR_model.predict(X_test)
accuracy_og_LR = accuracy_score(y_test, Y_pred_LR)
print("The accuracy score of the original Logistic Regression model: ")
print(accuracy_og_LR)
 

        ################################
        #Hyperparameter Tuning 
        ################################
 
#Defining the hyperparameters to be tuned using GridSearchCV
param_dict_LR = {"C" : [0.001, 0.01, 0.1, 1, 10, 100, 1000], #This is the regularizing parameter 
              "penalty" : ["l1", "l2", "elasticnet", "none"], 
              "solver" : ["liblinear","newton-cg","sag"]}

retrained_LR_model, LR_params = hyp_tuning("Logsitic Regression", trained_LR_model, param_dict_LR, skf, X_train, y_train)

        ################################
        #Evaluating the model 
        ################################

#Evaluating the new accuracy 
Y_pred_LR = retrained_LR_model.predict(X_test)
accuracy_LR = accuracy_score(y_test, Y_pred_LR)
print("The accuracy score of the optimized Logistic Regression model: ")
print(accuracy_LR)

#Confusion Matrix and Classification Rport
LR_conf_matrix, LR_class_report = eval_model("Logistic Regression", y_test, Y_pred_LR)

#Bootstrap confidence intervals 
accuracy_ci_LP, precision_ci_LR, recall_ci_LR, f1_ci_LR, auc_ci_LR, class_confidence_intervals_LR = bootstrapCI(retrained_LR_model, X_test, y_test, num_class=2 , n_iterations=1000)

#Measuring the log-loss of model 
#Log-loss measures how well the predicted probabilites match the true distribuiton of the class; 
y_prob_LR = retrained_LR_model.predict_proba(X_test)
logloss = log_loss(y_test, y_prob_LR)
print("Log-Loss of Final Logistic Regression Model")
print(logloss)
#Lower log-loss values indicate better agreement between predicted and actual probabilities (perfect model has a log-loss of 0)
    #Interpretation: 
            #Values very close to zero indicate great performance (or overfitting)
            #Values between 0-1 indicate intermediate log-loss --> reasonable predicitons (possible room for improvement)
            #Values >1 may indicate poor-calibration



        ################################
        #Feature Importance of LR Model (Coefficients)
        ################################

#Identifying the coefficients and corresponding feature names (column names)
coeffficients = retrained_LR_model.coef_[0]
feature_names = X_test.columns 

#Creating a sorted DF of features coefficients 
feature_imp_LR = pd.DataFrame({'Feature':feature_names, 'Coefficient': coeffficients})

#Adding another column of absolute coefficient & sorting on this (magnitude of strenth with predicing prem. mortality)
feature_imp_LR['Absolute Coefficient'] = feature_imp_LR['Coefficient'].abs()
feature_imp_LR = feature_imp_LR.sort_values(by='Absolute Coefficient', ascending=False)

#Get the names of the features 
feature_imp_LR.to_csv(FILE_PATH)

#Plot the importances - alternative plot layout
plt.figure(figsize=(8,6))
sns.barplot(x='Absolute Coefficient', y='Feature', data=feature_imp_LR)
plt.title('Feature Importances from Logistic Regression (Task 1) - Coefficients')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(FILE_PATH, dpi=700)
plt.show()




################################################
#Random Forest Classifier 
################################################

#Initializing the RF Classifier with 
random_forest = RandomForestClassifier() 

#Training the pipeline model on the training data
trained_model_RF = random_forest.fit(X_train, y_train)

#Predicting on the training dataset and computing the accuracy 
Y_pred_RF = trained_model_RF.predict(X_test)
accuracy_original_RF = accuracy_score(y_test, Y_pred_RF)
print("The accuracy score of the original Random forest model (testing data): " + str(accuracy_original_RF))
 
#Defining the hyperparameters to be tuned using GridSearchCV
param_dict_RF = {"n_estimators" : [50, 100, 200, 500], 
                 "max_depth" : [5, 10, 15, 18], 
                 "min_samples_leaf" : [5, 15, 35, 50], 
                 "min_samples_split" : [2,5,10,20], 
                 "bootstrap" : [True,False]}
        #Are there other parameters I shoud use?

retrained_RF, RF_params = hyp_tuning("Random Forest", trained_model_RF, param_dict_RF, skf, X_train, y_train)


        ################################
        #Evaluating the model 
        ################################

#Accuracy of retrained model 
Y_pred_RF = retrained_RF.predict(X_test)
accuracy_RF = accuracy_score(y_test, Y_pred_RF)
print("\n The accuracy score of the retrained Random Forest Model (testing data): ")
print(accuracy_LR)

#Confusion Matric and Classification Rport
RF_conf_matrix, RF_class_report = eval_model("Random Forest", y_test, Y_pred_RF)

#Bootstrap confidence intervals 
accuracy_ci_RF, precision_ci_RF, recall_ci_RF, f1_ci_RF, auc_ci_RF, class_confidence_intervals_RF = bootstrapCI(retrained_RF, X_test, y_test, num_class=2 , n_iterations=1000)

#Determining the size of the final forest
RF_forest_size = len(retrained_RF.estimators_)
print(f"The Random Forest has {RF_forest_size} trees")

#Feature Importance of Final RF Model 
importances_RF = retrained_RF.feature_importances_
importances_RF = pd.Series(importances_RF, index=X_test.columns).sort_values(ascending=False)
df_feature_imp_RF = pd.DataFrame(importances_RF)
df_feature_imp_RF['Condition'] = X_test.columns
df_feature_imp_RF.to_csv(FILE_PATH)


# Plot the importances - alternative plot layout
plt.figure(figsize=(6, 6))
sns.barplot(x=importances_RF, y=importances_RF.index)
plt.title('Feature Importances from Random Forest (Task 1) - Gini Impurity')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(FILE_PATH, dpi=700)
plt.show()

#Visualizing Tree Structure 
#Selecting a tree of high feature importance - these are more representative of the overall model 
tree_index = np.argmax(importances_RF)
selected_tree = retrained_RF.estimators_[tree_index] #selecting the tree
tree_rules = tree.export_text(selected_tree, feature_names=list(X_test.columns))
#Exporting the Tree 
tree_lines = tree_rules.split('\n')
df_tree = pd.DataFrame({'Tree':tree_lines})   #Converting the tree to a dataframe
df_tree.to_csv(FILE_PATH, index=False)




################################################
#XGBoost    
################################################

#Initializing the XGBoost Classifier with base parameters  
xgb_classifier = xgb.XGBClassifier()

#Training the Gradient Boosting model on the training 
model_xgb = xgb_classifier.fit(X_train, y_train)

#Predicting on the training dataset and computing the accuracy 
Y_pred_xgb = model_xgb.predict(X_test)
accuracy_original_XGB = accuracy_score(y_test, Y_pred_xgb)
print("The accuracy score of the original XGBoost model: " + str(accuracy_original_XGB))

#Defining the hyperparameters to be tuned using GridSearchCV
param_dict_XGB = {"learning_rate" : [0.01, 0.1, 0.2],       #Controls the contribution of each weak learner 
                 "n_estimators" : [50, 100, 200],           #Number of boosting stages to be run
                 "max_depth" : [5, 10],                 #Controls the depth of each tree in the ensemble 
                 "min_child_weight" : [1, 3, 5]}            #Controls the Minimum sum of instance weight (hessian) needed in a child. It helps control overfitting 
                 #"gamma" : [0, 0.t1, 0.2],                   #Minimum loss reduction required to make a further partition on a leaf node (encourages pruning of the tree)
                 #"subsample" : [0.8, 0.9, 1.0],             #Fraction of samples used for training each tree (prevents overfitting by introducing randomness)
                 #"reg_lambda" : [1, 2, 3],                  #L2 regularization (helps prevent overfitting)
                 #"reg_alpha" : [0, 0.1, 0.5]}               #L1 regularization (helps prevent overfitting)
        #Note I chose to omit "col_sample_bytree" - similar effect as subsample but applies to the features at each level of the tree - due to the computational cost / time 
        #Omitted "scale pos weights because the classes are not imbalanced 


retrained_XGB, XGB_params = hyp_tuning("XGBoost", model_xgb, param_dict_XGB, skf, X_train, y_train)


        ################################
        #Evaluating the model 
        ################################

#Accuracy of retrained model 
Y_pred_XGB = retrained_XGB.predict(X_test)
accuracy_XGB = accuracy_score(y_test, Y_pred_XGB)
print("The accuracy score of the retrained XGBoost Model: ")
print(accuracy_XGB)

#Confusion Matric and Classification Rport
XGB_conf_matrix, XGB_class_report = eval_model("XGBoost Model", y_test, Y_pred_XGB)

#Bootstrap confidence intervals 
accuracy_ci_XGB, precision_ci_XGB, recall_ci_XGB, f1_ci_XGB, auc_ci_XGB, class_confidence_intervals_XGB = bootstrapCI(retrained_XGB, X_test, y_test, num_class=2 , n_iterations=1000)

#Feature Importance for  XGB Model (Gini Importance)
importances_XBG = retrained_XGB.feature_importances_
importances_XBG = pd.Series(importances_XBG, index=X_test.columns).sort_values(ascending=False)
df_feature_imp_XGB = pd.DataFrame(importances_XBG)
df_feature_imp_XGB['Condition'] = X_test.columns
df_feature_imp_XGB.to_csv(FILE_PATH)

#Plot the importances 
plt.figure(figsize=(8,6))
sns.barplot(x=importances_XBG, y=importances_XBG.index)
plt.title('Feature Importances from XGBoost (Task 1) - Gini Impurity')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(FILE_PATH, dpi=700)
plt.show()



################# Exporting Dictionary of Best Parameters #################

#Create dataframes from dictionaries 
df_LR_params = pd.DataFrame(list(LR_params.items()), columns=['Key','LR'])
df_RF_params = pd.DataFrame(list(RF_params.items()), columns=['Key','RF'])
df_XGB_params = pd.DataFrame(list(XGB_params.items()), columns=['Key','XGB'])

#Merge dataframes sequentially 
merged_df = df_LR_params.merge(df_RF_params, on="Key", how='outer')
merged_df = merged_df.merge(df_XGB_params, on="Key", how='outer')

merged_df.to_csv(FILE_PATH)






################# AUROC Task 1 -- Plotting all Models  #################


def auc_model3(model1, model2, model3, X_train, y_train, X_test, y_test, plot_name):
    """
    model_name: string of the model 
    This function creates a graph of AUC

    """
    # Get the predicted probabilities for the positive class 
    y_model1_train_proba = model1.predict_proba(X_train)[:,1]
    y_model1_test_proba = model1.predict_proba(X_test)[:,1]
    y_model2_train_proba = model2.predict_proba(X_train)[:,1]
    y_model2_test_proba = model2.predict_proba(X_test)[:,1]
    y_model3_train_proba = model3.predict_proba(X_train)[:,1]
    y_model3_test_proba = model3.predict_proba(X_test)[:,1]
    
    # Model 1 - Calculate the ROC curve and AUC for the training and testing dataa
    fpr_train_model1, tpr_train_model1, _ = roc_curve(y_train, y_model1_train_proba)     #training data
    auc_train_model1 = auc(fpr_train_model1, tpr_train_model1)                           #training data
    print(f"LR1 AUC on Training Data: {auc_train_model1}")
    fpr_test_model1, tpr_test_model1, _ = roc_curve(y_test, y_model1_test_proba)         #testing data
    auc_test_model1 = auc(fpr_test_model1, tpr_test_model1)                              #testing data
    print(f"LR1 AUC on Testing Data: {auc_test_model1}")

    # Model 2 - Calculate the ROC curve and AUC for the training and testing dataa
    fpr_train_model2, tpr_train_model2, _ = roc_curve(y_train, y_model2_train_proba)     #training data
    auc_train_model2 = auc(fpr_train_model2, tpr_train_model2)                           #training data
    print(f"RF1 AUC on Training Data: {auc_train_model2}")
    fpr_test_model2, tpr_test_model2, _ = roc_curve(y_test, y_model2_test_proba)         #testing data
    auc_test_model2 = auc(fpr_test_model2, tpr_test_model2)                              #testing data
    print(f"RF1 AUC on Testing Data: {auc_test_model2}")

    # Model 3 - Calculate the ROC curve and AUC for the training and testing dataa
    fpr_train_model3, tpr_train_model3, _ = roc_curve(y_train, y_model3_train_proba)     #training data
    auc_train_model3 = auc(fpr_train_model3, tpr_train_model3)                           #training data
    print(f"XGB1 AUC on Training Data: {auc_train_model3}")
    fpr_test_model3, tpr_test_model3, _ = roc_curve(y_test, y_model3_test_proba)         #testing data
    auc_test_model3 = auc(fpr_test_model3, tpr_test_model3)                              #testing data
    print(f"XGB1 AUC on Testing Data: {auc_test_model3}")

    # Ploting ROC Curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_train_model1, tpr_train_model1, color='darkgray', lw=2, label=f'LR1: Training ROC (AUC = {auc_train_model1:.2f})')
    plt.plot(fpr_test_model1, tpr_test_model1, color='darkgray', lw=2, linestyle='--', label=f'LR1: Testing ROC (AUC = {auc_test_model1:.2f})')
    plt.plot(fpr_train_model2, tpr_train_model2, color='dodgerblue', lw=2, label=f'RF1: Training ROC (AUC = {auc_train_model2:.2f})')
    plt.plot(fpr_test_model2, tpr_test_model2, color='dodgerblue', lw=2, linestyle='--', label=f'RF1: Testing ROC (AUC = {auc_test_model2:.2f})')
    plt.plot(fpr_train_model3, tpr_train_model3, color='navy', lw=2, label=f'XGB1: Training ROC (AUC = {auc_train_model3:.2f})')
    plt.plot(fpr_test_model3, tpr_test_model3, color='navy', lw=2, linestyle='--', label=f'XGB1: Testing ROC (AUC = {auc_test_model3:.2f})')
    plt.plot([0, 1], [0, 1], color='lightgray', lw=2, label="Random Guess")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(plot_name, dpi=700)
    plt.show()

    return 


#Runnning the plot to determine 
auc_model3(retrained_LR_model, retrained_RF, retrained_XGB, X_train, y_train, X_test, y_test, FILE_PATH)





################# Calibration Curves for Task 1 #################

def calibration_curve_task(model1, model2, model3, X_test, y_test, plot_name): 

    #Calcualte the probaility of positive class 
    y_proba_model1 = model1.predict_proba(X_test)[:, 1] #Final slice gets the predicted probabilities for class 1
    y_proba_model2 = model2.predict_proba(X_test)[:, 1] 
    y_proba_model3 = model3.predict_proba(X_test)[:, 1] 
    
    #Create a calibration curve 
    prob_true_model1, prob_pred_model1 = calibration_curve(y_test, y_proba_model1, n_bins=50)
    prob_true_model2, prob_pred_model2 = calibration_curve(y_test, y_proba_model2, n_bins=50)
    prob_true_model3, prob_pred_model3 = calibration_curve(y_test, y_proba_model3, n_bins=50)
    
    #Plot calibration curve
    plt.figure(figsize=(8,8))
    plt.plot(prob_pred_model1, prob_true_model1, color='darkgray', linestyle = '-', label="Logistic Regression")
    plt.plot(prob_pred_model2, prob_true_model2, color='dodgerblue', linestyle = '-', label="Random Forest")
    plt.plot(prob_pred_model3, prob_true_model3,  color='navy', linestyle = '-', label="XGBoost")
    plt.plot([0,1], [0,1], linestyle='--', color='gray', label="Perfect calibration") #Diagnoal line representing perfect calibration 
    plt.xlabel('Predicted probability')
    plt.ylabel('Empirical probability')
    plt.title('Calibration Plot')
    plt.legend(loc='lower right')
    plt.savefig(plot_name, dpi=700)
    plt.show()

#Calibration curve for task 2 
calibration_curve_task(retrained_LR_model, retrained_RF, retrained_XGB, X_test, y_test, FILE_PATH)



################# SHAP Plots for the Models in Task 1  #################

#SHAP ANALYSIS
#Note on the SHAP analysis - the features that are still scaled. This should not affect 

#Creating a dataframe for the analysis - this allows use of the actual column names 
X2_test_df = pd.DataFrame(X_test, index=y_test.index, columns=list(df_test.columns))

#   LR Model
#Create explainer
explainer_LR = shap.LinearExplainer(retrained_LR_model, X_test)
shap_values_LR = explainer_LR.shap_values(X_test)

#Visualize the SHAP Values - shows how all features impact the predictions 
shap.summary_plot(shap_values_LR, X_test)
plt.savefig('/linux_home/gepostill/Files/u/gepostill/EXPORT/Task1_LR_shap_summary_plot.png')
plt.show()

#for feature in X_test.columns: #Plot how a specific feature impacts predeiciton 
#    shap.dependence_plot(feature, shap_values_LR, X_test)
#    plt.savefig('FILE_PATH_{feature}.png')
#    plt.show()


#   RF Model --- having a problem with this model 
#Create explainer
#explainer_RF = shap.TreeExplainer(retrained_RF)
##Alternative - #explainer_RF = shap.KernelExplainer(retrained_RF.predict, X_test)
#shap_values_RF = explainer_RF.shap_values(X_test, approximate=False, check_additivity=False)

#Visualize the SHAP Values - shows how all features impact the predictions 
#shap.summary_plot(shap_values_RF[1], X_test)
#plt.savefig(FILE_PATH)
#plt.show()


#   XGB Model
#Create explainer
explainer_XBG = shap.Explainer(retrained_XGB, X_test)
shap_values_XGB = explainer_XBG.shap_values(X_test)

#Visualize the SHAP Values - shows how all features impact the prediction
shap.summary_plot(shap_values_XGB, X_test)
plt.savefig(FILE_PATH)
plt.show()


################# Exporting the models #################

joblib.dump(retrained_LR_model, FILE_PATH)
joblib.dump(retrained_RF, FILE_PATH)
joblib.dump(retrained_XGB, FILE_PATH)


