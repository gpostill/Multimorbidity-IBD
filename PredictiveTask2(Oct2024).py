#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:35:35 2023

@author: gepostill
"""

#Note - for privacy all file names are written as FILE_NAME to conceal the file path of data and figures on the laptop used to create the code

#################################
#Importing the necessary packages
#################################
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import plotly.express as px -- we would need to get permission to install the package plotly

import xgboost as xgb
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, roc_curve, auc, log_loss, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, label_binarize, OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
    
    #Ploting ROC Curve
    plt.figure(figsize=(8,8))
    plt.plot(fpr_train, tpr_train, color='dodgerblue', lw=2, label='Train AUC = {:.2f}'.format(roc_auc_train))
    plt.plot(fpr_test, tpr_test, color='navy', lw=2, label='Test AUC = {:.2f}'.format(roc_auc_test))
    #plt.plot([0,1], [0,1], color='gray', lw=, label="Random Guess") 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(plot_name)
    plt.show()

    return accuracy_score_retrain, accuracy_score_retest


def feature_importance(model, plt_name):
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
    feature_names = X_train.columns # the format is not as nice so i will create improved format below
    
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
    return X.iloc[idx], y.iloc[idx]  #using different formats because X is an array and y is a series 


#Create a funciton to calculate the confidence intervals 
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
         'Renal_Disease','Diabetes','Osteoporosis','Rheumatoid_Arthritis','Osteoarthritis','Mood_disorder','Other_Mental_disorder']]

#Encoding the categorical variables 
df = df.replace({"No" : 0, "Yes" : 1})
df = df.replace({"M" : 0, "F" : 1})

print("The number of patients included in the model")
print(len(df))



################################################
#Sample Size Calculation 
################################################
from statsmodels.stats.power import TTestIndPower

#Parameters
effect_size = 0.5
alpha = 0.05
power = 0.80
ratio = 1 #Set to 1 because there is equal sample zie in both groups 

#Calculate sample size 
Analysis = TTestIndPower()
sample_size = Analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=ratio) 

print("Required sample size per group:", round(sample_size))

################################################
#Data Splitting 80:20  
################################################
#Using the same testing data as the other tasks

#Import the training and testing data to get the ID values
df_train = pd.read_csv('FILE_NAME)
df_test = pd.read_csv(FILE_NAME)

#Create a list of trianing and testing IDs
training_IDs = df_train['ID'].to_list()
testing_IDs = df_test['ID'].to_list()

df_train = df[df['ID'].isin(training_IDs)] #re-writing this with the pre-defined training value 
df_test = df[df['ID'].isin(testing_IDs)] #re-writing this with the the pre-defined testing value 

#Remove the label from dataset 
X_train = df_train.drop('ID',axis=1) #The label is: 'Premature'
X_train = X_train.drop('Premature',axis=1) #The label is: 'Premature'
y_train = df_train['Premature']

#Remove the label from dataset 
X_test = df_test.drop('ID',axis=1) #The label is: 'Premature'
X_test = X_test.drop('Premature',axis=1) #The label is: 'Premature'
y_test = df_test['Premature']


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

retrained_LR_model, LR_params = hyp_tuning("Logistic Regression", trained_LR_model, param_dict_LR, skf, X_train, y_train)

        ################################
        #Evaluating the model 
        ################################

#Evaluating the new accuracy 
Y_pred_LR = retrained_LR_model.predict(X_test)
accuracy_LR = accuracy_score(y_test, Y_pred_LR)
print("The accuracy score of the optimized Logistic Regression model: ")
print(accuracy_LR)


#Confusion Matrix and Classification Report
LR_conf_matrix, LR_class_report = eval_model("Logistic Regression", y_test, Y_pred_LR)

#Accuracy + AUC plot with the new retrained model
acc_train_LR, acc_test_LR = auc_model(retrained_LR_model, X_train, y_train, X_test, y_test, FILE_NAME) 

#Bootstrap confidence intervals 
accuracy_ci_LP, precision_ci_LR, recall_ci_LR, f1_ci_LR, auc_ci_LR, class_confidence_intervals_LR = bootstrapCI(retrained_LR_model, X_test, y_test, num_class=2 , n_iterations=1000)

#Measuring the log-loss of model 
#Log-loss measures how well the predicted probabilities match the true distribution of the class; 
y_prob_LR = retrained_LR_model.predict_proba(X_test)
logloss = log_loss(y_test, y_prob_LR)
print("Log-Loss of Final Logistic Regression Model")
print(logloss)
#Lower log-loss values indicate better agreement between predicted and actual probabilities (perfect model has a log-loss of 0)
    #Interpretation: 
            #Values very close to zero indicate great performance (or overfitting)
            #Values between 0-1 indicate intermediate log-loss --> reasonable predictions (possible room for improvement)
            #Values >1 may indicate poor-calibration



        ################################
        #Feature Importance of LR Model (Coefficients)
        ################################

#Identifying the coefficients and corresponding feature names (column names)
coeffficients = retrained_LR_model.coef_[0]
feature_names = X_test.columns 

#Creating a sorted DF of features coefficients 
feature_imp_LR = pd.DataFrame({'Feature':feature_names, 'Coefficient': coeffficients})

#Adding another column of absolute coefficient & sorting on this (magnitude of strength with predicting prem. mortality)
feature_imp_LR['Absolute Coefficient'] = feature_imp_LR['Coefficient'].abs()
feature_imp_LR = feature_imp_LR.sort_values(by='Absolute Coefficient', ascending=False)

#Get the names of the features 
feature_imp_LR.to_csv(FILE_NAME)


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
print("The accuracy score of the original Random forest model: " + str(accuracy_original_RF))
 
#Defining the hyperparameters to be tuned using GridSearchCV
param_dict_RF = {"n_estimators" : [50, 100, 200, 500], 
                 "max_depth" : [5, 10, 15, 18], 
                 "min_samples_leaf" : [5, 15, 35, 50], 
                 "min_samples_split" : [2,5,10,20], 
                 "bootstrap" : [True,False]}
        #Are there other parameters I should use?

retrained_RF, RF_params = hyp_tuning("Random Forest", trained_model_RF, param_dict_RF, skf, X_train, y_train)


        ################################
        #Evaluating the model 
        ################################

#Accuracy of retrained model 
Y_pred_RF = retrained_RF.predict(X_test)
accuracy_RF = accuracy_score(y_test, Y_pred_RF)
print("\n The accuracy score of the retrained Random Forest Model: ")
print(accuracy_LR)

#Confusion Matrix and Classification Report
RF_conf_matrix, RF_class_report = eval_model("Random Forest", y_test, Y_pred_RF)

#Accuracy + AUC plot with the new retrained model
acc_train_RF, acc_test_RF = auc_model(retrained_RF, X_train, y_train, X_test, y_test, FILE_NAME)

#Bootstrap confidence intervals 
accuracy_ci_RF, precision_ci_RF, recall_ci_RF, f1_ci_RF, auc_ci_RF, class_confidence_intervals_RF = bootstrapCI(retrained_RF, X_test, y_test, num_class=2 , n_iterations=1000)

#Determining the size of the final forest
RF_forest_size = len(retrained_RF.estimators_)
print(f"The Random Forest has {RF_forest_size} trees")

#Feature Importance of Final RF Model 
feature_imp_RF = feature_importance(retrained_RF, FILE_NAME)
df_feature_imp_RF = pd.DataFrame(feature_imp_RF)
df_feature_imp_RF['Condition'] = X_test.columns #Note that based on function the order of the fature importance is preserved as order of the columns
df_feature_imp_RF.to_csv(FILE_NAME)


#Visualizing Tree Structure 
#Selecting a tree of high feature importance - these are more representative of the overall model 
tree_index = np.argmax(feature_imp_RF)
selected_tree = retrained_RF.estimators_[tree_index] #selecting the tree
tree_rules = tree.export_text(selected_tree, feature_names=list(X_test.columns))
#Exporting the Tree 
tree_lines = tree_rules.split('\n')
df_tree = pd.DataFrame({'Tree':tree_lines})   #Converting the tree to a dataframe
df_tree.to_csv(FILE_NAME, index=False)



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

#Confusion Matrix and Classification Report
XGB_conf_matrix, XGB_class_report = eval_model("XGBoost Model", y_test, Y_pred_XGB)

#Accuracy + AUC plot with the new retrained model
acc_train_XGB, acc_test_XGB = auc_model(retrained_XGB, X_train, y_train, X_test, y_test, FILE_NAME)

#Bootstrap confidence intervals 
accuracy_ci_XGB, precision_ci_XGB, recall_ci_XGB, f1_ci_XGB, auc_ci_XGB, class_confidence_intervals_XGB = bootstrapCI(retrained_XGB, X_test, y_test, num_class=2 , n_iterations=1000)


#Feature Importance of Final RF Model 
feature_imp_XGB = feature_importance(retrained_XGB, FILE_NAME)
df_feature_imp_XGB = pd.DataFrame(feature_imp_XGB)
df_feature_imp_XGB['Condition'] = X_test.columns #Note that based on function the order of the feature importance is preserved as order of the columns
df_feature_imp_XGB.to_csv(FILE_NAME)





################# Exporting Dictionary of Best Parameters #################

#Create dataframes from dictionaries 
df_LR_params = pd.DataFrame(list(LR_params.items()), columns=['Key','LR'])
df_RF_params = pd.DataFrame(list(RF_params.items()), columns=['Key','RF'])
df_XGB_params = pd.DataFrame(list(XGB_params.items()), columns=['Key','XGB'])

#Merge dataframes sequentially 
merged_df = df_LR_params.merge(df_RF_params, on="Key", how='outer')
merged_df = merged_df.merge(df_XGB_params, on="Key", how='outer')

merged_df.to_csv(FILE_NAME)


