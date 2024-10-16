#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 07:46:52 2024

@author: gepostill
"""

#Note - for privacy all file names are written as FILE_NAME to conceal the file path of data and figures on the laptop used to create the code


#################################
# IMPORTING PACKAGES
################################

import pandas as pd
import tqdm as notebook_tqdm
import numpy as np
import itertools
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import hdbscan
from consensusclustering import ConsensusClustering
from tableone import TableOne

import plotly.graph_objects as go

from scipy.stats import entropy
from scipy.linalg import det
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances, silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
# from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, roc_curve, auc, log_loss
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


import warnings
warnings.filterwarnings('ignore')



#################################
# DEFINING FUNCTIONS
################################


#plot the heatmap for the optimal number of K 
def clustermap_plot(consensus_clusters, sex, train): 
    """

    Parameters
    ----------
    consensus_clusters : APPROACH TO CLUSTERING / CLUSTERING OBJECT 
    sex : STRING; this is to add to the end of the figure name  
        DESCRIPTION.

    Returns
    -------
    CREATES, SHOWS, AND SAVES FIGURE.

    """
    #List of K values 
    k_numbers = [2,3,4,5]
    
    #Iterate through the clusters 
    for k_number in k_numbers: 
        grid = consensus_clusters.plot_clustermap(k=k_number, 
            figsize=(5,5),
            dendrogram_ratio=0.05,                                         
            xticklabels=False, yticklabels=False, cbar=False,
            cmap='Blues')
        grid.cax.set_visible(True)
        plt.savefig(f'FILE_NAME_{k_number}_{sex}_{train}.png',dpi=700)
        plt.show()



def cumulative_distribuiton_plots(consensus_clusters, sex, train): 
    """
    Parameters
    ----------
    consensus_clusters : TYPE
        DESCRIPTION.
    sex : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #CUMMULATIVE DISTRIBUTION FUNCTION PLOTS 
    _, ax = plt.subplots(figsize=(4,4))
    consensus_clusters.plot_cdf(ax=ax)
    #ax.legend(bbox_to_anchor=(1,1))
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'FILE_NAME_{sex}_{train}.png',dpi=700)
    plt.show()
    
    #CHANGE IN AREA UNDER CDF 
    _, ax = plt.subplots(figsize=(4,4))
    consensus_clusters.plot_change_area_under_cdf(ax=ax)    
    plt.savefig(f'FILE_NAME_{sex}_{train}.png',dpi=700)
    plt.show()
    
    #CDF vs. K
    #Compare above value to the change point in the AUC of the CDF vs. K
    _, ax = plt.subplots(figsize=(4,4))
    consensus_clusters.plot_auc_cdf()
    plt.savefig(f'FILE_NAME_{sex}_{train}.png',dpi=700)
    plt.show()




# Describes the clusters

def describe_clusters(original_data, cluster_data, labels_OF_cluster, csv_name):
    """
    Parameters
    ----------
    original_data : DATAFRAME
        ORIGINAL DATAFRAME WITH ALL COLUMNS (INCLUDING THOSE DROPPED).
    cluster_data : DATAFRAME
        DATA FITTED BY THE MODEL.
    labels_OF_cluster : NUMPY ARRAY
        CLUSTER LABELS OUTPUTTED BY THE MODEL.
    csv_name : STRING
        NAME FOR DESCRIPTIVE DATAFRAME TO BE SAVED; INCLUDES FILE PATH.

    Returns
    -------
    None. Displays plot in console. Saves dataframe describing clusters as a CSV.

    """
    # Identifying the labels of the clusters
    cluster_data['Cluster'] = labels_OF_cluster
    df_cluster_descriptive = cluster_data.copy()
    add_to_df = ['Age_death', 'sex', 'AGE_IBD', 'DX_latest', 'death_yr', 'Rio_rank', 'incquint', 'Material_deprivation', 'Education_Quintile',
                 'Age_Asthma', 'Age_CHF', 'Age_COPD', 'Age_Myocardial_infarction', 'Age_Hypertension', 'Age_Arrhythmia', 'Age_CCS', 'Age_Stroke', 'Age_Cancer',
                 'Age_Dementia', 'Age_Renal_Disease', 'Age_Diabetes', 'Age_Osteoporosis', 'Age_Rheumatoid_Arthritis', 'Age_OsteoArthritis', 'Age_Mood_disorder',
                 'Age_Other_Mental_disorder', 'Degree', 'Premature', 'conditions']
    for column in add_to_df:
        df_cluster_descriptive[column] = original_data[column]
    grouped_data = df_cluster_descriptive.groupby('Cluster')

    #   DESCRIPTIVE STATISTICS OF OPTIMAL CLUSTERS: 4
    cluster_statistics = grouped_data.describe()
    cluster_statistics = cluster_statistics.transpose()
    cluster_statistics.to_csv(csv_name)


#   Counting the number of people with each conditions in a cluster
def describe_clusters_conditions(df_cluster):

    # list of the conditions to be tabulated
    conditions = ['Asthma', 'CHF', 'COPD', 'Myocardial_infarction', 'Hypertension', 'Arrhythmia', 'CCS',
                  'Stroke', 'Cancer', 'Dementia', 'Renal_Disease', 'Diabetes', 'Osteoporosis', 'Rheumatoid_Arthritis',
                  'Osteoarthritis', 'Mood_disorder', 'Other_Mental_disorder']

    # initializing an empty dataframe
    cluster_tabulated = pd.DataFrame()

    for condition in conditions:
        # Create counts of the condition
        aa = df_cluster.groupby([condition, 'Cluster']
                                ).size().reset_index(name='count')
        aa = aa.pivot_table(index=condition, columns='Cluster',
                            values='count', fill_value=0)

        # Add condition name
        aa['condition'] = condition

        # Append to dataframe
        cluster_tabulated = pd.concat([cluster_tabulated, aa])

    # Return the new dataframe with counts per condition
    return (cluster_tabulated)


def plot_tsne(df_train, cluster_labels_train, sex, train): 
    # Reducing the data wit t-SNE
    tsne = TSNE(n_components=2, perplexity=40, random_state=0)
    tsne_result = tsne.fit_transform(df_train)
    
    # Creating a scatter plot of t-SNE results colored by cluster
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels_train, cmap='Paired')
    
    # Adding a legned with custom labels
    legend_labels = ['Cluster A', 'Cluster B', 'Cluster C']
    legend = plt.legend(*scatter.legend_elements())
    for text, label in zip(legend.get_texts(), legend_labels):
        text.set_text(label)
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(f'FILE_NAME_{train}_{sex}.png', dpi=700)
    plt.show()



def umap_plot(df_train_age, cluster_labels_train, sex, train):

    # Compute UMPA Embeddings
    umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    # Need to check these hyperparameters
    embedding = umap_emb.fit_transform(df_train_age)
    
    # plot UMPA embeddings iwth clusters colored by subtype
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c=cluster_labels_train, cmap='Paired', s=10)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    # plt.colorbar(label='Cluster')
    plt.savefig(f'FILE_NAME_{train}_{sex}.png', dpi=700)
    plt.show()


################################################

#################################
# DATA IMPORT & CLEANING
################################


# importing the dataframe
df = pd.read_csv(FILE_NAME) 
df = df[df['Cohort'] == 'Case']

# Import the training and testing data
# Noting - training and testing datasets have already removed all controls.
df_train = pd.read_csv(FILE_NAME)
df_test = pd.read_csv(FILE_NAME) 

# Create a list of trianing and testing IDs
training_IDs = df_train['ID'].to_list()
testing_IDs = df_test['ID'].to_list()

# re-writing this with the pre-defined training value
df2_train = df[df['ID'].isin(training_IDs)]
# re-writing this with the the pre-defined testing value
df2_test = df[df['ID'].isin(testing_IDs)]


#################################
# DATA PRE-PROCESSING
################################

# Changing the "Yes" and "No" in dataset
df2_test = df2_test.replace({"No": 0, "Yes": 1})
df2_train = df2_train.replace({"No": 0, "Yes": 1})
df = df.replace({"No": 0, "Yes": 1})

# Defining the Dataframe to Cluster (use for later in prediction sensitivity analysis

#TRAINING DATA
#Subsetting female data
df_train = df2_train[['Asthma', 'CHF', 'COPD', 'Myocardial_infarction', 'Hypertension', 'Arrhythmia', 'CCS', 'Stroke', 'Cancer', 'Dementia',
                      'Renal_Disease', 'Diabetes', 'Osteoporosis', 'Rheumatoid_Arthritis', 'Osteoarthritis', 'Mood_disorder', 'Other_Mental_disorder']]

df_train_age = df2_train[['Age_Asthma', 'Age_CHF', 'Age_COPD', 'Age_Myocardial_infarction', 'Age_Hypertension', 'Age_Arrhythmia', 'Age_CCS', 'Age_Stroke', 'Age_Cancer', 'Age_Dementia',
                          'Age_Renal_Disease', 'Age_Diabetes', 'Age_Osteoporosis', 'Age_Rheumatoid_Arthritis', 'Age_OsteoArthritis', 'Age_Mood_disorder', 'Age_Other_Mental_disorder']]

# Apply Standard scalar
scaler = StandardScaler()
scaled_train = scaler.fit_transform(df_train_age)
scaled_train_array = np.nan_to_num(scaled_train, nan=-10)
df_train_age_EXPORT = df_train_age #Creating a copy that can be exported for the validation
df_train_age = pd.DataFrame(scaled_train_array, columns=df_train_age.columns, index=df_train_age.index) # Convert scaled data back to dataframe


# Sex Specific data
df2_train_M = df2_train[df2_train['sex'] == 'M']
df2_train_F = df2_train[df2_train['sex'] == 'F']

#Subsetting male data
df_train_M = df2_train_M[['Asthma', 'CHF', 'COPD', 'Myocardial_infarction', 'Hypertension', 'Arrhythmia', 'CCS', 'Stroke', 'Cancer', 'Dementia',
                      'Renal_Disease', 'Diabetes', 'Osteoporosis', 'Rheumatoid_Arthritis', 'Osteoarthritis', 'Mood_disorder', 'Other_Mental_disorder']]

df_train_age_M = df2_train_M[['Age_Asthma', 'Age_CHF', 'Age_COPD', 'Age_Myocardial_infarction', 'Age_Hypertension', 'Age_Arrhythmia', 'Age_CCS', 'Age_Stroke', 'Age_Cancer', 'Age_Dementia',
                          'Age_Renal_Disease', 'Age_Diabetes', 'Age_Osteoporosis', 'Age_Rheumatoid_Arthritis', 'Age_OsteoArthritis', 'Age_Mood_disorder', 'Age_Other_Mental_disorder']]

# Apply Standard scalar - Male
scaler_M = StandardScaler()
scaled_train_M = scaler_M.fit_transform(df_train_age_M)
scaled_train_array_M = np.nan_to_num(scaled_train_M, nan=-10)
df_train_age_M_EXPORT = df_train_age_M #Creating a copy that can be exported for the validation
df_train_age_M = pd.DataFrame(scaled_train_array_M, columns=df_train_age_M.columns, index=df_train_age_M.index) # Convert scaled data back to dataframe


#Subsetting female data
df_train_F = df2_train_F[['Asthma', 'CHF', 'COPD', 'Myocardial_infarction', 'Hypertension', 'Arrhythmia', 'CCS', 'Stroke', 'Cancer', 'Dementia',
                      'Renal_Disease', 'Diabetes', 'Osteoporosis', 'Rheumatoid_Arthritis', 'Osteoarthritis', 'Mood_disorder', 'Other_Mental_disorder']]

df_train_age_F = df2_train_F[['Age_Asthma', 'Age_CHF', 'Age_COPD', 'Age_Myocardial_infarction', 'Age_Hypertension', 'Age_Arrhythmia', 'Age_CCS', 'Age_Stroke', 'Age_Cancer', 'Age_Dementia',
                          'Age_Renal_Disease', 'Age_Diabetes', 'Age_Osteoporosis', 'Age_Rheumatoid_Arthritis', 'Age_OsteoArthritis', 'Age_Mood_disorder', 'Age_Other_Mental_disorder']]

# Apply Standard scalar - Female
scaler_F = StandardScaler()
scaled_train = scaler_F.fit_transform(df_train_age_F)
scaled_train_array_F = np.nan_to_num(scaled_train, nan=-10)
df_train_age_F_EXPORT = df_train_age_F #Creating a copy that can be exported for the validation
df_train_age_F = pd.DataFrame(scaled_train_array_F, columns=df_train_age_F.columns, index=df_train_age_F.index) # Convert scaled data back to dataframe



##############
# TESTING DATA

df_test = df2_test[['Asthma', 'CHF', 'COPD', 'Myocardial_infarction', 'Hypertension', 'Arrhythmia', 'CCS', 'Stroke', 'Cancer', 'Dementia',
                    'Renal_Disease', 'Diabetes', 'Osteoporosis', 'Rheumatoid_Arthritis', 'Osteoarthritis', 'Mood_disorder', 'Other_Mental_disorder']]

df_test_age = df2_test[['Age_Asthma', 'Age_CHF', 'Age_COPD', 'Age_Myocardial_infarction', 'Age_Hypertension', 'Age_Arrhythmia', 'Age_CCS', 'Age_Stroke', 'Age_Cancer', 'Age_Dementia',
                       'Age_Renal_Disease', 'Age_Diabetes', 'Age_Osteoporosis', 'Age_Rheumatoid_Arthritis', 'Age_OsteoArthritis', 'Age_Mood_disorder', 'Age_Other_Mental_disorder']]

# Apply Standard scalar
scaler = StandardScaler()
scaled_test = scaler.fit_transform(df_test_age)
scaled_test_array = np.nan_to_num(scaled_test, nan=-10)
df_test_age_EXPORT = df_test_age #Creating a copy that can be exported for the validation
df_test_age = pd.DataFrame(scaled_test_array, columns=df_test_age.columns, index=df_test_age.index) # Convert scaled data back to dataframe


# Sex Specific data
df2_test_M = df2_test[df2_test['sex'] == 'M']
df2_test_F = df2_test[df2_test['sex'] == 'F']

#Subsetting male data
df_test_M = df2_test_M[['Asthma', 'CHF', 'COPD', 'Myocardial_infarction', 'Hypertension', 'Arrhythmia', 'CCS', 'Stroke', 'Cancer', 'Dementia',
                      'Renal_Disease', 'Diabetes', 'Osteoporosis', 'Rheumatoid_Arthritis', 'Osteoarthritis', 'Mood_disorder', 'Other_Mental_disorder']]

df_test_age_M = df2_test_M[['Age_Asthma', 'Age_CHF', 'Age_COPD', 'Age_Myocardial_infarction', 'Age_Hypertension', 'Age_Arrhythmia', 'Age_CCS', 'Age_Stroke', 'Age_Cancer', 'Age_Dementia',
                          'Age_Renal_Disease', 'Age_Diabetes', 'Age_Osteoporosis', 'Age_Rheumatoid_Arthritis', 'Age_OsteoArthritis', 'Age_Mood_disorder', 'Age_Other_Mental_disorder']]

# Apply Standard scalar - Male
scaler_M = StandardScaler()
scaled_test_M = scaler_M.fit_transform(df_test_age_M)
scaled_test_array_M = np.nan_to_num(scaled_test_M, nan=-10)
df_test_age_M_EXPORT = df_test_age_M #Creating a copy that can be exported for the validation
df_test_age_M = pd.DataFrame(scaled_test_array_M, columns=df_test_age_M.columns, index=df_test_age_M.index) # Convert scaled data back to dataframe


#Subsetting female data
df_test_F = df2_test_F[['Asthma', 'CHF', 'COPD', 'Myocardial_infarction', 'Hypertension', 'Arrhythmia', 'CCS', 'Stroke', 'Cancer', 'Dementia',
                      'Renal_Disease', 'Diabetes', 'Osteoporosis', 'Rheumatoid_Arthritis', 'Osteoarthritis', 'Mood_disorder', 'Other_Mental_disorder']]

df_test_age_F = df2_test_F[['Age_Asthma', 'Age_CHF', 'Age_COPD', 'Age_Myocardial_infarction', 'Age_Hypertension', 'Age_Arrhythmia', 'Age_CCS', 'Age_Stroke', 'Age_Cancer', 'Age_Dementia',
                          'Age_Renal_Disease', 'Age_Diabetes', 'Age_Osteoporosis', 'Age_Rheumatoid_Arthritis', 'Age_OsteoArthritis', 'Age_Mood_disorder', 'Age_Other_Mental_disorder']]

# Apply Standard scalar - Female
scaler_F = StandardScaler()
scaled_test = scaler_F.fit_transform(df_test_age_F)
scaled_test_array_F = np.nan_to_num(scaled_test, nan=-10)
df_test_age_F_EXPORT = df_test_age_F #Creating a copy that can be exported for the validation
df_test_age_F = pd.DataFrame(scaled_test_array_F, columns=df_test_age_F.columns, index=df_test_age_F.index) # Convert scaled data back to dataframe

# Note we are doing this after scaling so that the imputed values do not have an impact on how the data is scaled


#################################
# PARTITIONING VS. CLUSTERING
################################

#   OPTICS Plot

# Perform OPTICS CLUSTERING
optics = OPTICS(min_samples=20, xi=0.2,
                min_cluster_size=0.1, metric='euclidean')
# Note regarding parameter values
# 'min_samples' parameter specifies the number of samples in a neighbourhood for a point to be considered as a core point
# 'xi' parameter determines the steepness threshold for identifying significant changes in the reachability plot
# 'min_cluster_size' parameter set the minimum size of clusters
optics.fit(df_train_age)

# Extract labels and reachability
labels = optics.labels_
reachability = optics.reachability_

# Sort the reachability distance
order = np.argsort(reachability)
sorted_reachability = reachability[order]
sorted_labels = labels[order]

# Plotting the results as Reachability plot
space = np.arange(len(df_train_age))
plt.bar(space, sorted_reachability, color='b')
plt.ylabel('Reachility (epsilon distance)')
plt.title('Reachability Plot')
#plt.ylim(0,100)
plt.savefig(FILE_NAME, dpi=700)
plt.show()

# Plotting a heatmap of the features
corr_matrix = df_train_age_M.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.savefig(FILE_NAME, dpi=700)
plt.show()

# Basted on the results of the optics plot (plot is stepped vs. smooth), it suggests that we should use hierarchical clustering



############################################
# DETERMINING THE OPTIMAL NUMBER OF CLUSTERS
############################################

#######################
#SETTING UP CLUSTERING
# To determine the optimal number of clusters, used a combination of cluster size (clinically relevant), characteristics of consensus CDF plots,  
#clear separation on consensus heatmaps, and adequate pairwise-consensus values between cluster members (>0.8)


# Creating a class of Kmeans wrapper for use:
class KMeansWrapper:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, init='k-means++',
                            n_init=10, max_iter=300, random_state=0)

    def fit_predict(self, X):
        return self.model.fit_predict(X)

    def set_params(self, **params):
        if 'n_clusters' in params:
            self.n_clusters = params['n_clusters']
            self.model = KMeans(n_clusters=self.n_clusters,
                                init='k-means++', n_init=10, max_iter=300, random_state=0)
        return self

clustering_obj = KMeansWrapper()
# clustering_obj = AgglomerativeClustering(affinity='euclidean', linkage='average')
exit()
#################
#FITTING FOR BOTH SEXES 
#Creating a clustering object to be used for both sexes -- ONLY 10 RESAMPLES 
consensus_clusters_BOTH = ConsensusClustering(clustering_obj=clustering_obj,
                                         min_clusters=2, max_clusters=6,
                                         n_resamples=10,  # Number should be enough to search the space of combinations but at a point will have diminishing returns
                                         resample_frac=0.8,
                                         k_param='n_clusters')  # number of resampling iteration

consensus_clusters_BOTH.fit(scaled_train_array, progress_bar=True)

#Consensus Cumulative Distribution Function plots
cumulative_distribuiton_plots(consensus_clusters_BOTH, sex='BOTH', train='train')

#Plot the consensus clustermap from 2-6 
#Iterate through the clusters 
#for k_number in [2,3,4,5]: 
#    grid = consensus_clusters_BOTH.plot_clustermap(k=k_number, 
#        figsize=(5,5), dendrogram_ratio=0.05,                                         
#        xticklabels=False, yticklabels=False, cbar=False,
#        cmap='Blues')
#    grid.cax.set_visible(True)
#    plt.savefig(f'FILE_NAME_{k_number}_BOTH_train.png',dpi=700)
#    plt.show()

#clustermap_plot(consensus_clusters_BOTH, 'BOTH', 'train')

#Currently set to plot the histogram for the consensus matrix for K=2, K=3, K=4
_, axes = plt.subplots(1, 5, figsize=(16,4))
for i, ax in enumerate(axes): 
    consensus_clusters_BOTH.plot_hist(i + 2, ax=ax)
    ax.set_title(i+2)
plt.tight_layout()
plt.savefig(FILE_NAME, dpi=700)
plt.show()


###############################################
# BOTH SEXES - TRAINING DATA - DESCRIBE CLUSTER WITH OPTIMAL K 

#The optimal number for K 
optimal_k_train = 3

#Fit KMeansWrapper with optimal clusters 
clustering_obj_BOTH = KMeansWrapper(n_clusters=optimal_k_train)
cluster_labels_train = clustering_obj_BOTH.fit_predict(scaled_train_array)

# add clusters labels to the training set
cluster_labelled_train = df2_train.copy() #Note we are using the dataset with YES/NO of conditions (DIFFERENT THAN CLUSTERING)
cluster_labelled_train['Cluster'] = cluster_labels_train

#Describe the final clusters - using Table 1 package 
#Create TableOne Object 
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
               'Osteoporosis', 'Rheumatoid_Arthritis', 'Oste_Arthritis', 'Mood_disorder',
               'Other_Mental_disorder', 'Degree_Cat'] 

nonnormal = ['AGE_IBD','Age_death', 'Age_Asthma','Age_CHF', 'Age_COPD', 'Age_Myocardial_infarction', 
             'Age_Hypertension', 'Age_Arrhythmia', 'Age_CCS', 'Age_Stroke', 'Age_Cancer', 'Age_Dementia', 'Age_Renal_Disease',
             'Age_Diabetes', 'Age_Osteoporosis', 'Age_Rheumatoid_Arthritis', 'Age_OsteoArthritis', 'Age_Mood_disorder',
             'Age_Other_Mental_disorder', 'Degree'] 

table1_train_BOTH = TableOne(cluster_labelled_train, columns=variables, categorical=categorical, nonnormal=nonnormal, groupby='Cluster', pval=True, smd=True)
print(table1_train_BOTH.tabulate(tablefmt = 'fancy_grid'))
table1_train_BOTH.to_csv(FILE_NAME)


# Describe the final clusters - using the function I created
described_clusters = describe_clusters_conditions(cluster_labelled_train)
described_clusters.to_csv(FILE_NAME)

#Export the labelled non-scaled data
df_train_age_EXPORT['Cluster'] = cluster_labels_train
df_train_age_EXPORT.to_csv(FILE_NAME)



####### TESTING DATA ############
#FITTING FOR BOTH SEXES 
#Creating a clustering object to be used for both sexes -- ONLY 10 RESAMPLES 
cc_BOTH_test = ConsensusClustering(clustering_obj=clustering_obj,
                                         min_clusters=2, max_clusters=6,
                                         n_resamples=10,  # Number should be enough to search the space of combinations but at a point will have diminishing returns
                                         resample_frac=0.8,
                                         k_param='n_clusters')  # number of resampling iteration

cc_BOTH_test.fit(scaled_test_array, progress_bar=True)

#Consensus Cumulative Distribution Function plots
cumulative_distribuiton_plots(cc_BOTH_test, sex='BOTH', train='test')

#Plot the conssot(cc_BOTH_test, 'BOTH', 'test')

#Currently set to plot the histogram for the consensus matrix for K=2, K=3, K=4
_, axes = plt.subplots(1, 5, figsize=(16,4))
for i, ax in enumerate(axes): 
    cc_BOTH_test.plot_hist(i + 2, ax=ax)
    ax.set_title(i+2)
plt.tight_layout()
plt.savefig(FILE_NAME, dpi=700)
plt.show()


################################################################################
###########BOTH SEXES - CREATING AND DESCRIBING THE FINAL CLUSTERS - TESTING DATA

#The optimal number for K 
optimal_k_test = 3

#Fit KMeansWrapper with optimal clusters 
clustering_obj_BOTH = KMeansWrapper(n_clusters=optimal_k_test)
cluster_labels_test = clustering_obj_BOTH.fit_predict(scaled_test_array)

# add clusters labels to the training set
cluster_labelled_test = df2_test.copy() #Note we are using the dataset with YES/NO of conditions (DIFFERENT THAN CLUSTERING)
cluster_labelled_test['Cluster'] = cluster_labels_test

# Describe the final clusters - using Table 1 package (and the predefined variable lists )
table1_test_BOTH = TableOne(cluster_labelled_test, columns=variables, categorical=categorical, nonnormal=nonnormal, groupby='Cluster', pval=True, smd=True)
print(table1_test_BOTH.tabulate(tablefmt = 'fancy_grid'))
table1_test_BOTH.to_csv(FILE_NAME)

# Describe the final clusters - using fucntion above
described_clusters_test = describe_clusters_conditions(cluster_labelled_test)
described_clusters_test.to_csv(FILE_NAME)

#Export the labelled non-scaled data
df_test_age_EXPORT['Cluster'] = cluster_labels_test
df_test_age_EXPORT.to_csv(FILE_NAME)



exit()

####################################
#FOR MALES 
####################################

#################
#FITTING FOR MALES 
#Creating a clustering object for males
consensus_clusters = ConsensusClustering(clustering_obj=clustering_obj,
                                         min_clusters=2, max_clusters=6,
                                         n_resamples=50,  # Number should be enough to search the space of combinations but at a point will have diminishing returns
                                         resample_frac=0.8,
                                         k_param='n_clusters')  # number of resampling iteration

consensus_clusters.fit(scaled_train_array_M, progress_bar=True)

#Consensus Cumulative Distribution Function plots
cumulative_distribuiton_plots(consensus_clusters, 'M', 'train')

#optimal K from change in auc
consensus_clusters.best_k('change_in_auc')

#optimal K from knee in auc
knee = consensus_clusters.best_k('knee')

# Get the Consensus Matrix
#consensus_matrix = consensus_clusters.consensus_matrices_

#Plot each matrix - for the different values of K 
#for index, matrix in enumerate(consensus_matrix):
#    plt.figure(figsize=(6,6))
#    plt.imshow(matrix, cmap='Blues', aspect='auto')
#    plt.colorbar()
#    plt.title(f'Consensus Matrix {index+1}')
#    plt.xlabel('Sample Index')
#    plt.ylabel('Sample Index')
    #plt.savefig(FILE_NAME,dpi=700)
    #plt.show()


#Plot the consensus clustermap from 2-6 
clustermap_plot(consensus_clusters, 'M', 'train')

#Currently set to plot the histogram for the consensus matrix for K=2, K=3, K=4
_, axes = plt.subplots(1, 5, figsize=(16,4))
for i, ax in enumerate(axes): 
    consensus_clusters.plot_hist(i + 2, ax=ax)
    ax.set_title(i+2)
plt.tight_layout()
plt.savefig(FILE_NAME, dpi=700)
plt.show()


###########MALES - CREATING THE FINAL CLUSTERS - TRAINING DATA###########

#The optimal number for K 
optimal_k_train_M = 3

#Fit KMeansWrapper with optimal clusters 
clustering_obj_M = KMeansWrapper(n_clusters=optimal_k_train_M)
cluster_labels_train_M = clustering_obj_M.fit_predict(scaled_train_array_M)

# add clusters labels to the training set
cluster_labelled_train_M = df2_train_M.copy() #Note we are using the dataset with YES/NO of conditions (DIFFERENT THAN CLUSTERING)
cluster_labelled_train_M['Cluster'] = cluster_labels_train_M

# Describe the final clusters - using Table 1 package (and the predefined variable lists )
table1_train_M = TableOne(cluster_labelled_train_M, columns=variables, categorical=categorical, nonnormal=nonnormal, groupby='Cluster', pval=True, smd=True)
print(table1_train_M.tabulate(tablefmt = 'fancy_grid'))
table1_train_M.to_csv(FIILE_NAME)

# Describe the final clusters - using fucntion above
described_clusters_M = describe_clusters_conditions(cluster_labelled_train_M)
described_clusters_M.to_csv(FILE_NAME)

#Export the labelled non-scaled data
df_train_age_M_EXPORT['Cluster'] = cluster_labels_train_M
df_train_age_M_EXPORT.to_csv(FILE_NAME)

################################################
####### TESTING DATA - FOR MALES ############
#Creating a clustering object for males
cc_male_test = ConsensusClustering(clustering_obj=clustering_obj,
                                         min_clusters=2, max_clusters=6,
                                         n_resamples=50,  # Number should be enough to search the space of combinations but at a point will have diminishing returns
                                         resample_frac=0.8,
                                         k_param='n_clusters')  # number of resampling iteration

cc_male_test.fit(scaled_test_array_M, progress_bar=True)

#Consensus Cumulative Distribution Function plots
cumulative_distribuiton_plots(cc_male_test, 'M', 'test')

#Plot the consensus clustermap from 2-6 
clustermap_plot(cc_male_test, 'M', 'test')

#Currently set to plot the histogram for the consensus matrix for K=2, K=3, K=4
_, axes = plt.subplots(1, 5, figsize=(16,4))
for i, ax in enumerate(axes): 
    cc_male_test.plot_hist(i + 2, ax=ax)
    ax.set_title(i+2)
plt.tight_layout()
plt.savefig(FILE_NAME, dpi=700)
plt.show()


##########
#CREATING THE FINAL CLUSTERS - MALE TESTING 

#The optimal number for K 
optimal_k_test_M = 3

#Fit KMeansWrapper with optimal clusters 
clustering_obj_M = KMeansWrapper(n_clusters=optimal_k_test_M)
cluster_labels_test_M = clustering_obj_M.fit_predict(scaled_test_array_M)

# add clusters labels to the training set
cluster_labelled_test_M = df2_test_M.copy() #Note we are using the dataset with YES/NO of conditions (DIFFERENT THAN CLUSTERING)
cluster_labelled_test_M['Cluster'] = cluster_labels_test_M

# Describe the final clusters - using Table 1 package (and teh predefined variable lists )
table1_test_M = TableOne(cluster_labelled_test_M, columns=variables, categorical=categorical, nonnormal=nonnormal, groupby='Cluster', pval=True, smd=True)
print(table1_test_M.tabulate(tablefmt = 'fancy_grid'))
table1_test_M.to_csv(FILE_NAME)

# Describe the final clusters - using function above
described_clusters_test_M = describe_clusters_conditions(cluster_labelled_test_M)
described_clusters_test_M.to_csv(FILE_NAME)

#Export the labelled non-scaled data
df_test_age_M_EXPORT['Cluster'] = cluster_labels_test_M
df_test_age_M_EXPORT.to_csv(FILE_NAME)


####################################
#FOR FEMALES 
####################################

#################
#FITTING FOR FEMALES ON THE TRAINING DATA
consensus_clusters_F = ConsensusClustering(clustering_obj=clustering_obj,
                                         min_clusters=2, max_clusters=6,
                                         n_resamples=50,  # Number should be enough to search the space of combinations but at a point will have diminishing returns
                                         resample_frac=0.8,
                                         k_param='n_clusters')  # number of resampling iteration

consensus_clusters_F.fit(scaled_train_array_F, progress_bar=True)

#Consensus Cumulative Distribution Function plots
cumulative_distribuiton_plots(consensus_clusters_F, 'F', 'train')

#Plot the consensus clustermap from 2-6 
clustermap_plot(consensus_clusters_F, 'F', 'train')

#Currently set to plot the histogram for the consensus matrix for K=2, K=3, K=4
_, axes = plt.subplots(1, 5, figsize=(16,4))
for i, ax in enumerate(axes): 
    consensus_clusters_F.plot_hist(i + 2, ax=ax)
    ax.set_title(i+2)
plt.tight_layout()
plt.savefig(FILE_NAME, dpi=700)
plt.show()


##########
#FEMALES - CREATING AND DESCRIBING THE FINAL CLUSTER STRUCTURE - TRAINING DATA

#The optimal number for K 
optimal_k_train_F = 3

#Fit KMeansWrapper with optimal clusters 
clustering_obj_F = KMeansWrapper(n_clusters=optimal_k_train_F)
cluster_labels_train_F = clustering_obj_F.fit_predict(scaled_train_array_F)

# add clusters labels to the training set
cluster_labelled_train_F = df2_train_F.copy() #Note we are using the dataset with YES/NO of conditions (DIFFERENT THAN CLUSTERING)
cluster_labelled_train_F['Cluster'] = cluster_labels_train_F

# Describe the final clusters - using Table 1 package (and teh predefined variable lists )
table1_train_F = TableOne(cluster_labelled_train_F, columns=variables, categorical=categorical, nonnormal=nonnormal, groupby='Cluster', pval=True, smd=True)
print(table1_train_F.tabulate(tablefmt = 'fancy_grid'))
table1_train_F.to_csv(FILE_NAME)

# Describe the final clusters - using function above
described_clusters_F = describe_clusters_conditions(cluster_labelled_train_F)
described_clusters_F.to_csv(FILE_NAME)

#Export the labelled non-scaled data
df_train_age_F_EXPORT['Cluster'] = cluster_labels_train_F
df_train_age_F_EXPORT.to_csv(FILE_NAME)


#####################################################
#FITTING FOR FEMALES ON THE TESTING DATA 
consensus_clusters_F = ConsensusClustering(clustering_obj=clustering_obj,
                                         min_clusters=2, max_clusters=6,
                                         n_resamples=50,  # Number should be enough to search the space of combinations but at a point will have diminishing returns
                                         resample_frac=0.8,
                                         k_param='n_clusters')  # number of resampling iteration

consensus_clusters_F.fit(scaled_test_array_F, progress_bar=True)

#Consensus Cumulative Distribution Function plots
cumulative_distribuiton_plots(consensus_clusters_F, 'F', 'test')

#Plot the consensus clustermap from 2-6 
clustermap_plot(consensus_clusters_F, 'F', 'test')

#Currently set to plot the histogram for the consensus matrix for K=2, K=3, K=4
_, axes = plt.subplots(1, 5, figsize=(16,4))
for i, ax in enumerate(axes): 
    consensus_clusters_F.plot_hist(i + 2, ax=ax)
    ax.set_title(i+2)
plt.tight_layout()
plt.savefig(FILE_NAME, dpi=700)
plt.show()


##########
#FEMALES - SELECTING THE FINAL / OPTIMAL CLUSTER NUMBER - TESTING DATA

#The optimal number for K 
optimal_k_test_F = 3

#Fit KMeansWrapper with optimal clusters 
clustering_obj_F = KMeansWrapper(n_clusters=optimal_k_test_F)
cluster_labels_test_F = clustering_obj_F.fit_predict(scaled_test_array_F)

# add clusters labels to the training set
cluster_labelled_test_F = df2_test_F.copy() #Note we are using the dataset with YES/NO of conditions (DIFFERENT THAN CLUSTERING)
cluster_labelled_test_F['Cluster'] = cluster_labels_test_F

# Describe the final clusters - using Table 1 package (and teh predefined variable lists )
table1_test_F = TableOne(cluster_labelled_test_F, columns=variables, categorical=categorical, nonnormal=nonnormal, groupby='Cluster', pval=True, smd=True)
print(table1_test_F.tabulate(tablefmt = 'fancy_grid'))
table1_test_F.to_csv(FILE_NAME)

# Describe the final clusters - using function above
described_clusters_test_F = describe_clusters_conditions(cluster_labelled_test_F)
described_clusters_test_F.to_csv(FILE_NAME)

#Export the labelled non-scaled data
df_test_age_F_EXPORT['Cluster'] = cluster_labels_test_F
df_test_age_F_EXPORT.to_csv(FILE_NAME)





########################
# VISUALIZING THE CLUSTERS
########################

#   TSNE VISUALIZATION

#Plot the tsne plots 
plot_tsne(df_train, cluster_labels_train, sex='BOTH', train='train')
plot_tsne(df_train_M, cluster_labels_train_M, sex='M', train='train')
plot_tsne(df_train_F, cluster_labels_train_F, sex='F', train='train')

plot_tsne(df_test, cluster_labels_test, sex='BOTH', train='test')
plot_tsne(df_test_M, cluster_labels_test_M, sex='M', train='test')
plot_tsne(df_test_F, cluster_labels_test_F, sex='F', train='test')


#   UMAP VISUALIZATION

#Plot the UMAP plots 
umap_plot(df_train_age, cluster_labels_train, sex='BOTH', train='train')
umap_plot(df_train_age_F, cluster_labels_train_F, sex='F', train='train')
umap_plot(df_train_age_M, cluster_labels_train_M, sex='M', train='train')

umap_plot(df_test_age, cluster_labels_test, sex='BOTH', train='test')
umap_plot(df_test_age_F, cluster_labels_test_F, sex='F', train='test')
umap_plot(df_test_age_M, cluster_labels_test_M, sex='M', train='test')

cluster_labels_train_F = np.where(cluster_labels_train_F==0, 4, cluster_labels_train_F)
cluster_labels_train_F = np.where(cluster_labels_train_F==1, 5, cluster_labels_train_F)
cluster_labels_train_F = np.where(cluster_labels_train_F==2, 3, cluster_labels_train_F)


#   CHORD DIAGRAM

# for cluster_id in cluster_labelled_train['Cluster'].unique():
#    plt.figure(figsize=(8,8))
#    cluster_df = cluster_labelled_train[cluster_labelled_train['Cluster'] == cluster_id]
#    chord_matrix = cluster_df.corr().values
#    features = cluster_df.columns[:-1]

# Create chord diagram
#    chord = circos.Chord(chord_matrix, labels=features)
#    chord.plot()
#    plt.title(f'Chord Diagram for Cluster {cluster_id}')
# plt.savefig('FILE_NAME_{cluster_id}.png', dpi=700)
#    plt.show()


