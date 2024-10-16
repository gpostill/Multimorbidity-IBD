# **Multimorbidity in Individuals with Inflammatory Bowel Disease - Supervised and Unsupervised Machine Learning** 

This page explores multimorbidity in individuals with Inflammatory Bowel Disease (IBD) using health administrative data from ICES. The subset of IBD individuals is drawn from the Ontario Crohn's and Colitis Cohort, a comprehensive database tracking health outcomes in individuals with IBD in Ontario. We take a life-course perspective with such research, assessing decedents with IBD who died between 2010-2020 and looking back on their entire life to analyze how conditions occur 

The project is divided into two sub-projects:
  1. **Predicting Premature Death from Non-IBD Chronic Conditions:** This sub-project examines the role of non-IBD chronic conditions in predicting premature mortality among individuals with IBD. Premature death is ddefined as death prior to age 75 years. By analyzing patterns of comorbidities, the goal is to understand which conditions contribute most significantly to early death in this population. 
  2. **Identifying Clusters of Chronic Conditions among those with IBD:** The second sub-project focuses on identifying clusters of non-IBD chronic conditions that commonly co-occur in individuals with IBD. This analysis aims to reveal patterns of multimorbidity that may influence health outcomes and healthcare needs. Our approach is unique in that we encode chronology of condition accumulation into the clusters ensuring that our clusters reflect not just what conditions occur together but how they occur. 

**Files pertaining to project 1 (predicting premature death):**
[Files to be added]
- Task 1 (Predicting premature mortality from all conditions): PredictiveTask1(Oct2024).py
- Task 2 (Predicting premature mortality from conditions developed before 60 years): PredictiveTask2(Oct2024).py
- Task 3 (Predicting premature mortality from conditions developed before 60 years using age of conditions): PredictiveTask3(Oct2024).py
- Cohort description: M1 Descritive Table.py

**Files pertaining to project 2 (multimorbidity clusters):**
- Cohort Description: M1 Descritive Table.py
- Accumulation of Chronic Conditions Figure: CummulativeConditions_Figure.py
- Cluster Derivation: Clustering_Age_Condition(July2024).py
- Cluster Validation: CC_Validation(Sep2024).py


The findings from this project will contribute to a deeper understanding of how multimorbidity affects individuals with IBD and help inform strategies for improving long-term health outcomes and care.
