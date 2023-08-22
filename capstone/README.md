### Project Title
Predict Job failures in Advance based on system Requirements and resource allocations

**Author**
Neetu Kulshrestha

#### Jupyter notebook

https://github.com/kneetu/AI-ML/blob/main/capstone/PredictJobFailureInSystem.ipynb

The **Data** directory contains 2 csv files that are used as input
- dcgm.csv - Low Level data for jobs
- scheduler_data.csv: High Level data set 
The **resource** directory contains the details files for certain features:
- JobStates.csv : Details of the job "states"

#### Executive summary
Many Jobs run in the system and some of them run for hours/days. It uses system resources unnecessarily and may bring system performance down. What if a model can predict if a certain job will fail when executed. 

#### Rationale
In a system where many jobs run on scheduled basis, high numbers of resources are allocated. The jobs are usually tested in a controlled test environment using manual testing methodologies. The testing though provide some level of confidence, it cannot guarntee or predict the resource requirements in Production environments where data is scaled and systems are more powerful.
The ML model can plug in values for initial estimate of system requirements and predict if a job will fail with those or not. This can be used to plan the resources in advance and increase reliability of the system.

#### Research Question
Can a ML model effectively predict the job failure based on system requirements and allocations?

#### Data Sources
https://www.kaggle.com/datasets/skylarkphantom/mit-datacenter-challenge-data
**About Dataset** Datacenter monitoring systems offer a variety of data streams and events. The Datacenter Challenge datasets are a combination of high-level data (e.g. Slurm Workload Manager scheduler data) and low-level job-specific time series data. The high-level data includes parameters such as the number of nodes requested, number of CPU/GPU/memory requests, exit codes, and run time data. The low-level time series data is collected on the order of seconds for each job. This granular time series data includes CPU/GPU/memory utilization, amount of disk I/O, and environmental parameters such as power drawn and temperature. Ideally, leveraging both high-level scheduler data and low-level time series data will facilitate the development of AI/ML algorithms which not only predict/detect failures, but also allow for the accurate determination of their cause.

#### Methodology
**Initial Modeling**
LogisticRegression will be used as initial modeling to find the important features. 
**Model comparision**
The project will try and compare the performance of 4 different classifiers, DecisionTree, Logisticregression AND SVC for the "mportant features" only
**Final Modeling**
The best model from above step will be used to understand teh best parameters and features

#### Results
**Summary**: DecisionTree gave 100% accurate results based on the state, that create questions about the model and overfitting it. The test dataset also returned 100% accuracy. All the various method of getting the importtant features shows high coeeficient for the feature "State". This is evident from LogisticRegression, and also from DecisionTree while plotting it using Plot_tree.
Upon further analysis, this make sense. The model and data prep missed one importtant point, that "state" will not be present for future jobs. Additionally, State indicate if Job was successful or not, hence such high correlation with exit_code is self-explanatory.

**Error In the Model**:  State is not available for any of the future jobs, so it should have been removed from the data set. Same applies to time_start, time_end etc. All these columns should be dropped before applying the models on it. 

#### Next steps
**Code cleanup**
Due to teh first attempt exploring teh dataset, there are quite a lot of repetion of code. The data cleanup/Exploration and prep part should be cleaned up.
**MOdel Improvement**
- adjust false-positives, false-negatives as per the problem
- Add Random Forest and XGBoost to the list of models to compare performance and try again
- Remove the first logistic regression, as this is causing rest of the models to work against a linearly biased model's results

#### Outline of project
- [Predict Job failures in advance](#Predict-Job-failures-in-advance)
  - [Data](#Data)
  - [Citation:](#Citation:)
- [Import Libraries](#Import-Libraries)
- [Load Data](#Load-Data)
    - [Data Exploration (low level DGCM Data)](#Data-Exploration-(low-level-DGCM-Data))
      - [BiVariant Analysis](#BiVariant-Analysis)
      - [Drop Columns with very high correlation](#Drop-Columns-with-very-high-correlation)
      - [Univariant Analysis](#Univariant-Analysis)
  - [Data Exploration (High level Schedular data)](#Data-Exploration-(High-level-Schedular-data))
- [Merge the 2 DataFrames](#Merge-the-2-DataFrames)
  - [id_job is the common field in both the databases, hence will be used for joining the DataFrames](#id_job-is-the-common-field-in-both-the-databases,-hence-will-be-used-for-joining-the-DataFrames)
    - [Explore id_job column in both DFs](#Explore-id_job-column-in-both-DFs)
  - [find and drop the duplicates](#find-and-drop-the-duplicates)
- [Cleanup Data on merged_df](#Cleanup-Data-on-merged_df)
    - [convert gres_alloc to int](#convert-gres_alloc-to-int)
    - [Drop highly skewed column 'constraint'](#Drop-highly-skewed-column-'constraint')
    - [Since Exit Code a target column, replace all of the code > 0 to 1, where 0 : Success and 1: Failure](#Since-Exit-Code-a-target-column,-replace-all-of-the-code->-0-to-1,-where-0-:-Success-and-1:-Failure)
  - [Final Exploration of merged table: confirm the columns and if any further cleanup is required](#Final-Exploration-of-merged-table:-confirm-the-columns-and-if-any-further-cleanup-is-required)
    - [Simplify job_type](#Simplify-job_type)
    - [Handle NaNs and Duplicates](#Handle-NaNs-and-Duplicates)
    - [Idnetify highly correlated features for target features and drop all others](#Idnetify-highly-correlated-features-for-target-features-and-drop-all-others)
- [Final data prep for modeling](#Final-data-prep-for-modeling)
  - [Scale the data](#Scale-the-data)
  - [Prepare Categorical Data](#Prepare-Categorical-Data)
- [Test Train Split](#Test-Train-Split)
- [Modeling](#Modeling)
  - [Find Significant Features using Logistic Regression](#Find-Significant-Features-using-Logistic-Regression)
    - [Train Test split with 10 most importance features](#Train-Test-split-with-10-most-importance-features)
  - [Apply Various models and compare the performances](#Apply-Various-models-and-compare-the-performances)
    - [Create Dataframe to capture performances of various models](#Create-Dataframe-to-capture-performances-of-various-models)
  - [Confusion Matrix Using decisionTreeClassifier](#Confusion-Matrix-Using-decisionTreeClassifier)
    - [DecisionTree Model Predictions](#DecisionTree-Model-Predictions)
  - [Get Decision Tree features](#Get-Decision-Tree-features)
  - [Plot Decision Tree](#Plot-Decision-Tree)
- [Conclusion](#Conclusion)


#### 
#### Citation:
S. Samsi et al., "The MIT Supercloud Dataset," 2021 IEEE High Performance Extreme Computing Conference (HPEC), Waltham, MA, USA, 2021, pp. 1-8, doi: 10.1109/HPEC49654.2021.9622850. Samsi, Siddharth, Weiss, Matthew, Bestor, David, et al. "The MIT Supercloud Dataset." 2021 IEEE High Performance Extreme Computing Conference (HPEC). IEEE, 2021.

#### References:
https://learn.microsoft.com/en-us/dotnet/api/system.management.automation.jobstate?view=powershellsdk-7.3.0
https://learn.microsoft.com/en-us/dotnet/api/microsoft.hpc.scheduler.properties.jobstate?view=hpc-sdk-5.1.6115
 


