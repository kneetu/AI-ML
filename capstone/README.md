### Project Title
Predict Job failures in Advance based on system Requirements and resource allocations

**Author**
Neetu Kulshrestha

#### Jupyter notebook

https://github.com/kneetu/AI-ML/blob/main/capstone/predict_job_failure_in_system.ipynb

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
 

#### EDA
EDA involves the exploring and cleaning up the data for:
- the 2 data sets provided
- Clean up the features from high level data set based on correlation, number of unique values, and null values
- merge 2 datasets
- Apply cleanup, get_dummies and data manipulation on the merged data set

#### Methodology

**Model comparision**
The project will try and compare the performance of 6 different classifiers, DecisionTree, Logisticregression, SVC, RandomForest and XGB 

**Final Modeling**
The best model from above step will be used to understand the confusion matrix,best parameters and features

#### Results
**Summary**: KGBClassifier gives the best performance and hence was used as final model to understand the important features

The model provide very low number of false positives and false-negatives compared to the correct predictions. For this problem statement, having slightly higher false positives are preferable as that prevent the production systems from unlikely circumstances. The program doesn't try to reduce the false-negatives though, because that may raise questions on the model's reliability. 

As per the model, best feature analysis, total_execution time is the feature with highest importance, with CPU_req and Memory_req following. This makes sense that if a job is running for prolonged period, it blocks the system resources for that long time, and hence may end up in failing state. 



#### Outline of project
 % ./printMarkDown.py kraftwerk/capstone/predict_job_failure_in_system.ipynb
 
- [Predict Job failures in advance for system jobs](#Predict-Job-failures-in-advance-for-system-jobs)
  - [Data](#Data)
  - [Citation:](#Citation:)
- [Import Libraries](#Import-Libraries)
- [Load Data](#Load-Data)
  - [Load High Level Data Set Scheduler](#Load-High-Level-Data-Set-Scheduler)
  - [Load Low level Data set DCGM](#Load-Low-level-Data-set-DCGM)
- [Exploratory Data Analysis (EDA)](#Exploratory-Data-Analysis-(EDA))
  - [Data Exploration (High level Schedular data)](#Data-Exploration-(High-level-Schedular-data))
    - [Bi-Variate Analysis](#Bi-Variate-Analysis)
      - [drop column time_submit and time_end, highly correlated with time_start](#drop-column-time_submit-and-time_end,-highly-correlated-with-time_start)
      - [Inspect and drop empty features](#Inspect-and-drop-empty-features)
    - [UniVariate Analysis](#UniVariate-Analysis)
      - [drop the columns that has all NaNs](#drop-the-columns-that-has-all-NaNs)
      - [Drop the columns with complicated values and high unique numbers](#Drop-the-columns-with-complicated-values-and-high-unique-numbers)
  - [Data Exploration (low level DGCM Data)](#Data-Exploration-(low-level-DGCM-Data))
- [Merge High level(Schedular) and Low level(DCGM) DataFrames](#Merge-High-level(Schedular)-and-Low-level(DCGM)-DataFrames)
  - [Explore id_job column in both DFs](#Explore-id_job-column-in-both-DFs)
    - [find and drop the duplicates](#find-and-drop-the-duplicates)
  - [Set Index to id_job for High level data set](#Set-Index-to-id_job-for-High-level-data-set)
  - [Set Index to id_job for Low Level Data set](#Set-Index-to-id_job-for-Low-Level-Data-set)
  - [Merge DFs](#Merge-DFs)
- [Data Exploration and Cleanup on merged_df](#Data-Exploration-and-Cleanup-on-merged_df)
  - [Drop columns with high number of Unique values](#Drop-columns-with-high-number-of-Unique-values)
  - [BiVariate Analysis](#BiVariate-Analysis)
    - [correlation heatmap for all columns](#correlation-heatmap-for-all-columns)
    - [Get Correaltion with taget 'exit_code'](#Get-Correaltion-with-taget-'exit_code')
    - [Drop highly correlated Columns with least correlation to target, Drop Columns with all NaN](#Drop-highly-correlated-Columns-with-least-correlation-to-target,-Drop-Columns-with-all-NaN)
    - [Drop the columns that are not available for new jobs](#Drop-the-columns-that-are-not-available-for-new-jobs)
  - [Univariate Analysis](#Univariate-Analysis)
    - [convert gres_alloc and gres_req to int](#convert-gres_alloc-and-gres_req-to-int)
    - [Drop highly skewed column 'constraint'](#Drop-highly-skewed-column-'constraint')
    - [Since Exit Code a target column, replace all of the code > 0 to 1, where 0 : Success and 1: Failure](#Since-Exit-Code-a-target-column,-replace-all-of-the-code->-0-to-1,-where-0-:-Success-and-1:-Failure)
    - [Simplify job_type column](#Simplify-job_type-column)
  - [Final Exploration of merged table: confirm the columns and if any further cleanup is required](#Final-Exploration-of-merged-table:-confirm-the-columns-and-if-any-further-cleanup-is-required)
    - [Handle NaNs and Duplicates](#Handle-NaNs-and-Duplicates)
    - [Identify highly correlated features for target features and drop all others](#Identify-highly-correlated-features-for-target-features-and-drop-all-others)
    - [Drop Duplicates](#Drop-Duplicates)
- [Data Preparation for modeling](#Data-Preparation-for-modeling)
  - [Prepare Categorical Data](#Prepare-Categorical-Data)
- [Test Train Split](#Test-Train-Split)
  - [Scale the data](#Scale-the-data)
- [Modeling](#Modeling)
  - [Apply Various models and compare the performances](#Apply-Various-models-and-compare-the-performances)
    - [Create Dataframe to capture performances of various models](#Create-Dataframe-to-capture-performances-of-various-models)
    - [Apply models and manually compare the performances](#Apply-models-and-manually-compare-the-performances)
    - [CONCLUSION](#CONCLUSION)
  - [Confusion Matrix Using Best Performing Model: XGBClassifier](#Confusion-Matrix-Using-Best-Performing-Model:-XGBClassifier)
    - [XGBClassifier Model Predictions](#XGBClassifier-Model-Predictions)
    - [Confusion Matrix for Training Data](#Confusion-Matrix-for-Training-Data)
    - [Confusion Matrix for Test Data](#Confusion-Matrix-for-Test-Data)
    - [CONCLUSION](#CONCLUSION)
  - [Get features by importance](#Get-features-by-importance)
    - [Use feature_importance_ to get the percentage weight](#Use-feature_importance_-to-get-the-percentage-weight)
    - [use plot_importance function provided by xgb to get the raw importance.](#use-plot_importance-function-provided-by-xgb-to-get-the-raw-importance.)

#### 
#### Citation:
S. Samsi et al., "The MIT Supercloud Dataset," 2021 IEEE High Performance Extreme Computing Conference (HPEC), Waltham, MA, USA, 2021, pp. 1-8, doi: 10.1109/HPEC49654.2021.9622850. Samsi, Siddharth, Weiss, Matthew, Bestor, David, et al. "The MIT Supercloud Dataset." 2021 IEEE High Performance Extreme Computing Conference (HPEC). IEEE, 2021.

#### References:
https://learn.microsoft.com/en-us/dotnet/api/system.management.automation.jobstate?view=powershellsdk-7.3.0
https://learn.microsoft.com/en-us/dotnet/api/microsoft.hpc.scheduler.properties.jobstate?view=hpc-sdk-5.1.6115
https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
 


