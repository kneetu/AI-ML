## Predict Job Failures Based on System Requirements and Resource Allocations


**Author**
Neetu Kulshrestha

### Jupyter notebook

https://github.com/kneetu/AI-ML/blob/main/capstone/predict_job_failure_in_system.ipynb

The **data** directory contains 2 csv files that are used as input
- dcgm.csv: Low Level data Set
- scheduler_data.csv: High Level data Set 

The **resource** directory contains the details files for certain features. The file is not directory used in the program, but was used for understanding the feature values in data set. 
- JobStates.csv : Details of the job "states"


### Executive summary

#### Project Background, Overview and Goals

**Background**: In a typical software system, there are mutiple 'Individual Scripts(AKA Jobs)' created and executed on a scheduled basis. These jobs are responsible of maintaining server states and are outside of the programatic boundary of microservices running on the system. Some examples of these jobs are: 
- copy the data from one disk to other
- compress the log files
- cleanup the file system by removing the old data/log files etc.

These jobs usually gets triggered either by scheduled time or the server state. In either case, they do consume server resources like CPU/GPU/Memory etc. These resources are shared by the microservices deployed on the same instances and hence may impact the functionality of the server. 

If a certain job fails, it may retry the operation, hence consume resources for longer period of time. In a worse case, a job may hold on to resources in a suspended state only to fail at the end. Please note some of these jobs runs for hours. 

Imagine you are watching an exciting match on TV and right at the final goal, your video stream froze. 

**Overview and Goals**: The goal of this project to identify the jobs that may fail in a production system and hence may affect the performance/functionality of the whole system.
The project focuses on certain features of the jobs that are known before it's even deployed to production and predict the failure for a certain job. Developers may fine-tune their script(job) and the system requirements based on the findings and hence prevent a production failure.
The project merges the 2 dataframes, 1) high level workload manager data set and 2) job specific detailed data set, in order to create a single Dataframe for modeling. Multiple Features are removed during the EDA process based on following criterian:
- If a column contains all 'Null' values
- If a column is NOT typically known for a new job
- If multiple columns are highly correlated to each other
-- Only the column with highest correlation with target column is kept
- If a column contains very high number of unique values, for example IDs of any sort
The project runs through a total of 6 different classifiers to find the best model and runs feature selection on it. The model's parameters will be adjusted based on teh confusion matrix's false-positives and false-negatives. 
Final conclusions are drawn based on the insights gained. 
Accuracy score, overfitting and fit time will be used for comparing the performances of the models. The weights are used for best feature selecion. 
The project utilizes the default settings for all the models, except for random state and expects to fine tune them in later stages.

#### Findings

Based on the accuracy score and fit time, XGBClassifier was the best model, which provides high accuracy with no overfitting. The train and test score for this model was 0.920085 and 0.906388 respectively and the fit time 1.406158. The Decision Tree and RandomForectClassifier both had very high train_score(~.99) , but comparatively lower test score(<.90), hence indicating overfitting. All other models stayed at or below .75 train/test scores, with SVC fit_time being the highest.
The confusion matrix was created only for the best model and it showed slightly higher false-positives than false-negatives, indicating that there could be some good jobs that may be marked as potential failing jobs. This works in the favor of the problem, as the problem statement benefits from a conservative approach, rather than a risky approach(false-negatives). 
Having said that, We don't want to increase the false-negatives as that would question the predictability of the model and would result in abandoning the model all-together.

The most important feature for the XGBClassifer is totalexecutiontime_sec, followed by timelimit, cpu_req and memory_req. This analysis shows that the total execution time for a job plays a significant role in job failure, if the job has been running for long time, then it's usually because it's struggling for resources. Combine it with timelimit set and cpu/memory requested in the begining of execution and we get our predictions. 

### Rationale

In a system where many jobs run on scheduled basis, high numbers of resources are allocated. The jobs are usually tested in a controlled test environment using manual testing methodologies. The testing though provide some level of confidence, it cannot guarntee or predict the resource requirements in production environments where data is scaled and systems are more powerful.
The ML model can plug in values for initial estimate of system requirements and predict if a job will fail with those or not. This can be used to plan the resources in advance and increase reliability of the system.

### Research Question

Can a ML model effectively predict the job failure based on system requirements and allocations? if yes, what are the important features, affecting the states?

### Data Sources
https://www.kaggle.com/datasets/skylarkphantom/mit-datacenter-challenge-data

**About Dataset** Datacenter monitoring systems offer a variety of data streams and events. The Datacenter Challenge datasets are a combination of high-level data (e.g. Slurm Workload Manager scheduler data) and low-level job-specific time series data. The high-level data includes parameters such as the number of nodes requested, number of CPU/GPU/memory requests, exit codes, and run time data. The low-level time series data is collected on the order of seconds for each job. This granular time series data includes CPU/GPU/memory utilization, amount of disk I/O, and environmental parameters such as power drawn and temperature. Ideally, leveraging both high-level scheduler data and low-level time series data will facilitate the development of AI/ML algorithms which not only predict/detect failures, but also allow for the accurate determination of their cause.

High level data set contains 287173 rows and 31 columns, while low level data set contains 96893 rows and 23 columns originally. Both of the data sets contain a common column id_job, that can be used as index for merging.

### EDA
EDA involves the exploring and cleaning up the data for:
- the 2 data sets provided
- Clean up the features from high level data set based on correlation, number of unique values, and null values
- merge 2 datasets
- Apply cleanup, get_dummies and data manipulation on the merged data set

**Target feature**: Since the project is predicting the job failure, the target feature is identified as exit_code. A particular job may fail due to various reasons, but for this project, the specific error code is not importatnt. Hence all non-zero error codes, that usually indicate failure are converted into 1, indicating failure. Success error code is represented by (default) 0.

**Data Exploration and Cleanup**: As mentioned above, mutually correlated columns were removed, since they all would produce the same results. Since high level data set contains the target feature, a correlation plot against 'exit_code" is used to select the feature to remain. Low level data set on the other hand does not contain the exit_code, hence the manual feature selection of this was delayed until both data sets are merged.

Some columns like gres_used and time_suspended contains only null values, hence these features were dropped. 

The cleanup of the data was conducted in multiple phases: 
phase 1: Cleanup of the data in high level data set
Phase 2: cleanup of merged data frame 
Phase 3: Final cleanup of data by removing duplicates. Although the original data set doesn't contain any duplicates, after dropping multiple features, the merged dataframe is bound to have some duplicate rows

Please note that the project takes an unusal decision on removing some features based on developer's knowledge of the system. During the initial phases of the model development, we were getting 100% score, which indicated issues with the model. Upon further analysis of best features, the error was evident. The low level data set contains many features from the job's execution in production system. These features are not going to be available for a job, that has never been executed in production. Hence a decision was taken to remove all such features from the data frame before modeling.

**Merged DataFrame**: id_job was used as index to merge the 2 databases using inner join. High Level data set contains duplicate values for id_job, which needed to be removed. Upon inspecting the data, it was found that the reason of duplicate values of id_job was that when a job failed, it retried until it succeeded. The data set contain all those occurances. Since the project study the job failures, the instance of id_job was kept, that shows error_code as 1. In simple terms a job failed at least once before getting successful. We need that 1 failed condition and throw away the success condition.

**Data Preparation**: get_dummy() function is used to convert categorical features into the dummy features. The dataset is then split into train, test data and further scaled usinf StandardScalar().

**Final Data Set**: Final data set contains 57044 rows and 35 columns, where all columns are of numeric types. "exit_code" serves as the target feature and doesn't seem to be heavily skewed. 

0    42276
1    14768
Name: exit_code, dtype: int64

### Methodology

**Model comparision**
The project compares the performance of 6 different classifiers, 1) DecisionTree, 2) Logisticregression, 3) SVC, 4) RandomForest, KNNeighers and 6) XGBoost 
The model performance will be determined based on the accuracy score of the train, test and the fit_time of the model. 

The default parameters will be used to determine the best model, but the parameters will be finetuned on the best model to get desired confusion matrix, and/or feature selection.

**Final Modeling**
The best model from above step will be used to understand the confusion matrix,best parameters and features

### Results

**Summary**: KGBClassifier gives the best performance and hence was used as final model to understand the important features.

- DecisionTree although generates highest Train score, it seems to be overfitting with train score > .99 and test score <.9
- The same is the case with RandomForextClassifier, which uses DecisionTree as well. Its high score, bt overfitting
- SVC model's average fit_time is very high (~50s), and the score is poor even though test score is very close to train score
- XGBClassifier is the best performing in terms of accuracy score without overfitting. The fitting time is higher than KNN and LogixticRegression, but it's in accepatble range(~1.14). 

The model provide very low number of false positives and false-negatives compared to the correct predictions. For this problem statement, having slightly higher false positives are preferable as that prevent the production systems from unlikely circumstances. The program doesn't try to reduce the false-negatives though, because that may raise questions on the model's reliability and will cause unnecessary delays in the production releases.

As per the model, best feature analysis, total execution time is the feature with highest importance, with time line, CPU_req and Memory_req following. This makes sense that if a job is running for prolonged period, it blocks the system resources for that long time, and hence may end up in failing state. 


### Outline of project
Project Markdown is copied in the text file: 
https://github.com/kneetu/AI-ML/blob/main/capstone/predict_job_failure_in_system_markdown.txt

### Next Steps and Recommendations

- Totalexecutiontime_sec is believed to be available from the test system. The model should be expanded to analyse if this feature is not available. 
- During the various phases of data cleanup, many of the features were dropped. Project can be further modified to see how many low level data set features are used, and what is their importance. Can the prediction be made solely from high level data set?
- Does local analysis provide any more insights to the results than what is already provided using the most importatnt features? if yes, extend the project to incluse the analysis.


### Citation:
S. Samsi et al., "The MIT Supercloud Dataset," 2021 IEEE High Performance Extreme Computing Conference (HPEC), Waltham, MA, USA, 2021, pp. 1-8, doi: 10.1109/HPEC49654.2021.9622850. Samsi, Siddharth, Weiss, Matthew, Bestor, David, et al. "The MIT Supercloud Dataset." 2021 IEEE High Performance Extreme Computing Conference (HPEC). IEEE, 2021.

### References:
https://learn.microsoft.com/en-us/dotnet/api/system.management.automation.jobstate?view=powershellsdk-7.3.0
https://learn.microsoft.com/en-us/dotnet/api/microsoft.hpc.scheduler.properties.jobstate?view=hpc-sdk-5.1.6115
https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
 



