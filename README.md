### AI-ML
#######################################################################################

## Assignment 17.1
# Link to the Jupityr Notebook:
https://github.com/kneetu/AI-ML/blob/main/assignment_17_1/practical_17_1.ipynb
# Summary of findings for Assignment 11_1:
**Objective**: The classification goal is to predict if the client will subscribe a term deposit. The goal of the exercise to compare the classification models in terms of their performance and accuracy 
**DataSet** : The data is related with direct marketing campaigns of a Portuguese banking institution. For this excercise we picked bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010)
RangeIndex: 41188 entries, 0 to 41187
Data columns (total 21 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   age             41188 non-null  int64  
 1   job             41188 non-null  object 
 2   marital         41188 non-null  object 
 3   education       41188 non-null  object 
 4   default         41188 non-null  object 
 5   housing         41188 non-null  object 
 6   loan            41188 non-null  object 
 7   contact         41188 non-null  object 
 8   month           41188 non-null  object 
 9   day_of_week     41188 non-null  object 
 10  duration        41188 non-null  int64  
 11  campaign        41188 non-null  int64  
 12  pdays           41188 non-null  int64  
 13  previous        41188 non-null  int64  
 14  poutcome        41188 non-null  object 
 15  emp.var.rate    41188 non-null  float64
 16  cons.price.idx  41188 non-null  float64
 17  cons.conf.idx   41188 non-null  float64
 18  euribor3m       41188 non-null  float64
 19  nr.employed     41188 non-null  float64
 20  y               41188 non-null  object 
dtypes: float64(5), int64(5), object(11)

Column 'Y' is the target or output column, that answers the binary Question: "has the client subscribed a term deposit?"

**Data cleanup**: The Original data set didn't include any NaN. In order to convert the data into meaningful Numeric data, we replces "yes", "no" and "unknown values to 1,0 and Nan. As a result there were NaNs created that were refilled with mean value of the column for Numeric column. rest of the Nans were dropped. Similarly all Duplicates were dropped as well. Resulting FInal 39180 entries.
Manual Feature cleanup: Based on teh correlation values, some of the features were dropped that were highly correlated to each other. Also dropped the features that showed 0 correlation to the Target.

**Process of analysis**: 
- analyse Numeric and Categorical data separtely
- identify the best Numeric and Categorical feature
- Apply the models on selected feature sets
- Compare the performance 

## Verdict
- As per the model comparision, it turns out to be that SVC is the slowest model. All of the models showed very high scores for both test and train data, indicating there overfitting was minimal.
model	train score	test score	average fit time
0	KNeighborsClassifier()	0.890488	0.882593	0.018238
1	LogisticRegression()	0.898724	0.895763	0.114167
2	DecisionTreeClassifier()	0.902978	0.896376	0.009241
3	SVC()	0.889740	0.885758	6.947943

- Based on above data, DecisionTreeClassifier is the best model for this data.
- According to the analysis the factors that affect the decision are:
   #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   contact_telephone  39180 non-null  uint8  
 1   month_aug          39180 non-null  uint8  
 2   month_jul          39180 non-null  uint8  
 3   month_may          39180 non-null  uint8  
 4   month_nov          39180 non-null  uint8  
 5   poutcome_success   39180 non-null  uint8  
 6   cons.price.idx     39180 non-null  float64

 Where the most impactful are: Consumer confidence index. If the rate is high then a consumer is most likely to subscribe. The 2 factors that seems to be affecting the confidence rate are euribor3m and emp.var.rate.
 It also matters if previous outcome was successful. 

**Next steps**
Need to go into deeper analysis for more features and understand how to get higher confidence rate



#######################################################################################
## Assignment 11.1
# Link to the Jupityr Notebook: 
https://github.com/kneetu/AI-ML/blob/main/assignment_11_1/what-drives-car-price.ipynb
# Summary of findings for Assignment 11_1:
**Objective**: understand what factors make a car more or less expensive. As a result of your analysis, provide clear recommendations to your client -- a used car dealership -- as to what consumers value in a used car.
**DataSet** : The data set consist of the 426K cars data, where price is used as the y_test parameter
**Data cleanup**: For the simplicity of the modeling and to focus on the process, all the NaNs were removed from the dataset, making the dataset even smaller ~35K
**Process of analysis**: 
- Apply multiple regression models and identify the best based on the MSE of the development data(X_test)
- Find best hyperparameters for Ridge model and compare it with other models
- feature selection to provide recommendation
- Analyze and provide recommendations
## Recommendation to the client
- The factors that affect the price of the cars are :
  'year': The newer the car, higher the price,
  'condition': a like_new car sell for higher price, but the correalation is weak here.
  'cylinderss': 8 Cylinder car sells for more price
  'fuel': Gas car sells for less price than the other type vehicles
  'drive': a front wheel drive seels to less price
   and 'size': A full size brings in the higher price
- Based on above analysis, the recommendation to the dealer would be to have inventory of the cars, that are newer models and has 8 cylinders. Dealer should keep less vehicles that Front wheel drive and uses gas for the fuel. Full_size vehicles are more popular and bring more price as well
  


########################################################################################
## Assigment 5.1
# Link to the Jupityr Notebook: 
https://github.com/kneetu/AI-ML/blob/main/assignment_5_1/prompt.ipynb

# Summary of findings for Assignment 5_1:

The analysis of this data was focused towards the effect of passangers, Temperature. Based on teh analysis, the driver with the kids as passanger will likely to go mostly a retaurant<20 or have a takeout.
If the temperature is high, then it will be a sit down restaurant, like a cheap restaurant.

so

- a driver with kids, in high temperature will likely go to a cheap retaurant
- Driver with kids will not likely to accept the bar coupon
- direction doesn't seem to play any big role in the coupon acceptance
- Person who has been to a bar more than 3 times a month and does NOT have kids as passenger will likely accept the bar coupon
- If the person is below the age of 30, then chances of it accepting bar coupon increases.

