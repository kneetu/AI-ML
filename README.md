### AI-ML
#######################################################################################
## Assignment 11.1
# Link to the Jupityr Notebook: 
https://github.com/kneetu/AI-ML/blob/main/assignment_11_1/what-drives-car-price.ipynb
# Summary of findings for Assignment 5_1:
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

