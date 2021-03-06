# Project Info
2022 Gachon University Data Science Term Project -YOUNGMIN OH Prof-<br>
We apply step of the end-to-end BigData process in this project

If you have any questions about this project, please contact us by email<br><br>
박태환 [chem602@naver.com ] Preprocessing<br>
김현종 [guswhd990507@naver.com] Regression<br>
유소연 [yeon000409@gmail.com] Knn<br>
김민준 [jmk7117@naver.com] K-Fold Validation <br>

# Fuel Consumption Ratings
We use 2022 Fuel Consumption Rations about vehicles<br>
Kaggle Data at [2022 Fuel Consumption Ratings](https://www.kaggle.com/datasets/rinichristy/2022-fuel-consumption-ratings)<br>
* Desciption : Dataset offers model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada in 2022<br><br>
* Objective : Analze how the vehicle type, engine specifiationsm,transmission, fuel, and fuel'efficiency rate to the cunt of carbon dioxide generated.<br>

# Directory Structure
```
FuelConsumptionRatings
    ├─.vscode
    ├─resources
    │   └─MY2022 Fuel Consumption Ratings.csv
    │  
    └─src
        └─tp_rating_knn.ipynb
        └─tp_rating_regression.ipynb
        └─.ipynb_checkpoints
```
* resources : The dataset we used [2022 Fuel Consumption Ratings](https://www.kaggle.com/datasets/rinichristy/2022-fuel-consumption-ratings)
* src : Jupyter notebook file with model (KNN , Poly Regression) training and testing

# A brief end to end process summary

* **Preprocessing** <br>
    * We use data Scaling and encoding
    * Use the **OrdinalEncoder** to encode the brand name, car class, fuel type, and transmission numerically <br>
    * Normalization is carried out using **StandardScaler**<br><br>
![image](https://user-images.githubusercontent.com/75360915/171775057-45ef870a-83b6-44b6-a003-de85f7492914.png)

* Poly regression
    * predict with Polynomial Regression Model<br><br>
![image](https://user-images.githubusercontent.com/75360915/171777765-121cb87e-d1f0-489e-94df-4f09c8f6d849.png)

* KNN
    * set k = n^(1/2). 
    * Accurancy of KNN classifier<br><br>
![image](https://user-images.githubusercontent.com/75360915/171777959-fc578497-213c-46cb-9b5d-44ec4d38082a.png)

* Evaluation
    * K-Fold Validation<br>
    * default K = 10<br><br>
    Poly regression - 10 K Fold Validation<br><br>
![image](https://user-images.githubusercontent.com/75360915/171776280-1c6d1ce8-1be9-435c-9bbf-a8c326042c7c.png)<br><br>
    KNN  - 10 K Fold Validation<br><br>
![image](https://user-images.githubusercontent.com/75360915/171776349-a50b2543-8a4d-4fb6-87d6-1dec2e78b921.png)<br><br>



**See the Jupyter Notebook files for more information.**

# Function
* After creating a Polynomial Regression model and learning with the values of training x and y, the model and predictions are returned by making predictions with test X.
```python
def predict_poly_regression(train_X, train_y, test_X, deg):
    regression_model = Pipeline([('poly', PolynomialFeatures(degree=deg)),
                        ('linear', LinearRegression(fit_intercept=False))])

    regression_model = regression_model.fit(train_X, train_y)
    predicted_co2 = regression_model.predict(test_X)
    return regression_model, predicted_co2
```
* From the 2 to the 10 degree, the regression model is predicted, and the R square adj value is calculated to return the degree with the largest R square adj value.
```python
def get_degree(train_X, train_y, test_X, test_Y):
    max_r2_adj = 0
    max_degree_r2_adj = 0
    for deg in range(2, 11):
        regression_model, predicted_co2 = predict_poly_regression(train_X, train_y, test_X, deg)
        r_score = r2_score(test_Y, predicted_co2)
        n = len(test_Y)
        r2_adj = 1 - (1 - r_score) * (n - 1) / (n - len(columns) - 1)
        
        if max_r2_adj < r2_adj:
            max_r2_adj = r2_adj
            max_degree_r2_adj = deg
            
    return max_degree_r2_adj
```
* Evaluate a score by cross-validation. <br>Model : estimator object implementing ‘fit’<br>X : features<br>y: The target variable to try to predict in the case of supervised learning <br>k : the number of folds  <br>return : Array of scores of the estimator for each run of the cross validation.
```python
def kfoldValidation(model,X,y,k):
    kfold = KFold(n_splits=k, shuffle=True ,random_state=0)
    scores = cross_val_score(model, X, y, cv=kfold)
    return scores
```
* The decision tree algorithm is added to the existing knn algorithm to predict the result with the voting algorithm and return the accuracy.
```python
def get_score_of_voting(depth):
    tree = DecisionTreeClassifier(max_depth=depth)
    voting = VotingClassifier(estimators=[('KNN', knn), ('tree', tree)], voting='soft')
    voting.fit(X_train, Y_train)
    pred = voting.predict(X_test)
    return accuracy_score(Y_test, pred)
```

# reference
* Polynomial Regression : [scikit-learn regression-extending-linear](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-function)
* Degree : [sklearn.metrics.r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)<br>[how-do-i-calculate-the-adjusted-r-squared-score-using-scikit-learn]( https://stackoverflow.com/questions/49381661/how-do-i-calculate-the-adjusted-r-squared-score-using-scikit-learn)
* KNN algorithm: [sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* K fold Cross Validation 1 : https://todayisbetterthanyesterday.tistory.com/21
* K fold Cross Validation 2 : https://techblog-history-younghunjo1.tistory.com/102)
* Voting algorithm : https://techblog-history-younghunjo1.tistory.com/102
