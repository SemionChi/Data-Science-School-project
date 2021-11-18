# Students:
# Yarden Levy - 205356074
# Rey Hadas - 313194748
# Semion Tchitchelnitski - 317226223

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#############################
# Question 1
print("Question 2 - Find an appropriate dataset.\n")
Data_churn = pd.read_csv("Churn_Modelling.csv")

#############################
# Question 2
print("*** Question 2 ***")
print("Verify there are no NA values:\n")
print(Data_churn.shape)
Data_churn = Data_churn.dropna(axis=0)
print(Data_churn.shape)

#############################
# Question 3:
print("\n*** Question 3 ***")
print("Convert categorical variables to dummy variables.\n")
catColumns = Data_churn.select_dtypes(['object']).columns
le = preprocessing.LabelEncoder()

for col in catColumns:
    n = len(Data_churn[col].unique())
    if n > 2:
        X = pd.get_dummies(Data_churn[col])
        Data_churn[X.columns] = X
        Data_churn.drop(col, axis=1, inplace=True)
    else:
        le.fit(Data_churn[col])
        Data_churn[col] = le.transform(Data_churn[col])

# Move 'Exited' back to the last column.
Last_column = Data_churn.pop("Exited")
Data_churn.insert(Data_churn.shape[1], 'Exited', Last_column)

#############################
# Question 4:
print("\n*** Question 4 ***")
print("Split the dataset to Train set (80%) and Test set (20%).\n")
x_train, x_test, y_train, y_test = train_test_split(Data_churn.iloc[:, 1:-1], Data_churn.iloc[:, -1],
                                                    test_size=0.2, random_state=0)

#############################
# Question 5:
print("\n*** Question 5 ***\n")
Accuracy_list = []
Precision_list = []

logisticRegr = LogisticRegression()
SVM = svm.SVC()
KNN = KNeighborsClassifier(n_neighbors=5)
XGB = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
Models = ['logisticRegr', 'SVM', 'KNN', 'XGB']

for model in Models:
    print("- Current model:", model)
    clf = eval(model)
    # Train the model
    clf.fit(x_train, y_train)
    # Predict
    predictions = clf.predict(x_test)
    # Calculate accuracy
    Accuracy = clf.score(x_test, y_test)
    print("Accuracy for Test is:", Accuracy, "in", model, "model")
    Accuracy_list.append(Accuracy)
    # Calculate precision
    All_measures = metrics.classification_report(y_test, predictions, output_dict=True)
    Precision = (All_measures['1']['precision'])
    print("Precision for Test is:", Precision, "in", model, "model\n")
    Precision_list.append(Precision)

print("Print Accuracy list:", Accuracy_list)
print("Print Precision list:", Precision_list)

#############################
# Question 6:
# The accuracy and precision scores show that the measurements system is neither accurate nor precise.
# The classifier returns a lot of false positives, meaning that the measurements do not agree.
# The company needs to be sure that customer X is churning before they invest in them, so we need to be confident
# of the true positives, so the right measurement is precision.

#############################
# Question 7:
print("\n*** Question 7 ***\n")
print("Create best_accuracy_classification function.")


def best_accuracy_classification(csv_name):
    function_data = pd.read_csv(csv_name)
    print(function_data.shape)
    function_data = function_data.dropna(axis=0)
    print(function_data.shape)

    # Convert categorical variables to dummy variables.
    categorical_cols = function_data.select_dtypes(['object']).columns
    label_encoder = preprocessing.LabelEncoder()

    for column in categorical_cols:
        size = len(function_data[column].unique())
        if size > 2:
            func_x = pd.get_dummies(function_data[column])
            function_data[func_x.columns] = func_x
            function_data.drop(column, axis=1, inplace=True)
        else:
            label_encoder.fit(function_data[column])
            function_data[column] = label_encoder.transform(function_data[column])

    # Move 'Exited' back to the last column.
    last_column = function_data.pop("Exited")
    function_data.insert(function_data.shape[1], 'Exited', last_column)

    # Split the dataset to Train set (80%) and Test set (20%).
    func_x_train, func_x_test, func_y_train, func_y_test = train_test_split(function_data.iloc[:, 1:-1],
                                                                            function_data.iloc[:, -1],
                                                                            test_size=0.2, random_state=0)

    # Run all models on the dataset and create an accuracy list
    func_accuracy_list = []

    logistic_regr = LogisticRegression()
    svm_model = svm.SVC()
    knn_model = KNeighborsClassifier(n_neighbors=5)
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    func_models = ['logistic_regr', 'svm_model', 'knn_model', 'xgb_model']

    for func_model in func_models:
        classification = eval(func_model)
        # Train the model
        classification.fit(func_x_train, func_y_train)
        # Calculate accuracy
        accuracy = classification.score(func_x_test, func_y_test)
        func_accuracy_list.append(accuracy)

    models_accuracy = np.row_stack((func_models, func_accuracy_list))
    max_acc_index = models_accuracy[1].argmax()
    print("The best model is", models_accuracy[0][max_acc_index], "and its accuracy is:",
          models_accuracy[1][max_acc_index])
    return models_accuracy


#############################
# Question 8:
print("\n*** Question 8 ***\n")
result = best_accuracy_classification('Churn_Modelling.csv')
print("\nPrint the function output:", result[0], "\n", result[1])

#######################################################################################################################
# Question 9:
print("\n*** Question 9 ***\n")
print("Filter 4 numeric continuous features.")
print("Loading new dataset.")
Dataset_new = pd.read_csv("USA_Housing.csv")
Dataset_numeric = Dataset_new[['Avg. Area Income', 'Avg. Area House Age', 'Area Population', 'Price']]

#############################
# Question 10:
print("\n*** Question 10 ***\n")
print("Feature to predict is CreditScore.")
features_lm = Dataset_numeric.columns[:-1]
X = Dataset_numeric[['Avg. Area Income', 'Avg. Area House Age', 'Area Population']]
Y = Dataset_numeric['Price']

#############################
# Question 11:
print("\n*** Question 11 ***\n")
print("Split the dataset to Train set (80%) and Test set (20%).\n")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Train the linear regression model
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# Predict with linear regression
y_test_predict = lin_model.predict(X_test)

#############################
# Question 12:
print("\n*** Question 12 ***\n")
print("Display a graph.")
features = ['Avg. Area Income', 'Avg. Area House Age', 'Area Population']
for i, f in enumerate(features):
    plt.subplot(1, len(features), i + 1)
    x = Dataset_numeric[f]
    y = Dataset_numeric["Price"]
    plt.scatter(x, y, marker='o')
    plt.title(f)
    plt.xlabel(f)
    plt.ylabel('Price')
plt.show()

#############################
# Question 13:
# There is a linear correlation between Y and all features in X, but the correlation coefficient changes between
# Y and each feature in X.

#############################
# Question 14:
print("\n*** Question 14 ***\n")
print("Polynomial Regression.")

# Polynomial transformation on X d=2:
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_poly_train, X_poly_test, Y_poly_train, Y_poly_test = train_test_split(X_poly, Y, test_size=0.2, random_state=5)

# Train the polynomial regression model
poly_lin_model = LinearRegression()
poly_lin_model.fit(X_poly_train, Y_poly_train)

# Predict with polynomial regression
y_poly_test_predict = poly_lin_model.predict(X_poly_test)

#############################
# Question 15:
print("\n*** Question 15 ***\n")
print("Compare the values of Linear and Polynomial Regressions:")
# Linear regression values
Lin_r2 = r2_score(Y_test, y_test_predict)
Lin_RMSE = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
Lin_MAE = mean_absolute_error(Y_test, y_test_predict)
Lin_MAPE = mean_absolute_percentage_error(Y_test, y_test_predict)
# Polynomial regression values
Poly_r2 = r2_score(Y_poly_test, y_poly_test_predict)
Poly_RMSE = (np.sqrt(mean_squared_error(Y_poly_test, y_poly_test_predict)))
Poly_MAE = mean_absolute_error(Y_poly_test, y_poly_test_predict)
Poly_MAPE = mean_absolute_percentage_error(Y_poly_test, y_poly_test_predict)

print("Measurements for the Linear Regression model:",
      "\nR2 =", Lin_r2,
      "\nRMSE =", Lin_RMSE,
      "\nMAE =", Lin_MAE,
      "\nMAPE =", Lin_MAPE)

print("\nMeasurements for the Polynomial Regression model:",
      "\nR2 =", Poly_r2,
      "\nRMSE =", Poly_RMSE,
      "\nMAE =", Poly_MAE,
      "\nMAPE =", Poly_MAPE)

#############################
# Question 16:
## Both models produced almost the same values for each measurement. Therefore, we would go with the simpler model
# which is the Linear Regression model.
## R2 is the correlation coefficient that measures linear association (between -1 and +1) between two variables, in our
# case it is quite high, so that indicates that each two variables are very related in a positive linear sense.
## RMSE is the standard deviation of the predicted values. Our RMSE units are in USD. The normalized RMSE equals
# to 0.127 which indicates that the fit of the model is pretty close to the observed data points.
## MAE is the mean of the absolute values of the difference between the forecasted value and the actual value.
# In our case, it tells us we can expect an error of 124,417USD from the forecast on average.
## MAPE represents accuracy as the percentage of the error (between 0-1). In our case, it indicates the the prediction
# is off by 11.14%.
