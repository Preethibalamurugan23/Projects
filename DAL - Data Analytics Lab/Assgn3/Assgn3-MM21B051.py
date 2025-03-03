import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt2
import math
import seaborn as sb
from lazypredict.Supervised import LazyRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


#importing the dataset
df = pd.read_csv('C:/Users/preet/Downloads/Assignment3.csv')
for col in df.columns:
    df[col] = df[col].astype(float)
    
df.head()

xx = np.array(df[['x1','x2','x3','x4','x5']])
yy = np.expand_dims(df['y'], 1)

#splitting the dataset into 80% training, 20% test
X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=42)

def SSE(y, yhat):
    return np.sum((y-yhat)**2)

#Task 1: performing OLS on the raw dataset
model1 = linear_model.LinearRegression()
model1.fit(X_train, y_train)
yhat = model1.predict(X_test)
loss = SSE(y_test, yhat)
print("Beta :", model1.coef_, " Bias :", model1.intercept_)
print("Loss :",loss)

# RMSE and loss for the OLS on raw dataset
rmse = np.sqrt(mean_squared_error(y_test, yhat))
print("RMSE :",rmse)
df.describe()

# Task 2: Analysing the dataset by the correlations between the features and y
corr = np.corrcoef(xx.T)
print(corr)

print(corr > 0.9)

all_data = np.concatenate((xx, yy), axis=1)
corr_all = np.corrcoef(all_data.T)
print(corr_all)

print(corr_all > 0.7)

sb.pairplot(df)
plt2.show()

# Task 3: fitting an OLS model to the relevant feautures after transformation
x5_sq = np.array(X_train[:, 4] ** 0.5).reshape(-1, 1) 
x1x2 = np.array(X_train[:, 0] * X_train[:,1]).reshape(-1, 1)

# Concatenate X_train with x5_sq to create a new feature matrix
xx_combined = np.concatenate((X_train[:,[1,3]], x5_sq, x1x2), axis=1)

x5_sq = np.array(X_test[:, 4] ** 0.5).reshape(-1, 1)  
x1x2 = np.array(X_test[:, 0] * X_test[:,1]).reshape(-1, 1)

# Concatenate X_test with x5_sq to create a new feature matrix
xx_combined_test = np.concatenate((X_test[:,[1,3]], x5_sq, x1x2), axis=1)

#checking relationship between transformed features and y using correlations and pairplots
all_data = np.concatenate((xx_combined, y_train), axis=1)
corr_all = np.corrcoef(all_data.T)
print(corr_all)

print(corr_all > 0.7)

column_names = [ 'x2', 'x4', 'x5_sq', 'x1x2', 'y']  
all_data_df = pd.DataFrame(all_data, columns=column_names)
sb.pairplot(all_data_df)
plt2.show()

# fitting OLS on the new dataset with trnasformed features
from sklearn import linear_model
model1 = linear_model.LinearRegression()
model1.fit(xx_combined, y_train)
yhat = model1.predict(xx_combined_test)
loss2 = SSE(y_test, yhat)
print("Beta :", model1.coef_, " Bias :", model1.intercept_)

# loss and RMSE for the new OLS model
print("Loss :",loss2)
rmse = np.sqrt(mean_squared_error(y_test, yhat))
print("RMSE :",rmse)

#Task 4: using lazyregressor and analyzing the output
lazy_regressor = LazyRegressor()

# Fit LazyRegressor
models = lazy_regressor.fit(X_train, X_test, y_train, y_test)

# Get results
results = models[0]  
results_reset = results.reset_index()

# Print the Model names and their corresponding RMSE
print(results_reset[['Model', 'RMSE']])

# Plotting the bar graph
plt2.figure(figsize=(10, 6))
plt2.barh(results_reset['Model'], results_reset['RMSE'], color='skyblue')
plt2.xlabel('RMSE')
plt2.ylabel('Model')
plt2.title('RMSE of Different Models')
plt2.gca().invert_yaxis() 
plt2.show()




