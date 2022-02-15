import numpy as np

import pandas as pd

import seaborn as sns

data=pd.read_csv('energydata_complete.csv')

#Question 12

new_data= data[['T2','T6']]

from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler()

normalised_df=pd.DataFrame(scaler.fit_transform(new_data),columns= new_data.columns)

x = normalised_df.drop(columns=['T6'])

y = normalised_df['T6']

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size= 0.3, random_state=1)

linear_model= LinearRegression()

linear_model.fit(x_train,y_train)

predicted_values= linear_model.predict(x_test)

from sklearn.metrics import r2_score

r2=r2_score(y_test, predicted_values)

round(r2,2)

#Question 13

data2= data.drop(columns="date")

normalised_data=pd.DataFrame(scaler.fit_transform(data2),columns= data2.columns)

features_data = normalised_df.drop(columns=['Appliances', 'lights'])

appliances = normalised_df['Appliances']

x_train1, x_test1, y_train1, y_test1=train_test_split(features_data, appliances,train_size=0.7, test_size= 0.3, random_state=42)

linear_model1= LinearRegression()

linear_model1.fit(x_train1,y_train1)

predicted_values1= linear_model1.predict(x_test1)

from sklearn.metrics import mean_absolute_error

mae=mean_absolute_error(y_test1, predicted_values1)

round(mae,2)

#Question 14

rss= np.sum(np.square(y_test1 - predicted_values1))

round(rss,2)

#Question 15

from sklearn.metrics import mean_squared_error

rmse=np.sqrt(mean_squared_error(y_test1, predicted_values1))

round(rmse,3)

#Question 16

r21=r2_score(y_test1, predicted_values1)

round(r21,2)

def get_weights_df(linear_model1, feat, col_name):

    weights= pd.Series(linear_model1.coef_, feat.columns).sort_values()
    
    weights_df = pd.DataFrame(weights).reset_index()
    
    weights_df.columns = ['Features', col_name]
    
    weights_df[col_name].round(3)
    
    return weights_df
    
    #Question 17
    
linear_model_weights = get_weights_df(linear_model1, x_train1, 'Linear_Model_Weight')

linear_model_weights

#Question 18

from sklearn.linear_model import Ridge

ridge_reg= Ridge(alpha=0.4)

ridge_reg.fit(x_train1,y_train1)

rmse=np.sqrt(mean_squared_error(y_test1, predicted_values1))

round(rmse,3)

#Question 19

from sklearn.linear_model import Lasso

lasso_reg= Lasso(alpha=0.001)

lasso_reg.fit(x_train1, y_train1)

lasso_weights_df = get_weights_df(lasso_reg, x_train1, 'Lasso_weight')

lasso_weights_df

#Question 20

rmse=np.sqrt(mean_squared_error(y_test1, predicted_values1))

round(rmse,3)
