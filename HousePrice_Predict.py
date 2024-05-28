import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
train_df = pd.read_csv('/Users/mac/Desktop/BTVN b7-8/train.csv')
train_df.shape
test_df = pd.read_csv('/Users/mac/Desktop/BTVN b7-8/test.csv')
test_df.shape

## Outliers
plt.figure(figsize=(15,3),dpi=150)
sns.boxplot(data=train_df['SalePrice'],orient='h');

train_df.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)

sns.scatterplot(data=train_df,x='OverallQual',y='SalePrice')

##>Xóa dữ liệu với OverallQual == 10 & SalePrice < 250000

##>Xóa dữ liệu với OverallQual == 8 & SalePrice > 500000

##>Xóa dữ liệu với OverallQual == 4 & SalePrice > 250000

drop_index = train_df[(train_df['OverallQual']==10) & (train_df['SalePrice'] < 250000)].index
train_df = train_df.drop(drop_index,axis=0)

drop_index = train_df[(train_df['OverallQual']==8) & (train_df['SalePrice'] > 500000)].index
train_df = train_df.drop(drop_index,axis=0)

drop_index = train_df[(train_df['OverallQual']==4) & (train_df['SalePrice'] > 220000)].index
train_df = train_df.drop(drop_index,axis=0)

sns.scatterplot(data=train_df, x='GrLivArea', y='SalePrice')

sns.scatterplot(data=train_df, x='GarageCars', y='SalePrice');

train_df.loc[train_df['GarageCars']==4,'GarageCars'] = 3

sns.scatterplot(data=train_df, x='GarageCars', y='SalePrice');

sns.scatterplot(data=train_df, x='TotalBsmtSF', y='SalePrice');

df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)

df = df.drop('Id',axis=1)

df.to_csv('df.csv',index=False)

df['MSSubClass'] = df['MSSubClass'].apply(str)
df['MoSold'] = df['MoSold'].apply(str)

def return_missing_values(df):
    missing_values = df.isnull().sum() / len(df) *100
    missing_values = missing_values[missing_values>0].sort_values(ascending=False)
    
    return missing_values
return_missing_values(df)

mode_cols = 'Electrical, SaleType, Exterior1st, Exterior2nd, KitchenQual, Utilities, Functional, MSZoning'.split(', ')

for col in mode_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

NA_cols = 'PoolQC, MiscFeature, Alley, Fence, FireplaceQu, GarageCond, GarageFinish, GarageQual, GarageType, BsmtExposure, BsmtCond, BsmtQual, BsmtFinType2, BsmtFinType1'.split(', ')

for col in NA_cols:
    df[col] = df[col].fillna("NA")
df['MasVnrType'] = df['MasVnrType'].fillna('None')
null_cols = 'TotalBsmtSF, BsmtUnfSF, GarageCars, GarageArea, BsmtFinSF2, BsmtFinSF1, BsmtHalfBath, BsmtFullBath, GarageYrBlt, MasVnrArea'.split(', ')

for col in null_cols: 
    df[col] = df[col].fillna(0)
mean_lot_frontage = df.groupby('Neighborhood')['LotFrontage'].mean()
mapping = dict(zip(mean_lot_frontage.index,mean_lot_frontage))

df['LotFrontage'] = df['LotFrontage'].fillna(df['Neighborhood'].map(mapping))
return_missing_values(df)   
df = pd.get_dummies(df,drop_first=True)
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop('SalePrice', axis=1)
X = train_df.drop('SalePrice',axis=1)
y = train_df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
## linear 
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)
y_lin_pred = linear_model.predict(X_test)
mae = mean_absolute_error(y_test, y_lin_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_lin_pred))
r2 = r2_score(y_test, y_lin_pred)

print(mae,rmse,r2)
## lasso
lasso_model = LassoCV(alphas=[1, 10, 100], cv=10, random_state=42, max_iter=100000)
lasso_model.fit(X_train,y_train)
y_lasso_pred = lasso_model.predict(X_test)
mae = mean_absolute_error(y_test, y_lasso_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_lasso_pred))
r2 = r2_score(y_test, y_lasso_pred)

print(mae,rmse,r2)
lasso_model.alpha_

## RidgeCV
ridge_model = RidgeCV(alphas=(0.1, 0.5, 1, 5, 10, 50, 100), scoring='neg_mean_squared_error')
ridge_model.fit(X_train,y_train)
y_ridge_pred = ridge_model.predict(X_test)
mae = mean_absolute_error(y_test, y_ridge_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_ridge_pred))
r2 = r2_score(y_test, y_ridge_pred)

print(mae,rmse,r2)

## ElasticNet
elastic_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=10)
elastic_model.fit(X_train, y_train)
y_elastic_pred = elastic_model.predict(X_test)

mae = mean_absolute_error(y_test, y_elastic_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_elastic_pred))
r2 = r2_score(y_test, y_elastic_pred)

print(mae,rmse,r2)

import pickle
model = linear_model
with open('House_Predict.pkl', 'wb') as f:
    pickle.dump(model, f)