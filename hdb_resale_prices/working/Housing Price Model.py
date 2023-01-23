import os
os.chdir('D:/User/Documents/R/Portfolio/Github/hdb_resale_prices')

import pandas as pd
import numpy as np
import scipy.stats as stat
import pylab
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

resale_df = pd.read_csv('data/resale_prices_final.csv')
resale_df['mid_storey'] = resale_df['mid_storey'].astype('float')
continuous_features = ['floor_area_sqm', 'nearest_mrt_dist', 'nearest_pri_sch_dist', 'nearest_sec_sch_dist', 'nearest_mall_dist', 'lease_deci', 'mid_storey']
categorical_features = ['town', 'flat_type', 'flat_model']

# Reducing features for towns
central = ['BISHAN', 'BUKIT MERAH', 'BUKIT TIMAH', 'CENTRAL AREA', 'DOWNTOWN CORE', 'GEYLANG', 'KALLANG/WHAMPOA', 'MARINA EAST', 'MARINA SOUTH', 'MARINE PARADE',
           'MUSEUM', 'NEWTON', 'NOVENA', 'ORCHARD', 'OUTRAM', 'QUEENSTOWN', 'RIVER VALLEY', 'ROCHOR', 'SINGAPORE RIVER', 'SOUTHERN ISLANDS',
           'STRAITS VIEW', 'TANGLIN', 'TOA PAYOH']
east = ['BEDOK', 'CHANGI', 'CHANGI BAY', 'PASIR RIS', 'PAYA LEBAR', 'TAMPINES']
north = ['CENTRAL WATER CATCHMENT', 'LIM CHU KANG', 'MANDAI', 'SEMBAWANG', 'SIMPANG', 'SUNGEI KADUT', 'WOODLANDS', 'YISHUN']
north_east = ['ANG MO KIO', 'HOUGANG', 'NORTH-EASTERN ISLANDS', 'PUNGGOL', 'SELETAR', 'SENGKANG', 'SERANGOON']
west = ['BOON LAY', 'BUKIT BATOK', 'BUKIT PANJANG', 'CHOA CHU KANG', 'CLEMENTI', 'JURONG EAST', 'JURONG WEST', 'PIONEER', 'TENGAH', 'TUAS',
        'WESTERN ISLANDS', 'WESTERN WATER CATCHMENT']

def town_to_region(town):
    if town in central:
        return 'central'
    elif town in east:
        return 'east'
    elif town in north:
        return 'north'
    elif town in north_east:
        return 'north_east'
    elif town in west:
        return 'west'
    else:
        return 'na'
    
town_to_region = np.vectorize(town_to_region)
region = town_to_region(resale_df['town'])
resale_df['region'] = region

reduced_categorical_features = ['region', 'flat_type', 'flat_model']

# Transforming skewed data

def plot_data(df, feature, method):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.subplot(1,2,2)
    stat.probplot(df[feature], dist='norm', plot=pylab)
    plt.title(method)
    plt.show()

plot_data(resale_df, 'inflation_price', 'raw')
temp_df = resale_df.copy()
temp_df['inflation_price'] = np.sqrt(temp_df['inflation_price'])
plot_data(temp_df, 'inflation_price', 'sqrt')

for feature in continuous_features:
    plot_data(resale_df, feature, feature)
    if 0 in temp_df[feature].unique():
        print(feature)
        pass
    else:
        temp_df = resale_df.copy()
        temp_df[feature] = np.log(temp_df[feature])
        plot_data(temp_df, feature, 'log '+feature)
        
# Standardisation
sc = StandardScaler()
temp_df = resale_df.copy()
temp_df = pd.DataFrame(temp_df[['floor_area_sqm', 'nearest_mrt_dist']])
plot_data(temp_df, 'floor_area_sqm', 'floor_area_sqm')
temp_df_scaled = sc.fit_transform(temp_df)
temp_df_scaled = pd.DataFrame(temp_df_scaled, columns=['floor_area_sqm', 'nearest_mrt_dist'])
plot_data(temp_df_scaled, 'floor_area_sqm', 'floor_area_sqm')

# Train Test Split
X = resale_df[continuous_features+categorical_features]
y = np.log1p(resale_df['inflation_price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Pipeline
transformer = FunctionTransformer(np.log1p, validate=False)

mapper = DataFrameMapper([
    (continuous_features, [transformer, StandardScaler()]),
    (categorical_features, OneHotEncoder())], df_out=True)

lr_pipeline = Pipeline(steps=[
    ('preprocessing', mapper),
    ('regressor', LinearRegression())])

lr_pipeline.fit(X_train, y_train)
print('lr model_score: {}'.format(lr_pipeline.score(X_test, y_test)))
print('lr mean square error: {}'.format(mean_squared_error(y_test, lr_pipeline.predict(X_test))))
print('lr mean absolute error: {}'.format(mean_absolute_error(y_test, lr_pipeline.predict(X_test))))

# Examining coefficients of Linear Regression
X_columns = lr_pipeline['preprocessing'].transform(X_train).columns
coeffs = lr_pipeline['regressor'].coef_
coeff_table = pd.DataFrame([X_columns, coeffs]).transpose()

# Ridge
r_pipeline = Pipeline(steps=[
    ('preprocessing', mapper),
    ('regressor', RidgeCV(fit_intercept=False))])

r_pipeline.fit(X_train, y_train)
print('model_score: {}'.format(r_pipeline.score(X_test, y_test)))
print('mean square error: {}'.format(mean_squared_error(y_test, r_pipeline.predict(X_test))))

# Random Forest
rf_mapper = DataFrameMapper([
    (continuous_features, None),
    (categorical_features, OneHotEncoder())], df_out=True)

rf_pipeline = Pipeline(steps=[
    ('preprocessing', rf_mapper),
    ('regressor', RandomForestRegressor())])

rf_pipeline.fit(X_train, y_train)
print('rf model_score: {}'.format(rf_pipeline.score(X_test, y_test)))

# RF Feature Importance
sns.barplot(y=rf_pipeline['preprocessing'].transform(X_train).columns, x=rf_pipeline['regressor'].feature_importances_)

feature_importances = pd.DataFrame(rf_pipeline['regressor'].feature_importances_,
                                index = rf_pipeline['preprocessing'].transform(X_train).columns,
                                columns=['importance']).sort_values('importance', ascending=False)

# Reduced Categories RF
X2 = resale_df[continuous_features+reduced_categorical_features]
y = np.log1p(resale_df['inflation_price'])
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=42)

rf_reduced_mapper = DataFrameMapper([
    (continuous_features, None),
    (reduced_categorical_features, OneHotEncoder())], df_out=True)

rf_reduced_encoding_pipeline = Pipeline(steps=[
    ('preprocessing', rf_reduced_mapper),
    ('regressor', RandomForestRegressor())])

rf_reduced_encoding_pipeline.fit(X2_train, y_train)
print('rf reduced model_score: {}'.format(rf_reduced_encoding_pipeline.score(X2_test, y_test)))

# Stick with normal features
# Hyperparamerter Optimisation
param_grid = {
    'regressor__max_depth': [30, 40, 50, 60, 70],
    'regressor__min_samples_leaf': [1, 3, 5],
    'regressor__min_samples_split': [2, 4, 8, 16],
    'regressor__n_estimators': [100, 200, 300, 1000]}

search = RandomizedSearchCV(rf_pipeline, param_grid, verbose=2, n_jobs=1, random_state=42)
search.fit(X_train, y_train)

rf_best_params = {'n_estimators': 100,
 'min_samples_split': 8,
 'min_samples_leaf': 1,
 'max_depth': 50}

rf_best = Pipeline(steps=[
    ('preprocessing', rf_mapper),
    ('regressor', RandomForestRegressor(**rf_best_params))])

rf_best.fit(X_train, y_train)
print('rf best model_score: {}'.format(rf_best.score(X_test, y_test)))

# XGBoost
xgb_pipeline = Pipeline(steps=[
    ('preprocessing', rf_mapper),
    ('regressor', XGBRegressor())])

xgb_pipeline.fit(X_train, y_train)
print('xgb model_score: {}'.format(xgb_pipeline.score(X_test, y_test)))

# Hyperparameter Optimisation
param_grid = {
    'regressor__learning_rate': [0.05, 0.10, 0.20, 0.30],
    'regressor__max_depth': [4, 6, 8, 10],
    'regressor__min_child_weight': [1, 3, 5],
    'regressor__gamma': [0.0, 0.1, 0.2, 0.3],
    'regressor__colsample_bytree': [0.3, 0.5, 0.7, 1]}

search = GridSearchCV(xgb_pipeline, param_grid, verbose=1, n_jobs=1, cv=3)
search.fit(X_train, y_train)

xgb_best_params = {'colsample_bytree': 0.7,
 'gamma': 0.0,
 'learning_rate': 0.2,
 'max_depth': 10,
 'min_child_weight': 5}

xgb_best = Pipeline(steps=[
    ('preprocessing', rf_mapper),
    ('regressor', XGBRegressor(**xgb_best_params))])

xgb_best.fit(X_train, y_train)
print('xgb best model_score: {}'.format(xgb_best.score(X_test, y_test)))

# Model Scores
model_scores = {'linear regression': lr_pipeline.score(X_test, y_test),
                'ridge regression': r_pipeline.score(X_test, y_test),
                'random forest': rf_pipeline.score(X_test, y_test),
                'random forest hyperparameter': rf_best.score(X_test, y_test),
                'xgboost': xgb_pipeline.score(X_test, y_test),
                'xgboost hyperparameter': xgb_best.score(X_test, y_test)}

mae_scores = {'linear regression': mean_absolute_error(y_test, lr_pipeline.predict(X_test)),
                'ridge regression': mean_absolute_error(y_test, r_pipeline.predict(X_test)),
                'random forest': mean_absolute_error(y_test, rf_pipeline.predict(X_test)),
                'random forest hyperparameter': mean_absolute_error(y_test, rf_best.predict(X_test)),
                'xgboost': mean_absolute_error(y_test, xgb_pipeline.predict(X_test)),
                'xgboost hyperparameter': mean_absolute_error(y_test, xgb_best.predict(X_test))}

mse_scores = {'linear regression': mean_squared_error(y_test, lr_pipeline.predict(X_test)),
                'ridge regression': mean_squared_error(y_test, r_pipeline.predict(X_test)),
                'random forest': mean_squared_error(y_test, rf_pipeline.predict(X_test)),
                'random forest hyperparameter': mean_squared_error(y_test, rf_best.predict(X_test)),
                'xgboost': mean_squared_error(y_test, xgb_pipeline.predict(X_test)),
                'xgboost hyperparameter': mean_squared_error(y_test, xgb_best.predict(X_test))}

print(sorted(list(zip(model_scores.keys(), model_scores.values())), key=lambda x: x[1], reverse=True))
print(sorted(list(zip(mae_scores.keys(), mae_scores.values())), key=lambda x: x[1]))
print(sorted(list(zip(mse_scores.keys(), mse_scores.values())), key=lambda x: x[1]))

# Saving model with pickle
pickle.dump(xgb_best, open('deployment/xgb_best.sav', 'wb'))
loaded_model = pickle.load(open('deployment/xgb_best.sav', 'rb'))

test = pd.DataFrame(X_test.iloc[0]).transpose()
test[continuous_features] = test[continuous_features].astype(float)
predicted = loaded_model.predict(test)
price = np.expm1(predicted)
