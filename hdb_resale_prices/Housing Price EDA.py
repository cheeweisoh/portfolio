import os
os.chdir('D:/User/Documents/R/Portfolio/Github/hdb_resale_prices')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat
import pylab
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 10)
sns.set_style("darkgrid")

resale_df = pd.read_csv('data/resale_prices.csv')
resale_ref_df = pd.read_csv('data/resale_prices.csv')
resale_df.info()

resale_df.drop(['street_name', '_id', 'resale_price', 'block', 'address', 'latitude', 'longitude', 'lease_commence_date', 'nearest_mall'], axis=1, inplace=True)

def plot_data(df, feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist()  
    plt.subplot(1,2,2)
    stat.probplot(df[feature], dist='norm', plot=pylab)
    plt.title(feature)
    plt.show()

##############################################################################

# Formating variables

# Creating a function to extract middle storey
def get_mid_storey(stories):
    low, high = stories.split(' TO ')
    mid = (int(high) + int(low)) // 2
    return mid

resale_df['mid_storey'] = resale_df['storey_range'].apply(get_mid_storey)

# Converting remaining lease to decimal number
def get_remaining_lease(lease):
    parts = lease.split(' ')
    if len(parts) == 2:
        return float(parts[0])
    else:
        return float(parts[0]) + (float(parts[2]) / 12)
    
resale_df['lease_deci'] = resale_df['remaining_lease'].apply(get_remaining_lease)

##############################################################################

# Exploratory Data Analysis

# Identify numerical features
numerical_features = [feature for feature in resale_df.columns if resale_df[feature].dtype != 'O']
continuous_features = [feature for feature in numerical_features if feature != 'mid_storey']

# Checking correlation of continuous variables
corrs = resale_df[continuous_features].corr()
mask = np.triu(np.ones_like(corrs, dtype=np.bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corrs, vmin=-1, vmax=1, annot=True, mask=mask, cmap=cmap)
plt.show()
plt.title('correlation map')

##############################################################################

# Floor area and flat type
ax = sns.scatterplot(data=resale_df, x='floor_area_sqm', y='inflation_price', hue='flat_type', hue_order=['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'])
ax.set(xlabel='Floor Area (Sq Meters)', ylabel = 'Resale Price', title='Inflation Price by Floor Area')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.show()

# Count of flat type
ax = sns.countplot(data=resale_df, y='flat_type', order=['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'])
ax.set(xlabel='Count', ylabel='Flat Type', title='Count of Flat Type')
plt.show()

# Model type
ax = sns.catplot(data=resale_df, y='flat_model', x='lease_deci', order=np.sort(resale_df['flat_model'].unique()))
ax.set(xlabel='Flat Model', ylabel='Remaining Lease', title='Remaining Lease by Flat Model')
plt.show()

temp_df = resale_df.copy()
temp_df = temp_df.groupby('flat_model', as_index=False)['inflation_price'].median()
ax = sns.barplot(y='flat_model', x='inflation_price', data=temp_df, order=np.sort(temp_df['flat_model'].unique()))
plt.show()

temp_df = resale_df.copy()
temp_df = resale_df[['flat_model', 'flat_type']]
flat_model_by_type = pd.pivot_table(temp_df, index='flat_model', columns='flat_type', aggfunc='size', fill_value='')

fig, axes = plt.subplots(nrows=3, ncols=2)
flat_types = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']
fig.delaxes(axes[2,1])

for ty, ax in zip(flat_types, axes.flat):
    temp_df = resale_df.copy()
    temp_df = temp_df[temp_df['flat_type'] == ty]
    temp_df = temp_df.groupby('flat_model', as_index=False)['inflation_price'].median()
    temp_df.sort_values(by='inflation_price', ascending=False, inplace=True)
    sns.barplot(data=temp_df, x='inflation_price', y='flat_model', order=temp_df['flat_model'], ax=ax)
    ax.set(xlabel='Resale Price', ylabel='Flat Model', title=ty)
    ax.tick_params(axis='y', labelsize = 6)

fig.tight_layout()
plt.show()

# Storey Range
temp_df = resale_df.copy()
temp_df = temp_df.groupby(by='town', as_index=False)['mid_storey'].max()
sns.barplot(data=temp_df, x='mid_storey', y='town')

# Distance to Amenities
temp_df = resale_df.copy()
temp_df['mrt_bins'] = pd.cut(temp_df.nearest_mrt_dist, bins=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
temp_df = temp_df.groupby(by='mrt_bins', as_index=False)['inflation_price'].mean()
sns.barplot(data=temp_df, x='mrt_bins', y='inflation_price')

##############################################################################

# Checking normalisation of continuous features
for feature in continuous_features:
    temp_df = resale_df.copy()
    if 0 in temp_df[feature].unique():
        pass
    else:
        plot_data(temp_df, feature)
        temp_df[feature] = np.log(temp_df[feature])
        plot_data(temp_df, feature)

# Checking outliers in continuous features
for feature in continuous_features:
    temp_df = resale_df.copy()
    if 0 in temp_df[feature].unique():
        pass
    else:
        temp_df[feature] = np.log(temp_df[feature])
        temp_df.boxplot(column=feature)
        plt.title(feature)
        plt.show()

# Identifying categorical features
categorical_features = [feature for feature in resale_df.columns if resale_df[feature].dtype == 'O']
categorical_features.remove('month')
categorical_features.remove('remaining_lease')
categorical_features.remove('storey_range')

# Number of categories in each categorical feature
for feature in categorical_features:
    print('{}: {}'.format(feature, len(resale_df[feature].unique())))

# Distribution of each categorical feature
for feature in categorical_features:
    temp_df = resale_df.copy()
    cat_count = temp_df[feature].value_counts()
    plt.bar(x=cat_count.index, height=cat_count)
    plt.xticks(rotation=90)
    plt.title(feature)
    plt.show()

# Distribution of resale price for each categorical feature    
for feature in categorical_features:
    temp_df = resale_df.copy()
    temp_df.groupby(feature)['inflation_price'].median().plot.bar()
    plt.xticks(rotation=90)
    plt.title(feature)
    plt.show()

resale_df.drop(['month', 'quarter', 'resale_price', 'remaining_lease', 'storey_range'], axis=1, inplace=True)

##############################################################################

# Feature Engineering

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(resale_df.drop('inflation_price', axis=1), resale_df['inflation_price'], test_size=0.1, random_state=1)

# One Hot Encoding
ohe = OneHotEncoder(sparse=False)
test = ohe.fit_transform(X_train[['town']])
