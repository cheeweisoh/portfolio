import os
os.chdir('D:/User/Documents/R/Portfolio/Github')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


resale_df = pd.read_csv('resale_prices.csv')
inflation_df= pd.read_csv('housing_cpi.csv')

price_df = resale_df[['month', 'resale_price']]
price_df['quarter'] = pd.to_datetime(price_df['month'].values, format='%Y-%m').astype('period[Q]').astype(str)

price_df = price_df.merge(right=inflation_df, on='quarter', how='left')
price_df['inflation_price'] = price_df['resale_price'] * price_df['inflation']

resale_df = pd.concat([resale_df, price_df['quarter'], price_df['inflation_price']], axis=1)
resale_df.to_csv('resale_prices.csv', index=False)
