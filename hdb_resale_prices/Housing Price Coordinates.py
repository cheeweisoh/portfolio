import requests
import os
import pandas as pd
import numpy as np
from tqdm import trange

os.chdir('D:/User/Documents/R/Portfolio/Github/hdb_resale_prices')

def onemap_geocoding_flat(address):
    result = requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+address+'&returnGeom=Y&getAddrDetails=Y&pageNum=1')
    result = eval(result.text)
    
    if result['found'] > 0:
        for i in result['results']:
            lat, long = i['LATITUDE'], i['LONGITUDE']
            r_block, r_street = i['BLK_NO'], i['ROAD_NAME']
            r_add = r_block + ' ' + r_street
            
            if r_add == address:
                return address, r_add, lat, long
            else:
                continue
        return address, 'none', '0', '0'
    else:
        pass

def geocode_series(addresses):
    coord_list = []
    failed_list = []
    failed = 0
    t = trange(len(addresses), desc='Geoding Addresses')
    for i in t:
        try:
            result = onemap_geocoding_flat(addresses[i])
            if len(result) > 0:
                coord_list.append(result)
                t.set_description('{} failed'.format(failed))
        except:
            failed_list.append(addresses[i])
            failed += 1
            t.set_description('{} failed'.format(failed))
    print(failed_list if failed_list else "All passed")
    return coord_list

# dataset = requests.get('https://data.gov.sg/api/action/datastore_search?resource_id=f1765b54-a209-4718-8d38-a39237f502b3&limit=108048').json()
resale_df = pd.read_csv('raw_resale_price.csv')
# resale_df = pd.DataFrame(dataset['result']['records'])
# resale_df.to_csv('raw_resale_price.csv', index=False)
resale_df['address'] = resale_df['block'] + ' ' + resale_df['street_name']
changes = [['ST. GEORGE', 'SAINT GEORGE'],
           [r'\bAVE\b', 'AVENUE'],
           [r'\bRD\b', 'ROAD'],
           [r'\bNTH\b', 'NORTH'],
           [r'\bSTH\b', 'SOUTH'],
           [r'\bST\b', 'STREET'],
           [r'\bCTRL\b', 'CENTRAL'],
           [r'\bDR\b', 'DRIVE'],
           [r'\bBT\b', 'BUKIT'],
           [r'\bJLN\b', 'JALAN'],
           [r'\bCRES\b', 'CRESCENT'],
           [r'\bLOR\b', 'LORONG'],
           [r"\bC'WEALTH\b", 'COMMONWEALTH'],
           [r'\bCL\b', 'CLOSE'],
           [r'\bGDNS\b', 'GARDENS'],
           [r'\bUPP\b', 'UPPER'],
           [r'\bHTS\b', 'HEIGHTS'],
           [r'\bTER\b', 'TERRACE'],
           [r'\bPL\b', 'PLACE'],
           [r'\bTG\b', 'TANJONG'],
           [r'\bPK\b', 'PARK'],
           [r'\bMKT\b', 'MARKET'],
           [r'\bKG\b', 'KAMPONG']]
for short, long in changes:
    resale_df['address'] = resale_df['address'].str.replace(short, long)


house_addresses = resale_df['address'].unique()
house_coord = geocode_series(house_addresses)
coord_df = pd.DataFrame(house_coord, columns = ['address', 'r_address', 'latitude', 'longitude'])

cck_exception = eval(requests.get('https://developers.onemap.sg/commonapi/search?searchVal=BLK 216 AND 215 CHOA CHU KANG CENTRAL&returnGeom=Y&getAddrDetails=Y&pageNum=1').text)['results'][0]
coord_df.loc[1283]['latitude'], coord_df.loc[1283]['longitude'] = cck_exception['LATITUDE'], cck_exception['LONGITUDE']
coord_df.loc[7150]['latitude'], coord_df.loc[7150]['longitude'] =cck_exception['LATITUDE'], cck_exception['LONGITUDE']

coord_df[['latitude','longitude']] = coord_df[['latitude', 'longitude']].astype(float)
resale_df = resale_df.merge(right=coord_df, on='address', how='left')
coord_df.to_csv('unique_house_coordinates.csv', index=False)

resale_df.to_csv('resale_prices.csv', index=False)
