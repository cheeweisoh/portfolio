import requests
import os
import pandas as pd
import numpy as np
from tqdm import trange
import collections

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 10)

os.chdir('D:/User/Documents/R/Portfolio/Github/hdb_resale_prices')

"""
Shape = (108048, 12)

To-do List

Verify latitude/longitude formula
Move unique house coordinates to data folder

"""

# Reset data
# resale_df = pd.read_csv('resale_prices.csv')
# resale_df = pd.read_csv('raw_resale_price.csv')
# coord_df = pd.read_csv('unique_house_coordinates.csv')
# resale_df = pd.read_csv('data/resale_df_geocode.csv')

# mrt_coord_df = pd.read_csv('data/mrt_coord.csv')

# malls_dist_df = pd.read_csv('nearest_mall.csv')

# Functions
def onemap_geocoding(address):
    result = requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+address+'&returnGeom=Y&getAddrDetails=Y&pageNum=1')
    result = eval(result.text)
    
    if result['found'] > 0:
        lat, long = result['results'][0]['LATITUDE'], result['results'][0]['LONGITUDE']
        return address, lat, long
    else:
        pass

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
    
def geocode_series(addresses, func):
    coord_list = []
    failed_list = []
    failed = 0
    t = trange(len(addresses), desc='Geoding Addresses')
    for i in t:
        try:
            result = func(addresses[i])
            if len(result) > 0:
                coord_list.append(result)
                t.set_description('{} failed'.format(failed))
        except:
            failed_list.append(addresses[i])
            failed += 1
            t.set_description('{} failed'.format(failed))
    print(failed_list if failed_list else "All passed")
    return coord_list

def calculate_dist(address_df, dist_df):
    temp = address_df.copy()
    for _, i in dist_df.iterrows():
        name, lat, long = i[0], float(i[1]), float(i[2])
        temp['{}'.format(name)] = (((address_df['latitude'] - lat) * 110.574) ** 2 + ((address_df['longitude'] - long) * 111.32) ** 2) ** 0.5
    nearest_name = temp.drop(['address', 'longitude', 'latitude'], axis=1).idxmin(axis=1)
    nearest_dist = temp.drop(['address', 'longitude', 'latitude'], axis=1).min(axis=1)
    nearest = pd.concat([temp['address'], nearest_name, nearest_dist], axis=1)
    return nearest

# test = calculate_dist(resale_df, mrt_coord_df)

def calculate_pri_score(address_df, dist_list, rank_list, dist_cutoff, num_cutoff, dist_weights={1: 0.1, 2: 0.5, 999: 1}):
    temp = address_df.copy()
    for i in dist_list:
        name, lat, long = i[0], float(i[1]), float(i[2])
        rank = int(rank_list.loc[rank_list['pri_sch_name'] == name]['rank'])
        temp[str(rank)] = (((address_df['latitude'] - lat) * 110.574) ** 2 + ((address_df['longitude'] - long) * 111.32) ** 2) ** 0.5
        
    t = trange(len(temp), desc='Address')
    distances = temp.drop(['address', 'latitude', 'longitude'], axis=1)
    result = {}
    for i in t:
        row = distances.iloc[i]
        row = row[row <= dist_cutoff][:num_cutoff]
        
        score = 0
        for j in row.index:
            dist = row[j]
            for ring in dist_weights.keys():
                if dist < ring:
                    weight = dist_weights[ring]
                    break
            score += (weight * int(j))

        result[temp.iloc[i]['address']] = score if score != 0 else 9999
        
    result = pd.DataFrame.from_dict(result, orient='index', columns=['pri_sch_score'])
    return result


##############################################################################

# Importing resale prices from data.gov API
dataset = requests.get('https://data.gov.sg/api/action/datastore_search?resource_id=f1765b54-a209-4718-8d38-a39237f502b3&limit=9999999').json()
resale_df = pd.DataFrame(dataset['result']['records'])
resale_df.to_csv('data/resale_df_raw.csv', index=False)

# Getting search field for geocoding
resale_df['address'] = resale_df['block'] + ' ' + resale_df['street_name']

# Renaming St. George's Rd to Saint George's Rd, for geoencoding problems later
resale_df['address'] = resale_df['address'].str.replace('ST. GEORGE', 'SAINT GEORGE')

# Convert short forms in address to long forms for better searching
changes = [[r'\bAVE\b', 'AVENUE'],
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

# Geocoding addresses with OneMap API
house_addresses = resale_df['address'].unique()
coord_list = geocode_series(house_addresses, onemap_geocoding_flat)
coord_df = pd.DataFrame(coord_list, columns=['address', 'search_address', 'latitude', 'longitude'])

# Check for addresses with no search results
none_check = coord_df[coord_df['search_address'] == 'none']

# Fixing two entries with no search result
cck_exception = eval(requests.get('https://developers.onemap.sg/commonapi/search?searchVal=BLK 216 AND 215 CHOA CHU KANG CENTRAL&returnGeom=Y&getAddrDetails=Y&pageNum=1').text)['results'][0]
for idx in coord_df[coord_df['search_address'] == 'none'].index:
    coord_df.loc[idx]['latitude'] = cck_exception['LATITUDE']
    coord_df.loc[idx]['longitude'] = cck_exception['LONGITUDE']

coord_df[['latitude','longitude']] = coord_df[['latitude', 'longitude']].astype(float)
coord_df.drop('search_address', axis=1, inplace=True)
coord_df.to_csv('unique_house_coordinates.csv', index=False)

resale_df = resale_df.merge(right=coord_df, on='address', how='left')
resale_df.to_csv('data/resale_df_geocode.csv', index=False)

##############################################################################

# Importing list of MRT stations
mrt_df = pd.read_csv('data/mrt_stations.csv')

# Geocoding MRT stations
mrt_names = mrt_df['stn_name'].unique()
mrt_coord = geocode_series(mrt_names, onemap_geocoding)
mrt_coord_df = pd.DataFrame(mrt_coord, columns = ['mrt_name', 'latitude', 'longitude'])
mrt_coord_df[['latitude', 'longitude']] = mrt_coord_df[['latitude', 'longitude']].astype(float)
mrt_coord_df.to_csv('data/mrt_coord.csv', index=False)

# Calculating nearest MRT station
mrt_dist_df = calculate_dist(coord_df, mrt_coord_df)
mrt_dist_df.columns = ['address', 'nearest_mrt', 'nearest_mrt_dist']
resale_df = resale_df.merge(right=mrt_dist_df, on='address', how='left')

##############################################################################

# # Importing list of schools from data.gov API
# schools = requests.get('https://data.gov.sg/api/action/datastore_search?resource_id=ede26d32-01af-4228-b1ed-f05c45a1d8ee&limit=347').json()
# schools_df = pd.DataFrame(schools['result']['records'])[['school_name', 'postal_code', 'mainlevel_code']]

# # Keeping only data for primary/secondary schools, and treating mixed levels schools
# ip_schools = ['ANGLO-CHINESE SCHOOL (INDEPENDENT)',
#               'DUNMAN HIGH SCHOOL',
#               'HWA CHONG INSTITUTION',
#               'NATIONAL JUNIOR COLLEGE',
#               'NUS HIGH SCHOOL OF MATHEMATICS AND SCIENCE',
#               'RAFFLES INSTITUTION',
#               'RIVER VALLEY HIGH SCHOOL',
#               'SCHOOL OF THE ARTS, SINGAPORE',
#               'SINGAPORE SPORTS SCHOOL',
#               "ST. JOSEPH'S INSTITUTION",
#               'TEMASEK JUNIOR COLLEGE']
# both_schools = ['CATHOLIC HIGH SCHOOL',
#                 "CHIJ ST. NICHOLAS GIRLS' SCHOOL",
#                 'MARIS STELLA HIGH SCHOOL']

# for school in ip_schools:
#     schools_df['mainlevel_code'] = np.where((schools_df['school_name'] == school), 'SECONDARY', schools_df['mainlevel_code'])
    
# both_schools_df = schools_df[schools_df['school_name'].isin(both_schools)]
# both_schools_df['mainlevel_code'] = 'SECONDARY'

# for school in both_schools:
#     schools_df['mainlevel_code'] = np.where((schools_df['school_name'] == school), 'PRIMARY', schools_df['mainlevel_code'])
    
# schools_df = pd.concat([schools_df, both_schools_df])

# schools_df.drop(schools_df[schools_df['mainlevel_code'] == 'JUNIOR COLLEGE'].index, inplace=True)
# schools_df.drop(schools_df[schools_df['mainlevel_code'] == 'CENTRALISED INSTITUTE'].index, inplace=True)

# # Geocoding schools with OneMap API
# school_postal = schools_df['postal_code'].unique()
# for i in range(len(school_postal)):
#     if len(school_postal[i]) < 6:
#         new_postal = '0' + school_postal[i]
#         school_postal[i] = new_postal

# school_coord = geocode_series(school_postal)
# school_coord_df = pd.DataFrame(school_coord, columns=['postal_code', 'latitude', 'longitude'])
# school_coord_df[['latitude', 'longitude']] = school_coord_df[['latitude', 'longitude']].astype(float)
# schools_df = schools_df.merge(right=school_coord_df, on='postal_code', how='left')

# # Splt into primary/secondary schools
# schools_pri_df = schools_df[schools_df['mainlevel_code'] == 'PRIMARY'][['school_name', 'latitude', 'longitude']]
# schools_pri = list(schools_pri_df.itertuples(index=False, name=None))
# schools_sec_df = schools_df[schools_df['mainlevel_code'] == 'SECONDARY'][['school_name', 'latitude', 'longitude']]
# schools_sec = list(schools_sec_df.itertuples(index=False, name=None))

# # Calculating nearest primary/secondary school
# schools_pri_dist_df = calculate_dist(coord_df, schools_pri)
# schools_pri_dist_df.columns = ['address', 'nearest_pri_sch', 'nearest_pri_sch_dist']
# resale_df = resale_df.merge(right=schools_pri_dist_df, on='address', how='left')
# schools_pri_dist_df.to_csv('nearest_pri_sch.csv', index=False)

# schools_sec_dist_df = calculate_dist(coord_df, schools_sec)
# schools_sec_dist_df.columns = ['address', 'nearest_sec_sch', 'nearest_sec_sch_dist']
# resale_df = resale_df.merge(right=schools_sec_dist_df, on='address', how='left')
# schools_sec_dist_df.to_csv('nearest_sec_sch.csv', index=False)

##############################################################################

# Constructing ranking index for primary schools
pri_sch_rank = pd.read_csv('data/pri_sch_ranking.csv',header=None)
pri_sch_rank = pd.DataFrame(pri_sch_rank.values.reshape(-1, 3), columns=['rank', 'pri_sch_name', 'town'])
pri_sch_rank.drop('town', axis=1, inplace=True)
pri_sch_rank['pri_sch_name'] = pri_sch_rank['pri_sch_name'].apply(lambda x: x.upper())
pri_sch_rank['pri_sch_name'] = pri_sch_rank['pri_sch_name'].str.replace(r'\bST. \b', 'SAINT ', regex=True)
pri_sch_rank['rank'] = pri_sch_rank['rank'].astype(int)

# Geocoding primary schools
pri_sch_names = pri_sch_rank['pri_sch_name'].unique()
pri_sch_coords = geocode_series(pri_sch_names, onemap_geocoding)
pri_sch_coords_df = pd.DataFrame(pri_sch_coords, columns=['pri_sch_name', 'latitude', 'longitude'])
pri_sch_coord_df = pri_sch_rank.merge(pri_sch_coords_df, on='pri_sch_name', how='left')
pri_sch_coord_df.to_csv('data/pri_sch_coord.csv', index=False)

# Calculating score for each address
pri_sch_dist_df = calculate_pri_score(coord_df, pri_sch_coords, pri_sch_rank, 4, 5)
resale_df = resale_df.merge(right=pri_sch_dist_df, left_on='address', right_index=True, how='left')

##############################################################################

# Importing list of shopping malls
malls_df = pd.read_csv('data/malls.csv')

# Geocoding shopping malls
malls_coord = geocode_series(malls_df['mall_name'], onemap_geocoding)
malls_coord_df = pd.DataFrame(malls_coord, columns=['mall_name', 'latitude', 'longitude'])
malls_coord_df.to_csv('data/malls_coord.csv', index=False)

# Calculating nearest shopping mall
malls_dist_df = calculate_dist(coord_df, malls_coord_df)
malls_dist_df.columns = ['address', 'nearest_mall', 'nearest_mall_dist']
resale_df = resale_df.merge(right=malls_dist_df, on='address', how='left')

resale_df.to_csv('data/resale_prices.csv', index=False)
