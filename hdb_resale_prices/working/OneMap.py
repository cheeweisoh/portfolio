import os
import requests
os.getcwd()

api_address = 'https://developers.onemap.sg'
key_path = '../../OneMap/OneMap2-Authentication-Modules/OneMap2-Authentication-Module_for_Windows_x64/authentication_module/authstore.txt'
key = open(key_path, mode='r').read().split(',')[0]

result = eval(requests.get(api_address + '/privateapi/themesvc/getAllThemesInfo?token={}&moreInfo=N'.format(key)).text)['Theme_Names']

for i in result:
    print(i['THEMENAME'])