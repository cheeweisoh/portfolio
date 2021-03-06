import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import pickle
import requests
import pandas as pd
import numpy as np

from models.house import House

# Create app object
app = FastAPI(
    title='Resale Price Prediction API',
    description='Simple API that uses XGB Boost model to predict prices of resale flats in Singapore',
    version='0.1'
)

templates = Jinja2Templates(directory='templates')

# Importing necessary external files
mrt_coord = pd.read_csv('models/mrt_coord.csv')
pri_sch_coord = pd.read_csv('models/pri_school_coord.csv')
sec_sch_coord = pd.read_csv('models/sec_school_coord.csv')
malls_coord = pd.read_csv('models/malls_coord.csv')
classifier = pickle.load(open('xgb_best.sav', 'rb'))

def to_capitalise(s: str):
    words = s.split(' ')
    out = []
    for i in words:
        if i in ('MRT', 'LRT'):
            out.append(i)
            continue
        i = i.capitalize()
        out.append(i)
    return ' '.join(out)

def geocoding(house: House):
    data = house.dict()
    try:
        result = requests.get(
            f'https://developers.onemap.sg/commonapi/search?searchVal={data["postal_code"]}&returnGeom=Y&getAddrDetails=Y&pageNum=1')
        result = result.json()
        if len(result) > 0:
            address = result['results'][0]
    except:
        return {"message": "Address not found."}

    lat, long = address["LATITUDE"], address["LONGITUDE"]
    return lat, long


def calculate_nearest(coordinates: tuple, dist_df: pd.DataFrame):
    add_lat, add_long = float(coordinates[0]), float(coordinates[1])
    temp = dist_df.copy().values.tolist()
    distances = []
    for name, lat, long in temp:
        distance = (((add_lat - float(lat)) * 110.574) ** 2 + ((add_long - float(long)) * 111.32) ** 2) ** 0.5
        distances.append([name, distance])
    distances = sorted(distances, key=lambda x: float(x[1]))
    nearest_name, nearest_dist = distances[0][0], distances[0][1]
    return nearest_name, nearest_dist


def predict_price(house: House):
    coords = geocoding(house)
    _, nearest_mrt = calculate_nearest(coords, mrt_coord)
    _, nearest_pri = calculate_nearest(coords, pri_sch_coord)
    _, nearest_sec = calculate_nearest(coords, sec_sch_coord)
    _, nearest_mall = calculate_nearest(coords, malls_coord)

    input = house.dict()
    input['mid_storey'] = input.pop('floor')
    input['lease_deci'] = input.pop('remaining_lease')
    input['nearest_mrt_dist'] = nearest_mrt
    input['nearest_pri_sch_dist'] = nearest_pri
    input['nearest_sec_sch_dist'] = nearest_sec
    input['nearest_mall_dist'] = nearest_mall
    input.pop('postal_code')

    input = pd.DataFrame(input, index=[0])

    predicted = classifier.predict(input).tolist()[0]
    price = np.expm1(predicted)
    return price

# Index route, opens on http://127.0.0.1:8000
@app.get('/', include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse('input.html', {
        'request': request
    })


@app.post('/nearest_mrt')
def get_nearest_mrt(house: House):
    coords = geocoding(house)
    mrt, _ = calculate_nearest(coords, mrt_coord)
    return mrt


@app.post('/nearest_pri_school')
def get_nearest_primary_sch(house: House):
    coords = geocoding(house)
    sch, _ = calculate_nearest(coords, pri_sch_coord)
    return sch


@app.post('/nearest_sec_school')
def get_nearest_secondary_sch(house: House):
    coords = geocoding(house)
    sch, _ = calculate_nearest(coords, sec_sch_coord)
    return sch


@app.post('/nearest_mall')
def get_nearest_mall(house: House):
    coords = geocoding(house)
    mall, _ = calculate_nearest(coords, malls_coord)
    return mall


@app.post('/predict')
def predict(house: House):
    price = predict_price(house)
    return price

@app.post('/submit', include_in_schema=False)
async def predict_main(request: Request,
                       postal_code: str=Form(...),
                       floor: str=Form(...),
                       remaining_lease: str=Form(...),
                       floor_area_sqm: str=Form(...),
                       flat_type: str=Form(...),
                       flat_model: str=Form(...),
                       town: str=Form(...)):
    house = House(postal_code=postal_code,
                  floor=float(floor),
                  floor_area_sqm=float(floor_area_sqm),
                  remaining_lease=float(remaining_lease),
                  flat_type=flat_type,
                  flat_model=flat_model,
                  town=town)

    coords = geocoding(house)
    mrt, _ = calculate_nearest(coords, mrt_coord)
    pri, _ = calculate_nearest(coords, pri_sch_coord)
    sec, _ = calculate_nearest(coords, sec_sch_coord)
    mall, _ = calculate_nearest(coords, malls_coord)
    price = predict_price(house)

    return templates.TemplateResponse('output.html', {
        'request': request,
        'postal_code': postal_code,
        'floor': floor,
        'floor_area_sqm': floor_area_sqm,
        'remaining_lease': remaining_lease,
        'flat_type': to_capitalise(flat_type),
        'flat_model': to_capitalise(flat_model),
        'town': to_capitalise(town),
        'nearest_mrt': to_capitalise(mrt),
        'nearest_pri': to_capitalise(pri),
        'nearest_sec': to_capitalise(sec),
        'nearest_mall': to_capitalise(mall),
        'price': round(price,2)
    })

# Runs the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)