import uvicorn
from fastapi import FastAPI
from sklearn.neighbors import KNeighborsClassifier
from haversine import haversine, Unit
from Locations import Location
import numpy as np
import pandas as pd
import pickle


        
app = FastAPI()
pickle_in = open("knn_classifier.pkl",'rb')
knn_classifier = pickle.load(pickle_in)


@app.get('/')
def index():
    return {'message': 'Hello IOT team'}


@app.get('/{name}')
def get_name(name: str):
    return {'Welcome to my first FastApi Work': f'{name}'}


@app.post('/predict')
def predict_clusters(data: Location):
    threshold_dist = 5.0
    data = data.dict()
    latitude = data['Latitude']
    longitude = data['Longitude']
    location_data = pd.read_csv('Location_data.csv')
    my_point_dist, my_point_ind = knn_classifier.kneighbors([[latitude, longitude]], 3, return_distance=True)
    near_index = int(my_point_ind[0:1, 0])
    near_lat = location_data.iloc[near_index]['Latitude']
    near_long = location_data.iloc[near_index]['Longitude']
    dist = haversine((near_lat, near_long), (latitude, longitude), unit=Unit.KILOMETERS)

    if dist > threshold_dist:
        response = "Outlier"
    else:
        response = knn_classifier.predict([[latitude, longitude]])
    return {'prediction': str(response[0])}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
