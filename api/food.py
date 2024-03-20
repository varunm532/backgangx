import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from flask_restful import Resource
from flask import Blueprint, jsonify 
from flask_restful import Api, Resource

food_api = Blueprint('food_api', __name__, url_prefix='/api/food')
api = Api(food_api)

class FoodAPI(Resource):
    def __init__(self):
        self.data = pd.read_csv('filtered_data.csv')

        # Preprocessing
        self.data['DateTime'] = pd.to_datetime(self.data['DateTime'])
        self.data['HourOfDay'] = self.data['DateTime'].dt.hour
        self.data['DayOfWeek'] = self.data['DateTime'].dt.dayofweek
        self.data['Weekend'] = self.data['DayType'].apply(lambda x: 1 if x == 'Weekend' else 0)

        # Encoding categorical variables
        self.enc = OneHotEncoder(drop='first')
        encoded_features = self.enc.fit_transform(self.data[['HourOfDay', 'DayOfWeek', 'Weekend']]).toarray()
        self.encoded_cols = self.enc.get_feature_names_out(['HourOfDay', 'DayOfWeek', 'Weekend'])
        self.data[self.encoded_cols] = encoded_features

        # Model training
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        X = self.data.drop(['Items', 'DateTime', 'DayType', 'HourOfDay', 'Weekend'], axis=1)
        y = self.data['Items']
        self.rf.fit(X, y)

    def predict_food_item(self, data):
        try:
            time_of_day = data['TimeOfDay']  # Update key to match frontend
            day_of_week = data['DayOfWeek']  # Update key to match frontend
            time = data['Time']

            # Encoding input data
            encoded_data = self.enc.transform([[time_of_day, day_of_week, 1 if day_of_week in [5, 6] else 0]]).toarray()
            input_features = np.hstack((time, encoded_data))

            # Predicting food item
            predicted_item = self.rf.predict([input_features])[0]
            return {'PredictedFoodItem': predicted_item}
        except Exception as e:
            return {'error': str(e)}
        
    def post(self):
        try:
            data = request.json
            result = self.predict_food_item(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
        
api.add_resource(FoodAPI, '/food')