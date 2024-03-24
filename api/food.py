from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import pandas as pd
from model.foods import food
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from flask_restful import Resource
from flask import Blueprint, jsonify, request 
from flask_restful import Api, Resource

food_api = Blueprint('food_api', __name__, url_prefix='/api/food')
api = Api(food_api)

class PredictItem(Resource):
    
    def __init__(self):
        self.model = food()  
    def post(self):
        try:
            # Get JSON data from the request
            payload = request.get_json()
            print(payload)
            foodModel = food.get_instance()
            # Predict the survival probability of the passenger
            response = foodModel.predict(payload)
            print(response)
            
            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)})
        
api.add_resource(PredictItem, '/predict')

#api.add_resource(PredictItem, '/predict')
#class FoodPredictor(Resource):
#    def __init__(self):
#        self.data = pd.read_csv('filtered_data.csv')
#        self.features = ['Time', 'DayPart', 'DayType']  # Define relevant features
#        self.label_encoder = LabelEncoder()
#        self.model = self.train_model()
#    def preprocess_data(self):
        # Encode categorical variables
#        for feature in self.features:
#            self.data[feature] = self.label_encoder.fit_transform(self.data[feature])
#    def train_model(self):
#        self.preprocess_data()
#        X = self.data[self.features]
#        y = self.data['Items']
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#        model = LogisticRegression()
#        model.fit(X_train, y_train)
#        return model
#    def predict_food(self, data):
#        try:
#            # Encode input data
#            encoded_data = [self.label_encoder.transform([data[feature]])[0] for feature in self.features]
#            # Predict using the trained model
#            food_prediction = self.model.predict([encoded_data])[0]
#            return {'food_prediction': food_prediction}
#        except Exception as e:
#            return {'error': str(e)}
#    def post(self):
#        try:
#            data = request.json
#            result = self.predict_food(data)
#            return jsonify(result)
#        except Exception as e:
#            return jsonify({'error': str(e)})

# api.add_resource(FoodPredictor, '/predict')