from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import pandas as pd
from model.food import food
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from flask_restful import Resource
from flask import Blueprint, jsonify, request 
from flask_restful import Api, Resource

food_apix = Blueprint('food_apix', __name__, url_prefix='/api/foodx')
api = Api(food_apix)

class PredictItem(Resource):
    
    def __init__(self):
        self.model = food()  
    def post(self):
        try:
            # Get JSON data from the request
            payload = request.get_json()
            print(payload)
            titanicModel = food.get_instance()
            # Predict the survival probability of the passenger
            response = titanicModel.predict(payload)
            print(response)
            
            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)})


api.add_resource(PredictItem, '/predict')

