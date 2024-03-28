from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import pandas as pd
from model.bakeries import bakery
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from flask_restful import Resource
from flask import Blueprint, jsonify, request 
from flask_restful import Api, Resource

bakery_api = Blueprint('bakery_api', __name__, url_prefix='/api/bakery')
api = Api(bakery_api)

class PredictNo(Resource):
    
    def __init__(self):
        self.model = bakery()  
    def post(self):
        try:
            # Get JSON data from the request
            payload = request.get_json()
            print(payload)
            bakeryModel = bakery.get_instance()
            # Predict item purchased from bakery excluding coffee
            response = bakeryModel.predict(payload)
            print(response)
            
            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)})
        
api.add_resource(PredictNo, '/predict')