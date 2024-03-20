from flask import Flask, request, jsonify
from flask import Blueprint
from flask_restful import Api, Resource
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)

class TitanicAPI(Resource):
    def __init__(self):
        titanic_data = sns.load_dataset('titanic')
        td = titanic_data.copy()
        td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
        td.dropna(inplace=True)
        td['sex'] = td['sex'].apply(lambda x: 1 if x == 'male' else 0)
        td['alone'] = td['alone'].apply(lambda x: 1 if x else 0)

        self.enc = OneHotEncoder(handle_unknown='ignore')
        embarked_encoded = self.enc.fit_transform(td[['embarked']].values.reshape(-1, 1))
        self.encoded_cols = self.enc.get_feature_names_out(['embarked'])

        td[self.encoded_cols] = embarked_encoded.toarray()
        td.drop(['embarked'], axis=1, inplace=True)

        self.logreg = LogisticRegression(max_iter=1000)
        X = td.drop('survived', axis=1)
        y = td['survived']
        self.logreg.fit(X, y)

    def predict_survival(self, data):
        try:
            passenger = pd.DataFrame([data]) 
            passenger['sex'] = passenger['sex'].apply(lambda x: 1 if x == 'male' else 0)
            passenger['alone'] = passenger['alone'].apply(lambda x: 1 if x else 0)

            embarked_encoded = self.enc.transform(passenger[['embarked']].values.reshape(-1, 1))
            passenger[self.encoded_cols] = embarked_encoded.toarray()
            passenger.drop(['embarked', 'name'], axis=1, inplace=True)

            dead_proba, alive_proba = np.squeeze(self.logreg.predict_proba(passenger))

            return {
                'Death probability': '{:.2%}'.format(dead_proba),
                'Survival probability': '{:.2%}'.format(alive_proba)
            }
        except Exception as e:
            return {'error': str(e)}

    def post(self):
        try:
            data = request.json
            result = self.predict_survival(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})

class FoodAPI(Resource):
    def __init__(self):
        self.data = pd.read_csv('filtered_data.csv')
        # Assuming the column names in 'filtered_data.csv' are 'DateTime', 'Daypart', 'DayType'

        # Preprocessing
        self.data['DateTime'] = self.data['DateTime'].apply(lambda x: 'Morning' if x < 12 else 'Afternoon')
        self.data['DayType'] = self.data['DayType'].apply(lambda x: 'Weekend' if x == 'Weekend' else 'Weekday')

        # Encoding categorical variables
        self.enc = OneHotEncoder(drop='first')
        self.enc.fit(self.data[['DateTime', 'DayType']])
        encoded_features = self.enc.transform(self.data[['DateTime', 'DayType']]).toarray()
        self.encoded_cols = self.enc.get_feature_names_out(['DateTime', 'DayType'])
        self.data[self.encoded_cols] = encoded_features

        # Model training
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        X = self.data.drop(['Items', 'DateTime', 'DateTime', 'DayType'], axis=1)
        y = self.data['Items']
        self.rf.fit(X, y)

    def predict_food_item(self, data):
        try:
            time_of_day = data['TimeOfDay']  # Update key to match frontend
            day_of_week = data['DayOfWeek']  # Update key to match frontend
            time = data['time']

            # Encoding input data
            encoded_data = self.enc.transform([[time_of_day, day_of_week]]).toarray()
            input_features = np.hstack((time, encoded_data))
            
            # Predicting food item
            predicted_item = self.rf.predict([input_features])[0]
            return {'Predicted Food Item': predicted_item}
        except Exception as e:
            return {'error': str(e)}

api.add_resource(TitanicAPI, '/predict')
api.add_resource(FoodAPI, '/food')