import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
# Import the required libraries for the TitanicModel class
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import seaborn as sns

class bakery:
    _instance = None
    
    def __init__(self):
        self.model = None
        self.dt = None
        self.features = ['Hour', 'DayPart', 'DayType']  # Features for prediction
        self.target = 'Items'  # Target variable to predict
        self.data = pd.read_csv('test_data.csv')
        
    def _clean(self):
        # Convert categorical variables to binary
        self.data['DayPart'] = self.data['DayPart'].apply(lambda x: 1 if x == 'Morning' else 0)
        self.data['DayType'] = self.data['DayType'].apply(lambda x: 1 if x == 'Weekend' else 0)
        self.data['Hour'] = pd.to_datetime(self.data['Time'], format='%H:%M:%S').dt.hour
        
        
    def _train(self):
        X = self.data[self.features]  # Features
        y = self.data[self.target]  # Target variable
        
        # Train a logistic regression model
        self.model = LogisticRegression(max_iter=400)
        self.model.fit(X, y)
        
        # Train a decision tree classifier
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)
        
    @classmethod
    def get_instance(cls):
        """Gets, and conditionally cleans and builds, the singleton instance of the Food model."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        return cls._instance
    
    def predict(self, payload):
        """Predict the item based on the given features."""
        # Convert input to DataFrame
        payload_df = pd.DataFrame(payload, index=[0])
        # Convert categorical variables to binary
        payload_df['DayPart'] = payload_df['DayPart'].apply(lambda x: 1 if x == 'Morning' else 0)
        payload_df['DayType'] = payload_df['DayType'].apply(lambda x: 1 if x == 'Weekend' else 0)
        payload_df['Hour'] = pd.to_datetime(payload_df['Time'], format='%H:%M:%S').dt.hour
        #payload_df['Hour'] = pd.to_datetime(payload_df['Time']).dt.hour
        # Predict item using the logistic regression model
        #item = self.model.predict(payload_df)
        item = self.model.predict(payload_df[self.features]) 
        #return {'item': item}
        return {'item': item.tolist()} 
    def feature_weights(self):
        """Get the feature weights
        The weights represent the relative importance of each feature in the prediction model.

        Returns:
            dictionary: contains each feature as a key and its weight of importance as a value
        """
        # extract the feature importances from the decision tree model
        importances = self.dt.feature_importances_
        # return the feature importances as a dictionary, using dictionary comprehension
        return {feature: importance for feature, importance in zip(self.features, importances)}
def initbakery():
    bakery.get_instance()