import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion

import pickle

from .utils import *

class FeatureSelector( BaseEstimator, TransformerMixin ):
    '''Selects list of features for transformation pipeline''' 
    def __init__( self, feature_type ):
        self.feature_type = feature_type 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None):
        print(f"Transforming {self.feature_type} features.")
        if self.feature_type == 'numerical':
            return X.select_dtypes(exclude='object')
        if self.feature_type == 'categorical':
            return X.select_dtypes(include='object')
        return X


class GeoRegionTransformer( BaseEstimator, TransformerMixin ):
    def __init__(self):
        from os import path
        basepath = path.dirname(__file__)
        clusterer_filepath = path.abspath(path.join(basepath, "assets", "latlong_custerer_6.pkl"))
        self.lat_long_clusterer = None
        with open(clusterer_filepath, 'rb') as f:
            self.lat_long_clusterer = pickle.load(f)
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        # Cluster Lat and Long to Geo_Region (6 clusters)
        X = identify_geo_region(X, self.lat_long_clusterer)
        X = X.drop(['Lat', 'Long', 'State', 'District'], axis=1)
        print('Clustered Lat-Long to Geo Region.')
        return X


class CropEncoder( BaseEstimator, TransformerMixin ):
    def __init__(self):
        # Weight of Evidence for Crop
        self.woe = None
    
    def fit(self, X, y):
        self.woe = crop_woe(pd.concat([X, y], axis=1), str(y.columns[0]), print_iv=False)
        return self
    
    def transform(self, X, y = None):
        # Encode Crop feature with WoE
        X['Crop'] = X['Crop'].apply(lambda x: self.woe[x])
        print('Encoded Crop using WoE.')
        return X


class CropDataProcessor():
    
    def __init__(self, data, excluded_features = [], has_yield_column = False):
        
        self.data = data.copy()
        
        self.excluded_features = excluded_features
        
        self.preprocessed = False
        
        if has_yield_column == False:
            # Create column - Yield = Production / Area
            self.data = self.convert_to_yield(self.data)
        
        self.target_transformer = PowerTransformer(method='box-cox', standardize=False)
        
        self.data = self.drop_excluded_features(self.data)
        
        num_scaler_pipeline = Pipeline(steps = [
            ('num-selector', FeatureSelector('numerical')),
            ('minmax-scaler', MinMaxScaler(feature_range=(1, 2)))
        ])
        
        cat_encoder_pipeline = Pipeline(steps = [
            ('cat-selector', FeatureSelector('categorical')),
            ('onehot-encoder', OneHotEncoder(sparse=False))
        ])
        
        feature_transformers = FeatureUnion(transformer_list= [
            ('num-scaler-pipeline', num_scaler_pipeline),
            ('cat-encoder-pipeline', cat_encoder_pipeline)
        ])
        
        self.feature_pipeline = Pipeline(steps=[
            ('geo-region-transformer', GeoRegionTransformer()),
            ('crop-woe-encoder', CropEncoder()),
            ('feature-transformers', feature_transformers)
        ])
        
        self.X = self.data.drop('Yield', axis=1)
        self.y = self.data[['Yield']]
    
    def drop_excluded_features(self, _data):
        data = _data.copy()
        if len(self.excluded_features) > 0:
            data = data.drop(self.excluded_features, axis=1)
        return data
        
    def convert_to_yield(self, _data):
        data = _data.copy()
        data['Yield'] = (data['Production'] / data['Area']) + 1
        data = data.drop(['Production', 'Area'], axis=1)
        return data
        
    def process_to_train(self):
        # Normalize distribution of target
        self.y = pd.DataFrame(self.target_transformer.fit_transform(self.y), columns=['Yield'])
        self.X = self.feature_pipeline.fit_transform(self.X, self.y)
        self.preprocessed = True
        
    def process_to_test(self, _data):
        test_data = _data.copy()
        
        test_data = self.drop_excluded_features(test_data)
        
        test_data = self.convert_to_yield(test_data)
        
        X = test_data.drop('Yield', axis=1)
        y = test_data[['Yield']]
        
        # Tranform distribution of target
        y = pd.DataFrame(self.target_transformer.transform(y), columns=['Yield'])
        X = self.feature_pipeline.transform(X)
        
        return X, y
        
    def process_to_predict(self, features):
        if len(self.excluded_features) > 0:
            features = features.drop(self.excluded_features, axis=1)
        return self.feature_pipeline.transform(features)
        
    def get_training_data(self):
        if self.preprocessed == False:
            print("Warning 1: Features are not processed yet.")
            print("Warning 2: Distribution of Yield may not be normal.")
        return self.X, self.y.values.flatten()
        