import os,sys,inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from common.preprocessing import CropDataProcessor
import pickle
    
class CropYielEstimator():
    def train(self, train_data):
        data_processor = CropDataProcessor(train_data, excluded_features=['Dew_Frost_Point', 'Year', 'State', 'District'])
        data_processor.process_to_train()
        X_train, y_train = data_processor.get_training_data()
        model = RandomForestRegressor(n_estimators=110,
                            max_depth=20,
                            min_samples_leaf=2,
                            min_samples_split=4,
                            n_jobs=-1,
                            random_state=101)
        model.fit(X_train, y_train)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('data_processor.pkl', 'wb') as f:
            pickle.dump(data_processor, f)
        print("Model trained!")
        print("Training score: ", model.score(X_train, y_train))
        
    def load_model(self):
        model = None
        data_processor = None
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('data_processor.pkl', 'rb') as f:
            data_processor = pickle.load(f)
        return model, data_processor
        
    def test(self, test_data):
        model = None
        data_processor = None
        try:
            model, data_processor = self.load_model()
            X_test, y_test = data_processor.process_to_test(test_data)
            print(f"Test score: {model.score(X_test, y_test)}")
        except:
            print("Model not found! Kindly train a model and retry.")
    
    def predict(self, _features):
        model = None
        data_processor = None
        try:
            model, data_processor = self.load_model()
        except:
            print("Model not found! Kindly train a model and retry.")
            return
        
        features = pd.DataFrame(_features)
        features = data_processor.process_to_predict(features)
        prediction = model.predict(features)

        # Get Yield from boxcox(yield)
        yld = data_processor.target_transformer.inverse_transform([prediction]) - 1

        return yld
    
    
    