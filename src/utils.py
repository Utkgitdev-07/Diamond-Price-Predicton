import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path, obj):  # making pkl file for given object and save it in the given path
    try:
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        logging.error(f"Error in saving the object: {e}")
        raise CustomException(e, sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models): # function to evaluate the model
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score # here we are storing the model name and its r2 score in the report dictionary

        return report # return the report dictionary 
    
    except Exception as e:
            logging.info('Exception occured during model training')
            raise CustomException(e,sys)
    


# function to load the object from the given path
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)