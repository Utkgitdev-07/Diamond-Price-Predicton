from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
import numpy as np
from dataclasses import dataclass

from src.utils import save_object


@dataclass
class DataTransformationconfig:  # Data Transformation Configuration use to store the configuration of the data transformation
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:  #perform data transformation
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig() # here we are creating the object of the DataTransformationconfig class
    
    def get_data_transformation_object(self): # basically this function is used to get the data transformation object
        try:
            logging.info("Data Transformation is started")
            # Define which column should be ordinal encoded or which column should be scaled
            categorical_cols=['cut', 'color', 'clarity']
            numerical_cols=['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Pipeline initiated")

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                 steps=[
                      ('imputer',SimpleImputer(strategy='median')),
                      ('scaler',StandardScaler())
                ]
            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                 steps=[
                      ('imputer',SimpleImputer(strategy='most_frequent')),
                      ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                      ('scaler',StandardScaler())
                 ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
                ])
            
            return preprocessor
            logging.info("Data Transformation is completed")

        except Exception as e:
            logging.error("Error occured in data transformation")
            raise CustomException(e,sys)
        

        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            # Reading the test and train data
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info("Reading the data is completed")   
            logging.info(f'Train Data Head :\n {train_df.head().to_string()}')
            logging.info(f'Test Data Head :\n {test_df.head().to_string()}')    

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformation_object()  # get the preprocessing object

            target_column_name='price'
            drop_columns=[target_column_name,'id']

            ## divide feature into dependent and independent features

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)  # Independent Features for training
            target_feature_train_df=train_df[target_column_name]  # Dependent Features for training

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)  # Independent Features for testing
            target_feature_test_df=test_df[target_column_name]  # Dependent Features    for testing

            ## apply the transformation

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info('applying processing object in train and test data is completed')

            # concat both input and target feature
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            ## save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessing pickle is saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.error("Error occured in data transformation")
            raise CustomException(e,sys)

