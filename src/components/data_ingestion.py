import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


## initialize the data ingestion configuration

@dataclass
class DataIngestionconfig :          # making path to save the data
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw')


## create a data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()  ## initialize the data ingestion configuration got all paths

    def initiate_data_ingestion(self):               ## here we do all task reading of data and train test split
        logging.info("Data Ingestion has started")
        try:
            df=pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info("Data has been read successfully as pandas dataframe") ## read the data

            os.makedirs(self.ingestion_config.raw_data_path,exist_ok=True) ## make directory to save the raw data if exits then ignore here it refer to artifacts/raw

            raw_data_file_path = os.path.join(self.ingestion_config.raw_data_path, 'raw_data.csv')
            df.to_csv(raw_data_file_path, index=False)  ## Save the raw data in the directory as a csv file and index is false so that it will not save the index
        
            logging.info("Train test split has started") ## train test split
            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42) ## split the data into train and test set

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) ## save the train data in the directory as csv file and index is false so that it will not save the index and header is true so that it will save the header
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) ## save the test data in the directory as csv file and index is false so that it will not save the index and header is true so that it will save the header

            logging.info("Ingestion of Data has been completed successfully") 

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        

        except Exception as e:
            logging.info("Error Occured in Data Ingestion Config")
            logging.error(str(e))
            raise CustomException(e, sys) ## raise the exception if any error occurs