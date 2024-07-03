# Diamond Price Prediction Project

This project predicts diamond prices using machine learning. The project is modular, with separate components for data ingestion, transformation, and model training, making it easy to understand and maintain.

## Description

This project aims to predict the price of diamonds using a machine learning model. The main functionalities include:

Data Ingestion: Reading and loading data.
Data Transformation: Cleaning and preprocessing data.
Model Training: Training machine learning models.
Prediction: Making predictions using the trained model.
Web Application: A Flask web app for interacting with the model.

## Folder Structure

1)artifacts/: Contains artifacts generated during the project, including the preprocessor, training and testing datasets, and the trained model.
2)notebooks/: Jupyter notebooks for Exploratory Data Analysis (EDA) and model training.
3)src/: Source code for the project.
4)components/: Contains Python scripts for data ingestion, data transformation, and model training.
5)pipeline/: Contains the training and prediction pipeline scripts.
6)templates/: Contains HTML templates for the web application.
7)application.py: The main Flask application file.
8)requirements.txt: List of Python dependencies required for the project.
9)setup.py: Script for packaging the project.
10).gitignore: Specifies files and directories to be ignored by Git.


## Installation

Clone the repository:
git clone https://github.com/Utkgitdev-07/diamond-price-prediction.git
cd diamond-price-prediction

Install the required packages:
pip install -r requirements.txt
Set up the project:


python setup.py install
Running the Application
Start the Flask application:


python application.py
Open your browser and navigate to:


http://127.0.0.1:5000/predictdata
The application will prompt for data input and display the predicted diamond price.

## Usage

1) Exploratory Data Analysis (EDA)
Explore the dataset using the EDA.ipynb notebook located in the notebooks/ directory.

2) Model Training
Train the model using the model_training.ipynb notebook located in the notebooks/ directory.

3) Data Ingestion
Ingest data using the data_ingestion.py script located in the src/components/ directory.

4) Data Transformation
Transform data using the data_transformation.py script located in the src/components/ directory.

5) Model Training
Train the model using the model_trainer.py script located in the src/components/ directory.

6) Training Pipeline
Run the training pipeline using the training_pipeline.py script located in the src/pipeline/ directory.

7) Prediction
Make predictions using the prediction.py script located in the src/pipeline/ directory.

## Project Structure Overview

1) Notebooks
EDA.ipynb: Exploratory Data Analysis.
model_training.ipynb: Model training notebook.

2) Components
data_ingestion.py: Handles data ingestion processes.
data_transformation.py: Handles data transformation processes.
model_trainer.py: Handles model training processes.

3) Pipeline
training_pipeline.py: Orchestrates the training pipeline.
prediction.py: Handles prediction logic.

4) Web Application
application.py: Flask web application for running predictions.
templates/: HTML templates for the web interface.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


## Project Structure


```plaintext
├── artifacts/
│   ├── preprocessor.pkl
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
├── notebooks/
│   ├── EDA.ipynb
│   ├── model_training.ipynb
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   ├── prediction.py
├── templates/
├── application.py
├── requirements.txt
├── setup.py
├── .gitignore

