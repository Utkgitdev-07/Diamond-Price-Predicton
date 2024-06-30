from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':            # if the request method is get then return the form.html page
        return render_template('form.html')
    
    else:                    # if the request method is post then get the data from the form and predict the result
        data=CustomData(
            carat=float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')
        )
        final_new_data=data.get_data_as_dataframe()  # getting the data as dataframe using the get_data_as_dataframe function of the CustomData class 
        predict_pipeline=PredictPipeline()     # creating the object of the PredictPipeline class to predict the data using the model and preprocessor 
        pred=predict_pipeline.predict(final_new_data) # predicting the data using the predict function of the PredictPipeline class

        results=round(pred[0],2)  # rounding the predicted value upto 2 decimal places

        return render_template('form.html',final_result=results) # returning the form.html page with the predicted result
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)