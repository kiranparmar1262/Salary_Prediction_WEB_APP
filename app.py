from flask import Flask,render_template,request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('salary_prediction_model.pkl')

df = pd.DataFrame()


@app.route('/',methods = ['POST','GET'])
def index():
    if request.method == 'POST':
        experience = request.form['experience']
        test_score = request.form['test_score']
        interview_score = request.form['interview_score']
        inputs = [experience,test_score,interview_score]
        input_features = [int(i) for i in inputs]
        feature_values = np.array(input_features)
        result_prediction = model.predict([feature_values])

        return render_template('index.html',prediction_text = 'Employee Salary Should be $ {}'.format(result_prediction[0][0].round(2)))

    return render_template('index.html')

@app.route('/preview')
def preview():
    df = pd.read_csv('data/salary_dataset.csv')
    return render_template('preview.html', df_view=df)

if __name__ == "__main__":
    app.run(debug=True)