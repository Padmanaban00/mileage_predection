import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load the model
regmodel = pickle.load(open('pickle_file.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html',show_predection =False)
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    input_features = [float(data['acceleration']), float(data['displacement']), float(data['horsepower']), float(data['weight'])]
    input_features = np.array(input_features).reshape(1, -1)
    print(data)
    print(np.array(list(data.values())).reshape(-1,1))
    new_data = scalar.transform(input_features)
    output = regmodel.predict(input_features)
    print(output[0])
    return jsonify(output[0])





@app.route('/predict',methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    input_features = [float(data[0]), float(data[1]), float(data[2]), float(data[3])]
    input_features = np.array(input_features).reshape(1, -1)
    final_input =scalar.transform(input_features)
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The predicted mileage is {}".format(output))

if __name__ =="__main__":
    app.run(debug=True)