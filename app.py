from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = [float(x) for x in request.form.values()]
        print(f"Received data: {data}")  # Debug information
        data = np.array(data).reshape(1, -1)

        # Scale the data
        data = scaler.transform(data)
        print(f"Scaled data: {data}")  # Debug information

        # Make prediction
        probability = model.predict_proba(data)[0][1]  # Get the probability of the positive class
        print(f"Prediction Probability: {probability}")  # Debug information

        return render_template('index.html', prediction=probability)
    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template('index.html', prediction="Error occurred during prediction")

if __name__ == '__main__':
    app.run(debug=True)

