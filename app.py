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
        # Get data from form, excluding 'weight' and 'height'
        data = [float(request.form.get(key)) for key in request.form.keys() if key not in ['weight', 'height']]
        print(f"Received data: {data}")  # Debug information

        # Ensure the length of data matches the expected number of features
        expected_num_features = 18  # Adjusted this based on the actual number of features
        if len(data) != expected_num_features:
            raise ValueError(f"Expected {expected_num_features} features, but got {len(data)}")

        data = np.array(data).reshape(1, -1)

        # Scale the data
        data = scaler.transform(data)
        print(f"Scaled data: {data}")  # Debug information

        # Make prediction
        probability = model.predict_proba(data)[0][1]  # Get the probability of the positive class
        prediction_percentage = probability * 100
        formatted_prediction = f"{prediction_percentage:.4f}%"
        print(f"Prediction Probability: {formatted_prediction}")  # Debug information

        return render_template('index.html', prediction=formatted_prediction)
    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template('index.html', prediction=f"Error occurred during prediction: {e}")

if __name__ == '__main__':
    app.run(debug=True)
