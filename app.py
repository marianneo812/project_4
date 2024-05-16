from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt

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
        prediction = model.predict(data)[0]
        print(f"Prediction: {prediction}")  # Debug information

        return render_template('index.html', prediction=prediction)
    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template('index.html', prediction="Error occurred during prediction")

# Create and save the feature importance plot
def save_feature_importances(model, feature_names, output_file):
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(output_file)
    plt.close()

# Feature names based on the form input order
feature_names = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", 
    "Diabetes", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", 
    "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", 
    "DiffWalk", "Sex", "Age", "Education", "Income"
]

# Save the feature importance plot
save_feature_importances(model, feature_names, 'static/feature_importances.png')

if __name__ == '__main__':
    app.run(debug=True)
