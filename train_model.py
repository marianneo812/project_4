import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import sqlite3
import csv

# Read data from SQLite database
conn = sqlite3.connect('heart_disease.db')
data = pd.read_sql_query("SELECT * FROM heart_disease", conn)
conn.close()

# Split the data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

# Perform grid search
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Save the best model and scaler
pickle.dump(best_model, open('heart_disease_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Log the model optimization process
with open('model_optimization_log.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Parameters", "Accuracy", "Notes"])
    writer.writerow(["1", f"C={grid_search.best_params_['C']}, solver={grid_search.best_params_['solver']}", accuracy, "Initial model with GridSearchCV"])

# Print overall model performance
print(f"Best model parameters: {grid_search.best_params_}")
print(f"Model accuracy: {accuracy}")
