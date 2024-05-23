# Project-4-Group-8 - Hazel Hwang, Marianne O'Reilly, Rachel Jiang
# Title: Predicting the Risk of Heart Disease

## Purpose
The purpose of this project is to train a machine learning model to recognise heart disease risk factors to predict whether an individual is at risk of developing heart disease. 

## Method
**Dataset:**
The dataset used esd taken from Kaggle using a csv file containing 21 'features' (with binary or continuous values) and one 'target' variable of Heart Disease or Attack (binary - 0 represents 'No' and 1 represents 'Yes').

**Data Model Implementation:**
- **Preprocessing:** As part of cleaning the data, irrelevant columns (e.g., income, have healthcare) were dropped. The remaining columns were split into X (features) and Y (target). The data was normalised and standardised, and then split into training and testing sets.
- **Initialising and Training the Model:** parameters were initialised and **Random Forest Classifier** was the model used to determine best fit of the model.
  ![image](https://github.com/marianneo812/project_4_group_8/assets/151903302/fe94b600-8396-48c8-b138-fca49bd86963)

- **Evaluating the Model:** An accuracy score and a Classification Report were generated to determine the accuracy of the model. This particular model generated a 90.8% accuracy score. The model was saved into a pickle file using joblib.
  ![image](https://github.com/marianneo812/project_4_group_8/assets/151903302/7336aaed-7b10-42fe-8daf-055ac026486e)


**Data Model Optimisation:**
- **Optimising using same model:**
- **Optimising using a different model:** Tensorflow model using three hidden layers was trained to attempt to achieve a higher accuracy score for predicting heart disease.
  
  ![image](https://github.com/marianneo812/project_4_group_8/assets/151903302/e958587d-5afa-41a9-97f5-e045c55a7203)
  
  Although a lower accuracy score was produced compared to the original model according to the Classification Report:
  ![image](https://github.com/marianneo812/project_4_group_8/assets/151903302/75d17e4c-9187-4ee5-9790-318faec82d7a)

## Interactive Webpage
### Run the application:

- To run the application locally, execute the following command: python data/app.py
- Open a web browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000).
- Fill out the form with your health information and click "Predict" to get the prediction of heart disease likelihood.
  
![image](https://github.com/marianneo812/project_4_group_8/assets/149750967/e2eed049-9478-44ec-8f5f-d7839f9432a2)


## Key Components

### app.py

The `app.py` script is built using the Flask framework. It handles the server-side logic of the application, including routes for rendering the main HTML page and processing form submissions.

### index.html

The `index.html` file is the front-end part of the application. It contains the user interface elements, such as form fields for users to enter their health information. It also includes a BMI calculator to help users calculate their Body Mass Index directly on the site.

### Feature Importance Graph

The feature importance graph provides valuable insights into which factors have the greatest influence on predicting heart disease. According to the graph, General Health, Age, and Stroke rank among the top features shaping the model's predictions. Visualising feature importance serves not only to aid in model interpretation but also in effectively communicating results to stakeholders.

The graph displayed below illustrates the significance of each feature in accurately predicting heart disease.

<img width="700" alt="image" src="https://github.com/marianneo812/project_4_group_8/assets/151903302/b13f6d10-e2a9-4353-a8c9-237d45ae876e">

## Conclusion
Although the overall accuracy score, and precision and recall scores for those classed as not having heart disease is high, the precision and recall scores for those who do have heart disease is low. Meaning according to the Classification Report of the original model, 61% of those who were predicted to have heart disease, actually had heart disease. And 7% of those who actually had heart disease, were predicted to have heart disease. Therefore to conclude, this model would be good to use for predicting those who do not have heart disease, but would not be a good model to use for those who do have heart disease. The model would need more data on those who do have heart disease to be able to be trained with a higher accuracy score.
