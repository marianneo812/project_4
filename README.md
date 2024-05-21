# Project-4-Group-8 - Hazel Hwang, Marianne O'Reilly, Rachel Jiang
# Title: Predicting the Risk of Heart Disease

## Purpose:
The purpose of this project is to train a machine learning model to recognise heart disease risk factors to predict whether an individual is at risk of developing heart disease. 

## Method:
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
- **Optimising using a different model:** Tensorflow model using three hidden layers was trained to achieve a higher accuracy score for predicting heart disease.
  
  ![image](https://github.com/marianneo812/project_4_group_8/assets/151903302/e958587d-5afa-41a9-97f5-e045c55a7203)
  
  Although a lower accuracy score was produced compared to the original model according to the Classification Report:
  ![image](https://github.com/marianneo812/project_4_group_8/assets/151903302/75d17e4c-9187-4ee5-9790-318faec82d7a)

## Interactive Webpage:
A webpage form was created (using app.py and index.html files) based on the model, whereby users input responses in fields for each 'feature' variable, and at the end of the form, the model will return a likelihood percentage of determining whether that user has or will develop heart disease (based on their responses). 
<img src="https://github.com/marianneo812/project_4_group_8/assets/151903302/5ea2cd3a-edcd-41c3-8525-26761cbbdd63" alt="Image" style="border: 1px solid black;">

The graph on the bottom shows the importance of each feature in determining accuracy of predicting heart disease. 
<img width="700" alt="image" src="https://github.com/marianneo812/project_4_group_8/assets/151903302/b13f6d10-e2a9-4353-a8c9-237d45ae876e">

## Conclusion:
  
