# Student Dropout Early Warning System
A Timeline-Based Machine Learning Approach

This project contains a Jupyter Notebook demonstrating an early warning system to predict students' dropout and academic success using Machine Learning techniques.

## Project Structure
- `data/`: Contains the dataset used for model training and evaluation (`Predict Students' Dropout and Academic Success.csv`).
- `notebooks/`: Contains the main Jupyter Notebook for data exploration, preprocessing, and model training.

## Getting Started

### Prerequisites
Make sure you have Python installed. You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Usage
Run the Jupyter Notebook to explore the code:

```bash
jupyter notebook notebooks/Predict_Students'_Dropout_and_Academic_Success_Lab.ipynb
```

## Key Features
- **Exploratory Data Analysis (EDA):** Visualizing key data factors leading to dropout.
- **Timeline-Based Predictions:** Training models from Day-1 data vs. End of Semester 1 data to show early warning capabilities.
- **Hyperparameter Tuning:** Utilizing `GridSearchCV` to optimize the Random Forest model for better accuracy.
- **Model Deployment:** Saving the trained model `predict_dropout_model.pkl` for easy integration into web applications or dashboards.

## Technologies Used
- **Pandas** for Data Manipulation
- **Matplotlib & Seaborn** for Data Visualization
- **Scikit-Learn** for Machine Learning (Random Forest, GridSearchCV, evaluation metrics)
- **Joblib** for Model Deployment Preparation
