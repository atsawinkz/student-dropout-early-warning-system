import nbformat
import sys

notebook_path = "notebooks/Predict_Students'_Dropout_and_Academic_Success_Lab.ipynb"

try:
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 1. Update the first markdown cell to include a professional project overview
    intro_markdown = """# **Predict Students' Dropout and Academic Success**
## Project Overview
This project aims to develop an early warning system to predict whether a student will dropout, enroll, or graduate based on various demographic, socio-economic, and academic performance factors. 
By employing a **Timeline-Based Machine Learning Approach**, we can understand the key factors contributing to student attrition and help institutions intervene early.

## Table of Contents
1. **Data Exploration & Visualization (EDA)**
2. **Feature Engineering**
3. **Model Training & Timeline Evaluation**
4. **Advanced Machine Learning & Deployment Preparation**
"""
    nb.cells[0].source = intro_markdown

    # 2. Append new cells for Advanced ML and Deployment
    # Markdown cell
    new_md_cell = nbformat.v4.new_markdown_cell(source="""# **4. Advanced Machine Learning & Deployment Preparation**
## 4.1 Hyperparameter Tuning
To ensure our model achieves the best possible performance, we will perform Hyperparameter Tuning on the Day-1 model using `GridSearchCV`. This process searches for the optimal combination of parameters for the Random Forest Classifier.
""")

    # Code cell for GridSearchCV
    new_code_cell_1 = nbformat.v4.new_code_cell(source="""from sklearn.model_selection import GridSearchCV
import joblib

print("Starting Hyperparameter Tuning for Day-1 Random Forest...")

# Defining parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Base model
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# GridSearchCV setup (cv=3 for faster execution during demonstration)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=1)

# Fit GridSearchCV 
grid_search.fit(X_train_d1, y_train)

# Best model
best_rf_day1 = grid_search.best_estimator_
print(f"\\nBest Parameters Found: {grid_search.best_params_}")

# Evaluate best model
y_pred_tuned = best_rf_day1.predict(X_test_d1)
print(f"\\nTuned Model Accuracy: {accuracy_score(y_test, y_pred_tuned)*100:.2f}%")
print("Classification Report (Tuned):")
print(classification_report(y_test, y_pred_tuned, target_names=['Dropout', 'Graduate']))
""")

    # Markdown cell
    new_md_cell_2 = nbformat.v4.new_markdown_cell(source="""## 4.2 Model Deployment Preparation
Now that we have an optimized model for Day-1 predictions, we can save it to disk. This serialized model (`predict_dropout_model.pkl`) can then be integrated into a web application or an internal dashboard for real-time predictions.
""")

    # Code cell for saving the model
    new_code_cell_2 = nbformat.v4.new_code_cell(source="""# Save the best trained model to disk
model_filename = 'predict_dropout_model.pkl'
joblib.dump(best_rf_day1, model_filename)

print(f"Model successfully saved to {model_filename}")
print("The model is now ready for deployment in a production environment!")
""")

    # Append the cells to the notebook
    nb.cells.extend([new_md_cell, new_code_cell_1, new_md_cell_2, new_code_cell_2])

    # Write back to the same file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("Notebook successfully updated with Advanced ML techniques and a professional intro!")

except Exception as e:
    print(f"An error occurred: {e}")
