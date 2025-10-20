# Autism Prediction

## Overview
This project focuses on developing a **machine learning model** to predict the likelihood of **Autism Spectrum Disorder (ASD)** based on behavioral and demographic features.  
By analyzing screening responses and background information, the system aims to provide early insights that can aid healthcare professionals and researchers in understanding ASD tendencies.

---

## 📂 Dataset Description

### Files Included
- **`train.csv`** – Training dataset containing labeled examples with input features and target labels.  
- **`test.csv`** – Testing dataset used to evaluate model performance on unseen data.

### Key Features
| Feature | Description |
|----------|-------------|
| `age` | Age of the individual |
| `gender` | Gender (Male/Female) |
| `ethnicity` | Ethnic background of the respondent |
| `jaundice` | Whether the individual had jaundice at birth |
| `family_history_with_asd` | Indicates if autism is present in the family history |
| `screening_answers` | Responses to a standardized behavioral screening questionnaire |
| `result_score` | Numerical score derived from screening answers |
| `class/target` | Target variable (Yes/No for autism) |

---

## Project Workflow

### 1️. Data Preprocessing
- Handle **missing values** and **inconsistent entries**.  
- Convert **categorical variables** into numerical representations using encoding techniques.  
- Apply **scaling/normalization** to numerical features.  
- Address **outliers** and maintain dataset integrity.

### 2️. Exploratory Data Analysis (EDA)
- Visualize distributions and identify correlations between variables.  
- Analyze relationships between **age**, **family history**, and **autism outcomes**.  
- Use plots such as histograms, pairplots, and correlation heatmaps to derive insights.

### 3️. Feature Engineering
- Derive new meaningful variables (e.g., total score from screening responses).  
- Perform **feature importance ranking** using tree-based models.  
- Select the most influential predictors to improve model performance.

### 4️. Model Development
Multiple supervised learning algorithms are implemented to identify the most effective model for autism prediction:
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- Gradient Boosting / XGBoost  

Hyperparameter optimization is performed using **GridSearchCV** or **RandomizedSearchCV** to maximize performance.

### 5️. Model Evaluation
Each model is assessed using several evaluation metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC–AUC Curve**

Performance comparison across models helps identify the most suitable one for deployment.

### 6️. Prediction and Model Deployment
- The best-performing model is applied to the test dataset for prediction.  
- Results are exported as a structured output file containing the predicted class labels and probabilities.  
- The model can later be integrated into a web interface for real-time prediction.

---

## 7. Expected Outcomes
- A reliable **classification model** capable of predicting ASD tendencies.  
- Insightful analysis highlighting **key behavioral and familial factors** linked to autism.  
- Comprehensive performance metrics validating model effectiveness and stability.

---

## Tools and Libraries

| Category | Libraries Used |
|-----------|----------------|
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Model Evaluation | `sklearn.metrics`, `roc_auc_score` |
| Deployment | `joblib` (for model saving and loading) |

---

## Future Enhancements
- Incorporate **explainable AI (XAI)** tools such as **SHAP** or **LIME** for model interpretability.  
- Develop an **interactive dashboard** for clinical or research use.  
- Integrate additional **genetic, environmental, or behavioral datasets** for improved generalization.  
- Apply **deep learning architectures** (e.g., MLPs or TabNet) for enhanced predictive power.

---

## Example Workflow Snippet

```python
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
train_df = pd.read_csv("train.csv")

# Split into features and target
X = train_df.drop("class", axis=1)
y = train_df["class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Import additional evaluation tools
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Accuracy and AUC
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"✅ Accuracy: {acc:.3f}")
print(f"✅ ROC–AUC: {auc:.3f}")

# ROC curve
# Import additional evaluation tools
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Accuracy and AUC
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f" Accuracy: {acc:.3f}")
print(f" ROC–AUC: {auc:.3f}")

#Feature Importance Visualization
# Display feature importances (for tree-based models)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Save and Load the Model
import joblib

# Save trained model
joblib.dump(model, "autism_rf_model.pkl")
print("Model saved as autism_rf_model.pkl")

# Later: Load the model again
loaded_model = joblib.load("autism_rf_model.pkl")

# Use for predictions
new_predictions = loaded_model.predict(X_test)

# Make Predictions on New or Test Data
# Load test dataset
test_df = pd.read_csv("test.csv")

# Predict autism likelihood
test_predictions = model.predict(test_df)
test_probabilities = model.predict_proba(test_df)[:, 1]

# Save results
submission = pd.DataFrame({
    "ID": test_df.index,
    "Predicted_Class": test_predictions,
    "Prediction_Probability": test_probabilities
})
submission.to_csv("autism_predictions.csv", index=False)
print("Predictions saved to autism_predictions.csv")
