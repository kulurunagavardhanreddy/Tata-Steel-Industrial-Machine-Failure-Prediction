# Tata_steel.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, auc, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv(r"C:\Users\nag15\Downloads\combined_data.csv")  # Replace with your actual path

# Data preprocessing function (unchanged)
def preprocess_data(df):
    df = df.drop_duplicates()
    df['Machine failure'] = df['Machine failure'].fillna(0).astype(int)
    return df

# Apply preprocessing
df = preprocess_data(df).reset_index(drop=True)

# Feature engineering function (unchanged)
def feature_engineering(df):
    df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Power Consumption'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
    df['Tool_Wear_Interaction'] = df['Tool wear [min]'] * df['Rotational speed [rpm]']
    return df

# Apply feature engineering
df = feature_engineering(df)

# Check class imbalance (unchanged)
def check_imbalance(df):
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=df["Machine failure"])
    ax.set_xticklabels(["No Failure (0)", "Failure (1)"])
    plt.title("Class Distribution of Machine Failure")
    plt.xlabel("Machine Failure")
    plt.ylabel("Count")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + 0.3, p.get_height() + 500), fontsize=12)
    plt.show()

check_imbalance(df)

# One-Hot Encoding for 'Type'
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded_type = encoder.fit_transform(df[['Type']])
encoded_df = pd.DataFrame(encoded_type, columns=encoder.get_feature_names_out(['Type']))

# Drop unnecessary columns and concatenate encoded features
df = df.drop(columns=['id', 'Product ID', 'Type'])
df = pd.concat([df, encoded_df], axis=1)

# Define features and target
X = df.drop(columns=['Machine failure'])
y = df['Machine failure']

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
feature_names = X.columns  # Store feature names

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split data (stratified)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- Improved Model Training ---
best_f1 = 0
best_model = None
best_smote_ratio = None
best_class_weight = None

for smote_ratio in [0.3, 0.5, 0.7]:
    for class_weight in [1, 2, 5, 10]:
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        model = XGBClassifier(scale_pos_weight=class_weight, random_state=42) # Added random state for reproducibility

        cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='f1')
        mean_cv_f1 = cv_scores.mean()

        if mean_cv_f1 > best_f1:
            best_f1 = mean_cv_f1
            best_model = model
            best_smote_ratio = smote_ratio
            best_class_weight = class_weight

# Train the best model on the full resampled training data
smote = SMOTE(sampling_strategy=best_smote_ratio, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
best_model.fit(X_train_resampled, y_train_resampled)

# Save the best model, scaler, and encoder
with open('best_xgb_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
with open('onehot_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(encoder, encoder_file)


print(f"Best SMOTE Ratio: {best_smote_ratio}")
print(f"Best Class Weight: {best_class_weight}")
print(f"Best Cross-Validation F1: {best_f1}")

# --- Evaluation ---
y_pred = best_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
plt.title("Confusion Matrix")
plt.show()

# Feature importance visualization
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# Adjust decision threshold (carefully - focus on F1)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
threshold = 0.3 # Example threshold - tune this!
y_pred_adjusted = (y_pred_proba > threshold).astype(int)

print(f"Classification Report (Adjusted Threshold):\n", classification_report(y_test, y_pred_adjusted))

# Training accuracy (of the best model)
y_train_pred = best_model.predict(X_train) # Predict on the training data *after* resampling
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', color='b', label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()

# Cross-validation results (on the *resampled* training data)
cv_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=5, scoring='f1')  # <--- Use resampled data here
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean Cross-validation F1 score: {cv_scores.mean():.4f}")


# ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='r')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()