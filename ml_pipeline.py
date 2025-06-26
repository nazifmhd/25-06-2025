import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
from datetime import datetime
import logging
import warnings
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('results'):
    os.makedirs('results')

def load_and_explore_data(filepath):
    """Load data and perform initial exploration"""
    logging.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Data types:\n{df.dtypes}")
    
    # Summary statistics
    summary_stats = df.describe().T
    logging.info(f"Summary statistics calculated")
    
    # Missing values analysis
    missing_values = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_data = pd.DataFrame({'Missing Values': missing_values, 
                                'Percentage': missing_percent})
    logging.info(f"Missing data analysis:\n{missing_data[missing_data['Missing Values'] > 0]}")
    
    return df

def clean_data(df):
    """Clean the dataset"""
    logging.info("Starting data cleaning process")
    
    # Store original shape for comparison
    original_shape = df.shape
    
    # Drop duplicates
    df = df.drop_duplicates()
    logging.info(f"Dropped {original_shape[0] - df.shape[0]} duplicate rows")
    
    # Handle missing values with appropriate strategy for each column
    # For numeric columns, use median (more robust to outliers than mean)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logging.info(f"Filled missing values in {col} with median: {median_val:.2f}")
    
    # Check for outliers in numeric columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        logging.info(f"Detected {outliers} outliers in {col}")
    
    return df

def feature_engineering(df):
    """Create new features and transform existing ones"""
    logging.info("Starting feature engineering")
    
    # Convert DOB to datetime
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    
    # Calculate age from DOB
    current_year = datetime.now().year
    df['Age'] = df['DOB'].apply(lambda dob: current_year - dob.year if pd.notnull(dob) else np.nan)
    logging.info("Created 'Age' feature from 'DOB'")
    
    # Create age groups
    bins = [0, 18, 30, 45, 60, 100]
    labels = ['Under 18', '18-30', '31-45', '46-60', 'Over 60']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
    logging.info("Created 'AgeGroup' categorical feature")
    
    # Salary ranges
    salary_bins = [0, 50000, 100000, 150000]
    salary_labels = ['Low', 'Medium', 'High']
    df['SalaryRange'] = pd.cut(df['Salary'], bins=salary_bins, labels=salary_labels)
    logging.info("Created 'SalaryRange' categorical feature")
    
    return df

def encode_and_split(df, target_col='Purchased'):
    """Encode categorical variables and split data"""
    logging.info("Encoding categorical features and preparing data for modeling")
    
    # Separate features and target
    X = df[['Gender', 'Salary', 'Age']]
    y = df[target_col]
    
    # Encode categorical target
    le_purchase = LabelEncoder()
    y = le_purchase.fit_transform(y)
    target_mapping = dict(zip(le_purchase.classes_, le_purchase.transform(le_purchase.classes_)))
    logging.info(f"Target encoding mapping: {target_mapping}")
    
    # Encode categorical features
    le_gender = LabelEncoder()
    X['Gender'] = le_gender.fit_transform(X['Gender'])
    gender_mapping = dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))
    logging.info(f"Gender encoding mapping: {gender_mapping}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test, le_gender, le_purchase

def scale_features(X_train, X_test):
    """Scale numeric features"""
    logging.info("Scaling numeric features")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models"""
    logging.info("Training and evaluating multiple models")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        logging.info(f"{name} CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logging.info(f"{name} Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Store results
        results[name] = {
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1
        }
        
        # Keep track of best model based on F1 score
        if f1 > best_score:
            best_model = model
            best_score = f1
            best_name = name
    
    logging.info(f"Best model: {best_name} with F1 score: {best_score:.4f}")
    
    # Save results
    with open('results/model_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return best_model, results

def save_model(model, scaler, le_gender, le_purchase):
    """Save model and preprocessing objects"""
    logging.info("Saving model and preprocessing objects")
    
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le_gender, 'models/le_gender.pkl')
    joblib.dump(le_purchase, 'models/le_purchase.pkl')
    
    logging.info("Model and preprocessing objects saved successfully")

def visualize_results(X_test, y_test, model, results):
    """Create visualizations for model performance"""
    logging.info("Creating visualizations")
    
    # Create directory for visualizations
    if not os.path.exists('results/visualizations'):
        os.makedirs('results/visualizations')
    
    # Feature importance (if available)
    try:
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            features = ['Gender', 'Salary', 'Age']
            importances = model.feature_importances_
            indices = np.argsort(importances)
            plt.title('Feature Importance')
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.savefig('results/visualizations/feature_importance.png')
            
        elif hasattr(model, 'coef_'):
            plt.figure(figsize=(10, 6))
            features = ['Gender', 'Salary', 'Age']
            importances = model.coef_[0]
            indices = np.argsort(importances)
            plt.title('Feature Coefficients')
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel('Coefficient Value')
            plt.savefig('results/visualizations/feature_coefficients.png')
    except Exception as e:
        logging.warning(f"Could not create feature importance plot: {e}")
    
    # ROC curve
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('results/visualizations/roc_curve.png')
    except Exception as e:
        logging.warning(f"Could not create ROC curve: {e}")
    
    # Confusion Matrix
    try:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('results/visualizations/confusion_matrix.png')
    except Exception as e:
        logging.warning(f"Could not create confusion matrix: {e}")
    
    # Model comparison
    try:
        model_names = list(results.keys())
        accuracies = [results[model]['test_accuracy'] for model in model_names]
        f1_scores = [results[model]['test_f1'] for model in model_names]
        
        plt.figure(figsize=(12, 8))
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Accuracy')
        plt.bar(x + width/2, f1_scores, width, label='F1 Score')
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Comparison: Accuracy vs F1 Score')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/visualizations/model_comparison.png')
    except Exception as e:
        logging.warning(f"Could not create model comparison plot: {e}")

def main():
    """Main execution function"""
    logging.info("Starting ML pipeline")
    
    # Load and explore data
    df = load_and_explore_data("data.csv")
    
    # Clean data
    df = clean_data(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Encode and split data
    X_train, X_test, y_train, y_test, le_gender, le_purchase = encode_and_split(df)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train and evaluate models
    best_model, results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Visualize results
    visualize_results(X_test_scaled, y_test, best_model, results)
    
    # Save model and preprocessing objects
    save_model(best_model, scaler, le_gender, le_purchase)
    
    logging.info("ML pipeline completed successfully")

if __name__ == "__main__":
    main()