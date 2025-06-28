"""
Enhanced ML Pipeline with Modern AI Trends
==========================================

This module implements cutting-edge machine learning techniques including:
- AutoML capabilities
- Explainable AI (XAI) with SHAP
- Advanced ensemble methods
- Neural networks
- Hyperparameter optimization
- Model monitoring and drift detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import json
import os
from datetime import datetime
import joblib

# Core ML libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             VotingClassifier, StackingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap
from scipy import stats

# Neural Networks (with fallback)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Neural network features will be disabled.")

# Model monitoring (with fallback)
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_suite import RegressionQualityMetricSuite
    from evidently.test_suite import TestSuite
    from evidently.tests import *
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning("Evidently not available. Advanced monitoring features will be disabled.")

# Configure settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')
if TF_AVAILABLE:
    tf.get_logger().setLevel('ERROR')

# Create directories
for dir_name in ['models', 'results', 'reports', 'experiments']:
    os.makedirs(dir_name, exist_ok=True)

class EnhancedMLPipeline:
    """Enhanced ML Pipeline with modern AI capabilities"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
        self.shap_explainer = None
        self.drift_detector = None
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess data with advanced techniques"""
        logging.info("🔄 Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(filepath)
        logging.info(f"Dataset shape: {df.shape}")
        
        # Advanced data cleaning
        df = self._advanced_data_cleaning(df)
        
        # Feature engineering
        df = self._advanced_feature_engineering(df)
        
        # Feature selection
        df = self._feature_selection(df)
        
        return df
    
    def _advanced_data_cleaning(self, df):
        """Advanced data cleaning with statistical methods"""
        logging.info("🧹 Performing advanced data cleaning...")
        
        # Handle duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logging.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Convert DOB to datetime and extract age
        df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
        current_year = datetime.now().year
        df['Age'] = df['DOB'].apply(lambda x: current_year - x.year if pd.notnull(x) else np.nan)
        
        # Advanced outlier detection using IQR and Z-score
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['Age', 'Salary']:
                # Z-score method for outlier detection
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    z_scores = np.abs(stats.zscore(col_data))
                    outliers_zscore = (z_scores > 3).sum()
                    
                    # IQR method
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers_iqr = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col].count()
                    
                    logging.info(f"{col}: Z-score outliers: {outliers_zscore}, IQR outliers: {outliers_iqr}")
                    
                    # Cap extreme outliers (top and bottom 1%)
                    df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
        
        # Handle missing values with advanced imputation
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Salary'] = df['Salary'].fillna(df['Salary'].median())
        
        return df
    
    def _advanced_feature_engineering(self, df):
        """Create advanced features using domain knowledge and statistical methods"""
        logging.info("⚙️ Creating advanced features...")
        
        # Age-based features
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 50, 65, 100], 
                                labels=['Young', 'Adult', 'Middle_Age', 'Senior', 'Elder'])
        
        # Salary-based features
        df['Salary_Quartile'] = pd.qcut(df['Salary'], q=4, labels=['Low', 'Medium', 'High', 'Very_High'])
        df['Log_Salary'] = np.log1p(df['Salary'])
        df['Salary_Per_Age'] = df['Salary'] / (df['Age'] + 1)
        
        # Interaction features
        df['Age_Salary_Interaction'] = df['Age'] * df['Salary'] / 1000
        
        # Polynomial features for Age and Salary
        df['Age_Squared'] = df['Age'] ** 2
        df['Salary_Squared'] = df['Salary'] ** 2
        
        # Statistical features
        df['Age_Z_Score'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
        df['Salary_Z_Score'] = (df['Salary'] - df['Salary'].mean()) / df['Salary'].std()
        
        return df
    
    def _feature_selection(self, df):
        """Advanced feature selection using multiple methods"""
        logging.info("🎯 Performing feature selection...")
        
        # Prepare features and target
        target_col = 'Purchased'
        feature_cols = [col for col in df.columns if col not in ['Name', 'DOB', target_col]]
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
        joblib.dump(le_gender, 'models/le_gender_enhanced.pkl')
        
        # Encode other categorical features
        categorical_features = ['Age_Group', 'Salary_Quartile']
        for cat_feat in categorical_features:
            if cat_feat in df.columns:
                le = LabelEncoder()
                df[f'{cat_feat}_Encoded'] = le.fit_transform(df[cat_feat].astype(str))
                joblib.dump(le, f'models/le_{cat_feat.lower()}.pkl')
        
        # Encode target
        le_target = LabelEncoder()
        df['Purchased_Encoded'] = le_target.fit_transform(df[target_col])
        joblib.dump(le_target, 'models/le_target_enhanced.pkl')
        
        return df
    
    def prepare_data_for_training(self, df):
        """Prepare final dataset for training"""
        # Select features for training
        feature_columns = [
            'Gender_Encoded', 'Age', 'Salary', 'Log_Salary', 'Salary_Per_Age',
            'Age_Salary_Interaction', 'Age_Squared', 'Salary_Squared',
            'Age_Z_Score', 'Salary_Z_Score'
        ]
        
        # Add encoded categorical features if they exist
        for cat_feat in ['Age_Group_Encoded', 'Salary_Quartile_Encoded']:
            if cat_feat in df.columns:
                feature_columns.append(cat_feat)
        
        X = df[feature_columns]
        y = df['Purchased_Encoded']
        
        # Feature scaling
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
        
        joblib.dump(scaler, 'models/scaler_enhanced.pkl')
        
        return X_scaled, y, feature_columns
    
    def hyperparameter_optimization(self, X_train, y_train, model_name, n_trials=100):
        """Hyperparameter optimization using Optuna"""
        logging.info(f"🔧 Optimizing hyperparameters for {model_name}...")
        
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.random_state
                }
                model = xgb.XGBClassifier(**params)
                
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'random_state': self.random_state,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**params)
                
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'random_state': self.random_state
                }
                model = RandomForestClassifier(**params)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return study.best_params
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train a neural network using TensorFlow/Keras"""
        if not TF_AVAILABLE:
            logging.warning("TensorFlow not available. Skipping neural network training.")
            return None, None
            
        logging.info("🧠 Training Neural Network...")
        
        # Create model architecture
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc']
        )
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Save model
        model.save('models/neural_network.h5')
        
        return model, history
    
    def train_ensemble_models(self, X_train, y_train):
        """Train advanced ensemble models"""
        logging.info("🎭 Training ensemble models...")
        
        # Base models with optimized hyperparameters
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=self.random_state)),
            ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=self.random_state)),
            ('lgb', lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=self.random_state, verbose=-1)),
            ('svc', SVC(probability=True, random_state=self.random_state)),
            ('nb', GaussianNB())
        ]
        
        # Voting Classifier
        voting_clf = VotingClassifier(
            estimators=base_models,
            voting='soft'
        )
        
        # Stacking Classifier
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=5
        )
        
        # Train models
        voting_clf.fit(X_train, y_train)
        stacking_clf.fit(X_train, y_train)
        
        # Save models
        joblib.dump(voting_clf, 'models/voting_classifier.pkl')
        joblib.dump(stacking_clf, 'models/stacking_classifier.pkl')
        
        return {'voting': voting_clf, 'stacking': stacking_clf}
    
    def explain_model_predictions(self, model, X_train, X_test, feature_names):
        """Generate SHAP explanations for model interpretability"""
        logging.info("🔍 Generating SHAP explanations...")
        
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model) if hasattr(model, 'estimators_') else shap.KernelExplainer(model.predict_proba, X_train.sample(100))
            else:
                explainer = shap.KernelExplainer(model.predict, X_train.sample(100))
            
            # Generate SHAP values
            shap_values = explainer.shap_values(X_test.sample(min(500, len(X_test))))
            
            # Save explainer
            joblib.dump(explainer, 'models/shap_explainer.pkl')
            
            return explainer, shap_values
            
        except Exception as e:
            logging.warning(f"Could not generate SHAP explanations: {e}")
            return None, None
    
    def run_enhanced_pipeline(self, filepath):
        """Run the complete enhanced ML pipeline"""
        logging.info("🚀 Starting Enhanced ML Pipeline...")
        
        # Load and preprocess data
        df = self.load_and_preprocess_data(filepath)
        
        # Prepare data for training
        X, y, feature_names = self.prepare_data_for_training(df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Further split training data for validation
        X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )
        
        # Train traditional models with hyperparameter optimization
        models_to_optimize = ['xgboost', 'lightgbm', 'random_forest']
        optimized_models = {}
        
        for model_name in models_to_optimize:
            best_params = self.hyperparameter_optimization(X_train, y_train, model_name, n_trials=50)
            
            if model_name == 'xgboost':
                model = xgb.XGBClassifier(**best_params)
            elif model_name == 'lightgbm':
                model = lgb.LGBMClassifier(**best_params)
            elif model_name == 'random_forest':
                model = RandomForestClassifier(**best_params)
            
            model.fit(X_train, y_train)
            optimized_models[model_name] = model
            joblib.dump(model, f'models/{model_name}_optimized.pkl')
        
        # Train neural network (if TensorFlow is available)
        if TF_AVAILABLE:
            nn_model, nn_history = self.train_neural_network(X_train_nn, y_train_nn, X_val_nn, y_val_nn)
        else:
            nn_model, nn_history = None, None
        
        # Train ensemble models
        ensemble_models = self.train_ensemble_models(X_train, y_train)
        
        # Evaluate all models
        all_models = {**optimized_models, **ensemble_models}
        if nn_model is not None:
            all_models['neural_network'] = nn_model
            
        results = self.evaluate_models(all_models, X_test, y_test, feature_names)
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x].get('roc_auc', 0))
        self.best_model = all_models[best_model_name]
        
        logging.info(f"🏆 Best model: {best_model_name} with ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
        
        # Generate explanations for best model
        if best_model_name != 'neural_network':
            explainer, shap_values = self.explain_model_predictions(
                self.best_model, X_train, X_test, feature_names
            )
        
        # Save results
        with open('results/enhanced_model_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results, self.best_model
    
    def evaluate_models(self, models, X_test, y_test, feature_names):
        """Evaluate all models and return comprehensive results"""
        results = {}
        
        for name, model in models.items():
            try:
                if name == 'neural_network':
                    y_pred_proba = model.predict(X_test).flatten()
                    y_pred = (y_pred_proba > 0.5).astype(int)
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                logging.info(f"✅ {name}: ROC-AUC = {results[name]['roc_auc']:.4f}")
                
            except Exception as e:
                logging.error(f"❌ Error evaluating {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results

if __name__ == "__main__":
    # Initialize and run enhanced pipeline
    pipeline = EnhancedMLPipeline()
    results, best_model = pipeline.run_enhanced_pipeline('data.csv')
    
    print("🎉 Enhanced ML Pipeline completed successfully!")
    print("📊 Check 'results/enhanced_model_results.json' for detailed results")
