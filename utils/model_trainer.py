import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Handles machine learning model training for market sentiment prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
    def prepare_features(self, data):
        """Prepare feature matrix and target vector"""
        try:
            # Define feature columns
            feature_columns = [
                'fii_net',
                'fii_change', 
                'pcr',
                'fii_3d_avg',
                'pcr_3d_avg',
                'fii_volatility',
                'pcr_change',
                'fii_momentum'
            ]
            
            # Ensure all feature columns exist
            missing_cols = [col for col in feature_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")
            
            # Extract features and target
            X = data[feature_columns].copy()
            y = data['target'].copy()
            
            # Handle any remaining NaN values
            X = X.fillna(X.median())
            
            # Remove any infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            return X, y, feature_columns
            
        except Exception as e:
            raise Exception(f"Error preparing features: {str(e)}")
    
    def train_model(self, data):
        """Train the logistic regression model"""
        try:
            # Prepare features
            X, y, feature_columns = self.prepare_features(data)
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train logistic regression model
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'  # Handle class imbalance
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            print(f"Model Training Complete:")
            print(f"Training Accuracy: {train_score:.3f}")
            print(f"Testing Accuracy: {test_score:.3f}")
            
            # Store trained components
            self.model = model
            self.scaler = scaler
            self.feature_columns = feature_columns
            
            return model, scaler, feature_columns
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def predict_sentiment(self, features):
        """Predict sentiment for given features"""
        try:
            if self.model is None or self.scaler is None:
                raise ValueError("Model not trained yet")
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get predictions and probabilities
            predictions = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)
            
            return predictions, probabilities
            
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            # For logistic regression, coefficients represent feature importance
            importance = self.model.coef_[0]
            feature_importance = dict(zip(self.feature_columns, importance))
            
            return feature_importance
            
        except Exception as e:
            raise Exception(f"Error getting feature importance: {str(e)}")
    
    def calculate_accuracy(self, y_true, y_pred):
        """Calculate model accuracy"""
        return accuracy_score(y_true, y_pred)
    
    def generate_classification_report(self, y_true, y_pred):
        """Generate detailed classification report"""
        return classification_report(y_true, y_pred)
    
    def validate_model_performance(self, data):
        """Validate model performance on the full dataset"""
        try:
            if self.model is None or self.scaler is None:
                raise ValueError("Model not trained yet")
            
            # Prepare features
            X, y, _ = self.prepare_features(data)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            y_pred = self.model.predict(X_scaled)
            y_prob = self.model.predict_proba(X_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            
            # Calculate additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            
            performance_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_prob
            }
            
            return performance_metrics
            
        except Exception as e:
            raise Exception(f"Error validating model performance: {str(e)}")
    
    def cross_validate_model(self, data, cv_folds=5):
        """Perform cross-validation on the model"""
        try:
            from sklearn.model_selection import cross_val_score
            
            # Prepare features
            X, y, _ = self.prepare_features(data)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create model
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='accuracy')
            
            cv_results = {
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'individual_scores': cv_scores
            }
            
            return cv_results
            
        except Exception as e:
            raise Exception(f"Error in cross-validation: {str(e)}")
