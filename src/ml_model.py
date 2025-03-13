import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.config import ML_FEATURES, ML_TEST_SIZE, ML_MAX_DEPTH, EARLY_STOPPING_ROUNDS
import os

class MLModel:
    def __init__(self, model_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'xgboost_model.json')):
        """Initialize the ML model, loading from file if it exists."""
        self.model = None
        self.model_path = model_path
        if os.path.exists(self.model_path):
            try:
                self.model = xgb.Booster()
                self.model.load_model(self.model_path)
                print(f"Loaded existing model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model from {self.model_path}: {e}")
                self.model = None
        else:
            print(f"No model file found at {self.model_path}. Model must be trained first.")

    def train(self, df):
        """Train the XGBoost model to predict >1% price increases."""
        print("Training ML Model...")
        # Define target: 1 if next close increases >1%, 0 otherwise
        df['target'] = (df['close'].shift(-1) / df['close'] - 1 > 0.01).astype(int)
        df = df.dropna()
        print(f"Training Data Shape after preprocessing: {df.shape}")
        print(f"Sample target distribution: {df['target'].value_counts(normalize=True)}")

        # Prepare features and target
        X = df[ML_FEATURES]
        y = df['target']

        # Split data: 70% train, 15% validation, 15% test (no shuffle for time series)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, shuffle=False)  # 0.1765 of 0.85 = 0.15 of total
        print(f"Train Set Shape: {X_train.shape}, Validation Set Shape: {X_val.shape}, Test Set Shape: {X_test.shape}")

        # Convert to DMatrix for xgboost.train
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Calculate scale_pos_weight to handle class imbalance
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        print(f"Scale Positive Weight: {scale_pos_weight:.2f}")

        # Define XGBoost parameters (tuned for 55.40% run)
        params = {
            'max_depth': ML_MAX_DEPTH,  # Tree depth (6) for complexity
            'learning_rate': 0.002,  # Small learning rate for постепенное обучение
            'objective': 'binary:logistic',  # Binary classification objective
            'eval_metric': 'logloss',  # Evaluation metric
            'scale_pos_weight': scale_pos_weight,  # Base value for balanced precision/recall
            'random_state': 42,  # Seed for reproducibility
            'min_child_weight': 15,  # Regularization to prevent overfitting
            'subsample': 0.8,  # 80% of data per tree to reduce overfitting
            'colsample_bytree': 0.7  # 70% of features per tree for diversity
        }

        # Train with early stopping to optimize performance
        evals = [(dtrain, 'train'), (dval, 'validation')]
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,  # Max rounds, stopped early if needed
            evals=evals,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,  # Stop after 50 rounds of no improvement
            verbose_eval=True  # Print training progress
        )

        # Evaluate on test set with threshold 0.55
        y_pred_prob = self.model.predict(dtest)
        threshold = 0.55  # Threshold for binary prediction
        y_pred_binary = [1 if p > threshold else 0 for p in y_pred_prob]
        accuracy = np.mean(np.array(y_pred_binary) == np.array(y_test))
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        print(f"Training completed. Test Accuracy: {accuracy:.2f}")
        print(f"Precision (positive class): {precision:.2f}")
        print(f"Recall (positive class): {recall:.2f}")
        print(f"F1-Score (positive class): {f1:.2f}")
        print(f"ROC-AUC Score: {roc_auc:.2f}")

        # Display feature importance for analysis
        importance = self.model.get_score(importance_type='weight')
        feature_importance = {k: float(v) for k, v in importance.items()}
        print("Feature Importance:")
        for feature, score in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {score:.4f}")

        # Save the trained model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)  # Ensure directory exists
        self.model.save_model(self.model_path)
        print(f"Model saved to {self.model_path}")
        return accuracy

    def predict(self, X):
        """Predict probabilities using the trained model."""
        if self.model is None:
            print("Warning: No trained model available for prediction. Returning zeros.")
            if not isinstance(X, xgb.DMatrix):
                raise ValueError("Input must be an xgboost.DMatrix object")
            return np.zeros(X.num_row())  # Return zeros array matching input size
        if not isinstance(X, xgb.DMatrix):
            raise ValueError("Input must be an xgboost.DMatrix object")
        try:
            return self.model.predict(X)
        except Exception as e:
            print(f"Prediction error: {e}. Returning zeros.")
            return np.zeros(X.num_row())  # Fallback to zeros