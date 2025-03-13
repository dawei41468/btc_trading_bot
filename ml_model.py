import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from config import ML_FEATURES, ML_TEST_SIZE, ML_MAX_DEPTH, EARLY_STOPPING_ROUNDS
import os

class MLModel:
    def __init__(self, model_path="xgboost_model.json"):
        """Initialize the ML model, loading from file if it exists."""
        self.model = None
        self.model_path = model_path
        if os.path.exists(model_path):
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            print(f"Loaded existing model from {model_path}")

    def train(self, df):
        """Train the XGBoost model to predict >1% price increases."""
        print("Training ML Model...")
        df['target'] = (df['close'].shift(-1) / df['close'] - 1 > 0.01).astype(int)
        df = df.dropna()
        print(f"Training Data Shape after preprocessing: {df.shape}")
        print(f"Sample target distribution: {df['target'].value_counts(normalize=True)}")

        X = df[ML_FEATURES]
        y = df['target']

        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, shuffle=False)
        print(f"Train Set Shape: {X_train.shape}, Validation Set Shape: {X_val.shape}, Test Set Shape: {X_test.shape}")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        print(f"Scale Positive Weight: {scale_pos_weight:.2f}")

        params = {
            'max_depth': ML_MAX_DEPTH,
            'learning_rate': 0.002,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'min_child_weight': 15,
            'subsample': 0.8,
            'colsample_bytree': 0.7
        }

        evals = [(dtrain, 'train'), (dval, 'validation')]
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=evals,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=True
        )

        y_pred_prob = self.model.predict(dtest)
        threshold = 0.55
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

        importance = self.model.get_score(importance_type='weight')
        feature_importance = {k: float(v) for k, v in importance.items()}
        print("Feature Importance:")
        for feature, score in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {score:.4f}")

        self.model.save_model(self.model_path)
        print(f"Model saved to {self.model_path}")
        return accuracy

    def predict(self, X):
        """Predict probabilities using the trained model."""
        if self.model:
            if not isinstance(X, xgb.DMatrix):
                raise ValueError("Input must be an xgboost.DMatrix object")
            return self.model.predict(X)
        print("Warning: No trained model available for prediction.")
        return None