"""
LightGBM Forecaster
===================
LightGBM-based time series forecaster với feature engineering.

LightGBM là gradient boosting framework hiệu quả, phù hợp cho:
    - Datasets lớn
    - Nhiều features
    - Non-linear relationships

Approach:
    - Sử dụng lag features và rolling statistics
    - Time Series Cross-Validation
    - Feature importance analysis

Usage:
    >>> from src.models.lightgbm_forecaster import LightGBMForecaster
    >>> model = LightGBMForecaster()
    >>> model.fit(X_train, y_train, X_val, y_val)
    >>> predictions = model.predict(X_test)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Dict, Optional, Tuple, Any
import pickle
import warnings


class LightGBMForecaster:
    """
    LightGBM-based time series forecaster.

    Features:
        - Time series cross-validation
        - Early stopping
        - Feature importance tracking
        - Recursive multi-step forecasting
        - Save/load functionality

    Attributes:
        params: LightGBM hyperparameters
        n_estimators: Number of boosting rounds
        early_stopping_rounds: Early stopping patience
        model: Trained LightGBM model
        feature_names: List of feature column names
        feature_importance: DataFrame với feature importances
    """

    DEFAULT_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
        'force_col_wise': True  # Avoid warning
    }

    def __init__(
        self,
        params: Optional[Dict] = None,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,
        forecast_horizon: int = 1
    ):
        """
        Khởi tạo LightGBM forecaster.

        Args:
            params: Custom LightGBM parameters (merged with defaults)
            n_estimators: Max number of boosting rounds
            early_stopping_rounds: Stop if no improvement after N rounds
            forecast_horizon: Steps ahead to forecast (for supervised learning setup)
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.forecast_horizon = forecast_horizon

        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.best_iteration = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
        verbose: int = 100
    ) -> 'LightGBMForecaster':
        """
        Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (cho early stopping)
            y_val: Validation target
            categorical_features: List các cột categorical
            verbose: Log evaluation every N rounds (0 = silent)

        Returns:
            self
        """
        self.feature_names = list(X_train.columns)

        # Create datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=categorical_features
        )

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
                categorical_feature=categorical_features
            )
            valid_sets.append(val_data)
            valid_names.append('valid')

        # Callbacks
        callbacks = []
        if verbose > 0:
            callbacks.append(lgb.log_evaluation(verbose))
        if self.early_stopping_rounds and X_val is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))

        # Train
        print(f"Training LightGBM with {len(self.feature_names)} features...")
        print(f"  Train samples: {len(X_train):,}")
        if X_val is not None:
            print(f"  Valid samples: {len(X_val):,}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )

        self.best_iteration = self.model.best_iteration

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        print(f"\nTraining complete!")
        print(f"  Best iteration: {self.best_iteration}")
        print(f"  Best score: {self.model.best_score}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Features DataFrame

        Returns:
            Array predictions
        """
        if self.model is None:
            raise ValueError("Model chưa được train. Gọi fit() trước.")

        predictions = self.model.predict(
            X,
            num_iteration=self.best_iteration
        )

        # Ensure non-negative
        predictions = np.maximum(predictions, 0)

        return predictions

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        test_size: int = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Time series cross-validation.

        Sử dụng expanding window approach để đảm bảo không có data leakage.

        Args:
            X: Features
            y: Target
            n_splits: Số folds
            test_size: Số samples trong mỗi test fold
            verbose: Print progress

        Returns:
            Dict với:
                - fold_scores: List scores từ mỗi fold
                - rmse_mean, rmse_std: Mean và std của RMSE
                - mae_mean, mae_std: Mean và std của MAE
        """
        if test_size is None:
            test_size = len(X) // (n_splits + 1)

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        scores = {'rmse': [], 'mae': [], 'mape': []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            if verbose:
                print(f"\nFold {fold + 1}/{n_splits}")

            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            # Train
            temp_model = LightGBMForecaster(
                params=self.params,
                n_estimators=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds
            )
            temp_model.fit(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                verbose=0
            )

            # Evaluate
            preds = temp_model.predict(X_val_fold)

            rmse = np.sqrt(np.mean((y_val_fold - preds) ** 2))
            mae = np.mean(np.abs(y_val_fold - preds))

            # MAPE (handle zeros)
            mask = y_val_fold != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_val_fold[mask] - preds[mask]) / y_val_fold[mask])) * 100
            else:
                mape = np.nan

            scores['rmse'].append(rmse)
            scores['mae'].append(mae)
            scores['mape'].append(mape)

            if verbose:
                print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

        return {
            'fold_scores': scores,
            'rmse_mean': np.mean(scores['rmse']),
            'rmse_std': np.std(scores['rmse']),
            'mae_mean': np.mean(scores['mae']),
            'mae_std': np.std(scores['mae']),
            'mape_mean': np.nanmean(scores['mape']),
            'mape_std': np.nanstd(scores['mape'])
        }

    def recursive_forecast(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        steps: int,
        update_func: callable = None
    ) -> np.ndarray:
        """
        Multi-step recursive forecasting.

        Mỗi prediction được sử dụng để update features cho step tiếp theo.

        Args:
            df: DataFrame với tất cả features
            target_col: Tên cột target
            feature_cols: List feature columns
            steps: Số steps cần forecast
            update_func: Function để update features sau mỗi step
                        Signature: update_func(df, new_pred, target_col) -> df

        Returns:
            Array với predictions cho N steps
        """
        forecasts = []
        current_df = df.copy()

        for step in range(steps):
            # Lấy features từ row cuối
            X_step = current_df[feature_cols].iloc[[-1]]

            # Predict
            pred = self.predict(X_step)[0]
            forecasts.append(pred)

            # Update dataframe nếu có update function
            if update_func and step < steps - 1:
                current_df = update_func(current_df, pred, target_col)

        return np.array(forecasts)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Lấy top N important features.

        Args:
            top_n: Số features cần lấy

        Returns:
            DataFrame với feature và importance
        """
        if self.feature_importance is None:
            raise ValueError("Model chưa được train.")

        return self.feature_importance.head(top_n)

    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple = (10, 8)):
        """
        Plot feature importance.

        Args:
            top_n: Số features hiển thị
            figsize: Figure size
        """
        import matplotlib.pyplot as plt

        fi = self.get_feature_importance(top_n)

        plt.figure(figsize=figsize)
        plt.barh(range(len(fi)), fi['importance'], align='center')
        plt.yticks(range(len(fi)), fi['feature'])
        plt.xlabel('Importance (Gain)')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def save(self, filepath: str):
        """
        Lưu model ra file.

        Args:
            filepath: Đường dẫn file
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'params': self.params,
                'n_estimators': self.n_estimators,
                'early_stopping_rounds': self.early_stopping_rounds,
                'forecast_horizon': self.forecast_horizon,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'best_iteration': self.best_iteration
            }, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'LightGBMForecaster':
        """
        Load model từ file.

        Args:
            filepath: Đường dẫn file

        Returns:
            LightGBMForecaster instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        forecaster = cls(
            params=data['params'],
            n_estimators=data['n_estimators'],
            early_stopping_rounds=data['early_stopping_rounds'],
            forecast_horizon=data['forecast_horizon']
        )
        forecaster.model = data['model']
        forecaster.feature_names = data['feature_names']
        forecaster.feature_importance = data['feature_importance']
        forecaster.best_iteration = data['best_iteration']

        return forecaster


if __name__ == "__main__":
    # Demo usage
    import sys
    import os
    sys.path.insert(0, os.path.abspath('..'))

    from src.data.preprocessor import load_timeseries, split_train_test
    from src.features.feature_engineering import TimeSeriesFeatureEngineer
    from src.models.evaluation import calculate_metrics

    # Load and prepare data
    print("Loading data...")
    df = load_timeseries('../data/processed/timeseries_15min.parquet')

    # Remove storm period
    df_clean = df[df['is_storm_period'] == 0]

    # Feature engineering
    print("\nCreating features...")
    fe = TimeSeriesFeatureEngineer(df_clean)
    df_features = fe.create_all_features(
        target_col='request_count',
        granularity='15min'
    )

    # Get feature columns
    feature_cols = fe.get_feature_columns(df_features)
    print(f"Number of features: {len(feature_cols)}")

    # Prepare supervised data
    X, y = fe.prepare_supervised(
        df_features,
        target_col='request_count',
        feature_cols=feature_cols,
        forecast_horizon=1
    )

    # Split train/test
    test_start = '1995-08-23'
    train_mask = X.index < test_start
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    print(f"\nTrain: {len(X_train):,} samples")
    print(f"Test: {len(X_test):,} samples")

    # Create validation set from end of train
    val_size = len(X_train) // 5
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_train = X_train.iloc[:-val_size]
    y_train = y_train.iloc[:-val_size]

    # Train model
    print("\n" + "="*50)
    model = LightGBMForecaster(
        n_estimators=1000,
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, X_val, y_val, verbose=100)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    metrics = calculate_metrics(y_test.values, predictions)
    print(f"\nTest Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Feature importance
    print("\nTop 10 Features:")
    print(model.get_feature_importance(10))

    # Save model
    model.save('../models/lightgbm_15min_demo.pkl')
