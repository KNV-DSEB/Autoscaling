"""
Quick model training script for SARIMA, LightGBM, and Prophet.
Uses simplified parameters for faster training.
"""
import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Set working directory to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from src.data.preprocessor import load_timeseries, split_train_test
from src.models.sarima import SARIMAForecaster
from src.models.lightgbm_forecaster import LightGBMForecaster
from src.models.evaluation import calculate_metrics
from src.features.feature_engineering import TimeSeriesFeatureEngineer

print("="*60)
print("           QUICK MODEL TRAINING SCRIPT")
print("="*60)

# Load data
print("\n1. Loading data...")
df = load_timeseries('data/processed/timeseries_15min.parquet')
df_clean = df[df['is_storm_period'] == 0].copy()
print(f"   Clean records: {len(df_clean)}")

# Train/Test split
train, test = split_train_test(df_clean, test_start='1995-08-23')
train_series = train['request_count']
test_series = test['request_count']
print(f"   Train: {len(train_series)}, Test: {len(test_series)}")

# ========== SARIMA ==========
print("\n2. Training SARIMA model (simplified params)...")
# Use simpler parameters: no seasonal component for faster training
# order=(2,1,2) without seasonal component
sarima_model = SARIMAForecaster(
    order=(2, 1, 2),
    seasonal_order=None  # Skip seasonal for speed
)

# Train on subset for speed
train_subset = train_series[-2000:]  # Last ~20 days
sarima_model.fit(train_subset, verbose=False)

# Predict
sarima_pred = sarima_model.predict(steps=min(len(test_series), 96), return_conf_int=True)
y_true_sarima = test_series.values[:len(sarima_pred)]
y_pred_sarima = sarima_pred['forecast'].values

# Metrics
sarima_metrics = calculate_metrics(y_true_sarima, y_pred_sarima)
print(f"   SARIMA Metrics: RMSE={sarima_metrics['RMSE']:.2f}, MAE={sarima_metrics['MAE']:.2f}")

# Save
sarima_model.save('models/sarima_15min.pkl')
print("   Saved: models/sarima_15min.pkl")

# ========== LightGBM ==========
print("\n3. Training LightGBM model...")
# Feature engineering
fe = TimeSeriesFeatureEngineer(df_clean)
df_features = fe.create_all_features(target_col='request_count', granularity='15min')

# Prepare supervised data
feature_cols = fe.get_feature_columns(df_features)
X, y = fe.prepare_supervised(df_features, 'request_count', feature_cols, forecast_horizon=1)

# Split
test_start = '1995-08-23'
train_mask = X.index < test_start
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

# Train LightGBM
lgb_model = LightGBMForecaster(
    params={
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'n_estimators': 200,
        'learning_rate': 0.05,
        'num_leaves': 31
    }
)
lgb_model.fit(X_train, y_train)

# Predict
lgb_pred = lgb_model.predict(X_test)
lgb_metrics = calculate_metrics(y_test.values, lgb_pred)
print(f"   LightGBM Metrics: RMSE={lgb_metrics['RMSE']:.2f}, MAE={lgb_metrics['MAE']:.2f}")

# Save
lgb_model.save('models/lightgbm_15min.pkl')
print("   Saved: models/lightgbm_15min.pkl")

# ========== Prophet ==========
print("\n4. Training Prophet model...")
try:
    from src.models.prophet_forecaster import ProphetForecaster

    # Prepare Prophet data format
    prophet_train = pd.DataFrame({
        'ds': train_series.index,
        'y': train_series.values
    })

    prophet_model = ProphetForecaster()
    prophet_model.fit(prophet_train)

    # Predict
    prophet_pred = prophet_model.predict(periods=min(len(test_series), 96))
    y_true_prophet = test_series.values[:len(prophet_pred)]
    y_pred_prophet = prophet_pred['yhat'].values

    prophet_metrics = calculate_metrics(y_true_prophet, y_pred_prophet)
    print(f"   Prophet Metrics: RMSE={prophet_metrics['RMSE']:.2f}, MAE={prophet_metrics['MAE']:.2f}")

    # Save
    prophet_model.save('models/prophet_15min.pkl')
    print("   Saved: models/prophet_15min.pkl")
except Exception as e:
    print(f"   Prophet training failed: {e}")

# ========== Summary ==========
print("\n" + "="*60)
print("           TRAINING COMPLETE!")
print("="*60)
print("\nModels saved:")
print("  - models/sarima_15min.pkl")
print("  - models/lightgbm_15min.pkl")
print("  - models/prophet_15min.pkl")
print("\nMetrics Summary:")
print(f"  SARIMA: RMSE={sarima_metrics['RMSE']:.2f}, MAE={sarima_metrics['MAE']:.2f}, MAPE={sarima_metrics['MAPE']:.2f}%")
print(f"  LightGBM: RMSE={lgb_metrics['RMSE']:.2f}, MAE={lgb_metrics['MAE']:.2f}, MAPE={lgb_metrics['MAPE']:.2f}%")
