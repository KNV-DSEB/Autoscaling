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
test_start = '1995-08-23'
print(f"   Train: {len(train_series)}, Test: {len(test_series)}")

# ========== SARIMA ==========
print("\n2. Training SARIMA model (simplified params)...")
sarima_model = SARIMAForecaster(
    order=(2, 1, 2),
    seasonal_order=None  # Skip seasonal for speed
)

train_subset = train_series[-2000:]
sarima_model.fit(train_subset, verbose=False)

sarima_pred = sarima_model.predict(steps=min(len(test_series), 96), return_conf_int=True)
y_true_sarima = test_series.values[:len(sarima_pred)]
y_pred_sarima = sarima_pred['forecast'].values

sarima_metrics = calculate_metrics(y_true_sarima, y_pred_sarima)
print(f"   SARIMA Metrics: RMSE={sarima_metrics['RMSE']:.2f}, MAE={sarima_metrics['MAE']:.2f}")

sarima_model.save('models/sarima_15min.pkl')
print("   Saved: models/sarima_15min.pkl")

# ========== LightGBM ==========
print("\n3. Training LightGBM model...")
fe = TimeSeriesFeatureEngineer(df_clean)
df_features = fe.create_all_features(target_col='request_count', granularity='15min')

feature_cols = fe.get_feature_columns(df_features)
X, y = fe.prepare_supervised(df_features, 'request_count', feature_cols, forecast_horizon=1)

train_mask = X.index < test_start
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

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

lgb_pred = lgb_model.predict(X_test)
lgb_metrics = calculate_metrics(y_test.values, lgb_pred)
print(f"   LightGBM Metrics: RMSE={lgb_metrics['RMSE']:.2f}, MAE={lgb_metrics['MAE']:.2f}")

lgb_model.save('models/lightgbm_15min.pkl')
print("   Saved: models/lightgbm_15min.pkl")

# ========== Prophet ==========
print("\n4. Training Prophet model...")
prophet_metrics = None
try:
    from src.models.prophet_forecaster import ProphetForecaster

    prophet_train = pd.DataFrame({
        'ds': train_series.index,
        'y': train_series.values
    })

    prophet_model = ProphetForecaster()
    prophet_model.fit(prophet_train)

    prophet_pred = prophet_model.predict(periods=min(len(test_series), 96))
    y_true_prophet = test_series.values[:len(prophet_pred)]
    y_pred_prophet = prophet_pred['yhat'].values

    prophet_metrics = calculate_metrics(y_true_prophet, y_pred_prophet)
    print(f"   Prophet Metrics: RMSE={prophet_metrics['RMSE']:.2f}, MAE={prophet_metrics['MAE']:.2f}")

    prophet_model.save('models/prophet_15min.pkl')
    print("   Saved: models/prophet_15min.pkl")
except Exception as e:
    print(f"   Prophet training failed: {e}")

# ========== BYTES_TOTAL MODELS ==========
print("\n" + "="*60)
print("           TRAINING BYTES_TOTAL MODELS")
print("="*60)

train_bytes = train['bytes_total']
test_bytes = test['bytes_total']
print(f"\nbytes_total Train: {len(train_bytes)}, Test: {len(test_bytes)}")

# ========== SARIMA for bytes_total ==========
print("\n5. Training SARIMA on bytes_total...")
sarima_bytes = SARIMAForecaster(
    order=(2, 1, 2),
    seasonal_order=None  # Skip seasonal for speed
)

train_bytes_subset = train_bytes[-2000:]
sarima_bytes.fit(train_bytes_subset, verbose=False)

sarima_bytes_pred = sarima_bytes.predict(steps=min(len(test_bytes), 96), return_conf_int=True)
y_true_sarima_bytes = test_bytes.values[:len(sarima_bytes_pred)]
y_pred_sarima_bytes = sarima_bytes_pred['forecast'].values

sarima_bytes_metrics = calculate_metrics(y_true_sarima_bytes, y_pred_sarima_bytes)
print(f"   SARIMA bytes_total: RMSE={sarima_bytes_metrics['RMSE']:.2f}, MAE={sarima_bytes_metrics['MAE']:.2f}")

sarima_bytes.save('models/sarima_bytes_15min.pkl')
print("   Saved: models/sarima_bytes_15min.pkl")

# ========== LightGBM for bytes_total ==========
print("\n6. Training LightGBM on bytes_total...")
fe_bytes = TimeSeriesFeatureEngineer(df_clean)
df_features_bytes = fe_bytes.create_all_features(target_col='bytes_total', granularity='15min')

feature_cols_bytes = fe_bytes.get_feature_columns(df_features_bytes)
X_bytes, y_bytes = fe_bytes.prepare_supervised(df_features_bytes, 'bytes_total', feature_cols_bytes, forecast_horizon=1)

train_mask_bytes = X_bytes.index < test_start
X_train_bytes, X_test_bytes = X_bytes[train_mask_bytes], X_bytes[~train_mask_bytes]
y_train_bytes, y_test_bytes = y_bytes[train_mask_bytes], y_bytes[~train_mask_bytes]

lgb_bytes = LightGBMForecaster(
    params={
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'n_estimators': 200,
        'learning_rate': 0.05,
        'num_leaves': 31
    }
)
lgb_bytes.fit(X_train_bytes, y_train_bytes)

lgb_bytes_pred = lgb_bytes.predict(X_test_bytes)
lgb_bytes_metrics = calculate_metrics(y_test_bytes.values, lgb_bytes_pred)
print(f"   LightGBM bytes_total: RMSE={lgb_bytes_metrics['RMSE']:.2f}, MAE={lgb_bytes_metrics['MAE']:.2f}")

lgb_bytes.save('models/lightgbm_bytes_15min.pkl')
print("   Saved: models/lightgbm_bytes_15min.pkl")

# ========== Prophet for bytes_total ==========
print("\n7. Training Prophet on bytes_total...")
prophet_bytes_metrics = None
try:
    from src.models.prophet_forecaster import ProphetForecaster

    prophet_train_bytes = pd.DataFrame({
        'ds': train_bytes.index,
        'y': train_bytes.values
    })

    prophet_bytes = ProphetForecaster()
    prophet_bytes.fit(prophet_train_bytes)

    prophet_bytes_pred = prophet_bytes.predict(periods=min(len(test_bytes), 96))
    y_true_prophet_bytes = test_bytes.values[:len(prophet_bytes_pred)]
    y_pred_prophet_bytes = prophet_bytes_pred['yhat'].values

    prophet_bytes_metrics = calculate_metrics(y_true_prophet_bytes, y_pred_prophet_bytes)
    print(f"   Prophet bytes_total: RMSE={prophet_bytes_metrics['RMSE']:.2f}, MAE={prophet_bytes_metrics['MAE']:.2f}")

    prophet_bytes.save('models/prophet_bytes_15min.pkl')
    print("   Saved: models/prophet_bytes_15min.pkl")
except Exception as e:
    print(f"   Prophet bytes_total training failed: {e}")

# ========== Summary ==========
print("\n" + "="*60)
print("           TRAINING COMPLETE!")
print("="*60)
print("\nrequest_count Models saved:")
print("  - models/sarima_15min.pkl")
print("  - models/lightgbm_15min.pkl")
if prophet_metrics:
    print("  - models/prophet_15min.pkl")
print("\nbytes_total Models saved:")
print("  - models/sarima_bytes_15min.pkl")
print("  - models/lightgbm_bytes_15min.pkl")
if prophet_bytes_metrics:
    print("  - models/prophet_bytes_15min.pkl")
print("\nMetrics Summary (request_count):")
print(f"  SARIMA: RMSE={sarima_metrics['RMSE']:.2f}, MAE={sarima_metrics['MAE']:.2f}, MAPE={sarima_metrics['MAPE']:.2f}%")
print(f"  LightGBM: RMSE={lgb_metrics['RMSE']:.2f}, MAE={lgb_metrics['MAE']:.2f}, MAPE={lgb_metrics['MAPE']:.2f}%")
if prophet_metrics:
    print(f"  Prophet: RMSE={prophet_metrics['RMSE']:.2f}, MAE={prophet_metrics['MAE']:.2f}, MAPE={prophet_metrics['MAPE']:.2f}%")
print("\nMetrics Summary (bytes_total):")
print(f"  SARIMA: RMSE={sarima_bytes_metrics['RMSE']:.2f}, MAE={sarima_bytes_metrics['MAE']:.2f}, MAPE={sarima_bytes_metrics['MAPE']:.2f}%")
print(f"  LightGBM: RMSE={lgb_bytes_metrics['RMSE']:.2f}, MAE={lgb_bytes_metrics['MAE']:.2f}, MAPE={lgb_bytes_metrics['MAPE']:.2f}%")
if prophet_bytes_metrics:
    print(f"  Prophet: RMSE={prophet_bytes_metrics['RMSE']:.2f}, MAE={prophet_bytes_metrics['MAE']:.2f}, MAPE={prophet_bytes_metrics['MAPE']:.2f}%")
print("="*60)
