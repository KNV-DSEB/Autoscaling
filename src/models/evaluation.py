"""
Model Evaluation Module
=======================
Module đánh giá và so sánh các forecasting models.

Metrics:
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error

Usage:
    >>> from src.models.evaluation import calculate_metrics, compare_models
    >>> metrics = calculate_metrics(y_true, y_pred)
    >>> print(metrics)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ''
) -> Dict[str, float]:
    """
    Tính toán tất cả evaluation metrics.

    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        prefix: Tiền tố cho tên metrics (vd: 'train_', 'test_')

    Returns:
        Dict với các metrics:
            - MSE: Mean Squared Error
            - RMSE: Root Mean Squared Error
            - MAE: Mean Absolute Error
            - MAPE: Mean Absolute Percentage Error (%)

    Note:
        MAPE xử lý trường hợp y_true = 0 bằng cách loại bỏ các điểm đó
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE - xử lý division by zero
    # Loại bỏ các điểm có y_true = 0
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {
        f'{prefix}MSE': mse,
        f'{prefix}RMSE': rmse,
        f'{prefix}MAE': mae,
        f'{prefix}MAPE': mape
    }


def calculate_metrics_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model'
) -> pd.DataFrame:
    """
    Tính metrics và trả về dưới dạng DataFrame.

    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        model_name: Tên model

    Returns:
        DataFrame với 1 row chứa các metrics
    """
    metrics = calculate_metrics(y_true, y_pred)
    metrics['Model'] = model_name
    return pd.DataFrame([metrics])


def compare_models(
    results: Dict[str, Dict[str, float]],
    primary_metric: str = 'RMSE',
    ascending: bool = True
) -> pd.DataFrame:
    """
    So sánh nhiều models dựa trên metrics.

    Args:
        results: Dict của {model_name: {metric: value}}
        primary_metric: Metric để sort (mặc định: RMSE)
        ascending: Sort ascending (True = lower is better)

    Returns:
        DataFrame với so sánh các models, sorted by primary_metric

    Example:
        >>> results = {
        ...     'SARIMA': {'RMSE': 10.5, 'MAE': 8.2},
        ...     'LightGBM': {'RMSE': 9.3, 'MAE': 7.1}
        ... }
        >>> compare_models(results)
    """
    df = pd.DataFrame(results).T
    df.index.name = 'Model'
    df = df.reset_index()

    # Sort by primary metric
    if primary_metric in df.columns:
        df = df.sort_values(primary_metric, ascending=ascending)

    # Add rank
    df['Rank'] = range(1, len(df) + 1)

    return df


def cross_validation_scores(
    cv_results: List[Dict[str, float]],
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Tổng hợp kết quả cross-validation.

    Args:
        cv_results: List các dict metrics từ mỗi fold
        metrics: Danh sách metrics cần tổng hợp

    Returns:
        DataFrame với mean và std của mỗi metric
    """
    metrics = metrics or ['RMSE', 'MAE', 'MAPE']

    summary = {}
    for metric in metrics:
        values = [r.get(metric, np.nan) for r in cv_results]
        summary[f'{metric}_mean'] = np.nanmean(values)
        summary[f'{metric}_std'] = np.nanstd(values)
        summary[f'{metric}_min'] = np.nanmin(values)
        summary[f'{metric}_max'] = np.nanmax(values)

    return pd.DataFrame([summary])


def calculate_forecast_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_pct: float = 10.0
) -> Dict[str, float]:
    """
    Tính các metrics về forecast accuracy.

    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        threshold_pct: Ngưỡng % để coi là "accurate"

    Returns:
        Dict với các accuracy metrics:
            - accuracy_within_threshold: % dự đoán trong ngưỡng
            - mean_error: Mean error (có thể âm = under-predict)
            - max_overpredict: Max over-prediction
            - max_underpredict: Max under-prediction
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    errors = y_pred - y_true
    pct_errors = np.abs(errors / np.where(y_true != 0, y_true, 1)) * 100

    within_threshold = (pct_errors <= threshold_pct).mean() * 100

    return {
        'accuracy_within_threshold': within_threshold,
        'mean_error': errors.mean(),
        'mean_pct_error': pct_errors.mean(),
        'max_overpredict': errors.max(),
        'max_underpredict': (-errors).max()
    }


def print_metrics_table(
    results: Dict[str, Dict[str, float]],
    title: str = "Model Comparison"
):
    """
    In bảng so sánh metrics đẹp.

    Args:
        results: Dict của {model_name: {metric: value}}
        title: Tiêu đề bảng
    """
    df = compare_models(results)

    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    # Best model
    best_model = df.iloc[0]['Model']
    best_rmse = df.iloc[0].get('RMSE', 'N/A')
    print(f"\nBest Model: {best_model} (RMSE: {best_rmse:.4f})")


def create_prediction_summary(
    y_true: pd.Series,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex = None
) -> pd.DataFrame:
    """
    Tạo DataFrame tổng hợp predictions.

    Args:
        y_true: Series giá trị thực tế
        y_pred: Array giá trị dự đoán
        timestamps: Index thời gian (nếu có)

    Returns:
        DataFrame với actual, predicted, error, pct_error
    """
    if timestamps is None:
        timestamps = y_true.index if hasattr(y_true, 'index') else range(len(y_true))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'actual': y_true.values if hasattr(y_true, 'values') else y_true,
        'predicted': y_pred,
        'error': y_pred - (y_true.values if hasattr(y_true, 'values') else y_true),
    })

    df['abs_error'] = df['error'].abs()
    df['pct_error'] = np.where(
        df['actual'] != 0,
        df['abs_error'] / df['actual'] * 100,
        np.nan
    )

    return df


if __name__ == "__main__":
    # Demo usage
    np.random.seed(42)

    # Generate sample data
    y_true = np.random.randint(50, 150, 100)
    y_pred_model1 = y_true + np.random.normal(0, 10, 100)
    y_pred_model2 = y_true + np.random.normal(0, 15, 100)
    y_pred_model3 = y_true + np.random.normal(5, 12, 100)

    # Calculate metrics
    results = {
        'SARIMA': calculate_metrics(y_true, y_pred_model1),
        'LightGBM': calculate_metrics(y_true, y_pred_model2),
        'Prophet': calculate_metrics(y_true, y_pred_model3)
    }

    # Print comparison
    print_metrics_table(results, "Demo Model Comparison")

    # Forecast accuracy
    print("\nForecast Accuracy (SARIMA):")
    accuracy = calculate_forecast_accuracy(y_true, y_pred_model1)
    for k, v in accuracy.items():
        print(f"  {k}: {v:.2f}")
