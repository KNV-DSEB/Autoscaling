"""
Test Models Module
==================
Unit tests cho forecasting models và evaluation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.evaluation import (
    calculate_metrics,
    compare_models,
    calculate_forecast_accuracy
)


class TestMetricsCalculation:
    """Test cases cho metrics calculation."""

    def test_perfect_prediction(self):
        """Test metrics với perfect prediction."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([100, 200, 300, 400, 500])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics['MSE'] == 0
        assert metrics['RMSE'] == 0
        assert metrics['MAE'] == 0
        assert metrics['MAPE'] == 0

    def test_constant_error(self):
        """Test metrics với constant error."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 210, 310, 410, 510])  # Constant error of 10

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics['MSE'] == 100  # 10^2
        assert metrics['RMSE'] == 10
        assert metrics['MAE'] == 10

    def test_mape_with_zeros(self):
        """Test MAPE handling của zeros."""
        y_true = np.array([0, 100, 200, 0, 300])
        y_pred = np.array([10, 110, 210, 10, 310])

        metrics = calculate_metrics(y_true, y_pred)

        # MAPE should only use non-zero values
        assert not np.isnan(metrics['MAPE'])

    def test_mape_all_zeros(self):
        """Test MAPE khi tất cả y_true = 0."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([10, 20, 30, 40, 50])

        metrics = calculate_metrics(y_true, y_pred)

        # MAPE should be NaN when all true values are 0
        assert np.isnan(metrics['MAPE'])

    def test_prefix(self):
        """Test prefix cho metric names."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 210, 310])

        metrics = calculate_metrics(y_true, y_pred, prefix='test_')

        assert 'test_MSE' in metrics
        assert 'test_RMSE' in metrics
        assert 'test_MAE' in metrics
        assert 'test_MAPE' in metrics


class TestModelComparison:
    """Test cases cho model comparison."""

    def test_compare_two_models(self):
        """Test comparison của 2 models."""
        results = {
            'Model A': {'RMSE': 10.0, 'MAE': 8.0, 'MAPE': 5.0},
            'Model B': {'RMSE': 15.0, 'MAE': 12.0, 'MAPE': 8.0}
        }

        comparison = compare_models(results)

        assert len(comparison) == 2
        assert comparison.iloc[0]['Model'] == 'Model A'  # Lower RMSE
        assert 'Rank' in comparison.columns

    def test_compare_multiple_models(self):
        """Test comparison của nhiều models."""
        results = {
            'SARIMA': {'RMSE': 50.0, 'MAE': 40.0},
            'LightGBM': {'RMSE': 45.0, 'MAE': 35.0},
            'Prophet': {'RMSE': 55.0, 'MAE': 45.0}
        }

        comparison = compare_models(results, primary_metric='RMSE')

        assert len(comparison) == 3
        assert comparison.iloc[0]['Model'] == 'LightGBM'  # Best RMSE
        assert comparison.iloc[0]['Rank'] == 1


class TestForecastAccuracy:
    """Test cases cho forecast accuracy."""

    def test_accuracy_within_threshold(self):
        """Test accuracy calculation."""
        y_true = np.array([100, 100, 100, 100, 100])
        y_pred = np.array([105, 95, 110, 90, 100])  # Within 10%

        accuracy = calculate_forecast_accuracy(y_true, y_pred, threshold_pct=10)

        assert accuracy['accuracy_within_threshold'] == 100.0  # All within 10%

    def test_accuracy_mixed(self):
        """Test accuracy với mixed results."""
        y_true = np.array([100, 100, 100, 100])
        y_pred = np.array([105, 120, 95, 80])  # 2 within 10%, 2 outside

        accuracy = calculate_forecast_accuracy(y_true, y_pred, threshold_pct=10)

        assert accuracy['accuracy_within_threshold'] == 50.0  # 50% within threshold

    def test_mean_error(self):
        """Test mean error calculation."""
        y_true = np.array([100, 100, 100])
        y_pred = np.array([110, 120, 130])  # Over-prediction

        accuracy = calculate_forecast_accuracy(y_true, y_pred)

        assert accuracy['mean_error'] > 0  # Positive = over-prediction

    def test_max_predictions(self):
        """Test max over/under prediction."""
        y_true = np.array([100, 100, 100])
        y_pred = np.array([150, 80, 100])  # +50, -20, 0

        accuracy = calculate_forecast_accuracy(y_true, y_pred)

        assert accuracy['max_overpredict'] == 50
        assert accuracy['max_underpredict'] == 20


class TestEdgeCases:
    """Test edge cases."""

    def test_single_value(self):
        """Test với single value."""
        y_true = np.array([100])
        y_pred = np.array([110])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics['MAE'] == 10

    def test_large_arrays(self):
        """Test với large arrays."""
        np.random.seed(42)
        y_true = np.random.randint(50, 200, 10000)
        y_pred = y_true + np.random.normal(0, 10, 10000)

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics['RMSE'] < 15  # Should be close to std of noise
        assert metrics['MAE'] < 10

    def test_negative_predictions(self):
        """Test handling của negative predictions."""
        y_true = np.array([100, 50, 30])
        y_pred = np.array([90, 40, -10])  # One negative

        metrics = calculate_metrics(y_true, y_pred)

        # Should still calculate metrics
        assert not np.isnan(metrics['RMSE'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
