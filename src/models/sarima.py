"""
SARIMA Forecaster
=================
SARIMA (Seasonal ARIMA) model cho time series forecasting.

SARIMA(p,d,q)(P,D,Q,m):
    - p: AR order (autoregressive)
    - d: Differencing order
    - q: MA order (moving average)
    - P: Seasonal AR order
    - D: Seasonal differencing order
    - Q: Seasonal MA order
    - m: Seasonal period (60 cho hourly trong 1-min data, 96 cho daily trong 15-min data)

Usage:
    >>> from src.models.sarima import SARIMAForecaster
    >>> model = SARIMAForecaster(order=(2,1,2), seasonal_order=(1,1,1,96))
    >>> model.fit(train_series)
    >>> predictions = model.predict(steps=96)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from typing import Tuple, Dict, Optional, List
import warnings
import pickle


class SARIMAForecaster:
    """
    SARIMA model wrapper với các tiện ích cho forecasting.

    Features:
        - Automatic stationarity checking
        - Grid search cho optimal parameters
        - Confidence intervals
        - Model diagnostics
        - Save/load functionality

    Attributes:
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, m)
        model: SARIMAX model object
        fitted_model: Fitted model results

    Recommended Parameters by Granularity:
        - 1min: order=(2,1,2), seasonal_order=(1,1,1,60) - hourly seasonality
        - 5min: order=(2,1,2), seasonal_order=(1,1,1,12) - hourly seasonality
        - 15min: order=(2,1,2), seasonal_order=(1,1,1,96) - daily seasonality
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (2, 1, 2),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 96),
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False
    ):
        """
        Khởi tạo SARIMA model.

        Args:
            order: (p, d, q) - ARIMA orders
            seasonal_order: (P, D, Q, m) - Seasonal orders
            enforce_stationarity: Enforce AR parameters stationarity
            enforce_invertibility: Enforce MA parameters invertibility
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        self.model = None
        self.fitted_model = None
        self.train_data = None
        self.history = []

    def check_stationarity(
        self,
        series: pd.Series,
        significance: float = 0.05
    ) -> Dict:
        """
        Kiểm tra tính dừng của series bằng ADF test.

        Args:
            series: Time series cần kiểm tra
            significance: Ngưỡng p-value (mặc định 0.05)

        Returns:
            Dict với test statistics và kết luận
        """
        # Loại bỏ NaN
        clean_series = series.dropna()

        result = adfuller(clean_series, autolag='AIC')

        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'used_lag': result[2],
            'n_obs': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < significance,
            'conclusion': 'Stationary' if result[1] < significance else 'Non-stationary'
        }

    def auto_select_d(self, series: pd.Series, max_d: int = 2) -> int:
        """
        Tự động chọn differencing order d.

        Args:
            series: Time series
            max_d: Maximum d để thử

        Returns:
            Optimal d value
        """
        for d in range(max_d + 1):
            if d == 0:
                test_series = series
            else:
                test_series = series.diff(d).dropna()

            stat_test = self.check_stationarity(test_series)
            if stat_test['is_stationary']:
                return d

        return max_d

    def fit(
        self,
        train_series: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        verbose: bool = True,
        maxiter: int = 200
    ) -> 'SARIMAForecaster':
        """
        Fit SARIMA model với training data.

        Args:
            train_series: Target time series
            exog: Exogenous variables (optional)
            verbose: In progress
            maxiter: Max iterations cho optimizer

        Returns:
            self
        """
        self.train_data = train_series.copy()

        if verbose:
            print(f"Fitting SARIMA{self.order}x{self.seasonal_order}...")
            print(f"  Training samples: {len(train_series):,}")

        # Tạo model
        self.model = SARIMAX(
            train_series,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility
        )

        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fitted_model = self.model.fit(
                disp=verbose,
                maxiter=maxiter,
                method='lbfgs'
            )

        if verbose:
            print(f"  AIC: {self.fitted_model.aic:.2f}")
            print(f"  BIC: {self.fitted_model.bic:.2f}")
            print(f"  Log-likelihood: {self.fitted_model.llf:.2f}")

        return self

    def predict(
        self,
        steps: int,
        exog: Optional[pd.DataFrame] = None,
        return_conf_int: bool = True,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate forecasts.

        Args:
            steps: Số bước cần dự đoán
            exog: Exogenous variables cho forecast period
            return_conf_int: Trả về confidence intervals
            alpha: Significance level cho CI (0.05 = 95% CI)

        Returns:
            DataFrame với columns:
                - forecast: Giá trị dự đoán
                - lower: Lower bound của CI
                - upper: Upper bound của CI
        """
        if self.fitted_model is None:
            raise ValueError("Model chưa được fit. Gọi fit() trước.")

        # Get forecast
        forecast = self.fitted_model.get_forecast(steps=steps, exog=exog)

        result = pd.DataFrame({
            'forecast': forecast.predicted_mean
        })

        if return_conf_int:
            conf_int = forecast.conf_int(alpha=alpha)
            result['lower'] = conf_int.iloc[:, 0]
            result['upper'] = conf_int.iloc[:, 1]

        # Ensure non-negative predictions (requests can't be negative)
        result['forecast'] = result['forecast'].clip(lower=0)
        if 'lower' in result.columns:
            result['lower'] = result['lower'].clip(lower=0)

        return result

    def forecast_in_sample(self) -> pd.Series:
        """
        Lấy in-sample fitted values.

        Returns:
            Series với fitted values
        """
        if self.fitted_model is None:
            raise ValueError("Model chưa được fit.")

        return self.fitted_model.fittedvalues

    def get_residuals(self) -> pd.Series:
        """
        Lấy model residuals.

        Returns:
            Series với residuals
        """
        if self.fitted_model is None:
            raise ValueError("Model chưa được fit.")

        return self.fitted_model.resid

    def get_diagnostics(self) -> Dict:
        """
        Lấy model diagnostics.

        Returns:
            Dict với các diagnostic metrics
        """
        if self.fitted_model is None:
            raise ValueError("Model chưa được fit.")

        residuals = self.get_residuals()

        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'llf': self.fitted_model.llf,
            'params': self.fitted_model.params.to_dict(),
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std(),
            'residual_skew': residuals.skew(),
            'residual_kurtosis': residuals.kurtosis()
        }

    def summary(self):
        """In model summary."""
        if self.fitted_model is None:
            raise ValueError("Model chưa được fit.")

        print(self.fitted_model.summary())

    def save(self, filepath: str):
        """
        Lưu model ra file.

        Args:
            filepath: Đường dẫn file (vd: 'models/sarima_15min.pkl')
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'enforce_stationarity': self.enforce_stationarity,
                'enforce_invertibility': self.enforce_invertibility,
                'fitted_model': self.fitted_model,
                'train_data': self.train_data
            }, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SARIMAForecaster':
        """
        Load model từ file.

        Args:
            filepath: Đường dẫn file

        Returns:
            SARIMAForecaster instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        forecaster = cls(
            order=data['order'],
            seasonal_order=data['seasonal_order'],
            enforce_stationarity=data['enforce_stationarity'],
            enforce_invertibility=data['enforce_invertibility']
        )
        forecaster.fitted_model = data['fitted_model']
        forecaster.train_data = data['train_data']

        return forecaster


def grid_search_sarima(
    series: pd.Series,
    p_range: range = range(0, 3),
    d_range: range = range(0, 2),
    q_range: range = range(0, 3),
    seasonal_period: int = 96,
    verbose: bool = True
) -> Tuple[Tuple, Tuple, float]:
    """
    Grid search để tìm optimal SARIMA parameters.

    CẢNH BÁO: Rất chậm với data lớn. Chỉ dùng với sample nhỏ.

    Args:
        series: Time series
        p_range: Range cho AR order
        d_range: Range cho differencing order
        q_range: Range cho MA order
        seasonal_period: Seasonal period m
        verbose: In progress

    Returns:
        Tuple (best_order, best_seasonal_order, best_aic)
    """
    from itertools import product

    best_aic = np.inf
    best_order = None
    best_seasonal = None

    # Simplified grid - chỉ thử một số combinations
    combos = list(product(p_range, d_range, q_range))
    total = len(combos)

    print(f"Testing {total} ARIMA combinations...")

    for i, (p, d, q) in enumerate(combos):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = SARIMAX(
                    series,
                    order=(p, d, q),
                    seasonal_order=(1, 1, 1, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                result = model.fit(disp=False, maxiter=50)

                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)
                    best_seasonal = (1, 1, 1, seasonal_period)

                    if verbose:
                        print(f"  [{i+1}/{total}] SARIMA{best_order}x{best_seasonal}: AIC={best_aic:.2f}")

        except Exception as e:
            continue

    print(f"\nBest model: SARIMA{best_order}x{best_seasonal}")
    print(f"Best AIC: {best_aic:.2f}")

    return best_order, best_seasonal, best_aic


if __name__ == "__main__":
    # Demo usage
    import sys
    import os
    sys.path.insert(0, os.path.abspath('..'))

    from src.data.preprocessor import load_timeseries, split_train_test
    from src.models.evaluation import calculate_metrics

    # Load data
    print("Loading data...")
    df = load_timeseries('../data/processed/timeseries_15min.parquet')

    # Remove storm period
    df_clean = df[df['is_storm_period'] == 0]

    # Split train/test
    train, test = split_train_test(df_clean, test_start='1995-08-23')

    # Use request_count as target
    train_series = train['request_count']
    test_series = test['request_count']

    print(f"\nTrain: {len(train_series):,} samples")
    print(f"Test: {len(test_series):,} samples")

    # Create and fit model
    print("\n" + "="*50)
    model = SARIMAForecaster(
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 96)  # Daily seasonality for 15-min data
    )

    # Check stationarity
    print("\nStationarity check:")
    stat_result = model.check_stationarity(train_series)
    print(f"  ADF statistic: {stat_result['test_statistic']:.4f}")
    print(f"  p-value: {stat_result['p_value']:.4f}")
    print(f"  Conclusion: {stat_result['conclusion']}")

    # Fit model
    model.fit(train_series, verbose=True)

    # Predict
    predictions = model.predict(steps=len(test_series))

    # Evaluate
    metrics = calculate_metrics(test_series.values, predictions['forecast'].values)
    print(f"\nTest Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Save model
    model.save('../models/sarima_15min_demo.pkl')
