"""
Prophet Forecaster
==================
Facebook Prophet wrapper cho time series forecasting.

Prophet là model được thiết kế cho:
    - Business time series với strong seasonality
    - Missing data và outliers
    - Trend changes (changepoints)

Ưu điểm:
    - Tự động detect seasonality
    - Handle missing data tốt
    - Dễ tune và interpretable
    - Uncertainty intervals

Usage:
    >>> from src.models.prophet_forecaster import ProphetForecaster
    >>> model = ProphetForecaster()
    >>> model.fit(df, target_col='request_count')
    >>> predictions = model.predict(periods=96)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import pickle
import warnings

# Prophet import với error handling
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not installed. Install with: pip install prophet")


class ProphetForecaster:
    """
    Facebook Prophet forecaster wrapper.

    Features:
        - Multiple seasonalities (daily, weekly, hourly)
        - Automatic changepoint detection
        - Holiday effects (optional)
        - Uncertainty intervals
        - Component decomposition

    Attributes:
        seasonality_mode: 'additive' hoặc 'multiplicative'
        model: Prophet model object
        forecast: Last forecast DataFrame

    Note:
        Prophet yêu cầu data format đặc biệt:
        - Column 'ds': datetime
        - Column 'y': target value
    """

    def __init__(
        self,
        seasonality_mode: str = 'multiplicative',
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        add_hourly_seasonality: bool = True
    ):
        """
        Khởi tạo Prophet forecaster.

        Args:
            seasonality_mode: 'additive' hoặc 'multiplicative'
            yearly_seasonality: Thêm yearly seasonality (False vì chỉ có 2 tháng data)
            weekly_seasonality: Thêm weekly seasonality
            daily_seasonality: Thêm daily seasonality
            changepoint_prior_scale: Flexibility của trend (cao = flexible hơn)
            seasonality_prior_scale: Strength của seasonality
            add_hourly_seasonality: Thêm custom hourly seasonality
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not installed. Install with: pip install prophet")

        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.add_hourly_seasonality = add_hourly_seasonality

        self.model = None
        self.forecast = None
        self.train_data = None

    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """
        Convert DataFrame sang Prophet format.

        Args:
            df: Input DataFrame với DatetimeIndex
            target_col: Tên cột target

        Returns:
            DataFrame với columns 'ds' và 'y'
        """
        prophet_df = pd.DataFrame({
            'ds': df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index),
            'y': df[target_col].values
        })

        return prophet_df

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'request_count',
        verbose: bool = True
    ) -> 'ProphetForecaster':
        """
        Fit Prophet model.

        Args:
            df: DataFrame với DatetimeIndex
            target_col: Tên cột target
            verbose: Print progress

        Returns:
            self
        """
        # Prepare data
        prophet_df = self._prepare_data(df, target_col)
        self.train_data = prophet_df.copy()

        if verbose:
            print(f"Fitting Prophet model...")
            print(f"  Training samples: {len(prophet_df):,}")
            print(f"  Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")

        # Create model
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale
        )

        # Add custom hourly seasonality nếu cần
        if self.add_hourly_seasonality:
            # Period = 1/24 day = 1 hour
            self.model.add_seasonality(
                name='hourly',
                period=1/24,
                fourier_order=5
            )

        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(prophet_df)

        if verbose:
            print("  Training complete!")

        return self

    def predict(
        self,
        periods: int,
        freq: str = '15min',
        include_history: bool = False
    ) -> pd.DataFrame:
        """
        Generate forecasts.

        Args:
            periods: Số periods cần forecast
            freq: Frequency (vd: '1min', '5min', '15min', '1h')
            include_history: Include historical predictions

        Returns:
            DataFrame với columns:
                - ds: datetime
                - yhat: prediction
                - yhat_lower: lower bound
                - yhat_upper: upper bound
        """
        if self.model is None:
            raise ValueError("Model chưa được fit. Gọi fit() trước.")

        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=include_history
        )

        # Predict
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.forecast = self.model.predict(future)

        # Ensure non-negative predictions
        result = self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        result['yhat'] = result['yhat'].clip(lower=0)
        result['yhat_lower'] = result['yhat_lower'].clip(lower=0)

        return result

    def get_components(self) -> pd.DataFrame:
        """
        Lấy các components của forecast (trend, seasonality, etc.).

        Returns:
            DataFrame với trend và các seasonality components
        """
        if self.forecast is None:
            raise ValueError("Chưa có forecast. Gọi predict() trước.")

        components = ['ds', 'trend']

        if self.daily_seasonality:
            components.append('daily')
        if self.weekly_seasonality:
            components.append('weekly')
        if self.add_hourly_seasonality and 'hourly' in self.forecast.columns:
            components.append('hourly')

        available_cols = [c for c in components if c in self.forecast.columns]

        return self.forecast[available_cols]

    def plot_forecast(
        self,
        figsize: Tuple[int, int] = (14, 6),
        include_history: bool = True
    ):
        """
        Plot forecast với uncertainty intervals.

        Args:
            figsize: Figure size
            include_history: Hiển thị historical data
        """
        if self.forecast is None:
            raise ValueError("Chưa có forecast. Gọi predict() trước.")

        # Use Prophet's built-in plotting
        fig = self.model.plot(self.forecast, figsize=figsize)
        return fig

    def plot_components(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Plot các components riêng lẻ.

        Args:
            figsize: Figure size
        """
        if self.forecast is None:
            raise ValueError("Chưa có forecast. Gọi predict() trước.")

        fig = self.model.plot_components(self.forecast, figsize=figsize)
        return fig

    def cross_validate(
        self,
        initial: str = '30 days',
        period: str = '7 days',
        horizon: str = '1 days',
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Time series cross-validation.

        Args:
            initial: Initial training period
            period: Period between cutoff dates
            horizon: Forecast horizon

        Returns:
            DataFrame với cv results
        """
        from prophet.diagnostics import cross_validation, performance_metrics

        if self.model is None:
            raise ValueError("Model chưa được fit.")

        if verbose:
            print(f"Running cross-validation...")
            print(f"  Initial: {initial}, Period: {period}, Horizon: {horizon}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_cv = cross_validation(
                self.model,
                initial=initial,
                period=period,
                horizon=horizon
            )

        if verbose:
            metrics = performance_metrics(df_cv)
            print(f"\nCV Metrics:")
            print(metrics[['horizon', 'mse', 'rmse', 'mae', 'mape']].tail())

        return df_cv

    def get_changepoints(self) -> pd.DataFrame:
        """
        Lấy các changepoints được detect.

        Returns:
            DataFrame với changepoint dates và magnitudes
        """
        if self.model is None:
            raise ValueError("Model chưa được fit.")

        changepoints = self.model.changepoints
        deltas = self.model.params['delta'].mean(axis=0)

        return pd.DataFrame({
            'ds': changepoints,
            'delta': deltas[:len(changepoints)]
        })

    def save(self, filepath: str):
        """
        Lưu model ra file.

        Args:
            filepath: Đường dẫn file
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'seasonality_mode': self.seasonality_mode,
                'yearly_seasonality': self.yearly_seasonality,
                'weekly_seasonality': self.weekly_seasonality,
                'daily_seasonality': self.daily_seasonality,
                'changepoint_prior_scale': self.changepoint_prior_scale,
                'seasonality_prior_scale': self.seasonality_prior_scale,
                'add_hourly_seasonality': self.add_hourly_seasonality,
                'train_data': self.train_data
            }, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ProphetForecaster':
        """
        Load model từ file.

        Args:
            filepath: Đường dẫn file

        Returns:
            ProphetForecaster instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        forecaster = cls(
            seasonality_mode=data['seasonality_mode'],
            yearly_seasonality=data['yearly_seasonality'],
            weekly_seasonality=data['weekly_seasonality'],
            daily_seasonality=data['daily_seasonality'],
            changepoint_prior_scale=data['changepoint_prior_scale'],
            seasonality_prior_scale=data['seasonality_prior_scale'],
            add_hourly_seasonality=data['add_hourly_seasonality']
        )
        forecaster.model = data['model']
        forecaster.train_data = data['train_data']

        return forecaster


if __name__ == "__main__":
    # Demo usage
    if not PROPHET_AVAILABLE:
        print("Prophet not available. Skipping demo.")
        exit()

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

    print(f"\nTrain: {len(train):,} samples")
    print(f"Test: {len(test):,} samples")

    # Create and fit model
    print("\n" + "="*50)
    model = ProphetForecaster(
        seasonality_mode='multiplicative',
        weekly_seasonality=True,
        daily_seasonality=True,
        add_hourly_seasonality=True
    )
    model.fit(train, target_col='request_count')

    # Predict
    predictions = model.predict(periods=len(test), freq='15min')

    # Align predictions với test data
    pred_values = predictions['yhat'].values[:len(test)]
    actual_values = test['request_count'].values

    # Evaluate
    metrics = calculate_metrics(actual_values, pred_values)
    print(f"\nTest Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Components
    print("\nForecast components:")
    components = model.get_components()
    print(components.head())

    # Save model
    model.save('../models/prophet_15min_demo.pkl')
