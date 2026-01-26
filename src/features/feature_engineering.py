"""
Feature Engineering Module
==========================
Module tạo features cho time series forecasting.

Features categories:
    1. Temporal: hour, day_of_week, is_weekend, cyclical encoding
    2. Lag: lag_1, lag_2, ..., lag_n
    3. Rolling: rolling_mean, rolling_std, rolling_max, rolling_min
    4. Derived: diff, pct_change, ewm (exponential weighted mean)

Usage:
    >>> fe = TimeSeriesFeatureEngineer(df)
    >>> df_features = fe.create_all_features(target_col='request_count')
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple


class TimeSeriesFeatureEngineer:
    """
    Feature engineering cho time series forecasting.

    Tạo các loại features:
        - Temporal features từ datetime index
        - Lag features để capture autocorrelation
        - Rolling statistics để capture trends
        - Derived features (diff, pct_change, ewm)

    Attributes:
        df: DataFrame với DatetimeIndex
        target_cols: Các cột target cần tạo features

    Example:
        >>> fe = TimeSeriesFeatureEngineer(df, target_cols=['request_count', 'bytes_total'])
        >>> df_with_features = fe.create_all_features(target_col='request_count')
        >>> print(df_with_features.columns.tolist())
    """

    # Default lag và window configurations theo granularity
    GRANULARITY_CONFIGS = {
        '1min': {
            'lags': [1, 2, 3, 5, 10, 15, 30, 60, 1440],  # Up to 1 day
            'windows': [5, 15, 30, 60, 1440],
            'diff_periods': [1, 60, 1440],
            'ewm_spans': [5, 15, 60, 1440]
        },
        '5min': {
            'lags': [1, 2, 3, 6, 12, 288],  # Up to 1 day (288 * 5min = 24h)
            'windows': [3, 6, 12, 288],
            'diff_periods': [1, 12, 288],
            'ewm_spans': [3, 12, 288]
        },
        '15min': {
            'lags': [1, 2, 4, 8, 96],  # Up to 1 day (96 * 15min = 24h)
            'windows': [4, 8, 96],
            'diff_periods': [1, 4, 96],
            'ewm_spans': [4, 96]
        }
    }

    def __init__(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None
    ):
        """
        Khởi tạo feature engineer.

        Args:
            df: DataFrame với DatetimeIndex
            target_cols: Danh sách các cột target (mặc định: ['request_count', 'bytes_total'])
        """
        self.df = df.copy()
        self.target_cols = target_cols or ['request_count', 'bytes_total']

        # Đảm bảo index là DatetimeIndex
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame phải có DatetimeIndex")

    def add_temporal_features(self) -> pd.DataFrame:
        """
        Thêm temporal features từ datetime index.

        Features tạo:
            - hour: 0-23
            - day_of_week: 0-6 (Monday=0)
            - day_of_month: 1-31
            - week_of_year: 1-52
            - month: 1-12
            - is_weekend: 0/1
            - is_business_hour: 0/1 (9-17, weekdays)
            - hour_sin, hour_cos: Cyclical encoding của hour
            - dow_sin, dow_cos: Cyclical encoding của day_of_week

        Returns:
            DataFrame với temporal features
        """
        df = self.df.copy()

        # Basic temporal features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['month'] = df.index.month

        # Derived boolean features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hour'] = (
            (df['hour'] >= 9) & (df['hour'] <= 17) &
            (df['day_of_week'] < 5)
        ).astype(int)

        # Cyclical encoding (để capture circular nature của time)
        # Sin và Cos encoding giúp model hiểu 23:00 gần với 00:00
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def add_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Thêm lag features cho target column.

        Args:
            df: Input DataFrame
            target_col: Tên cột target
            lags: Danh sách các lag periods (mặc định theo granularity)

        Returns:
            DataFrame với lag features

        Note:
            Lag features sẽ tạo NaN ở đầu series (số rows = max(lags))
        """
        if lags is None:
            lags = [1, 2, 3, 5, 10, 15, 30, 60]

        for lag in lags:
            col_name = f'{target_col}_lag_{lag}'
            df[col_name] = df[target_col].shift(lag)

        return df

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Thêm rolling statistics features.

        Features cho mỗi window:
            - rolling_mean_{window}
            - rolling_std_{window}
            - rolling_max_{window}
            - rolling_min_{window}

        Args:
            df: Input DataFrame
            target_col: Tên cột target
            windows: Danh sách các window sizes

        Returns:
            DataFrame với rolling features
        """
        if windows is None:
            windows = [5, 15, 30, 60]

        for window in windows:
            # Rolling mean
            df[f'{target_col}_rolling_mean_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).mean()
            )
            # Rolling std
            df[f'{target_col}_rolling_std_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).std().fillna(0)
            )
            # Rolling max
            df[f'{target_col}_rolling_max_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).max()
            )
            # Rolling min
            df[f'{target_col}_rolling_min_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).min()
            )

        return df

    def add_diff_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Thêm difference features để capture trends.

        Features:
            - diff_{period}: First difference (y_t - y_{t-period})
            - pct_change_{period}: Percentage change

        Args:
            df: Input DataFrame
            target_col: Tên cột target
            periods: Danh sách các periods

        Returns:
            DataFrame với diff features
        """
        if periods is None:
            periods = [1, 60, 1440]

        for period in periods:
            # First difference
            df[f'{target_col}_diff_{period}'] = df[target_col].diff(period)
            # Percentage change
            df[f'{target_col}_pct_change_{period}'] = df[target_col].pct_change(period)

        # Replace inf values từ pct_change (khi y_t-1 = 0)
        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    def add_ewm_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        spans: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Thêm Exponentially Weighted Mean features.

        EWM cho weight cao hơn cho các observations gần nhất,
        hữu ích để capture recent trends.

        Args:
            df: Input DataFrame
            target_col: Tên cột target
            spans: Danh sách các span values

        Returns:
            DataFrame với EWM features
        """
        if spans is None:
            spans = [5, 15, 60]

        for span in spans:
            df[f'{target_col}_ewm_{span}'] = (
                df[target_col].ewm(span=span, adjust=False).mean()
            )

        return df

    def create_all_features(
        self,
        target_col: str = 'request_count',
        lags: Optional[List[int]] = None,
        windows: Optional[List[int]] = None,
        diff_periods: Optional[List[int]] = None,
        ewm_spans: Optional[List[int]] = None,
        granularity: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Tạo toàn bộ feature set cho một target column.

        Args:
            target_col: Tên cột target
            lags: Custom lag values (nếu None, dùng default)
            windows: Custom window sizes (nếu None, dùng default)
            diff_periods: Custom diff periods
            ewm_spans: Custom EWM spans
            granularity: '1min', '5min', hoặc '15min' để auto-config

        Returns:
            DataFrame với tất cả features

        Example:
            >>> fe = TimeSeriesFeatureEngineer(df)
            >>> df_features = fe.create_all_features(
            ...     target_col='request_count',
            ...     granularity='15min'
            ... )
        """
        # Auto-config dựa trên granularity nếu được cung cấp
        if granularity and granularity in self.GRANULARITY_CONFIGS:
            config = self.GRANULARITY_CONFIGS[granularity]
            lags = lags or config['lags']
            windows = windows or config['windows']
            diff_periods = diff_periods or config['diff_periods']
            ewm_spans = ewm_spans or config['ewm_spans']

        # Defaults nếu không có config
        lags = lags or [1, 2, 3, 5, 10, 15, 30, 60]
        windows = windows or [5, 15, 30, 60]
        diff_periods = diff_periods or [1, 60]
        ewm_spans = ewm_spans or [5, 15, 60]

        print(f"Creating features for '{target_col}'...")
        print(f"  Lags: {lags}")
        print(f"  Windows: {windows}")
        print(f"  Diff periods: {diff_periods}")
        print(f"  EWM spans: {ewm_spans}")

        # Step 1: Temporal features
        df = self.add_temporal_features()
        print(f"  + Temporal features added")

        # Step 2: Lag features
        df = self.add_lag_features(df, target_col, lags)
        print(f"  + Lag features added")

        # Step 3: Rolling features
        df = self.add_rolling_features(df, target_col, windows)
        print(f"  + Rolling features added")

        # Step 4: Diff features
        df = self.add_diff_features(df, target_col, diff_periods)
        print(f"  + Diff features added")

        # Step 5: EWM features
        df = self.add_ewm_features(df, target_col, ewm_spans)
        print(f"  + EWM features added")

        print(f"\nTotal features: {len(df.columns)}")
        print(f"NaN rows (due to lags): {df.isna().any(axis=1).sum()}")

        return df

    @staticmethod
    def get_feature_columns(
        df: pd.DataFrame,
        exclude_targets: bool = True,
        exclude_storm: bool = True
    ) -> List[str]:
        """
        Lấy danh sách các feature columns (loại bỏ targets và metadata).

        Args:
            df: DataFrame với features
            exclude_targets: Loại bỏ các cột target
            exclude_storm: Loại bỏ cột is_storm_period

        Returns:
            List các tên feature columns
        """
        # Các cột không phải features
        exclude_cols = set()

        if exclude_targets:
            exclude_cols.update([
                'request_count', 'bytes_total', 'bytes_mean', 'bytes_std',
                'success_count', 'error_count', 'server_error_count',
                'error_rate', 'avg_request_size', 'unique_hosts'
            ])

        if exclude_storm:
            exclude_cols.add('is_storm_period')

        # Các cột categorical không dùng trực tiếp
        exclude_cols.update(['time_of_day', 'day_name', 'date'])

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        return feature_cols

    @staticmethod
    def prepare_supervised(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        forecast_horizon: int = 1,
        dropna: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Chuẩn bị data cho supervised learning.

        Tạo X (features) và y (target shifted by forecast_horizon).
        Điều này cho phép model học cách dự đoán y_{t+horizon} từ features tại t.

        Args:
            df: DataFrame với features
            target_col: Tên cột target
            feature_cols: Danh sách các feature columns
            forecast_horizon: Số steps ahead để dự đoán
            dropna: Loại bỏ rows với NaN

        Returns:
            Tuple (X, y) với:
                - X: DataFrame features
                - y: Series target (shifted)
        """
        X = df[feature_cols].copy()
        y = df[target_col].shift(-forecast_horizon)

        if dropna:
            # Tạo mask cho valid rows
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]

        return X, y


def create_features_for_all_targets(
    df: pd.DataFrame,
    targets: List[str] = None,
    granularity: str = '15min'
) -> pd.DataFrame:
    """
    Tiện ích: Tạo features cho nhiều targets cùng lúc.

    Args:
        df: Input DataFrame
        targets: List các target columns
        granularity: Granularity để auto-config

    Returns:
        DataFrame với features cho tất cả targets
    """
    targets = targets or ['request_count', 'bytes_total']

    fe = TimeSeriesFeatureEngineer(df, target_cols=targets)

    # Tạo features cho target đầu tiên
    result = fe.create_all_features(targets[0], granularity=granularity)

    # Thêm features cho các targets còn lại
    for target in targets[1:]:
        fe_temp = TimeSeriesFeatureEngineer(result, target_cols=[target])
        config = fe.GRANULARITY_CONFIGS.get(granularity, {})

        result = fe_temp.add_lag_features(result, target, config.get('lags'))
        result = fe_temp.add_rolling_features(result, target, config.get('windows'))

    return result


if __name__ == "__main__":
    # Demo usage
    import sys
    import os
    sys.path.insert(0, os.path.abspath('..'))

    from src.data.preprocessor import load_timeseries

    # Load data
    print("Loading data...")
    df = load_timeseries('../data/processed/timeseries_15min.parquet')

    # Create features
    print("\nCreating features...")
    fe = TimeSeriesFeatureEngineer(df)
    df_features = fe.create_all_features(
        target_col='request_count',
        granularity='15min'
    )

    # Show result
    print(f"\nResult shape: {df_features.shape}")
    print(f"\nFeature columns:")
    feature_cols = fe.get_feature_columns(df_features)
    for col in feature_cols[:20]:
        print(f"  - {col}")
    print(f"  ... và {len(feature_cols) - 20} features khác")

    # Prepare supervised data
    X, y = fe.prepare_supervised(
        df_features,
        target_col='request_count',
        feature_cols=feature_cols,
        forecast_horizon=1
    )

    print(f"\nSupervised data:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
