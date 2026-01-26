"""
Anomaly Detector Module
=======================
Module phát hiện anomaly trong traffic data.

Các phương pháp:
    - Z-score: Statistical deviation detection
    - IQR: Interquartile range based detection
    - Isolation Forest: ML-based outlier detection
    - Rolling: Comparison with rolling statistics
    - Spike: Sudden change detection

Usage:
    >>> from src.anomaly.detector import AnomalyDetector
    >>> detector = AnomalyDetector()
    >>> anomalies = detector.detect_zscore(traffic, threshold=3.0)
    >>> report = detector.generate_report(traffic)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AnomalyResult:
    """
    Kết quả phát hiện anomaly.

    Attributes:
        method: Tên phương pháp sử dụng
        anomaly_mask: Boolean Series đánh dấu anomalies
        anomaly_count: Số lượng anomalies
        anomaly_rate: Tỷ lệ anomalies (%)
        scores: Anomaly scores (nếu có)
    """
    method: str
    anomaly_mask: pd.Series
    anomaly_count: int
    anomaly_rate: float
    scores: Optional[pd.Series] = None


class AnomalyDetector:
    """
    Anomaly Detector cho traffic data.

    Cung cấp nhiều phương pháp phát hiện anomaly:
        - Statistical: Z-score, IQR
        - ML-based: Isolation Forest
        - Time-series: Rolling window, Spike detection

    Example:
        >>> detector = AnomalyDetector()
        >>> zscore_anomalies = detector.detect_zscore(traffic, threshold=3.0)
        >>> combined = detector.get_combined_anomaly_score(traffic)
    """

    def __init__(self):
        """Khởi tạo Anomaly Detector."""
        self.results = {}

    def detect_zscore(
        self,
        series: pd.Series,
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Phát hiện anomaly bằng Z-score.

        Z-score = (x - mean) / std
        Anomaly nếu |z| > threshold

        Args:
            series: Time series data
            threshold: Z-score threshold (mặc định 3.0 = 99.7%)

        Returns:
            Boolean Series đánh dấu anomalies
        """
        mean = series.mean()
        std = series.std()

        z_scores = (series - mean) / std
        anomalies = np.abs(z_scores) > threshold

        self.results['zscore'] = AnomalyResult(
            method='Z-Score',
            anomaly_mask=anomalies,
            anomaly_count=anomalies.sum(),
            anomaly_rate=anomalies.sum() / len(series) * 100,
            scores=z_scores
        )

        return anomalies

    def detect_iqr(
        self,
        series: pd.Series,
        k: float = 1.5
    ) -> pd.Series:
        """
        Phát hiện anomaly bằng IQR (Interquartile Range).

        Anomaly nếu x < Q1 - k*IQR hoặc x > Q3 + k*IQR

        Args:
            series: Time series data
            k: IQR multiplier (mặc định 1.5)

        Returns:
            Boolean Series đánh dấu anomalies
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        anomalies = (series < lower_bound) | (series > upper_bound)

        self.results['iqr'] = AnomalyResult(
            method='IQR',
            anomaly_mask=anomalies,
            anomaly_count=anomalies.sum(),
            anomaly_rate=anomalies.sum() / len(series) * 100
        )

        return anomalies

    def detect_rolling(
        self,
        series: pd.Series,
        window: int = 24,
        threshold: float = 2.5
    ) -> pd.Series:
        """
        Phát hiện anomaly so với rolling statistics.

        So sánh mỗi điểm với rolling mean và std.

        Args:
            series: Time series data
            window: Rolling window size
            threshold: Number of std deviations

        Returns:
            Boolean Series đánh dấu anomalies
        """
        rolling_mean = series.rolling(window, min_periods=1).mean()
        rolling_std = series.rolling(window, min_periods=1).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, series.std())

        z_scores = (series - rolling_mean) / rolling_std
        anomalies = np.abs(z_scores) > threshold

        self.results['rolling'] = AnomalyResult(
            method='Rolling Window',
            anomaly_mask=anomalies,
            anomaly_count=anomalies.sum(),
            anomaly_rate=anomalies.sum() / len(series) * 100,
            scores=z_scores
        )

        return anomalies

    def detect_spikes(
        self,
        series: pd.Series,
        window: int = 4,
        threshold: float = 2.0
    ) -> pd.Series:
        """
        Phát hiện sudden spikes (tăng đột biến).

        Spike = tăng nhanh so với rolling mean gần đây.

        Args:
            series: Time series data
            window: Window để tính baseline
            threshold: Threshold (multiplier của rolling mean)

        Returns:
            Boolean Series đánh dấu spikes
        """
        rolling_mean = series.rolling(window, min_periods=1).mean().shift(1)
        rolling_std = series.rolling(window, min_periods=1).std().shift(1)

        # Fill NaN với overall stats
        rolling_mean = rolling_mean.fillna(series.mean())
        rolling_std = rolling_std.fillna(series.std())

        # Spike if value > mean + threshold * std
        spikes = series > (rolling_mean + threshold * rolling_std)

        # Also check for sudden increase
        pct_change = series.pct_change().abs()
        sudden_increase = pct_change > 1.0  # More than 100% increase

        combined_spikes = spikes & sudden_increase

        self.results['spike'] = AnomalyResult(
            method='Spike Detection',
            anomaly_mask=combined_spikes,
            anomaly_count=combined_spikes.sum(),
            anomaly_rate=combined_spikes.sum() / len(series) * 100
        )

        return combined_spikes

    def detect_drops(
        self,
        series: pd.Series,
        window: int = 4,
        threshold: float = 0.5
    ) -> pd.Series:
        """
        Phát hiện sudden drops (giảm đột biến).

        Args:
            series: Time series data
            window: Window để tính baseline
            threshold: Threshold (dưới % của rolling mean)

        Returns:
            Boolean Series đánh dấu drops
        """
        rolling_mean = series.rolling(window, min_periods=1).mean().shift(1)
        rolling_mean = rolling_mean.fillna(series.mean())

        drops = series < (rolling_mean * threshold)

        self.results['drop'] = AnomalyResult(
            method='Drop Detection',
            anomaly_mask=drops,
            anomaly_count=drops.sum(),
            anomaly_rate=drops.sum() / len(series) * 100
        )

        return drops

    def detect_ddos_pattern(
        self,
        traffic: pd.Series,
        unique_hosts: pd.Series,
        traffic_threshold: float = 2.0,
        host_ratio_threshold: float = 0.5
    ) -> pd.Series:
        """
        Phát hiện DDoS-like pattern.

        DDoS: High traffic + Low unique hosts ratio
        (Nhiều requests từ ít sources)

        Args:
            traffic: Request count series
            unique_hosts: Unique hosts count series
            traffic_threshold: Traffic z-score threshold
            host_ratio_threshold: Max ratio of hosts/requests

        Returns:
            Boolean Series đánh dấu suspicious periods
        """
        # High traffic
        traffic_mean = traffic.mean()
        traffic_std = traffic.std()
        high_traffic = (traffic - traffic_mean) / traffic_std > traffic_threshold

        # Low host diversity (few hosts making many requests)
        host_ratio = unique_hosts / traffic
        host_ratio = host_ratio.replace([np.inf, -np.inf], 1.0).fillna(1.0)
        low_diversity = host_ratio < host_ratio_threshold

        suspicious = high_traffic & low_diversity

        self.results['ddos'] = AnomalyResult(
            method='DDoS Pattern',
            anomaly_mask=suspicious,
            anomaly_count=suspicious.sum(),
            anomaly_rate=suspicious.sum() / len(traffic) * 100
        )

        return suspicious

    def detect_isolation_forest(
        self,
        series: pd.Series,
        contamination: float = 0.05,
        n_estimators: int = 100
    ) -> pd.Series:
        """
        Phát hiện anomaly bằng Isolation Forest.

        Args:
            series: Time series data
            contamination: Expected proportion of outliers
            n_estimators: Number of trees

        Returns:
            Boolean Series đánh dấu anomalies
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            print("sklearn not available for Isolation Forest")
            return pd.Series(False, index=series.index)

        # Prepare features
        features = pd.DataFrame({
            'value': series,
            'lag_1': series.shift(1),
            'diff': series.diff(),
            'rolling_mean': series.rolling(4).mean()
        }).dropna()

        # Train model
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )

        predictions = model.fit_predict(features)
        anomalies = pd.Series(predictions == -1, index=features.index)

        # Extend to full series
        full_anomalies = pd.Series(False, index=series.index)
        full_anomalies[anomalies.index] = anomalies

        self.results['isolation_forest'] = AnomalyResult(
            method='Isolation Forest',
            anomaly_mask=full_anomalies,
            anomaly_count=anomalies.sum(),
            anomaly_rate=anomalies.sum() / len(features) * 100
        )

        return full_anomalies

    def get_combined_anomaly_score(
        self,
        series: pd.Series,
        methods: List[str] = None,
        weights: Dict[str, float] = None
    ) -> pd.Series:
        """
        Tính combined anomaly score từ nhiều phương pháp.

        Score = sum of individual detection flags

        Args:
            series: Time series data
            methods: List methods to use ['zscore', 'iqr', 'rolling', 'spike']
            weights: Optional weights for each method

        Returns:
            Series với anomaly scores (0-N)
        """
        if methods is None:
            methods = ['zscore', 'iqr', 'rolling', 'spike']

        if weights is None:
            weights = {m: 1.0 for m in methods}

        # Run all detection methods
        detections = {}

        if 'zscore' in methods:
            detections['zscore'] = self.detect_zscore(series)
        if 'iqr' in methods:
            detections['iqr'] = self.detect_iqr(series)
        if 'rolling' in methods:
            detections['rolling'] = self.detect_rolling(series)
        if 'spike' in methods:
            detections['spike'] = self.detect_spikes(series)
        if 'drop' in methods:
            detections['drop'] = self.detect_drops(series)

        # Combine scores
        combined = pd.Series(0.0, index=series.index)
        for method, detection in detections.items():
            combined += detection.astype(float) * weights.get(method, 1.0)

        return combined

    def generate_report(
        self,
        series: pd.Series,
        methods: List[str] = None
    ) -> str:
        """
        Generate text report về anomaly detection.

        Args:
            series: Time series data
            methods: Methods to include

        Returns:
            Formatted report string
        """
        if methods is None:
            methods = ['zscore', 'iqr', 'rolling', 'spike']

        # Run detections
        self.get_combined_anomaly_score(series, methods)

        report = f"""
{'='*60}
                ANOMALY DETECTION REPORT
{'='*60}

DATA SUMMARY:
  Total Records: {len(series):,}
  Date Range: {series.index.min()} to {series.index.max()}
  Mean: {series.mean():.2f}
  Std: {series.std():.2f}
  Min: {series.min():.2f}
  Max: {series.max():.2f}

DETECTION RESULTS:
{'-'*60}
"""
        for method, result in self.results.items():
            report += f"""
  {result.method}:
    - Anomalies: {result.anomaly_count}
    - Rate: {result.anomaly_rate:.2f}%
"""

        # Top anomalies
        combined = self.get_combined_anomaly_score(series, methods)
        severe = combined[combined >= 3]

        report += f"""
{'-'*60}
SEVERE ANOMALIES (score >= 3):
  Count: {len(severe)}
"""
        if len(severe) > 0:
            report += f"  Timestamps:\n"
            for ts in severe.index[:10]:  # Top 10
                report += f"    - {ts}: score={combined[ts]:.0f}, value={series[ts]:.0f}\n"

        report += f"""
{'='*60}
"""
        return report

    def get_anomaly_timestamps(
        self,
        series: pd.Series,
        method: str = 'combined',
        threshold: int = 2
    ) -> pd.DatetimeIndex:
        """
        Lấy danh sách timestamps của anomalies.

        Args:
            series: Time series data
            method: Detection method hoặc 'combined'
            threshold: Threshold cho combined score

        Returns:
            DatetimeIndex của anomalies
        """
        if method == 'combined':
            combined = self.get_combined_anomaly_score(series)
            return combined[combined >= threshold].index
        else:
            if method not in self.results:
                # Run detection
                if method == 'zscore':
                    self.detect_zscore(series)
                elif method == 'iqr':
                    self.detect_iqr(series)
                elif method == 'rolling':
                    self.detect_rolling(series)
                elif method == 'spike':
                    self.detect_spikes(series)

            if method in self.results:
                return series.index[self.results[method].anomaly_mask]

        return pd.DatetimeIndex([])


if __name__ == "__main__":
    # Demo
    import sys
    import os
    sys.path.insert(0, os.path.abspath('..'))

    from src.data.preprocessor import load_timeseries

    # Load data
    print("Loading data...")
    df = load_timeseries('../data/processed/timeseries_15min.parquet')
    traffic = df['request_count']

    # Create detector
    detector = AnomalyDetector()

    # Run detection
    print("\nRunning anomaly detection...")

    # Z-score
    zscore = detector.detect_zscore(traffic, threshold=3.0)
    print(f"Z-score anomalies: {zscore.sum()}")

    # IQR
    iqr = detector.detect_iqr(traffic, k=1.5)
    print(f"IQR anomalies: {iqr.sum()}")

    # Rolling
    rolling = detector.detect_rolling(traffic, window=96)
    print(f"Rolling anomalies: {rolling.sum()}")

    # Spikes
    spikes = detector.detect_spikes(traffic)
    print(f"Spikes: {spikes.sum()}")

    # Report
    report = detector.generate_report(traffic)
    print(report)
