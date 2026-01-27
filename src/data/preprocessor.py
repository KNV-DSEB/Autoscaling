"""
Log Preprocessor
================
Module xử lý và aggregate log data thành time series.

Chức năng chính:
    - Aggregate logs theo các time window (1min, 5min, 15min)
    - Tính toán các metrics: request_count, bytes_total, error_rate, etc.
    - Xử lý missing data (storm gap)
    - Tạo derived features

Storm Gap:
    - Khoảng thời gian server offline: 01/Aug/1995 14:52:01 → 03/Aug/1995 04:36:13
    - Xử lý: Đánh dấu is_storm_period=1, fill với 0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class LogPreprocessor:
    """
    Preprocessor để aggregate parsed logs thành time series.

    Xử lý:
        - Storm gap (server offline period)
        - Missing timestamps
        - Error rate calculation
        - Multiple granularities (1min, 5min, 15min)

    Attributes:
        df: DataFrame gốc với parsed logs
        STORM_START: Timestamp bắt đầu storm gap
        STORM_END: Timestamp kết thúc storm gap

    Usage:
        >>> preprocessor = LogPreprocessor(parsed_df)
        >>> ts_1min = preprocessor.aggregate_timeseries('1min')
        >>> all_ts = preprocessor.create_all_granularities()
    """

    # Storm period (server offline do bão)
    # Dựa trên phân tích dữ liệu thực tế
    STORM_START = pd.Timestamp('1995-08-01 14:52:01')
    STORM_END = pd.Timestamp('1995-08-03 04:36:13')

    def __init__(self, df: pd.DataFrame):
        """
        Khởi tạo preprocessor với DataFrame đã parse.

        Args:
            df: DataFrame từ NASALogParser với các cột:
                host, timestamp, timezone, method, url, protocol, status_code, bytes
        """
        self.df = df.copy()

        # Đảm bảo timestamp là index
        if 'timestamp' in self.df.columns:
            self.df.set_index('timestamp', inplace=True)

        # Đảm bảo index là DatetimeIndex
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)

        # Sort theo thời gian
        self.df.sort_index(inplace=True)

    def add_derived_columns(self) -> pd.DataFrame:
        """
        Thêm các cột derived từ dữ liệu gốc.

        Returns:
            DataFrame với các cột mới:
                - status_category: 1xx, 2xx, 3xx, 4xx, 5xx
                - is_success: True nếu status < 400
                - is_error: True nếu status >= 400
                - is_server_error: True nếu status >= 500
                - url_path: Phần path của URL (không có query string)
                - url_extension: Extension của file (gif, html, etc.)
        """
        df = self.df.copy()

        # Status code categories
        df['status_category'] = pd.cut(
            df['status_code'],
            bins=[0, 199, 299, 399, 499, 599],
            labels=['1xx', '2xx', '3xx', '4xx', '5xx'],
            include_lowest=True
        )

        # Boolean flags
        df['is_success'] = (df['status_code'] >= 200) & (df['status_code'] < 400)
        df['is_error'] = df['status_code'] >= 400
        df['is_server_error'] = df['status_code'] >= 500

        # URL parsing
        df['url_path'] = df['url'].str.split('?').str[0]
        df['url_extension'] = df['url_path'].str.extract(r'\.(\w+)$')[0].fillna('none')

        return df

    def aggregate_timeseries(
        self,
        freq: str = '1min',
        fill_gaps: bool = True,
        mark_storm: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate log data thành time series với frequency cụ thể.

        Args:
            freq: Pandas frequency string:
                - '1min': 1 phút
                - '5min': 5 phút
                - '15min': 15 phút
                - '1h': 1 giờ
            fill_gaps: Fill các timestamps missing với 0
            mark_storm: Đánh dấu storm period với cột is_storm_period

        Returns:
            DataFrame với các cột:
                - request_count: Số requests trong window
                - bytes_total: Tổng bytes
                - bytes_mean: Bytes trung bình per request
                - bytes_std: Độ lệch chuẩn bytes
                - success_count: Số requests thành công (2xx, 3xx)
                - error_count: Số requests lỗi (4xx, 5xx)
                - server_error_count: Số server errors (5xx)
                - error_rate: error_count / request_count
                - unique_hosts: Số unique hosts/IPs
                - is_storm_period: 1 nếu trong storm gap
        """
        df = self.add_derived_columns()

        # Tính unique hosts trước khi resample
        unique_hosts = df.groupby(pd.Grouper(freq=freq))['host'].nunique()

        # Aggregate các metrics chính
        agg_dict = {
            'host': 'count',  # Total requests
            'bytes': ['sum', 'mean', 'std'],
            'is_success': 'sum',
            'is_error': 'sum',
            'is_server_error': 'sum'
        }

        ts = df.resample(freq).agg(agg_dict)

        # Flatten column names
        ts.columns = [
            'request_count',
            'bytes_total', 'bytes_mean', 'bytes_std',
            'success_count', 'error_count', 'server_error_count'
        ]

        # Thêm unique hosts
        ts['unique_hosts'] = unique_hosts

        # Fill NaN từ std calculation (khi chỉ có 1 value)
        ts['bytes_std'] = ts['bytes_std'].fillna(0)

        # Tính error rate (tránh division by zero)
        ts['error_rate'] = np.where(
            ts['request_count'] > 0,
            ts['error_count'] / ts['request_count'],
            0
        )

        # Tính average request size
        ts['avg_request_size'] = np.where(
            ts['request_count'] > 0,
            ts['bytes_total'] / ts['request_count'],
            0
        )

        if fill_gaps:
            # Tạo complete time range
            full_range = pd.date_range(
                start=ts.index.min(),
                end=ts.index.max(),
                freq=freq
            )

            # Reindex và fill missing với 0
            ts = ts.reindex(full_range, fill_value=0)

        if mark_storm:
            # Đánh dấu storm period
            ts['is_storm_period'] = (
                (ts.index >= self.STORM_START) &
                (ts.index <= self.STORM_END)
            ).astype(int)

        return ts

    def create_all_granularities(
        self,
        fill_gaps: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Tạo time series với tất cả các granularities cần thiết.

        Args:
            fill_gaps: Fill missing timestamps với 0

        Returns:
            Dict với keys: '1min', '5min', '15min'
                  values: DataFrame tương ứng
        """
        print("Tạo time series với các granularities...")

        granularities = {
            '1min': '1min',
            '5min': '5min',
            '15min': '15min'
        }

        results = {}

        for name, freq in granularities.items():
            print(f"  - Đang xử lý {name}...", end=" ")
            ts = self.aggregate_timeseries(freq, fill_gaps=fill_gaps)
            results[name] = ts
            print(f"Done! ({len(ts):,} records)")

        return results

    def get_data_summary(self) -> Dict:
        """
        Lấy thống kê tổng quan về dữ liệu.

        Returns:
            Dict với các thống kê chính
        """
        df = self.df

        return {
            'total_records': len(df),
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max(),
                'duration_days': (df.index.max() - df.index.min()).days
            },
            'requests': {
                'total': len(df),
                'success_2xx': (df['status_code'].between(200, 299)).sum(),
                'redirect_3xx': (df['status_code'].between(300, 399)).sum(),
                'client_error_4xx': (df['status_code'].between(400, 499)).sum(),
                'server_error_5xx': (df['status_code'].between(500, 599)).sum()
            },
            'bytes': {
                'total': df['bytes'].sum(),
                'mean': df['bytes'].mean(),
                'median': df['bytes'].median(),
                'max': df['bytes'].max()
            },
            'unique_hosts': df['host'].nunique(),
            'methods': df['method'].value_counts().to_dict(),
            'storm_period': {
                'start': self.STORM_START,
                'end': self.STORM_END,
                'duration_hours': (self.STORM_END - self.STORM_START).total_seconds() / 3600
            }
        }

    def detect_gaps(self, freq: str = '1min', threshold_minutes: int = 5) -> pd.DataFrame:
        """
        Phát hiện các gaps trong dữ liệu (ngoài storm period).

        Args:
            freq: Frequency để check gaps
            threshold_minutes: Số phút tối thiểu để coi là gap

        Returns:
            DataFrame với các gaps phát hiện được
        """
        ts = self.aggregate_timeseries(freq, fill_gaps=False)

        # Tính time diff giữa các records liên tiếp
        time_diff = ts.index.to_series().diff()

        # Convert threshold sang timedelta
        threshold = pd.Timedelta(minutes=threshold_minutes)

        # Tìm gaps
        gaps = time_diff[time_diff > threshold]

        if len(gaps) > 0:
            gap_info = []
            for end_time in gaps.index:
                start_time = ts.index[ts.index.get_loc(end_time) - 1]
                duration = (end_time - start_time).total_seconds() / 60

                # Check nếu là storm period
                is_storm = (start_time >= self.STORM_START - pd.Timedelta(hours=1) and
                           end_time <= self.STORM_END + pd.Timedelta(hours=1))

                gap_info.append({
                    'gap_start': start_time,
                    'gap_end': end_time,
                    'duration_minutes': duration,
                    'is_storm_gap': is_storm
                })

            return pd.DataFrame(gap_info)

        return pd.DataFrame()

    def save_processed(
        self,
        output_dir: str,
        granularities: Dict[str, pd.DataFrame] = None
    ):
        """
        Lưu processed data dưới dạng parquet files.

        Args:
            output_dir: Thư mục output
            granularities: Dict các DataFrame (nếu None, sẽ tạo mới)
        """
        import os

        # Tạo thư mục nếu chưa có
        os.makedirs(output_dir, exist_ok=True)

        # Tạo granularities nếu chưa có
        if granularities is None:
            granularities = self.create_all_granularities()

        # Lưu từng file
        for name, df in granularities.items():
            filepath = os.path.join(output_dir, f'timeseries_{name}.parquet')
            df.to_parquet(filepath, engine='pyarrow')
            print(f"Đã lưu: {filepath} ({len(df):,} records)")

        print(f"\nĐã lưu {len(granularities)} files vào {output_dir}")


def load_timeseries(filepath: str) -> pd.DataFrame:
    """
    Load time series từ parquet file.

    Args:
        filepath: Đường dẫn tới file parquet

    Returns:
        DataFrame với DatetimeIndex
    """
    df = pd.read_parquet(filepath)

    # Đảm bảo index là DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index)

    return df


def split_train_test(
    df: pd.DataFrame,
    test_start: str = '1995-08-23'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series thành train và test sets.

    Theo đề bài:
        - Train: Jul 1995 + Aug 1-22, 1995
        - Test: Aug 23-31, 1995

    Args:
        df: DataFrame với DatetimeIndex
        test_start: Ngày bắt đầu test set

    Returns:
        Tuple (train_df, test_df)
    """
    test_start_ts = pd.Timestamp(test_start)

    train = df[df.index < test_start_ts]
    test = df[df.index >= test_start_ts]

    return train, test


if __name__ == "__main__":
    # Demo usage
    from parser import NASALogParser

    # Parse data
    print("Parsing log file...")
    parser = NASALogParser()
    df = parser.parse_file("DATA/train.txt")

    # Preprocess
    print("\nPreprocessing...")
    preprocessor = LogPreprocessor(df)

    # Show summary
    summary = preprocessor.get_data_summary()
    print(f"\nData Summary:")
    print(f"  Total records: {summary['total_records']:,}")
    print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"  Unique hosts: {summary['unique_hosts']:,}")

    # Create all granularities
    granularities = preprocessor.create_all_granularities()

    # Save
    preprocessor.save_processed('data/processed', granularities)

    # Show sample
    print("\nSample 1-min data:")
    print(granularities['1min'].head(10))
