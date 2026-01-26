"""
Autoscaling Policy Module
=========================
Module định nghĩa các rules và logic cho autoscaling.

Scaling Strategies:
    1. Reactive: Scale dựa trên metrics hiện tại
    2. Predictive: Scale dựa trên forecast

Anti-flapping mechanisms:
    - Cooldown period: Chờ sau mỗi scaling action
    - Hysteresis: Ngưỡng khác nhau cho scale-out và scale-in
    - Consecutive breaches: Yêu cầu nhiều lần vượt ngưỡng liên tiếp

Usage:
    >>> config = ServerConfig(max_requests_per_min=1000)
    >>> policy = ScalingPolicy(scale_out_threshold=0.8)
    >>> engine = AutoscalingEngine(config, policy)
    >>> result = engine.step(current_time, requests, bytes)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ScaleAction(Enum):
    """Các actions scaling có thể thực hiện."""
    NONE = "none"
    SCALE_OUT = "scale_out"  # Thêm servers
    SCALE_IN = "scale_in"    # Giảm servers


@dataclass
class ServerConfig:
    """
    Cấu hình capacity của server.

    Attributes:
        max_requests_per_min: Số requests tối đa mỗi server xử lý được / phút
        max_bytes_per_min: Bytes tối đa mỗi server xử lý được / phút
        min_servers: Số servers tối thiểu
        max_servers: Số servers tối đa
        cost_per_server_hour: Chi phí mỗi server / giờ (USD)
        startup_time_seconds: Thời gian khởi động server mới
    """
    max_requests_per_min: int = 1000
    max_bytes_per_min: int = 50_000_000  # 50MB/min
    min_servers: int = 1
    max_servers: int = 100
    cost_per_server_hour: float = 0.10  # USD
    startup_time_seconds: int = 60


@dataclass
class ScalingPolicy:
    """
    Cấu hình scaling policy.

    Attributes:
        scale_out_threshold: % capacity để trigger scale-out (vd: 0.8 = 80%)
        scale_in_threshold: % capacity để trigger scale-in (vd: 0.3 = 30%)
        cooldown_period: Số phút chờ giữa các scaling actions
        consecutive_breaches: Số lần vượt ngưỡng liên tiếp trước khi scale
        scale_out_increment: Số servers thêm khi scale-out
        scale_in_decrement: Số servers giảm khi scale-in
        prediction_horizon: Số phút nhìn trước cho predictive scaling
        predictive_buffer: Buffer % thêm vào predicted load (vd: 0.2 = 20%)
    """
    scale_out_threshold: float = 0.8
    scale_in_threshold: float = 0.3
    cooldown_period: int = 5  # minutes
    consecutive_breaches: int = 3
    scale_out_increment: int = 2
    scale_in_decrement: int = 1
    prediction_horizon: int = 15  # minutes
    predictive_buffer: float = 0.2  # 20% safety margin


@dataclass
class ScalingEvent:
    """Record của một scaling event."""
    timestamp: pd.Timestamp
    action: str
    old_servers: int
    new_servers: int
    trigger: str  # 'reactive' or 'predictive'
    reason: str
    utilization: float


class AutoscalingEngine:
    """
    Engine chính cho autoscaling decisions.

    Implements:
        - Reactive scaling dựa trên actual metrics
        - Predictive scaling dựa trên forecasts
        - Cooldown và hysteresis để tránh flapping
        - Tracking scaling history

    Attributes:
        server_config: ServerConfig instance
        policy: ScalingPolicy instance
        current_servers: Số servers hiện tại
        use_predictive: Sử dụng predictive scaling
        scaling_history: Lịch sử các scaling events

    Example:
        >>> engine = AutoscalingEngine(ServerConfig(), ScalingPolicy())
        >>> for timestamp, row in df.iterrows():
        ...     result = engine.step(timestamp, row['requests'], row['bytes'],
        ...                          predicted_requests=row['pred_requests'])
        ...     print(f"{timestamp}: {result['action']}, servers={result['servers']}")
    """

    def __init__(
        self,
        server_config: ServerConfig,
        policy: ScalingPolicy,
        use_predictive: bool = True,
        initial_servers: int = None
    ):
        """
        Khởi tạo autoscaling engine.

        Args:
            server_config: Cấu hình server
            policy: Cấu hình policy
            use_predictive: Sử dụng predictive scaling
            initial_servers: Số servers ban đầu (mặc định: min_servers)
        """
        self.server_config = server_config
        self.policy = policy
        self.use_predictive = use_predictive

        # State
        self.current_servers = initial_servers or server_config.min_servers
        self.last_scale_time = None
        self.breach_counter = 0
        self.breach_type = None  # 'out' hoặc 'in'

        # History
        self.scaling_history: List[ScalingEvent] = []

    def reset(self, initial_servers: int = None):
        """Reset engine về trạng thái ban đầu."""
        self.current_servers = initial_servers or self.server_config.min_servers
        self.last_scale_time = None
        self.breach_counter = 0
        self.breach_type = None
        self.scaling_history = []

    def calculate_utilization(
        self,
        requests: float,
        bytes_total: float
    ) -> Tuple[float, float, float]:
        """
        Tính utilization metrics.

        Args:
            requests: Số requests hiện tại
            bytes_total: Bytes hiện tại

        Returns:
            Tuple (request_util, bytes_util, max_util)
        """
        capacity_requests = self.current_servers * self.server_config.max_requests_per_min
        capacity_bytes = self.current_servers * self.server_config.max_bytes_per_min

        request_util = requests / capacity_requests if capacity_requests > 0 else 1.0
        bytes_util = bytes_total / capacity_bytes if capacity_bytes > 0 else 1.0
        max_util = max(request_util, bytes_util)

        return request_util, bytes_util, max_util

    def calculate_required_servers(
        self,
        requests: float,
        bytes_total: float,
        buffer: float = 0.0
    ) -> int:
        """
        Tính số servers cần thiết cho load.

        Args:
            requests: Số requests
            bytes_total: Bytes total
            buffer: Buffer % thêm vào

        Returns:
            Số servers cần thiết
        """
        servers_for_requests = np.ceil(
            requests * (1 + buffer) / self.server_config.max_requests_per_min
        )
        servers_for_bytes = np.ceil(
            bytes_total * (1 + buffer) / self.server_config.max_bytes_per_min
        )

        required = max(servers_for_requests, servers_for_bytes, 1)

        return int(np.clip(
            required,
            self.server_config.min_servers,
            self.server_config.max_servers
        ))

    def check_cooldown(self, current_time: pd.Timestamp) -> bool:
        """
        Kiểm tra đã hết cooldown period chưa.

        Args:
            current_time: Thời điểm hiện tại

        Returns:
            True nếu có thể scale
        """
        if self.last_scale_time is None:
            return True

        elapsed = (current_time - self.last_scale_time).total_seconds() / 60
        return elapsed >= self.policy.cooldown_period

    def _evaluate_reactive(
        self,
        requests: float,
        bytes_total: float,
        current_time: pd.Timestamp
    ) -> Tuple[ScaleAction, str]:
        """
        Đánh giá reactive scaling.

        Args:
            requests: Requests hiện tại
            bytes_total: Bytes hiện tại
            current_time: Thời điểm hiện tại

        Returns:
            Tuple (action, reason)
        """
        _, _, max_util = self.calculate_utilization(requests, bytes_total)

        # Check scale-out
        if max_util >= self.policy.scale_out_threshold:
            if self.breach_type == 'out':
                self.breach_counter += 1
            else:
                self.breach_type = 'out'
                self.breach_counter = 1

            if self.breach_counter >= self.policy.consecutive_breaches:
                if self.check_cooldown(current_time):
                    reason = f"Utilization {max_util:.1%} >= {self.policy.scale_out_threshold:.0%} for {self.breach_counter} consecutive periods"
                    return ScaleAction.SCALE_OUT, reason

        # Check scale-in
        elif max_util <= self.policy.scale_in_threshold:
            if self.current_servers > self.server_config.min_servers:
                if self.breach_type == 'in':
                    self.breach_counter += 1
                else:
                    self.breach_type = 'in'
                    self.breach_counter = 1

                if self.breach_counter >= self.policy.consecutive_breaches:
                    if self.check_cooldown(current_time):
                        reason = f"Utilization {max_util:.1%} <= {self.policy.scale_in_threshold:.0%} for {self.breach_counter} consecutive periods"
                        return ScaleAction.SCALE_IN, reason

        # Normal range - reset breach counter
        else:
            self.breach_counter = 0
            self.breach_type = None

        return ScaleAction.NONE, ""

    def _evaluate_predictive(
        self,
        predicted_requests: float,
        predicted_bytes: float,
        current_time: pd.Timestamp
    ) -> Tuple[ScaleAction, str]:
        """
        Đánh giá predictive scaling.

        Args:
            predicted_requests: Predicted requests
            predicted_bytes: Predicted bytes
            current_time: Thời điểm hiện tại

        Returns:
            Tuple (action, reason)
        """
        required_servers = self.calculate_required_servers(
            predicted_requests,
            predicted_bytes,
            buffer=self.policy.predictive_buffer
        )

        if required_servers > self.current_servers:
            if self.check_cooldown(current_time):
                reason = f"Predicted load requires {required_servers} servers (current: {self.current_servers})"
                return ScaleAction.SCALE_OUT, reason

        elif required_servers < self.current_servers - 1:  # Hysteresis for scale-in
            if self.check_cooldown(current_time):
                reason = f"Predicted load only needs {required_servers} servers (current: {self.current_servers})"
                return ScaleAction.SCALE_IN, reason

        return ScaleAction.NONE, ""

    def _execute_scaling(
        self,
        action: ScaleAction,
        current_time: pd.Timestamp,
        trigger: str,
        reason: str,
        utilization: float
    ) -> int:
        """
        Thực hiện scaling action.

        Args:
            action: ScaleAction
            current_time: Thời điểm
            trigger: 'reactive' hoặc 'predictive'
            reason: Lý do scale
            utilization: Current utilization

        Returns:
            Số servers mới
        """
        old_servers = self.current_servers

        if action == ScaleAction.SCALE_OUT:
            self.current_servers = min(
                self.current_servers + self.policy.scale_out_increment,
                self.server_config.max_servers
            )
        elif action == ScaleAction.SCALE_IN:
            self.current_servers = max(
                self.current_servers - self.policy.scale_in_decrement,
                self.server_config.min_servers
            )

        # Update state
        if self.current_servers != old_servers:
            self.last_scale_time = current_time
            self.breach_counter = 0
            self.breach_type = None

            # Record event
            event = ScalingEvent(
                timestamp=current_time,
                action=action.value,
                old_servers=old_servers,
                new_servers=self.current_servers,
                trigger=trigger,
                reason=reason,
                utilization=utilization
            )
            self.scaling_history.append(event)

        return self.current_servers

    def step(
        self,
        current_time: pd.Timestamp,
        actual_requests: float,
        actual_bytes: float,
        predicted_requests: Optional[float] = None,
        predicted_bytes: Optional[float] = None
    ) -> Dict:
        """
        Xử lý một time step của autoscaling.

        Args:
            current_time: Thời điểm hiện tại
            actual_requests: Requests thực tế
            actual_bytes: Bytes thực tế
            predicted_requests: Requests dự đoán (optional)
            predicted_bytes: Bytes dự đoán (optional)

        Returns:
            Dict với:
                - action: Action đã thực hiện
                - servers: Số servers hiện tại
                - trigger: 'reactive', 'predictive', hoặc None
                - reason: Lý do (nếu có action)
                - utilization: Current utilization
        """
        _, _, max_util = self.calculate_utilization(actual_requests, actual_bytes)

        # Try predictive scaling first
        if self.use_predictive and predicted_requests is not None:
            pred_bytes = predicted_bytes if predicted_bytes is not None else predicted_requests * 5000

            action, reason = self._evaluate_predictive(
                predicted_requests,
                pred_bytes,
                current_time
            )

            if action != ScaleAction.NONE:
                new_servers = self._execute_scaling(
                    action, current_time, 'predictive', reason, max_util
                )
                return {
                    'action': action.value,
                    'servers': new_servers,
                    'trigger': 'predictive',
                    'reason': reason,
                    'utilization': max_util
                }

        # Fall back to reactive scaling
        action, reason = self._evaluate_reactive(
            actual_requests,
            actual_bytes,
            current_time
        )

        if action != ScaleAction.NONE:
            new_servers = self._execute_scaling(
                action, current_time, 'reactive', reason, max_util
            )
            return {
                'action': action.value,
                'servers': new_servers,
                'trigger': 'reactive',
                'reason': reason,
                'utilization': max_util
            }

        return {
            'action': 'none',
            'servers': self.current_servers,
            'trigger': None,
            'reason': '',
            'utilization': max_util
        }

    def get_scaling_history(self) -> pd.DataFrame:
        """Lấy scaling history dưới dạng DataFrame."""
        if not self.scaling_history:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'timestamp': e.timestamp,
                'action': e.action,
                'old_servers': e.old_servers,
                'new_servers': e.new_servers,
                'trigger': e.trigger,
                'reason': e.reason,
                'utilization': e.utilization
            }
            for e in self.scaling_history
        ])

    def get_stats(self) -> Dict:
        """Lấy thống kê về scaling."""
        history_df = self.get_scaling_history()

        if len(history_df) == 0:
            return {
                'total_events': 0,
                'scale_out_count': 0,
                'scale_in_count': 0,
                'predictive_count': 0,
                'reactive_count': 0
            }

        return {
            'total_events': len(history_df),
            'scale_out_count': (history_df['action'] == 'scale_out').sum(),
            'scale_in_count': (history_df['action'] == 'scale_in').sum(),
            'predictive_count': (history_df['trigger'] == 'predictive').sum(),
            'reactive_count': (history_df['trigger'] == 'reactive').sum()
        }


if __name__ == "__main__":
    # Demo usage
    import sys
    import os
    sys.path.insert(0, os.path.abspath('..'))

    from src.data.preprocessor import load_timeseries

    # Load data
    print("Loading data...")
    df = load_timeseries('../data/processed/timeseries_15min.parquet')

    # Use 1 day of data for demo
    sample = df.iloc[:96]  # 24 hours * 4 (15-min intervals)

    # Create engine
    config = ServerConfig(max_requests_per_min=100)  # Lower for demo
    policy = ScalingPolicy(
        scale_out_threshold=0.8,
        scale_in_threshold=0.3,
        consecutive_breaches=2,
        cooldown_period=3
    )
    engine = AutoscalingEngine(config, policy, use_predictive=False)

    # Simulate
    print("\nSimulating autoscaling...")
    for timestamp, row in sample.iterrows():
        result = engine.step(
            current_time=timestamp,
            actual_requests=row['request_count'],
            actual_bytes=row['bytes_total']
        )

        if result['action'] != 'none':
            print(f"{timestamp}: {result['action']} -> {result['servers']} servers")
            print(f"  Reason: {result['reason']}")

    # Stats
    stats = engine.get_stats()
    print(f"\nScaling Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
