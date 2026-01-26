"""
Autoscaling Simulator
=====================
Module simulate autoscaling behavior trên historical hoặc predicted data.

Cho phép:
    - Test các scaling policies khác nhau
    - So sánh reactive vs predictive scaling
    - Phân tích performance và cost

Usage:
    >>> simulator = AutoscalingSimulator(config, policy)
    >>> results = simulator.simulate(actual_data, predictions)
    >>> events = simulator.get_scaling_events(results)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from .policy import AutoscalingEngine, ServerConfig, ScalingPolicy


class AutoscalingSimulator:
    """
    Simulator cho autoscaling.

    Chạy simulation trên historical data để:
        - Đánh giá hiệu quả của policy
        - So sánh các configurations khác nhau
        - Tính toán chi phí và performance

    Attributes:
        server_config: ServerConfig
        policy: ScalingPolicy
        forecaster: Optional forecaster function

    Example:
        >>> sim = AutoscalingSimulator(ServerConfig(), ScalingPolicy())
        >>> results = sim.simulate(df, predictions=pred_df)
        >>> print(f"Total cost: ${sim.calculate_total_cost(results):.2f}")
    """

    def __init__(
        self,
        server_config: ServerConfig,
        policy: ScalingPolicy,
        forecaster=None
    ):
        """
        Khởi tạo simulator.

        Args:
            server_config: Cấu hình server
            policy: Cấu hình policy
            forecaster: Optional function để generate predictions
        """
        self.server_config = server_config
        self.policy = policy
        self.forecaster = forecaster

    def simulate(
        self,
        actual_data: pd.DataFrame,
        predictions: Optional[pd.DataFrame] = None,
        use_predictive: bool = True,
        initial_servers: int = None
    ) -> pd.DataFrame:
        """
        Chạy autoscaling simulation.

        Args:
            actual_data: DataFrame với 'request_count' và 'bytes_total'
            predictions: DataFrame với predicted values
            use_predictive: Sử dụng predictive scaling
            initial_servers: Số servers ban đầu

        Returns:
            DataFrame với simulation results:
                - timestamp
                - actual_requests, actual_bytes
                - servers
                - action
                - trigger
                - utilization
                - capacity
                - is_overloaded
        """
        engine = AutoscalingEngine(
            self.server_config,
            self.policy,
            use_predictive=use_predictive,
            initial_servers=initial_servers
        )

        results = []
        prediction_horizon = self.policy.prediction_horizon

        for i, (timestamp, row) in enumerate(actual_data.iterrows()):
            # Get prediction if available
            pred_requests = None
            pred_bytes = None

            if predictions is not None and use_predictive:
                # Look ahead by prediction_horizon
                # Assuming index is aligned
                future_idx = min(i + prediction_horizon, len(predictions) - 1)
                if future_idx < len(predictions):
                    pred_row = predictions.iloc[future_idx]
                    pred_requests = pred_row.get('request_count', pred_row.get('forecast', None))
                    pred_bytes = pred_row.get('bytes_total', None)

            # Execute one step
            step_result = engine.step(
                current_time=timestamp,
                actual_requests=row['request_count'],
                actual_bytes=row['bytes_total'],
                predicted_requests=pred_requests,
                predicted_bytes=pred_bytes
            )

            # Calculate capacity
            capacity = step_result['servers'] * self.server_config.max_requests_per_min

            # Check if overloaded
            is_overloaded = row['request_count'] > capacity

            # Calculate dropped requests if overloaded
            dropped = max(0, row['request_count'] - capacity) if is_overloaded else 0

            results.append({
                'timestamp': timestamp,
                'actual_requests': row['request_count'],
                'actual_bytes': row['bytes_total'],
                'servers': step_result['servers'],
                'action': step_result['action'],
                'trigger': step_result['trigger'],
                'utilization': step_result['utilization'],
                'capacity': capacity,
                'is_overloaded': is_overloaded,
                'dropped_requests': dropped
            })

        return pd.DataFrame(results).set_index('timestamp')

    def get_scaling_events(self, simulation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract scaling events từ simulation results.

        Args:
            simulation_df: DataFrame từ simulate()

        Returns:
            DataFrame chỉ chứa các scaling events
        """
        events = simulation_df[simulation_df['action'] != 'none'].copy()
        events['server_change'] = events['servers'].diff().fillna(0).astype(int)
        return events

    def calculate_metrics(self, simulation_df: pd.DataFrame) -> Dict:
        """
        Tính các metrics từ simulation.

        Args:
            simulation_df: DataFrame từ simulate()

        Returns:
            Dict với các metrics
        """
        total_minutes = len(simulation_df)
        total_hours = total_minutes / 60 if 'request_count' in simulation_df.columns else total_minutes / 4  # 15-min intervals

        # Server metrics
        total_server_minutes = simulation_df['servers'].sum()
        total_server_hours = total_server_minutes / 60 if len(simulation_df) > 100 else total_server_minutes / 4
        avg_servers = simulation_df['servers'].mean()
        max_servers = simulation_df['servers'].max()
        min_servers = simulation_df['servers'].min()

        # Cost
        total_cost = total_server_hours * self.server_config.cost_per_server_hour

        # Scaling events
        events = self.get_scaling_events(simulation_df)
        scale_out_count = (events['action'] == 'scale_out').sum() if len(events) > 0 else 0
        scale_in_count = (events['action'] == 'scale_in').sum() if len(events) > 0 else 0

        # Performance
        overloaded_count = simulation_df['is_overloaded'].sum()
        overload_rate = overloaded_count / len(simulation_df) * 100

        total_dropped = simulation_df['dropped_requests'].sum()
        total_requests = simulation_df['actual_requests'].sum()
        drop_rate = total_dropped / total_requests * 100 if total_requests > 0 else 0

        # Utilization
        avg_utilization = simulation_df['utilization'].mean()
        max_utilization = simulation_df['utilization'].max()

        return {
            'total_hours': total_hours,
            'total_server_hours': total_server_hours,
            'total_cost': total_cost,
            'avg_servers': avg_servers,
            'max_servers': max_servers,
            'min_servers': min_servers,
            'scale_out_count': scale_out_count,
            'scale_in_count': scale_in_count,
            'total_scaling_events': scale_out_count + scale_in_count,
            'overloaded_periods': overloaded_count,
            'overload_rate_pct': overload_rate,
            'total_dropped_requests': total_dropped,
            'drop_rate_pct': drop_rate,
            'avg_utilization': avg_utilization,
            'max_utilization': max_utilization
        }

    def compare_strategies(
        self,
        actual_data: pd.DataFrame,
        predictions: Optional[pd.DataFrame] = None,
        strategies: Dict[str, dict] = None
    ) -> pd.DataFrame:
        """
        So sánh nhiều strategies khác nhau.

        Args:
            actual_data: Actual traffic data
            predictions: Predicted traffic
            strategies: Dict của {name: {'use_predictive': bool, 'policy': ScalingPolicy}}

        Returns:
            DataFrame so sánh các strategies
        """
        if strategies is None:
            # Default strategies
            strategies = {
                'Reactive Only': {
                    'use_predictive': False,
                    'policy': self.policy
                },
                'Predictive': {
                    'use_predictive': True,
                    'policy': self.policy
                },
                'Aggressive Scale-out': {
                    'use_predictive': True,
                    'policy': ScalingPolicy(
                        scale_out_threshold=0.7,
                        scale_in_threshold=0.3,
                        consecutive_breaches=2
                    )
                },
                'Conservative': {
                    'use_predictive': True,
                    'policy': ScalingPolicy(
                        scale_out_threshold=0.9,
                        scale_in_threshold=0.2,
                        consecutive_breaches=5
                    )
                }
            }

        results = []

        for name, config in strategies.items():
            print(f"Simulating: {name}...")

            sim = AutoscalingSimulator(
                self.server_config,
                config['policy']
            )

            sim_results = sim.simulate(
                actual_data,
                predictions,
                use_predictive=config['use_predictive']
            )

            metrics = sim.calculate_metrics(sim_results)
            metrics['strategy'] = name
            results.append(metrics)

        df = pd.DataFrame(results)

        # Reorder columns
        cols = ['strategy'] + [c for c in df.columns if c != 'strategy']
        df = df[cols]

        # Sort by cost
        df = df.sort_values('total_cost')

        return df

    def run_sensitivity_analysis(
        self,
        actual_data: pd.DataFrame,
        predictions: Optional[pd.DataFrame] = None,
        param_name: str = 'scale_out_threshold',
        param_values: List = None
    ) -> pd.DataFrame:
        """
        Chạy sensitivity analysis cho một parameter.

        Args:
            actual_data: Actual data
            predictions: Predictions
            param_name: Tên parameter để vary
            param_values: Các giá trị để test

        Returns:
            DataFrame với results cho mỗi parameter value
        """
        if param_values is None:
            if param_name == 'scale_out_threshold':
                param_values = [0.6, 0.7, 0.8, 0.9, 0.95]
            elif param_name == 'scale_in_threshold':
                param_values = [0.1, 0.2, 0.3, 0.4, 0.5]
            elif param_name == 'cooldown_period':
                param_values = [1, 3, 5, 10, 15]
            elif param_name == 'consecutive_breaches':
                param_values = [1, 2, 3, 5, 10]
            else:
                param_values = [0.5, 1.0, 1.5, 2.0]

        results = []

        for value in param_values:
            print(f"Testing {param_name}={value}...")

            # Create policy with modified parameter
            policy_dict = {
                'scale_out_threshold': self.policy.scale_out_threshold,
                'scale_in_threshold': self.policy.scale_in_threshold,
                'cooldown_period': self.policy.cooldown_period,
                'consecutive_breaches': self.policy.consecutive_breaches,
                'scale_out_increment': self.policy.scale_out_increment,
                'scale_in_decrement': self.policy.scale_in_decrement,
                'prediction_horizon': self.policy.prediction_horizon,
                'predictive_buffer': self.policy.predictive_buffer
            }
            policy_dict[param_name] = value

            test_policy = ScalingPolicy(**policy_dict)

            sim = AutoscalingSimulator(self.server_config, test_policy)
            sim_results = sim.simulate(actual_data, predictions)
            metrics = sim.calculate_metrics(sim_results)
            metrics[param_name] = value
            results.append(metrics)

        return pd.DataFrame(results)


if __name__ == "__main__":
    # Demo
    import sys
    import os
    sys.path.insert(0, os.path.abspath('..'))

    from src.data.preprocessor import load_timeseries

    # Load data
    print("Loading data...")
    df = load_timeseries('../data/processed/timeseries_15min.parquet')

    # Use subset for demo
    sample = df.iloc[:96*7]  # 1 week

    # Create simulator
    config = ServerConfig(max_requests_per_min=100)
    policy = ScalingPolicy()
    simulator = AutoscalingSimulator(config, policy)

    # Run simulation
    print("\nRunning simulation...")
    results = simulator.simulate(sample, use_predictive=False)

    # Get metrics
    metrics = simulator.calculate_metrics(results)
    print("\nSimulation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Scaling events
    events = simulator.get_scaling_events(results)
    print(f"\nScaling Events: {len(events)}")
    if len(events) > 0:
        print(events[['action', 'servers', 'utilization']].head(10))
