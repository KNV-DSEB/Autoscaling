"""
Cost Analyzer Module
====================
Module phân tích và so sánh chi phí của các scaling strategies.

Phân tích:
    - Fixed provisioning vs Dynamic scaling
    - Cost breakdown
    - Savings calculations
    - ROI analysis

Usage:
    >>> analyzer = CostAnalyzer(server_config)
    >>> fixed_cost = analyzer.analyze_fixed_provisioning(demand, servers=10)
    >>> comparison = analyzer.compare_strategies(demand, strategies)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .policy import ServerConfig


@dataclass
class CostMetrics:
    """
    Chi phí và metrics cho một strategy.

    Attributes:
        total_server_hours: Tổng server-hours sử dụng
        total_cost: Tổng chi phí ($)
        average_servers: Số servers trung bình
        peak_servers: Số servers cao nhất
        min_servers: Số servers thấp nhất
        overprovisioning_cost: Chi phí do over-provisioning
        underprovisioning_events: Số lần under-provisioned (overload)
        dropped_requests: Số requests bị drop do overload
        sla_violation_rate: % thời gian vi phạm SLA
    """
    total_server_hours: float
    total_cost: float
    average_servers: float
    peak_servers: int
    min_servers: int
    overprovisioning_cost: float
    underprovisioning_events: int
    dropped_requests: int
    sla_violation_rate: float


class CostAnalyzer:
    """
    Phân tích chi phí cho autoscaling.

    Cho phép:
        - So sánh fixed vs dynamic provisioning
        - Tính toán savings
        - Phân tích ROI của predictive scaling

    Attributes:
        server_config: ServerConfig với cost thông tin
        penalty_per_dropped_request: Chi phí penalty cho mỗi dropped request

    Example:
        >>> analyzer = CostAnalyzer(ServerConfig())
        >>> fixed = analyzer.analyze_fixed_provisioning(demand_series, 10)
        >>> dynamic = analyzer.analyze_dynamic_provisioning(demand_series, server_series)
        >>> print(f"Savings: ${fixed.total_cost - dynamic.total_cost:.2f}")
    """

    def __init__(
        self,
        server_config: ServerConfig,
        penalty_per_dropped_request: float = 0.001
    ):
        """
        Khởi tạo cost analyzer.

        Args:
            server_config: Cấu hình server với cost
            penalty_per_dropped_request: Penalty cost cho mỗi dropped request
        """
        self.server_config = server_config
        self.penalty_per_dropped_request = penalty_per_dropped_request

    def _calculate_interval_hours(self, series_length: int, freq_minutes: int = 15) -> float:
        """Tính tổng số giờ từ số intervals."""
        return series_length * freq_minutes / 60

    def analyze_fixed_provisioning(
        self,
        demand_series: pd.Series,
        fixed_servers: int,
        freq_minutes: int = 15
    ) -> CostMetrics:
        """
        Phân tích chi phí cho fixed server allocation.

        Args:
            demand_series: Series requests per interval
            fixed_servers: Số servers cố định
            freq_minutes: Số phút mỗi interval

        Returns:
            CostMetrics instance
        """
        total_hours = self._calculate_interval_hours(len(demand_series), freq_minutes)
        capacity = fixed_servers * self.server_config.max_requests_per_min * freq_minutes

        # Calculate metrics
        total_server_hours = fixed_servers * total_hours
        total_cost = total_server_hours * self.server_config.cost_per_server_hour

        # Overprovisioning (unused capacity)
        used_capacity = demand_series.sum()
        total_capacity = capacity * len(demand_series)
        unused = max(0, total_capacity - used_capacity)
        overprov_ratio = unused / total_capacity if total_capacity > 0 else 0
        overprov_cost = total_cost * overprov_ratio

        # Underprovisioning (overload)
        overloaded = (demand_series > capacity)
        underprov_events = overloaded.sum()
        dropped = (demand_series - capacity).clip(lower=0).sum()
        sla_violation = underprov_events / len(demand_series) * 100

        return CostMetrics(
            total_server_hours=total_server_hours,
            total_cost=total_cost,
            average_servers=fixed_servers,
            peak_servers=fixed_servers,
            min_servers=fixed_servers,
            overprovisioning_cost=overprov_cost,
            underprovisioning_events=int(underprov_events),
            dropped_requests=int(dropped),
            sla_violation_rate=sla_violation
        )

    def analyze_dynamic_provisioning(
        self,
        demand_series: pd.Series,
        server_series: pd.Series,
        freq_minutes: int = 15
    ) -> CostMetrics:
        """
        Phân tích chi phí cho dynamic (autoscaled) provisioning.

        Args:
            demand_series: Series requests per interval
            server_series: Series số servers tại mỗi interval
            freq_minutes: Số phút mỗi interval

        Returns:
            CostMetrics instance
        """
        # Ensure aligned
        assert len(demand_series) == len(server_series), "Series must have same length"

        total_hours = self._calculate_interval_hours(len(demand_series), freq_minutes)

        # Calculate capacity at each point
        capacity_series = server_series * self.server_config.max_requests_per_min * freq_minutes

        # Server-hours (sum of servers across all intervals, converted to hours)
        total_server_intervals = server_series.sum()
        total_server_hours = total_server_intervals * freq_minutes / 60

        total_cost = total_server_hours * self.server_config.cost_per_server_hour

        # Overprovisioning
        unused = (capacity_series - demand_series).clip(lower=0).sum()
        total_capacity = capacity_series.sum()
        overprov_ratio = unused / total_capacity if total_capacity > 0 else 0
        overprov_cost = total_cost * overprov_ratio

        # Underprovisioning
        overloaded = demand_series > capacity_series
        underprov_events = overloaded.sum()
        dropped = (demand_series - capacity_series).clip(lower=0).sum()
        sla_violation = underprov_events / len(demand_series) * 100

        return CostMetrics(
            total_server_hours=total_server_hours,
            total_cost=total_cost,
            average_servers=server_series.mean(),
            peak_servers=int(server_series.max()),
            min_servers=int(server_series.min()),
            overprovisioning_cost=overprov_cost,
            underprovisioning_events=int(underprov_events),
            dropped_requests=int(dropped),
            sla_violation_rate=sla_violation
        )

    def calculate_optimal_fixed_servers(
        self,
        demand_series: pd.Series,
        sla_target: float = 99.0,
        freq_minutes: int = 15
    ) -> Tuple[int, CostMetrics]:
        """
        Tìm số servers tối ưu cho fixed provisioning.

        Args:
            demand_series: Demand series
            sla_target: Target SLA % (vd: 99.0 = 99%)
            freq_minutes: Interval duration

        Returns:
            Tuple (optimal_servers, cost_metrics)
        """
        capacity_per_server = self.server_config.max_requests_per_min * freq_minutes
        max_demand = demand_series.max()

        # Start from minimum needed for peak
        min_servers = int(np.ceil(max_demand / capacity_per_server))

        for servers in range(min_servers, self.server_config.max_servers + 1):
            metrics = self.analyze_fixed_provisioning(
                demand_series, servers, freq_minutes
            )

            if 100 - metrics.sla_violation_rate >= sla_target:
                return servers, metrics

        # If can't meet SLA, return max servers
        return self.server_config.max_servers, self.analyze_fixed_provisioning(
            demand_series, self.server_config.max_servers, freq_minutes
        )

    def compare_strategies(
        self,
        demand_series: pd.Series,
        strategies: Dict[str, pd.Series],
        freq_minutes: int = 15
    ) -> pd.DataFrame:
        """
        So sánh nhiều scaling strategies.

        Args:
            demand_series: Actual demand
            strategies: Dict của {name: server_series}
                       Sử dụng int cho fixed strategies
            freq_minutes: Interval duration

        Returns:
            DataFrame so sánh các strategies
        """
        results = []

        for name, server_series in strategies.items():
            if isinstance(server_series, (int, float)):
                # Fixed provisioning
                metrics = self.analyze_fixed_provisioning(
                    demand_series,
                    int(server_series),
                    freq_minutes
                )
            else:
                # Dynamic provisioning
                metrics = self.analyze_dynamic_provisioning(
                    demand_series,
                    server_series,
                    freq_minutes
                )

            results.append({
                'Strategy': name,
                'Total Cost ($)': metrics.total_cost,
                'Avg Servers': metrics.average_servers,
                'Peak Servers': metrics.peak_servers,
                'Min Servers': metrics.min_servers,
                'Overprov Cost ($)': metrics.overprovisioning_cost,
                'Underprov Events': metrics.underprovisioning_events,
                'Dropped Requests': metrics.dropped_requests,
                'SLA Violation (%)': metrics.sla_violation_rate,
                'Server Hours': metrics.total_server_hours
            })

        df = pd.DataFrame(results)

        # Add savings vs baseline (first strategy)
        baseline_cost = df.iloc[0]['Total Cost ($)']
        df['Savings ($)'] = baseline_cost - df['Total Cost ($)']
        df['Savings (%)'] = (df['Savings ($)'] / baseline_cost * 100).round(2)

        return df.sort_values('Total Cost ($)')

    def calculate_roi(
        self,
        baseline_cost: float,
        optimized_cost: float,
        implementation_cost: float = 0,
        period_months: int = 12
    ) -> Dict:
        """
        Tính ROI của việc implement autoscaling.

        Args:
            baseline_cost: Chi phí baseline (fixed provisioning) cho period
            optimized_cost: Chi phí sau optimization
            implementation_cost: Chi phí implement solution
            period_months: Số tháng để tính

        Returns:
            Dict với ROI metrics
        """
        monthly_savings = (baseline_cost - optimized_cost) / period_months
        total_savings = baseline_cost - optimized_cost - implementation_cost
        payback_months = implementation_cost / monthly_savings if monthly_savings > 0 else float('inf')
        roi_pct = (total_savings / implementation_cost * 100) if implementation_cost > 0 else float('inf')

        return {
            'baseline_cost': baseline_cost,
            'optimized_cost': optimized_cost,
            'implementation_cost': implementation_cost,
            'total_savings': total_savings,
            'monthly_savings': monthly_savings,
            'payback_months': payback_months,
            'roi_percent': roi_pct
        }

    def generate_cost_report(
        self,
        demand_series: pd.Series,
        server_series: pd.Series,
        freq_minutes: int = 15
    ) -> str:
        """
        Generate text report cho cost analysis.

        Args:
            demand_series: Demand data
            server_series: Server allocation data
            freq_minutes: Interval duration

        Returns:
            Formatted report string
        """
        # Calculate various provisioning costs
        max_demand = demand_series.max()
        capacity_per_server = self.server_config.max_requests_per_min * freq_minutes

        peak_servers = int(np.ceil(max_demand / capacity_per_server))
        p90_servers = int(np.ceil(demand_series.quantile(0.9) / capacity_per_server))
        avg_servers = int(np.ceil(demand_series.mean() / capacity_per_server))

        strategies = {
            'Fixed (Peak)': peak_servers,
            'Fixed (P90)': p90_servers,
            'Fixed (Average)': avg_servers,
            'Autoscaled': server_series
        }

        comparison = self.compare_strategies(demand_series, strategies, freq_minutes)

        report = f"""
{'='*70}
                    COST ANALYSIS REPORT
{'='*70}

CONFIGURATION:
  Server Capacity: {self.server_config.max_requests_per_min:,} requests/min
  Cost per Server-Hour: ${self.server_config.cost_per_server_hour:.2f}
  Analysis Period: {len(demand_series)} intervals ({self._calculate_interval_hours(len(demand_series), freq_minutes):.1f} hours)

DEMAND STATISTICS:
  Total Requests: {demand_series.sum():,.0f}
  Average Demand: {demand_series.mean():,.1f} / interval
  Peak Demand: {demand_series.max():,.0f} / interval
  P90 Demand: {demand_series.quantile(0.9):,.0f} / interval

STRATEGY COMPARISON:
{comparison.to_string(index=False)}

KEY FINDINGS:
"""
        # Find best strategy
        best = comparison.iloc[0]
        worst = comparison.iloc[-1]

        report += f"""
  - Best Strategy: {best['Strategy']}
    - Total Cost: ${best['Total Cost ($)']:.2f}
    - Avg Servers: {best['Avg Servers']:.1f}
    - SLA Violations: {best['SLA Violation (%)']:.2f}%

  - Highest Cost: {worst['Strategy']}
    - Total Cost: ${worst['Total Cost ($)']:.2f}
    - Max Savings vs this: ${worst['Total Cost ($)'] - best['Total Cost ($)']:.2f}

{'='*70}
"""
        return report


if __name__ == "__main__":
    # Demo
    import sys
    import os
    sys.path.insert(0, os.path.abspath('..'))

    from src.data.preprocessor import load_timeseries

    # Load data
    print("Loading data...")
    df = load_timeseries('../data/processed/timeseries_15min.parquet')

    # Use subset
    sample = df.iloc[:96*7]  # 1 week
    demand = sample['request_count']

    # Create analyzer
    config = ServerConfig(max_requests_per_min=100)
    analyzer = CostAnalyzer(config)

    # Generate simulated server series (example)
    np.random.seed(42)
    server_series = pd.Series(
        np.clip(np.ceil(demand / 1500) + np.random.randint(-1, 2, len(demand)), 1, 20),
        index=demand.index
    ).astype(int)

    # Generate report
    report = analyzer.generate_cost_report(demand, server_series)
    print(report)
