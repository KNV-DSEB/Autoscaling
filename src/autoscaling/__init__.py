"""
Autoscaling Module
==================
Hệ thống autoscaling với reactive và predictive scaling.

Classes:
- AutoscalingEngine: Engine chính xử lý scaling decisions
- ServerConfig: Cấu hình server capacity
- ScalingPolicy: Cấu hình scaling policy
- AutoscalingSimulator: Simulator cho testing
- CostAnalyzer: Phân tích chi phí

Enums:
- ScaleAction: NONE, SCALE_OUT, SCALE_IN
"""

from .policy import (
    AutoscalingEngine,
    ServerConfig,
    ScalingPolicy,
    ScaleAction
)
from .simulator import AutoscalingSimulator
from .cost_analyzer import CostAnalyzer, CostMetrics

__all__ = [
    'AutoscalingEngine',
    'ServerConfig',
    'ScalingPolicy',
    'ScaleAction',
    'AutoscalingSimulator',
    'CostAnalyzer',
    'CostMetrics'
]
