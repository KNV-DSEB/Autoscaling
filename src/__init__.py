"""
AUTOSCALING ANALYSIS PROJECT
============================
Hệ thống dự báo traffic và tự động điều chỉnh tài nguyên
dựa trên dữ liệu log NASA web server từ tháng 7-8/1995.

Modules:
- data: Parser và preprocessor cho log files
- features: Feature engineering cho time series
- models: SARIMA, LightGBM, Prophet forecasters
- autoscaling: Policy, simulator, cost analyzer
- anomaly: Anomaly detection (DDoS, spikes)
"""

__version__ = "1.0.0"
__author__ = "Autoscaling Analysis Team"
