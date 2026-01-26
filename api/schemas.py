"""
API Schemas
===========
Pydantic schemas cho FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ModelType(str, Enum):
    """Enum cho các model types."""
    SARIMA = "sarima"
    LIGHTGBM = "lightgbm"
    PROPHET = "prophet"


class Granularity(str, Enum):
    """Enum cho time granularity."""
    ONE_MIN = "1min"
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"


class TargetType(str, Enum):
    """Enum cho target variables."""
    REQUESTS = "request_count"
    BYTES = "bytes_total"


# =============================================================================
# Forecast Schemas
# =============================================================================

class ForecastRequest(BaseModel):
    """Request schema cho forecast endpoint."""
    model: ModelType = Field(
        default=ModelType.LIGHTGBM,
        description="Model sử dụng cho forecasting"
    )
    horizon: int = Field(
        default=96,
        ge=1,
        le=672,
        description="Số intervals cần forecast (max 7 ngày)"
    )
    granularity: Granularity = Field(
        default=Granularity.FIFTEEN_MIN,
        description="Time granularity"
    )
    target: TargetType = Field(
        default=TargetType.REQUESTS,
        description="Target variable để forecast"
    )
    include_confidence: bool = Field(
        default=True,
        description="Include confidence intervals"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "lightgbm",
                "horizon": 96,
                "granularity": "15min",
                "target": "request_count",
                "include_confidence": True
            }
        }


class ForecastPoint(BaseModel):
    """Single forecast point."""
    timestamp: datetime
    value: float
    lower: Optional[float] = None
    upper: Optional[float] = None


class ForecastResponse(BaseModel):
    """Response schema cho forecast endpoint."""
    model: str
    granularity: str
    target: str
    horizon: int
    forecasts: List[ForecastPoint]
    generated_at: datetime
    metrics: Optional[Dict[str, float]] = None


# =============================================================================
# Scaling Schemas
# =============================================================================

class ScalingAction(str, Enum):
    """Scaling actions."""
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    MAINTAIN = "maintain"


class ScalingRecommendRequest(BaseModel):
    """Request schema cho scaling recommendation."""
    current_requests: float = Field(
        ge=0,
        description="Current requests per interval"
    )
    current_servers: int = Field(
        ge=1,
        description="Current number of servers"
    )
    predicted_requests: Optional[float] = Field(
        default=None,
        description="Predicted requests (từ forecast)"
    )
    server_capacity: int = Field(
        default=15000,
        description="Requests per server per interval (15min = 1000/min * 15)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "current_requests": 45000,
                "current_servers": 5,
                "predicted_requests": 52000,
                "server_capacity": 15000
            }
        }


class ScalingRecommendResponse(BaseModel):
    """Response schema cho scaling recommendation."""
    action: ScalingAction
    current_servers: int
    recommended_servers: int
    current_utilization: float
    predicted_utilization: Optional[float]
    reason: str
    confidence: float


# =============================================================================
# Metrics Schemas
# =============================================================================

class HistoricalMetricsRequest(BaseModel):
    """Request parameters cho historical metrics."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    granularity: Granularity = Granularity.FIFTEEN_MIN


class MetricsSummary(BaseModel):
    """Summary statistics cho metrics."""
    total_requests: int
    average_requests: float
    peak_requests: float
    min_requests: float
    total_bytes: int
    average_bytes: float
    error_rate: float
    unique_periods: int


class HistoricalMetricsResponse(BaseModel):
    """Response cho historical metrics."""
    summary: MetricsSummary
    data: List[Dict[str, Any]]
    start_date: datetime
    end_date: datetime
    granularity: str


# =============================================================================
# Simulation Schemas
# =============================================================================

class SimulationRequest(BaseModel):
    """Request cho autoscaling simulation."""
    strategy: str = Field(
        default="predictive",
        description="Scaling strategy: 'reactive' hoặc 'predictive'"
    )
    initial_servers: int = Field(
        default=5,
        ge=1,
        description="Số servers ban đầu"
    )
    scale_out_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Threshold để scale out"
    )
    scale_in_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=0.5,
        description="Threshold để scale in"
    )
    predictive_buffer: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Safety buffer cho predictive scaling"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "strategy": "predictive",
                "initial_servers": 5,
                "scale_out_threshold": 0.8,
                "scale_in_threshold": 0.3,
                "predictive_buffer": 0.2
            }
        }


class SimulationResponse(BaseModel):
    """Response cho autoscaling simulation."""
    strategy: str
    total_cost: float
    average_servers: float
    peak_servers: int
    min_servers: int
    scaling_events: int
    sla_violations: int
    dropped_requests: int
    timeline_summary: Dict[str, Any]


# =============================================================================
# Anomaly Schemas
# =============================================================================

class AnomalyDetectionRequest(BaseModel):
    """Request cho anomaly detection."""
    method: str = Field(
        default="combined",
        description="Detection method: zscore, iqr, rolling, combined"
    )
    threshold: float = Field(
        default=3.0,
        description="Detection threshold"
    )
    window: int = Field(
        default=96,
        description="Rolling window size (intervals)"
    )


class AnomalyPoint(BaseModel):
    """Single anomaly point."""
    timestamp: datetime
    value: float
    score: float
    type: str


class AnomalyDetectionResponse(BaseModel):
    """Response cho anomaly detection."""
    method: str
    total_anomalies: int
    anomaly_rate: float
    anomalies: List[AnomalyPoint]
    threshold_used: float


# =============================================================================
# Health & Info Schemas
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    models_loaded: Dict[str, bool]
    data_available: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    name: str
    type: str
    granularity: str
    target: str
    metrics: Dict[str, float]
    last_trained: Optional[datetime]
    features_count: Optional[int]


# =============================================================================
# Cost Analysis Schemas
# =============================================================================

class CostComparisonRequest(BaseModel):
    """Request cho cost comparison."""
    include_fixed_peak: bool = True
    include_fixed_p90: bool = True
    include_reactive: bool = True
    include_predictive: bool = True


class StrategyCost(BaseModel):
    """Cost cho một strategy."""
    strategy: str
    total_cost: float
    average_servers: float
    sla_violations: int
    savings_vs_peak: float
    savings_percent: float


class CostComparisonResponse(BaseModel):
    """Response cho cost comparison."""
    period_hours: float
    strategies: List[StrategyCost]
    best_strategy: str
    max_savings: float
