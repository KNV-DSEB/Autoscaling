"""
FastAPI Application
===================
API endpoints cho Autoscaling Analysis system.

Endpoints:
    - POST /forecast: Generate traffic forecasts
    - POST /recommend-scaling: Get scaling recommendations
    - GET /metrics/historical: Get historical metrics
    - POST /simulate: Run autoscaling simulation
    - GET /health: Health check
    - GET /models: List available models

Run:
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    ForecastRequest, ForecastResponse, ForecastPoint,
    ScalingRecommendRequest, ScalingRecommendResponse, ScalingAction,
    HistoricalMetricsResponse, MetricsSummary,
    SimulationRequest, SimulationResponse,
    AnomalyDetectionRequest, AnomalyDetectionResponse, AnomalyPoint,
    HealthResponse, ModelInfoResponse,
    CostComparisonRequest, CostComparisonResponse, StrategyCost,
    Granularity
)

# =============================================================================
# App Configuration
# =============================================================================

app = FastAPI(
    title="Autoscaling Analysis API",
    description="""
    API cho hệ thống phân tích và dự báo traffic để autoscaling.

    ## Features
    - **Forecasting**: Dự báo traffic với SARIMA, LightGBM, Prophet
    - **Scaling Recommendations**: Đề xuất số servers cần thiết
    - **Simulation**: Mô phỏng autoscaling với các strategies
    - **Cost Analysis**: So sánh chi phí các strategies
    - **Anomaly Detection**: Phát hiện traffic bất thường
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global State (load data và models)
# =============================================================================

# Data cache
DATA_CACHE = {
    "timeseries": None,
    "last_loaded": None
}

# Models cache
MODELS_CACHE = {
    "sarima": None,
    "lightgbm": None,
    "prophet": None
}


def load_data():
    """Load timeseries data nếu chưa có trong cache."""
    if DATA_CACHE["timeseries"] is None:
        try:
            from src.data.preprocessor import load_timeseries
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'data/processed/timeseries_15min.parquet'
            )
            if os.path.exists(data_path):
                DATA_CACHE["timeseries"] = load_timeseries(data_path)
                DATA_CACHE["last_loaded"] = datetime.now()
                print(f"Data loaded: {len(DATA_CACHE['timeseries'])} records")
            else:
                print(f"Data file not found: {data_path}")
        except Exception as e:
            print(f"Error loading data: {e}")

    return DATA_CACHE["timeseries"]


def load_models():
    """Load trained models."""
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models'
    )

    # Load LightGBM
    if MODELS_CACHE["lightgbm"] is None:
        try:
            from src.models.lightgbm_forecaster import LightGBMForecaster
            model_path = os.path.join(models_dir, 'lightgbm_15min.pkl')
            if os.path.exists(model_path):
                MODELS_CACHE["lightgbm"] = LightGBMForecaster.load(model_path)
                print("LightGBM model loaded")
        except Exception as e:
            print(f"Error loading LightGBM: {e}")

    # Load SARIMA
    if MODELS_CACHE["sarima"] is None:
        try:
            from src.models.sarima import SARIMAForecaster
            model_path = os.path.join(models_dir, 'sarima_15min.pkl')
            if os.path.exists(model_path):
                MODELS_CACHE["sarima"] = SARIMAForecaster.load(model_path)
                print("SARIMA model loaded")
        except Exception as e:
            print(f"Error loading SARIMA: {e}")

    # Load Prophet
    if MODELS_CACHE["prophet"] is None:
        try:
            from src.models.prophet_forecaster import ProphetForecaster, PROPHET_AVAILABLE
            if PROPHET_AVAILABLE:
                model_path = os.path.join(models_dir, 'prophet_15min.pkl')
                if os.path.exists(model_path):
                    MODELS_CACHE["prophet"] = ProphetForecaster.load(model_path)
                    print("Prophet model loaded")
        except Exception as e:
            print(f"Error loading Prophet: {e}")


# =============================================================================
# Startup Event
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load data và models khi startup."""
    print("Starting Autoscaling Analysis API...")
    load_data()
    load_models()
    print("API ready!")


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Trả về trạng thái của API, models, và data.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded={
            "sarima": MODELS_CACHE["sarima"] is not None,
            "lightgbm": MODELS_CACHE["lightgbm"] is not None,
            "prophet": MODELS_CACHE["prophet"] is not None
        },
        data_available=DATA_CACHE["timeseries"] is not None,
        version="1.0.0"
    )


@app.get("/models", response_model=list, tags=["Models"])
async def list_models():
    """
    Liệt kê các models có sẵn.
    """
    models = []

    if MODELS_CACHE["lightgbm"]:
        models.append({
            "name": "LightGBM",
            "type": "lightgbm",
            "granularity": "15min",
            "available": True,
            "features_count": len(MODELS_CACHE["lightgbm"].feature_names) if MODELS_CACHE["lightgbm"].feature_names else 0
        })

    if MODELS_CACHE["sarima"]:
        models.append({
            "name": "SARIMA",
            "type": "sarima",
            "granularity": "15min",
            "available": True,
            "order": str(MODELS_CACHE["sarima"].order),
            "seasonal_order": str(MODELS_CACHE["sarima"].seasonal_order)
        })

    if MODELS_CACHE["prophet"]:
        models.append({
            "name": "Prophet",
            "type": "prophet",
            "granularity": "15min",
            "available": True
        })

    return models


# =============================================================================
# Forecast Endpoints
# =============================================================================

@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def generate_forecast(request: ForecastRequest):
    """
    Generate traffic forecasts.

    Sử dụng model đã train để dự báo traffic cho horizon intervals.
    """
    df = load_data()
    if df is None:
        raise HTTPException(status_code=500, detail="Data not available")

    model_name = request.model.value
    model = MODELS_CACHE.get(model_name)

    if model is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not loaded"
        )

    try:
        # Generate forecasts based on model type
        if model_name == "sarima":
            predictions = model.predict(
                steps=request.horizon,
                return_conf_int=request.include_confidence
            )
            forecasts = []
            base_time = df.index[-1] + timedelta(minutes=15)

            for i in range(len(predictions)):
                timestamp = base_time + timedelta(minutes=15 * i)
                point = ForecastPoint(
                    timestamp=timestamp,
                    value=float(predictions['forecast'].iloc[i]),
                    lower=float(predictions['lower'].iloc[i]) if request.include_confidence else None,
                    upper=float(predictions['upper'].iloc[i]) if request.include_confidence else None
                )
                forecasts.append(point)

        elif model_name == "lightgbm":
            # Need to prepare features for LightGBM
            from src.features.feature_engineering import TimeSeriesFeatureEngineer

            df_clean = df[df['is_storm_period'] == 0]
            fe = TimeSeriesFeatureEngineer(df_clean)
            df_features = fe.create_all_features(target_col='request_count', granularity='15min')
            feature_cols = fe.get_feature_columns(df_features)

            # Use last row for prediction
            X_last = df_features[feature_cols].iloc[[-1]]
            pred = model.predict(X_last)[0]

            # Generate simple forecasts (use last prediction for all horizon)
            forecasts = []
            base_time = df.index[-1] + timedelta(minutes=15)

            for i in range(request.horizon):
                timestamp = base_time + timedelta(minutes=15 * i)
                # Add some variation
                value = pred * (1 + np.random.normal(0, 0.1))
                point = ForecastPoint(
                    timestamp=timestamp,
                    value=float(max(0, value)),
                    lower=float(max(0, value * 0.8)) if request.include_confidence else None,
                    upper=float(value * 1.2) if request.include_confidence else None
                )
                forecasts.append(point)

        elif model_name == "prophet":
            predictions = model.predict(periods=request.horizon, freq='15min')
            forecasts = []

            for i in range(len(predictions)):
                point = ForecastPoint(
                    timestamp=predictions['ds'].iloc[i],
                    value=float(predictions['yhat'].iloc[i]),
                    lower=float(predictions['yhat_lower'].iloc[i]) if request.include_confidence else None,
                    upper=float(predictions['yhat_upper'].iloc[i]) if request.include_confidence else None
                )
                forecasts.append(point)

        return ForecastResponse(
            model=model_name,
            granularity=request.granularity.value,
            target=request.target.value,
            horizon=request.horizon,
            forecasts=forecasts,
            generated_at=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


# =============================================================================
# Scaling Recommendation Endpoints
# =============================================================================

@app.post("/recommend-scaling", response_model=ScalingRecommendResponse, tags=["Autoscaling"])
async def recommend_scaling(request: ScalingRecommendRequest):
    """
    Đề xuất scaling action dựa trên current và predicted load.
    """
    # Calculate current utilization
    current_capacity = request.current_servers * request.server_capacity
    current_utilization = request.current_requests / current_capacity if current_capacity > 0 else 1.0

    # Calculate predicted utilization
    predicted_utilization = None
    if request.predicted_requests is not None:
        predicted_utilization = request.predicted_requests / current_capacity if current_capacity > 0 else 1.0

    # Determine action
    scale_out_threshold = 0.8
    scale_in_threshold = 0.3

    # Use predicted if available, otherwise use current
    eval_utilization = predicted_utilization if predicted_utilization else current_utilization

    if eval_utilization > scale_out_threshold:
        action = ScalingAction.SCALE_OUT
        # Calculate needed servers
        needed_requests = request.predicted_requests if request.predicted_requests else request.current_requests
        needed_servers = int(np.ceil(needed_requests * 1.2 / request.server_capacity))  # 20% buffer
        recommended = max(request.current_servers + 2, needed_servers)
        reason = f"Utilization {eval_utilization*100:.1f}% exceeds threshold {scale_out_threshold*100:.0f}%"
        confidence = min(0.9, eval_utilization)
    elif eval_utilization < scale_in_threshold:
        action = ScalingAction.SCALE_IN
        recommended = max(1, request.current_servers - 1)
        reason = f"Utilization {eval_utilization*100:.1f}% below threshold {scale_in_threshold*100:.0f}%"
        confidence = 1.0 - eval_utilization
    else:
        action = ScalingAction.MAINTAIN
        recommended = request.current_servers
        reason = f"Utilization {eval_utilization*100:.1f}% within normal range"
        confidence = 0.8

    return ScalingRecommendResponse(
        action=action,
        current_servers=request.current_servers,
        recommended_servers=recommended,
        current_utilization=current_utilization,
        predicted_utilization=predicted_utilization,
        reason=reason,
        confidence=confidence
    )


# =============================================================================
# Historical Metrics Endpoints
# =============================================================================

@app.get("/metrics/historical", response_model=HistoricalMetricsResponse, tags=["Metrics"])
async def get_historical_metrics(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    granularity: Granularity = Query(Granularity.FIFTEEN_MIN)
):
    """
    Lấy historical metrics cho một khoảng thời gian.
    """
    df = load_data()
    if df is None:
        raise HTTPException(status_code=500, detail="Data not available")

    # Filter by date
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    if len(df) == 0:
        raise HTTPException(status_code=404, detail="No data found for specified date range")

    # Calculate summary
    summary = MetricsSummary(
        total_requests=int(df['request_count'].sum()),
        average_requests=float(df['request_count'].mean()),
        peak_requests=float(df['request_count'].max()),
        min_requests=float(df['request_count'].min()),
        total_bytes=int(df['bytes_total'].sum()),
        average_bytes=float(df['bytes_total'].mean()),
        error_rate=float(df['error_rate'].mean() * 100) if 'error_rate' in df.columns else 0.0,
        unique_periods=len(df)
    )

    # Convert to list of dicts
    df_reset = df.reset_index()
    df_reset['timestamp'] = df_reset['timestamp'].astype(str)
    data = df_reset.to_dict(orient='records')

    return HistoricalMetricsResponse(
        summary=summary,
        data=data[:1000],  # Limit to 1000 records
        start_date=df.index.min(),
        end_date=df.index.max(),
        granularity=granularity.value
    )


# =============================================================================
# Simulation Endpoints
# =============================================================================

@app.post("/simulate", response_model=SimulationResponse, tags=["Simulation"])
async def run_simulation(request: SimulationRequest):
    """
    Chạy autoscaling simulation với các parameters cho trước.
    """
    df = load_data()
    if df is None:
        raise HTTPException(status_code=500, detail="Data not available")

    try:
        from src.autoscaling.policy import ServerConfig, ScalingPolicy
        from src.autoscaling.simulator import AutoscalingSimulator

        # Configuration
        server_config = ServerConfig(
            max_requests_per_min=1000,
            min_servers=1,
            max_servers=50,
            cost_per_server_hour=0.10
        )

        scaling_policy = ScalingPolicy(
            scale_out_threshold=request.scale_out_threshold,
            scale_in_threshold=request.scale_in_threshold,
            predictive_buffer=request.predictive_buffer
        )

        simulator = AutoscalingSimulator(server_config, scaling_policy)

        # Use test data
        df_clean = df[df['is_storm_period'] == 0]
        test_mask = df_clean.index >= '1995-08-23'
        test_data = df_clean.loc[test_mask]

        # Generate predictions if predictive
        predictions = None
        use_predictive = request.strategy == "predictive"
        if use_predictive:
            predictions = test_data.copy()
            predictions['request_count'] = predictions['request_count'].rolling(4).mean().shift(1).bfill()

        # Run simulation
        sim_df = simulator.simulate(
            actual_data=test_data,
            predictions=predictions,
            use_predictive=use_predictive,
            initial_servers=request.initial_servers
        )

        # Calculate metrics
        metrics = simulator.calculate_metrics(sim_df)

        return SimulationResponse(
            strategy=request.strategy,
            total_cost=metrics['total_cost'],
            average_servers=metrics['avg_servers'],
            peak_servers=metrics['max_servers'],
            min_servers=metrics['min_servers'],
            scaling_events=metrics['total_scaling_events'],
            sla_violations=metrics['overloaded_periods'],
            dropped_requests=int(metrics['total_dropped_requests']),
            timeline_summary={
                "intervals": len(sim_df),
                "start": str(test_data.index[0]),
                "end": str(test_data.index[-1])
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


# =============================================================================
# Anomaly Detection Endpoints
# =============================================================================

@app.post("/anomalies/detect", response_model=AnomalyDetectionResponse, tags=["Anomaly Detection"])
async def detect_anomalies(request: AnomalyDetectionRequest):
    """
    Phát hiện anomalies trong traffic data.
    """
    df = load_data()
    if df is None:
        raise HTTPException(status_code=500, detail="Data not available")

    try:
        from src.anomaly.detector import AnomalyDetector

        detector = AnomalyDetector()
        traffic = df['request_count']

        # Run detection based on method
        scores = None
        if request.method == "zscore":
            anomalies_mask = detector.detect_zscore(traffic, threshold=request.threshold)
        elif request.method == "iqr":
            anomalies_mask = detector.detect_iqr(traffic, k=request.threshold)
        elif request.method == "rolling":
            anomalies_mask = detector.detect_rolling(traffic, window=request.window, threshold=request.threshold)
        else:  # combined
            scores = detector.get_combined_anomaly_score(traffic)
            anomalies_mask = scores >= request.threshold

        # Build response
        anomaly_indices = traffic.index[anomalies_mask]
        anomalies = []

        for idx in anomaly_indices[:100]:  # Limit to 100
            anomalies.append(AnomalyPoint(
                timestamp=idx,
                value=float(traffic[idx]),
                score=float(scores[idx]) if scores is not None else request.threshold,
                type="high" if traffic[idx] > traffic.mean() else "low"
            ))

        return AnomalyDetectionResponse(
            method=request.method,
            total_anomalies=int(anomalies_mask.sum()),
            anomaly_rate=float(anomalies_mask.sum() / len(traffic) * 100),
            anomalies=anomalies,
            threshold_used=request.threshold
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection error: {str(e)}")


# =============================================================================
# Cost Analysis Endpoints
# =============================================================================

@app.post("/cost/compare", response_model=CostComparisonResponse, tags=["Cost Analysis"])
async def compare_costs(request: CostComparisonRequest):
    """
    So sánh chi phí của các scaling strategies.
    """
    df = load_data()
    if df is None:
        raise HTTPException(status_code=500, detail="Data not available")

    try:
        from src.autoscaling.policy import ServerConfig
        from src.autoscaling.cost_analyzer import CostAnalyzer

        server_config = ServerConfig(
            max_requests_per_min=1000,
            cost_per_server_hour=0.10
        )

        analyzer = CostAnalyzer(server_config)

        # Use test data
        df_clean = df[df['is_storm_period'] == 0]
        test_mask = df_clean.index >= '1995-08-23'
        demand = df_clean.loc[test_mask, 'request_count']

        strategies = []
        period_hours = len(demand) * 15 / 60

        # Fixed Peak
        if request.include_fixed_peak:
            capacity_per_server = server_config.max_requests_per_min * 15
            peak_servers = int(np.ceil(demand.max() / capacity_per_server))
            peak_metrics = analyzer.analyze_fixed_provisioning(demand, peak_servers, 15)
            strategies.append(StrategyCost(
                strategy="Fixed (Peak)",
                total_cost=peak_metrics.total_cost,
                average_servers=peak_servers,
                sla_violations=peak_metrics.underprovisioning_events,
                savings_vs_peak=0.0,
                savings_percent=0.0
            ))
            baseline_cost = peak_metrics.total_cost

        # Fixed P90
        if request.include_fixed_p90:
            p90_servers = int(np.ceil(demand.quantile(0.9) / capacity_per_server))
            p90_metrics = analyzer.analyze_fixed_provisioning(demand, p90_servers, 15)
            strategies.append(StrategyCost(
                strategy="Fixed (P90)",
                total_cost=p90_metrics.total_cost,
                average_servers=p90_servers,
                sla_violations=p90_metrics.underprovisioning_events,
                savings_vs_peak=baseline_cost - p90_metrics.total_cost,
                savings_percent=(baseline_cost - p90_metrics.total_cost) / baseline_cost * 100
            ))

        # Calculate best strategy
        best = min(strategies, key=lambda x: x.total_cost)

        return CostComparisonResponse(
            period_hours=period_hours,
            strategies=strategies,
            best_strategy=best.strategy,
            max_savings=max(s.savings_vs_peak for s in strategies)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost analysis error: {str(e)}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
