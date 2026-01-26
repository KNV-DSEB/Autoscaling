# Autoscaling Analysis - NASA Web Server Logs

Hệ thống phân tích và dự báo traffic để tự động điều chỉnh tài nguyên (autoscaling) dựa trên dữ liệu log NASA web server tháng 7-8/1995.

## Mục Lục

- [Tổng Quan](#tổng-quan)
- [Cấu Trúc Project](#cấu-trúc-project)
- [Cài Đặt](#cài-đặt)
- [Dữ Liệu](#dữ-liệu)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Components](#components)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)

---

## Tổng Quan

### Bài Toán
Phân bổ tài nguyên server cố định (fixed provisioning) dẫn đến:
- **Over-provisioning**: Lãng phí chi phí khi traffic thấp
- **Under-provisioning**: Quá tải, drop requests khi traffic cao

### Giải Pháp
Xây dựng hệ thống autoscaling thông minh:
1. **Dự báo traffic** bằng các models (SARIMA, LightGBM, Prophet)
2. **Đề xuất số servers** cần thiết dựa trên dự báo
3. **Mô phỏng và so sánh** các scaling strategies
4. **Phân tích chi phí** để tối ưu hóa

### Kết Quả Mong Đợi
- Giảm chi phí > 30% so với fixed provisioning
- Duy trì SLA (< 1% requests bị drop)
- Giảm số scaling events không cần thiết

---

## Cấu Trúc Project

```
AUTOSCALING ANALYSIS/
├── README.md                   # File này
├── requirements.txt            # Python dependencies
│
├── DATA/                       # Raw data (train.txt, test.txt)
│
├── data/
│   └── processed/              # Processed parquet files
│       ├── timeseries_1min.parquet
│       ├── timeseries_5min.parquet
│       └── timeseries_15min.parquet
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_sarima.ipynb
│   ├── 05_modeling_lightgbm.ipynb
│   ├── 06_model_comparison.ipynb
│   ├── 07_autoscaling_simulation.ipynb
│   └── 08_anomaly_detection.ipynb
│
├── src/                        # Source code
│   ├── data/
│   │   ├── parser.py           # Log parsing
│   │   └── preprocessor.py     # Time series aggregation
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── sarima.py
│   │   ├── lightgbm_forecaster.py
│   │   ├── prophet_forecaster.py
│   │   └── evaluation.py
│   ├── autoscaling/
│   │   ├── policy.py           # Scaling rules
│   │   ├── simulator.py        # Simulation engine
│   │   └── cost_analyzer.py    # Cost analysis
│   └── anomaly/
│       └── detector.py         # Anomaly detection
│
├── api/                        # FastAPI
│   ├── main.py
│   └── schemas.py
│
├── dashboard/                  # Streamlit
│   └── app.py
│
├── models/                     # Saved trained models
│   ├── sarima_15min.pkl
│   ├── lightgbm_15min.pkl
│   └── prophet_15min.pkl
│
└── reports/
    └── figures/                # Generated plots
```

---

## Cài Đặt

### Yêu Cầu
- Python 3.8+
- 8GB RAM (recommended)

### Cài Đặt Dependencies

```bash
# Clone/download project
cd "AUTOSCALING ANALYSIS"

# Tạo virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Dependencies Chính
- **Data Processing**: pandas, numpy, pyarrow
- **Modeling**: statsmodels, lightgbm, prophet
- **Visualization**: matplotlib, seaborn, plotly
- **API**: fastapi, uvicorn
- **Dashboard**: streamlit

---

## Dữ Liệu

### Source
NASA-HTTP Web Server Access Logs (July-August 1995)

### Format (Common Log Format)
```
199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245
│              │  │                          │                                │    │
Host           │  Timestamp                  Request                          Code Bytes
               User (always "-")
```

### Statistics
| Dataset | Entries | Period |
|---------|---------|--------|
| Train | 2,934,960 | 01/Jul - 22/Aug/1995 |
| Test | 526,650 | 23/Aug - 31/Aug/1995 |

### Storm Period
- **Gap**: 01/Aug/1995 14:52 → 03/Aug/1995 04:36 (~38 giờ)
- **Nguyên nhân**: Server offline do hurricane
- **Xử lý**: Đánh dấu `is_storm_period = 1`

---

## Hướng Dẫn Sử Dụng

### 1. Data Pipeline

Chạy notebook `01_data_ingestion.ipynb` hoặc:

```python
from src.data.parser import NASALogParser
from src.data.preprocessor import LogPreprocessor

# Parse logs
parser = NASALogParser()
df = parser.parse_file('DATA/train.txt')

# Aggregate to time series
preprocessor = LogPreprocessor(df)
ts_15min = preprocessor.aggregate_timeseries(freq='15min')
ts_15min.to_parquet('data/processed/timeseries_15min.parquet')
```

### 2. Train Models

Chạy notebooks 04-05 hoặc:

```python
from src.models.lightgbm_forecaster import LightGBMForecaster
from src.features.feature_engineering import TimeSeriesFeatureEngineer

# Prepare features
fe = TimeSeriesFeatureEngineer(df)
df_features = fe.create_all_features(target_col='request_count')
X, y = fe.prepare_supervised(df_features, 'request_count', feature_cols)

# Train
model = LightGBMForecaster()
model.fit(X_train, y_train, X_val, y_val)
model.save('models/lightgbm_15min.pkl')
```

### 3. Run Simulation

```python
from src.autoscaling.simulator import AutoscalingSimulator
from src.autoscaling.policy import ServerConfig, ScalingPolicy

# Configure
server_config = ServerConfig(max_requests_per_min=1000)
scaling_policy = ScalingPolicy(scale_out_threshold=0.8)

# Simulate
simulator = AutoscalingSimulator(server_config, scaling_policy)
result = simulator.run_simulation(
    actual_demand=demand_series,
    predicted_demand=predicted_series
)

print(f"Total Cost: ${result.total_cost:.2f}")
print(f"SLA Violations: {result.sla_violations}")
```

### 4. Start API

```bash
cd "AUTOSCALING ANALYSIS"
uvicorn api.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 5. Start Dashboard

```bash
cd "AUTOSCALING ANALYSIS"
streamlit run dashboard/app.py
```

Dashboard: http://localhost:8501

---

## Components

### Models

| Model | Type | Best For | RMSE* |
|-------|------|----------|-------|
| SARIMA | Statistical | Seasonal patterns, confidence intervals | ~50 |
| LightGBM | ML | Complex features, fast inference | ~45 |
| Prophet | Statistical | Missing data, holidays | ~55 |

*Approximate RMSE on test set (15min granularity)

### Autoscaling Policy

```python
ScalingPolicy(
    scale_out_threshold=0.8,    # Scale out at 80% utilization
    scale_in_threshold=0.3,     # Scale in at 30% utilization
    cooldown_period=5,          # 5 intervals (75min) cooldown
    consecutive_breaches=3,     # Need 3 consecutive breaches
    predictive_buffer=0.2       # 20% safety margin
)
```

### Cost Model

```python
ServerConfig(
    max_requests_per_min=1000,    # Server capacity
    cost_per_server_hour=0.10,    # $0.10/hour
    min_servers=1,
    max_servers=50
)
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/models` | List available models |
| POST | `/forecast` | Generate forecasts |
| POST | `/recommend-scaling` | Get scaling recommendation |
| GET | `/metrics/historical` | Get historical metrics |
| POST | `/simulate` | Run autoscaling simulation |
| POST | `/anomalies/detect` | Detect anomalies |
| POST | `/cost/compare` | Compare strategy costs |

### Example: Forecast

```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lightgbm",
    "horizon": 96,
    "granularity": "15min",
    "include_confidence": true
  }'
```

---

## Dashboard

### 5 Tabs

1. **Overview**: KPIs, traffic time series
2. **Traffic Analysis**: Hourly/daily patterns, heatmap
3. **Forecasting**: Model selection, predictions
4. **Autoscaling Simulator**: Policy tuning, simulation
5. **Cost Analysis**: Strategy comparison, savings

---

## Notebooks

| # | Notebook | Nội Dung |
|---|----------|----------|
| 01 | data_ingestion | Parse logs, create time series |
| 02 | eda | Exploratory data analysis |
| 03 | feature_engineering | Create features |
| 04 | modeling_sarima | SARIMA model |
| 05 | modeling_lightgbm | LightGBM model |
| 06 | model_comparison | Compare all models |
| 07 | autoscaling_simulation | Simulate scaling strategies |
| 08 | anomaly_detection | Detect traffic anomalies |

---

## Kết Quả Dự Kiến

### Cost Savings

| Strategy | Cost | Savings vs Peak |
|----------|------|-----------------|
| Fixed (Peak) | $XX.XX | 0% |
| Fixed (P90) | $XX.XX | ~15% |
| Reactive | $XX.XX | ~25% |
| Predictive | $XX.XX | ~35% |

### Model Performance

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| SARIMA | ~50 | ~35 | ~15% |
| LightGBM | ~45 | ~30 | ~12% |
| Prophet | ~55 | ~40 | ~18% |

---

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

---

## License

MIT License

---

## Contact

- Project: Autoscaling Analysis
- Data Source: NASA HTTP Logs (http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html)
