"""
Streamlit Dashboard
===================
Dashboard cho Autoscaling Analysis với 5 tabs:
    1. Overview - KPIs và traffic overview
    2. Traffic Analysis - Patterns và heatmaps
    3. Forecasting - Model predictions
    4. Autoscaling Simulator - Policy testing
    5. Cost Analysis - Strategy comparison

Run:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="Autoscaling Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Data Loading Functions
# =============================================================================

@st.cache_data
def load_data():
    """Load timeseries data."""
    try:
        from src.data.preprocessor import load_timeseries
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data/processed/timeseries_15min.parquet'
        )
        if os.path.exists(data_path):
            df = load_timeseries(data_path)
            return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
    return None


@st.cache_resource
def load_models():
    """Load trained models."""
    models = {}
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models'
    )

    # LightGBM
    try:
        from src.models.lightgbm_forecaster import LightGBMForecaster
        model_path = os.path.join(models_dir, 'lightgbm_15min.pkl')
        if os.path.exists(model_path):
            models['lightgbm'] = LightGBMForecaster.load(model_path)
    except:
        pass

    # SARIMA
    try:
        from src.models.sarima import SARIMAForecaster
        model_path = os.path.join(models_dir, 'sarima_15min.pkl')
        if os.path.exists(model_path):
            models['sarima'] = SARIMAForecaster.load(model_path)
    except:
        pass

    return models


# =============================================================================
# Helper Functions
# =============================================================================

def format_number(num):
    """Format number với comma separator."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(int(num))


# =============================================================================
# Tab 1: Overview
# =============================================================================

def render_overview_tab(df):
    """Render Overview tab."""
    st.header("📊 Traffic Overview")

    if df is None:
        st.warning("Dữ liệu chưa được load. Vui lòng chạy data pipeline trước.")
        return

    # Remove storm period for analysis
    df_clean = df[df['is_storm_period'] == 0]

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Requests",
            format_number(df_clean['request_count'].sum()),
            help="Tổng số requests trong toàn bộ dataset"
        )

    with col2:
        st.metric(
            "Avg Requests/Interval",
            format_number(df_clean['request_count'].mean()),
            help="Trung bình requests mỗi 15 phút"
        )

    with col3:
        st.metric(
            "Peak Requests",
            format_number(df_clean['request_count'].max()),
            help="Request count cao nhất"
        )

    with col4:
        error_rate = df_clean['error_rate'].mean() * 100 if 'error_rate' in df_clean.columns else 0
        st.metric(
            "Avg Error Rate",
            f"{error_rate:.2f}%",
            help="Tỷ lệ lỗi trung bình"
        )

    st.markdown("---")

    # Traffic over time
    st.subheader("Traffic Over Time")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_clean.index,
        y=df_clean['request_count'],
        mode='lines',
        name='Request Count',
        line=dict(color='steelblue')
    ))

    # Highlight storm period
    storm_data = df[df['is_storm_period'] == 1]
    if len(storm_data) > 0:
        storm_start = storm_data.index.min()
        storm_end = storm_data.index.max()
        fig.add_vrect(
            x0=storm_start, x1=storm_end,
            fillcolor="red", opacity=0.2,
            annotation_text="Storm Period",
            annotation_position="top left"
        )

    fig.update_layout(
        title="Request Count Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Requests",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribution
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df_clean,
            x='request_count',
            nbins=50,
            title="Request Count Distribution"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Stats table
        st.subheader("Statistics")
        stats = df_clean['request_count'].describe()
        stats_df = pd.DataFrame({
            'Metric': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
            'Value': [
                format_number(stats['count']),
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                format_number(stats['min']),
                format_number(stats['25%']),
                format_number(stats['50%']),
                format_number(stats['75%']),
                format_number(stats['max'])
            ]
        })
        st.dataframe(stats_df, hide_index=True)


# =============================================================================
# Tab 2: Traffic Analysis
# =============================================================================

def render_traffic_analysis_tab(df):
    """Render Traffic Analysis tab."""
    st.header("📈 Traffic Analysis")

    if df is None:
        st.warning("Dữ liệu chưa được load.")
        return

    df_clean = df[df['is_storm_period'] == 0].copy()

    # Add time features
    df_clean['hour'] = df_clean.index.hour
    df_clean['day_of_week'] = df_clean.index.dayofweek
    df_clean['day_name'] = df_clean.index.day_name()

    col1, col2 = st.columns(2)

    with col1:
        # Hourly pattern
        hourly = df_clean.groupby('hour')['request_count'].mean()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hourly.index,
            y=hourly.values,
            marker_color='steelblue'
        ))
        fig.update_layout(
            title="Average Traffic by Hour",
            xaxis_title="Hour",
            yaxis_title="Avg Requests",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Day of week pattern
        daily = df_clean.groupby('day_of_week')['request_count'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=days,
            y=daily.values,
            marker_color='coral'
        ))
        fig.update_layout(
            title="Average Traffic by Day of Week",
            xaxis_title="Day",
            yaxis_title="Avg Requests",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.subheader("Traffic Heatmap (Hour x Day)")

    # Create pivot table
    heatmap_data = df_clean.pivot_table(
        values='request_count',
        index='day_of_week',
        columns='hour',
        aggfunc='mean'
    )

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Hour", y="Day", color="Requests"),
        x=list(range(24)),
        y=days,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        title="Traffic Heatmap",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Trend decomposition (simplified)
    st.subheader("Traffic Trend")

    # Rolling averages
    df_clean['rolling_24h'] = df_clean['request_count'].rolling(96).mean()
    df_clean['rolling_1h'] = df_clean['request_count'].rolling(4).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_clean.index,
        y=df_clean['request_count'],
        mode='lines',
        name='Actual',
        opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=df_clean.index,
        y=df_clean['rolling_1h'],
        mode='lines',
        name='1-Hour Rolling Avg',
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df_clean.index,
        y=df_clean['rolling_24h'],
        mode='lines',
        name='24-Hour Rolling Avg',
        line=dict(width=2, color='red')
    ))

    fig.update_layout(
        title="Traffic with Rolling Averages",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Tab 3: Forecasting
# =============================================================================

def render_forecasting_tab(df, models):
    """Render Forecasting tab."""
    st.header("🔮 Traffic Forecasting")

    if df is None:
        st.warning("Dữ liệu chưa được load.")
        return

    df_clean = df[df['is_storm_period'] == 0]

    # Model selection
    col1, col2, col3 = st.columns(3)

    with col1:
        available_models = list(models.keys()) if models else ['No models loaded']
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            help="Chọn model để forecast"
        )

    with col2:
        horizon = st.slider(
            "Forecast Horizon (intervals)",
            min_value=4,
            max_value=192,
            value=96,
            help="Số intervals (15min) cần forecast"
        )

    with col3:
        show_confidence = st.checkbox("Show Confidence Interval", value=True)

    if st.button("Generate Forecast", type="primary"):
        if selected_model in models:
            with st.spinner("Generating forecast..."):
                try:
                    model = models[selected_model]

                    if selected_model == 'sarima':
                        predictions = model.predict(
                            steps=horizon,
                            return_conf_int=show_confidence
                        )

                        # Plot
                        fig = go.Figure()

                        # Historical
                        recent = df_clean['request_count'].iloc[-200:]
                        fig.add_trace(go.Scatter(
                            x=recent.index,
                            y=recent.values,
                            mode='lines',
                            name='Historical'
                        ))

                        # Forecast
                        base_time = df_clean.index[-1] + timedelta(minutes=15)
                        forecast_times = [base_time + timedelta(minutes=15*i) for i in range(horizon)]

                        fig.add_trace(go.Scatter(
                            x=forecast_times,
                            y=predictions['forecast'].values,
                            mode='lines',
                            name='Forecast',
                            line=dict(dash='dash', color='red')
                        ))

                        if show_confidence:
                            fig.add_trace(go.Scatter(
                                x=forecast_times + forecast_times[::-1],
                                y=list(predictions['upper'].values) + list(predictions['lower'].values[::-1]),
                                fill='toself',
                                fillcolor='rgba(255,0,0,0.1)',
                                line=dict(color='rgba(255,0,0,0)'),
                                name='95% CI'
                            ))

                        fig.update_layout(
                            title=f"{selected_model.upper()} Forecast ({horizon} intervals)",
                            xaxis_title="Timestamp",
                            yaxis_title="Requests",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Stats
                        st.subheader("Forecast Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Avg Forecast", f"{predictions['forecast'].mean():.0f}")
                        with col2:
                            st.metric("Max Forecast", f"{predictions['forecast'].max():.0f}")
                        with col3:
                            st.metric("Min Forecast", f"{predictions['forecast'].min():.0f}")

                    else:
                        st.info(f"Forecast với {selected_model} chưa được implement trong dashboard.")

                except Exception as e:
                    st.error(f"Error generating forecast: {e}")
        else:
            st.warning("Model không khả dụng. Vui lòng train models trước.")


# =============================================================================
# Tab 4: Autoscaling Simulator
# =============================================================================

def render_autoscaling_tab(df):
    """Render Autoscaling Simulator tab."""
    st.header("⚙️ Autoscaling Simulator")

    if df is None:
        st.warning("Dữ liệu chưa được load.")
        return

    df_clean = df[df['is_storm_period'] == 0]

    # Policy configuration
    st.subheader("Scaling Policy Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        scale_out_threshold = st.slider(
            "Scale Out Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            help="Scale out khi utilization vượt ngưỡng này"
        )

    with col2:
        scale_in_threshold = st.slider(
            "Scale In Threshold",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            help="Scale in khi utilization dưới ngưỡng này"
        )

    with col3:
        predictive_buffer = st.slider(
            "Predictive Buffer",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            help="Safety buffer cho predictive scaling"
        )

    col1, col2 = st.columns(2)
    with col1:
        initial_servers = st.number_input(
            "Initial Servers",
            min_value=1,
            max_value=50,
            value=5
        )

    with col2:
        strategy = st.selectbox(
            "Strategy",
            ["Reactive", "Predictive"],
            help="Chọn scaling strategy"
        )

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            try:
                from src.autoscaling.policy import ServerConfig, ScalingPolicy
                from src.autoscaling.simulator import AutoscalingSimulator

                server_config = ServerConfig(
                    max_requests_per_min=1000,
                    min_servers=1,
                    max_servers=50,
                    cost_per_server_hour=0.10
                )

                scaling_policy = ScalingPolicy(
                    scale_out_threshold=scale_out_threshold,
                    scale_in_threshold=scale_in_threshold,
                    predictive_buffer=predictive_buffer
                )

                simulator = AutoscalingSimulator(server_config, scaling_policy)

                # Use test data
                test_mask = df_clean.index >= '1995-08-23'
                test_data = df_clean.loc[test_mask]

                # Predictions
                predictions = None
                use_predictive = strategy == "Predictive"
                if use_predictive:
                    predictions = test_data.copy()
                    predictions['request_count'] = predictions['request_count'].rolling(4).mean().shift(1).bfill()

                sim_df = simulator.simulate(
                    actual_data=test_data,
                    predictions=predictions,
                    use_predictive=use_predictive,
                    initial_servers=initial_servers
                )

                # Calculate metrics
                metrics = simulator.calculate_metrics(sim_df)

                # Display results
                st.subheader("Simulation Results")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Cost", f"${metrics['total_cost']:.2f}")
                with col2:
                    st.metric("Avg Servers", f"{metrics['avg_servers']:.1f}")
                with col3:
                    st.metric("Scaling Events", metrics['total_scaling_events'])
                with col4:
                    st.metric("SLA Violations", metrics['overloaded_periods'])

                # Timeline plot
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    subplot_titles=("Traffic Demand", "Server Allocation")
                )

                # Demand
                fig.add_trace(
                    go.Scatter(
                        x=sim_df.index,
                        y=sim_df['actual_requests'].values,
                        name="Demand",
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )

                # Servers
                fig.add_trace(
                    go.Scatter(
                        x=sim_df.index,
                        y=sim_df['servers'].values,
                        name="Servers",
                        line=dict(color='orange')
                    ),
                    row=2, col=1
                )

                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Simulation error: {e}")


# =============================================================================
# Tab 5: Cost Analysis
# =============================================================================

def render_cost_analysis_tab(df):
    """Render Cost Analysis tab."""
    st.header("💰 Cost Analysis")

    if df is None:
        st.warning("Dữ liệu chưa được load.")
        return

    df_clean = df[df['is_storm_period'] == 0]

    st.subheader("Strategy Comparison")

    try:
        from src.autoscaling.policy import ServerConfig
        from src.autoscaling.cost_analyzer import CostAnalyzer

        server_config = ServerConfig(
            max_requests_per_min=1000,
            cost_per_server_hour=0.10
        )

        analyzer = CostAnalyzer(server_config)

        # Use test data
        test_mask = df_clean.index >= '1995-08-23'
        demand = df_clean.loc[test_mask, 'request_count']

        capacity_per_server = server_config.max_requests_per_min * 15

        # Calculate costs for different strategies
        peak_servers = int(np.ceil(demand.max() / capacity_per_server))
        p90_servers = int(np.ceil(demand.quantile(0.9) / capacity_per_server))
        avg_servers = int(np.ceil(demand.mean() / capacity_per_server))

        # Simulate autoscaled
        autoscaled_servers = pd.Series(
            np.clip(np.ceil(demand / capacity_per_server * 1.2), 1, 50),
            index=demand.index
        ).astype(int)

        strategies = {
            'Fixed (Peak)': peak_servers,
            'Fixed (P90)': p90_servers,
            'Fixed (Average)': avg_servers,
            'Autoscaled': autoscaled_servers
        }

        comparison = analyzer.compare_strategies(demand, strategies, 15)

        # Display as table
        st.dataframe(
            comparison.style.format({
                'Total Cost ($)': '${:.2f}',
                'Avg Servers': '{:.1f}',
                'Overprov Cost ($)': '${:.2f}',
                'SLA Violation (%)': '{:.2f}%',
                'Savings ($)': '${:.2f}',
                'Savings (%)': '{:.2f}%'
            }),
            hide_index=True
        )

        # Chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=comparison['Strategy'],
            y=comparison['Total Cost ($)'],
            name='Total Cost',
            marker_color='steelblue'
        ))

        fig.update_layout(
            title="Cost Comparison by Strategy",
            xaxis_title="Strategy",
            yaxis_title="Cost ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Savings chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=comparison['Strategy'],
            y=comparison['Savings (%)'],
            marker_color=np.where(comparison['Savings (%)'] >= 0, 'green', 'red')
        ))
        fig2.update_layout(
            title="Savings vs Fixed Peak Provisioning",
            xaxis_title="Strategy",
            yaxis_title="Savings (%)",
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Key findings
        st.subheader("Key Findings")
        best_strategy = comparison.iloc[0]['Strategy']
        best_cost = comparison.iloc[0]['Total Cost ($)']
        max_savings = comparison['Savings ($)'].max()

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Best Strategy:** {best_strategy}")
            st.info(f"**Lowest Cost:** ${best_cost:.2f}")
        with col2:
            st.success(f"**Max Savings:** ${max_savings:.2f}")
            st.info(f"**Period:** {len(demand) * 15 / 60:.1f} hours")

    except Exception as e:
        st.error(f"Error in cost analysis: {e}")


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application."""
    # Sidebar
    st.sidebar.title("📊 Autoscaling Analysis")
    st.sidebar.markdown("---")

    # Load data
    df = load_data()
    models = load_models()

    # Data status
    if df is not None:
        st.sidebar.success(f"✓ Data loaded: {len(df):,} records")
        st.sidebar.info(f"📅 {df.index.min().date()} → {df.index.max().date()}")
    else:
        st.sidebar.error("✗ Data not loaded")

    # Models status
    st.sidebar.markdown("**Models:**")
    for name in ['sarima', 'lightgbm', 'prophet']:
        if name in models:
            st.sidebar.success(f"✓ {name.upper()}")
        else:
            st.sidebar.warning(f"○ {name.upper()}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with Streamlit")

    # Main content - Tabs
    tabs = st.tabs([
        "📊 Overview",
        "📈 Traffic Analysis",
        "🔮 Forecasting",
        "⚙️ Autoscaling",
        "💰 Cost Analysis"
    ])

    with tabs[0]:
        render_overview_tab(df)

    with tabs[1]:
        render_traffic_analysis_tab(df)

    with tabs[2]:
        render_forecasting_tab(df, models)

    with tabs[3]:
        render_autoscaling_tab(df)

    with tabs[4]:
        render_cost_analysis_tab(df)


if __name__ == "__main__":
    main()
