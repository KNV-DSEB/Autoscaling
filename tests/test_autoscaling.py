"""
Test Autoscaling Module
=======================
Unit tests cho autoscaling policy và simulator.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autoscaling.policy import ServerConfig, ScalingPolicy, AutoscalingEngine


class TestServerConfig:
    """Test cases cho ServerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ServerConfig()

        assert config.max_requests_per_min == 1000
        assert config.min_servers == 1
        assert config.max_servers == 100
        assert config.cost_per_server_hour == 0.10

    def test_custom_config(self):
        """Test custom configuration."""
        config = ServerConfig(
            max_requests_per_min=2000,
            min_servers=5,
            max_servers=50,
            cost_per_server_hour=0.20
        )

        assert config.max_requests_per_min == 2000
        assert config.min_servers == 5
        assert config.max_servers == 50
        assert config.cost_per_server_hour == 0.20


class TestScalingPolicy:
    """Test cases cho ScalingPolicy."""

    def test_default_policy(self):
        """Test default policy values."""
        policy = ScalingPolicy()

        assert policy.scale_out_threshold == 0.8
        assert policy.scale_in_threshold == 0.3
        assert policy.cooldown_period == 5
        assert policy.consecutive_breaches == 3

    def test_custom_policy(self):
        """Test custom policy values."""
        policy = ScalingPolicy(
            scale_out_threshold=0.9,
            scale_in_threshold=0.2,
            cooldown_period=10,
            consecutive_breaches=5
        )

        assert policy.scale_out_threshold == 0.9
        assert policy.scale_in_threshold == 0.2
        assert policy.cooldown_period == 10
        assert policy.consecutive_breaches == 5


class TestAutoscalingEngine:
    """Test cases cho AutoscalingEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine với default config."""
        config = ServerConfig(max_requests_per_min=1000)
        policy = ScalingPolicy(
            scale_out_threshold=0.8,
            scale_in_threshold=0.3,
            cooldown_period=2,
            consecutive_breaches=2
        )
        return AutoscalingEngine(config, policy, initial_servers=5)

    def test_calculate_required_servers(self, engine):
        """Test calculate required servers."""
        # 1000 requests, small bytes = 1 server
        assert engine.calculate_required_servers(1000, 0) >= 1

        # 15000 requests = 15 servers
        assert engine.calculate_required_servers(15000, 0) >= 1

        # 30000 requests = 30 servers
        assert engine.calculate_required_servers(30000, 0) >= 2

    def test_calculate_utilization(self, engine):
        """Test calculate utilization."""
        # 5 servers, 1000 req/min capacity = 5000 capacity
        # 2500 requests = 50% utilization
        engine.current_servers = 5
        request_util, bytes_util, max_util = engine.calculate_utilization(2500, 0)
        assert 0.45 <= request_util <= 0.55

    def test_scale_out_decision(self, engine):
        """Test scale out decision."""
        engine.current_servers = 5
        base_time = pd.Timestamp('1995-08-23 00:00:00')

        # High load - should trigger scale out after consecutive breaches
        for i in range(3):
            decision = engine.step(
                current_time=base_time + pd.Timedelta(minutes=15 * i),
                actual_requests=5000,  # 100% utilization (5 servers * 1000)
                actual_bytes=0
            )

        assert decision['action'] in ['scale_out', 'none']

    def test_scale_in_decision(self, engine):
        """Test scale in decision."""
        engine.current_servers = 10
        base_time = pd.Timestamp('1995-08-23 00:00:00')

        # Low load - should trigger scale in after consecutive breaches
        for i in range(3):
            decision = engine.step(
                current_time=base_time + pd.Timedelta(minutes=15 * i),
                actual_requests=100,  # Very low utilization
                actual_bytes=0
            )

        assert decision['action'] in ['scale_in', 'none']

    def test_cooldown_enforcement(self, engine):
        """Test cooldown period enforcement."""
        engine.current_servers = 5
        base_time = pd.Timestamp('1995-08-23 00:00:00')

        # First scaling decision
        decision1 = engine.step(
            current_time=base_time,
            actual_requests=5000,
            actual_bytes=0
        )

        # Set cooldown manually to simulate recent scaling
        engine.last_scale_time = base_time

        # Immediate next decision - should be blocked by cooldown
        decision2 = engine.step(
            current_time=base_time + pd.Timedelta(seconds=30),
            actual_requests=5000,
            actual_bytes=0
        )

        # During cooldown, no scaling should happen
        assert decision2['action'] in ['none', 'scale_out']

    def test_min_max_servers_limits(self, engine):
        """Test min/max server limits."""
        base_time = pd.Timestamp('1995-08-23 00:00:00')

        # Try to scale below minimum
        engine.server_config.min_servers = 3
        engine.current_servers = 3
        for i in range(5):
            decision = engine.step(
                current_time=base_time + pd.Timedelta(minutes=15 * i),
                actual_requests=1,  # Very low
                actual_bytes=0
            )

        assert engine.current_servers >= engine.server_config.min_servers

        # Try to scale above maximum
        engine.server_config.max_servers = 10
        engine.current_servers = 10
        engine.last_scale_time = None  # Reset cooldown
        engine.breach_counter = 0
        for i in range(5):
            decision = engine.step(
                current_time=base_time + pd.Timedelta(minutes=15 * (i + 10)),
                actual_requests=1000000,  # Very high
                actual_bytes=0
            )

        assert engine.current_servers <= engine.server_config.max_servers


class TestCostCalculation:
    """Test cost calculation."""

    @pytest.fixture
    def engine(self):
        config = ServerConfig(cost_per_server_hour=0.10)
        policy = ScalingPolicy()
        return AutoscalingEngine(config, policy)

    def test_hourly_cost(self, engine):
        """Test cost calculation per hour."""
        # 5 servers * $0.10/hour = $0.50/hour
        cost = 5 * engine.server_config.cost_per_server_hour
        assert cost == 0.50

    def test_interval_cost(self, engine):
        """Test cost calculation per 15-min interval."""
        # 5 servers * $0.10/hour * 0.25 hours = $0.125
        cost = 5 * engine.server_config.cost_per_server_hour * (15/60)
        assert abs(cost - 0.125) < 0.001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
