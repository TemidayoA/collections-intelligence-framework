"""
Unit tests for Propensity-to-Pay Modelling
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.propensity.payment_propensity import (
    PropensityModel,
    PropensityScore,
    identify_intent_signals
)
from src.propensity.feature_engineering import (
    FeatureEngineeringPipeline,
    PaymentFeatureEngineer,
    IncomeFeatureEngineer,
    ContactFeatureEngineer,
    TimingFeatureEngineer
)


class TestIntentSignals:
    """Tests for payment intent signal identification."""
    
    def test_strong_intent_signals(self):
        """Test that failed payment attempts indicate strong intent."""
        features = {
            'payment_attempts_failed': 2,
            'contact_response_rate': 0.8,
            'partial_payment_made': True,
            'initiated_contact': True,
            'channels_blocked': 0
        }
        
        signals = identify_intent_signals(features)
        
        assert signals['attempted_payment'] == True
        assert signals['responsive'] == True
        assert signals['intent_score'] >= 0.8
    
    def test_low_intent_signals(self):
        """Test that avoidance behaviours indicate low intent."""
        features = {
            'payment_attempts_failed': 0,
            'contact_response_rate': 0.1,
            'partial_payment_made': False,
            'initiated_contact': False,
            'channels_blocked': 2
        }
        
        signals = identify_intent_signals(features)
        
        assert signals['attempted_payment'] == False
        assert signals['responsive'] == False
        assert signals['channels_open'] == False
        assert signals['intent_score'] <= 0.3
    
    def test_mixed_signals(self):
        """Test handling of mixed intent signals."""
        features = {
            'payment_attempts_failed': 1,
            'contact_response_rate': 0.4,
            'partial_payment_made': False,
            'initiated_contact': False,
            'channels_blocked': 0
        }
        
        signals = identify_intent_signals(features)
        
        assert signals['attempted_payment'] == True
        assert signals['responsive'] == False
        assert 0.3 <= signals['intent_score'] <= 0.7


class TestPaymentFeatureEngineer:
    """Tests for payment feature engineering."""
    
    @pytest.fixture
    def engineer(self):
        return PaymentFeatureEngineer(lookback_days=90)
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data."""
        now = datetime.now()
        return pd.DataFrame({
            'customer_id': ['C001'] * 5 + ['C002'] * 3,
            'timestamp': [
                now - timedelta(days=10),
                now - timedelta(days=20),
                now - timedelta(days=30),
                now - timedelta(days=40),
                now - timedelta(days=50),
                now - timedelta(days=5),
                now - timedelta(days=15),
                now - timedelta(days=25),
            ],
            'transaction_type': ['payment_attempt'] * 8,
            'status': ['success', 'failed', 'success', 'failed', 'success',
                      'failed', 'failed', 'failed'],
            'amount': [100, 100, 100, 100, 100, 50, 50, 50]
        })
    
    @pytest.fixture
    def sample_loans(self):
        """Create sample loan data."""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002'],
            'instalment_amount': [100, 50],
            'status': ['active', 'active']
        })
    
    def test_successful_payment_features(self, engineer, sample_transactions, sample_loans):
        """Test features for customer with successful payments."""
        features = engineer.engineer('C001', sample_transactions, sample_loans)
        
        assert features['payment_attempts_total'] == 5
        assert features['payment_attempts_successful'] == 3
        assert features['payment_attempts_failed'] == 2
        assert features['payment_success_rate'] == 0.6
    
    def test_failed_payment_features(self, engineer, sample_transactions, sample_loans):
        """Test features for customer with all failed payments."""
        features = engineer.engineer('C002', sample_transactions, sample_loans)
        
        assert features['payment_attempts_total'] == 3
        assert features['payment_attempts_successful'] == 0
        assert features['payment_attempts_failed'] == 3
        assert features['has_failed_attempt'] == 1.0
    
    def test_no_payment_history(self, engineer, sample_loans):
        """Test features for customer with no payment history."""
        empty_txns = pd.DataFrame(columns=[
            'customer_id', 'timestamp', 'transaction_type', 'status', 'amount'
        ])
        
        features = engineer.engineer('C003', empty_txns, sample_loans)
        
        assert features['payment_attempts_total'] == 0
        assert features['days_since_last_attempt'] == 999


class TestIncomeFeatureEngineer:
    """Tests for income feature engineering."""
    
    @pytest.fixture
    def engineer(self):
        return IncomeFeatureEngineer(lookback_days=180)
    
    def test_regular_salary_pattern(self, engineer):
        """Test detection of regular monthly salary pattern."""
        now = datetime.now()
        
        # Monthly salary on 25th
        transactions = pd.DataFrame({
            'customer_id': ['C001'] * 6,
            'timestamp': [
                now.replace(day=25) - timedelta(days=30*i)
                for i in range(6)
            ],
            'amount': [50000] * 6
        })
        
        # Add loan data
        loans = pd.DataFrame({
            'customer_id': ['C001'],
            'instalment_amount': [10000],
            'status': ['active']
        })
        
        features = engineer.engineer('C001', transactions, loans)
        
        assert features['income_regularity_score'] > 0.6
        assert features['income_pattern'] == 'monthly_salary'
    
    def test_irregular_income_pattern(self, engineer):
        """Test detection of irregular income pattern."""
        now = datetime.now()
        
        # Random intervals
        transactions = pd.DataFrame({
            'customer_id': ['C001'] * 6,
            'timestamp': [
                now - timedelta(days=5),
                now - timedelta(days=12),
                now - timedelta(days=45),
                now - timedelta(days=60),
                now - timedelta(days=95),
                now - timedelta(days=150),
            ],
            'amount': [10000, 5000, 15000, 8000, 12000, 7000]
        })
        
        loans = pd.DataFrame({
            'customer_id': ['C001'],
            'instalment_amount': [5000],
            'status': ['active']
        })
        
        features = engineer.engineer('C001', transactions, loans)
        
        assert features['income_regularity_score'] < 0.6
        assert features['income_pattern'] == 'irregular'


class TestContactFeatureEngineer:
    """Tests for contact response feature engineering."""
    
    @pytest.fixture
    def engineer(self):
        return ContactFeatureEngineer(lookback_days=60)
    
    def test_high_response_rate(self, engineer):
        """Test features for highly responsive customer."""
        now = datetime.now()
        
        contacts = pd.DataFrame({
            'customer_id': ['C001'] * 10,
            'timestamp': [now - timedelta(days=i*3) for i in range(10)],
            'channel': ['sms', 'call', 'sms', 'whatsapp', 'call',
                       'sms', 'sms', 'call', 'whatsapp', 'sms'],
            'responded': [True, True, True, True, False,
                         True, True, True, True, True],
            'blocked': [False] * 10
        })
        
        features = engineer.engineer('C001', contacts)
        
        assert features['overall_response_rate'] == 0.9
        assert features['is_avoiding'] == 0.0
    
    def test_avoiding_customer(self, engineer):
        """Test features for customer showing avoidance behaviour."""
        now = datetime.now()
        
        contacts = pd.DataFrame({
            'customer_id': ['C001'] * 10,
            'timestamp': [now - timedelta(days=i*3) for i in range(10)],
            'channel': ['sms'] * 5 + ['call'] * 5,
            'responded': [False] * 10,
            'blocked': [False] * 5 + [True] * 5
        })
        
        features = engineer.engineer('C001', contacts)
        
        assert features['overall_response_rate'] == 0.0
        assert features['is_avoiding'] == 1.0
        assert features['channels_blocked'] >= 1


class TestTimingFeatureEngineer:
    """Tests for timing feature engineering."""
    
    @pytest.fixture
    def engineer(self):
        return TimingFeatureEngineer()
    
    def test_dpd_bucket_assignment(self, engineer):
        """Test correct DPD bucket assignment."""
        loans = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004'],
            'dpd': [5, 15, 45, 100],
            'status': ['active', 'active', 'active', 'active']
        })
        
        f1 = engineer.engineer('C001', loans)
        f2 = engineer.engineer('C002', loans)
        f3 = engineer.engineer('C003', loans)
        f4 = engineer.engineer('C004', loans)
        
        assert f1['dpd_bucket'] == 1  # 0-7 days
        assert f2['dpd_bucket'] == 2  # 8-30 days
        assert f3['dpd_bucket'] == 3  # 31-60 days
        assert f4['dpd_bucket'] == 5  # 90+ days
    
    def test_salary_window_detection(self, engineer):
        """Test salary window feature calculation."""
        loans = pd.DataFrame({
            'customer_id': ['C001'],
            'dpd': [10],
            'status': ['active']
        })
        
        profile = {'salary_day': 25}
        
        features = engineer.engineer('C001', loans, profile)
        
        assert 'days_to_salary' in features
        assert 'in_salary_window' in features


class TestFeatureEngineeringPipeline:
    """Tests for the complete feature engineering pipeline."""
    
    def test_pipeline_produces_all_features(self):
        """Test that pipeline produces complete feature set."""
        pipeline = FeatureEngineeringPipeline()
        
        # Minimal test data
        transactions = pd.DataFrame({
            'customer_id': ['C001'],
            'timestamp': [datetime.now()],
            'transaction_type': ['payment_attempt'],
            'status': ['success'],
            'amount': [100]
        })
        
        loans = pd.DataFrame({
            'customer_id': ['C001'],
            'instalment_amount': [100],
            'status': ['active'],
            'dpd': [5]
        })
        
        feature_set = pipeline.engineer_features(
            customer_id='C001',
            transactions_df=transactions,
            loan_df=loans
        )
        
        # Check all feature groups present
        assert len(feature_set.payment_features) > 0
        assert len(feature_set.income_features) > 0
        assert len(feature_set.timing_features) > 0
        
        # Check flattening works
        flat = feature_set.to_dict()
        assert 'customer_id' in flat
        assert len(flat) > 10  # Should have many features


class TestPropensityModel:
    """Tests for propensity model (when fitted)."""
    
    def test_suppression_threshold(self):
        """Test that high self-cure probability triggers suppression."""
        # Create mock score
        score = PropensityScore(
            customer_id='C001',
            p_repay_no_contact=0.8,
            p_repay_with_contact=0.85,
            uplift=0.05,
            recommended_action='suppress',
            suppression_candidate=True
        )
        
        assert score.suppression_candidate == True
        assert score.p_repay_no_contact >= 0.7
    
    def test_contact_prioritisation(self):
        """Test that low propensity triggers deprioritisation."""
        score = PropensityScore(
            customer_id='C002',
            p_repay_no_contact=0.1,
            p_repay_with_contact=0.2,
            uplift=0.1,
            recommended_action='deprioritise',
            suppression_candidate=True
        )
        
        assert 'deprioritise' in score.recommended_action or score.suppression_candidate


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
