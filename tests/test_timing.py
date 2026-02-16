"""
Unit tests for Contact Timing Optimisation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.contact_optimization.timing_optimizer import (
    ContactTimingOptimizer,
    LiquidityPatternDetector,
    LiquidityPattern,
    ContactWindow,
    TimingRecommendation
)
from src.contact_optimization.channel_selector import (
    ChannelSelector,
    ChannelScorer,
    Channel,
    ChannelRecommendation,
    ChannelProfile
)


class TestLiquidityPatternDetector:
    """Tests for liquidity pattern detection."""
    
    @pytest.fixture
    def detector(self):
        return LiquidityPatternDetector(min_observations=3)
    
    def test_monthly_salary_detection(self, detector):
        """Test detection of monthly salary pattern."""
        # Create transactions on 25th of each month
        transactions = pd.DataFrame({
            'customer_id': ['C001'] * 6,
            'timestamp': pd.to_datetime([
                '2024-01-25', '2024-02-25', '2024-03-25',
                '2024-04-25', '2024-05-25', '2024-06-25'
            ]),
            'amount': [50000] * 6
        })
        
        pattern, details = detector.detect_pattern(transactions, 'C001')
        
        assert pattern == LiquidityPattern.MONTHLY_SALARY
        assert details['salary_day'] == 25
        assert details['confidence'] >= 0.6
    
    def test_weekly_wage_detection(self, detector):
        """Test detection of weekly wage pattern."""
        # Create transactions every Friday
        base_date = datetime(2024, 1, 5)  # A Friday
        transactions = pd.DataFrame({
            'customer_id': ['C001'] * 8,
            'timestamp': [base_date + timedelta(weeks=i) for i in range(8)],
            'amount': [10000] * 8
        })
        
        pattern, details = detector.detect_pattern(transactions, 'C001')
        
        assert pattern == LiquidityPattern.WEEKLY_WAGE
        assert details['wage_day'] == 4  # Friday = 4
    
    def test_irregular_pattern_detection(self, detector):
        """Test detection of irregular income pattern."""
        # Random intervals
        transactions = pd.DataFrame({
            'customer_id': ['C001'] * 6,
            'timestamp': pd.to_datetime([
                '2024-01-05', '2024-01-20', '2024-02-15',
                '2024-03-01', '2024-03-28', '2024-04-10'
            ]),
            'amount': [5000, 8000, 3000, 12000, 6000, 9000]
        })
        
        pattern, details = detector.detect_pattern(transactions, 'C001')
        
        assert pattern == LiquidityPattern.GIG_IRREGULAR
    
    def test_insufficient_data(self, detector):
        """Test handling of insufficient transaction data."""
        transactions = pd.DataFrame({
            'customer_id': ['C001'],
            'timestamp': pd.to_datetime(['2024-01-15']),
            'amount': [5000]
        })
        
        pattern, details = detector.detect_pattern(transactions, 'C001')
        
        assert pattern == LiquidityPattern.UNKNOWN


class TestContactTimingOptimizer:
    """Tests for contact timing optimisation."""
    
    @pytest.fixture
    def optimizer(self):
        return ContactTimingOptimizer(
            window_before_days=1,
            window_after_days=3
        )
    
    @pytest.fixture
    def salary_transactions(self):
        """Create transactions showing monthly salary on 25th."""
        return pd.DataFrame({
            'customer_id': ['C001'] * 6,
            'timestamp': pd.to_datetime([
                '2024-01-25', '2024-02-25', '2024-03-25',
                '2024-04-25', '2024-05-25', '2024-06-25'
            ]),
            'amount': [50000] * 6
        })
    
    def test_contact_now_within_window(self, optimizer, salary_transactions):
        """Test recommendation to contact when within salary window."""
        optimizer.learn_patterns(salary_transactions, ['C001'])
        
        # Test date: July 26 (one day after salary)
        test_date = datetime(2024, 7, 26)
        
        rec = optimizer.get_recommendation('C001', test_date)
        
        assert rec.current_window == ContactWindow.CONTACT_NOW
        assert rec.days_until_window == 0
    
    def test_wait_before_window(self, optimizer, salary_transactions):
        """Test recommendation to wait when before salary window."""
        optimizer.learn_patterns(salary_transactions, ['C001'])
        
        # Test date: July 15 (10 days before salary)
        test_date = datetime(2024, 7, 15)
        
        rec = optimizer.get_recommendation('C001', test_date)
        
        assert rec.current_window == ContactWindow.WAIT_FOR_WINDOW
        assert rec.days_until_window > 0
    
    def test_unknown_customer(self, optimizer):
        """Test handling of unknown customer."""
        rec = optimizer.get_recommendation('UNKNOWN', datetime.now())
        
        assert rec.current_window == ContactWindow.NO_PATTERN
        assert rec.confidence == 0.0
    
    def test_get_contactable_now(self, optimizer, salary_transactions):
        """Test batch filtering for contactable customers."""
        optimizer.learn_patterns(salary_transactions, ['C001'])
        
        # Within salary window
        test_date = datetime(2024, 7, 26)
        
        contactable = optimizer.get_contactable_now(['C001', 'C002'], test_date)
        
        assert 'C001' in contactable


class TestChannelScorer:
    """Tests for channel scoring."""
    
    @pytest.fixture
    def scorer(self):
        return ChannelScorer(
            fatigue_window_hours=48,
            min_contacts_for_preference=3
        )
    
    def test_channel_response_rates(self, scorer):
        """Test calculation of channel response rates."""
        now = datetime.now()
        
        contacts = pd.DataFrame({
            'customer_id': ['C001'] * 10,
            'channel': ['sms'] * 5 + ['call'] * 5,
            'timestamp': [now - timedelta(days=i*5) for i in range(10)],
            'responded': [True, True, True, False, False,  # SMS: 60%
                         True, False, False, False, False]  # Call: 20%
        })
        
        profile = scorer.score_channels('C001', contacts)
        
        assert profile.channel_scores['sms'] > profile.channel_scores['call']
        assert profile.preferred_channel == 'sms'
    
    def test_fatigue_detection(self, scorer):
        """Test detection of fatigued channels."""
        now = datetime.now()
        
        # Recent SMS contact (should be fatigued)
        contacts = pd.DataFrame({
            'customer_id': ['C001'] * 3,
            'channel': ['sms', 'sms', 'call'],
            'timestamp': [
                now - timedelta(hours=12),  # Very recent
                now - timedelta(days=5),
                now - timedelta(days=10)
            ],
            'responded': [True, True, True]
        })
        
        profile = scorer.score_channels('C001', contacts)
        
        assert 'sms' in profile.fatigued_channels
    
    def test_blocked_channel_handling(self, scorer):
        """Test handling of blocked channels."""
        now = datetime.now()
        
        contacts = pd.DataFrame({
            'customer_id': ['C001'] * 5,
            'channel': ['call'] * 5,
            'timestamp': [now - timedelta(days=i*3) for i in range(5)],
            'responded': [False] * 5,
            'blocked': [False, False, True, True, True]
        })
        
        profile = scorer.score_channels('C001', contacts)
        
        assert 'call' in profile.blocked_channels
        assert profile.channel_scores['call'] == 0.0


class TestChannelSelector:
    """Tests for channel selection."""
    
    @pytest.fixture
    def selector(self):
        return ChannelSelector()
    
    @pytest.fixture
    def sample_history(self):
        """Create sample contact history."""
        now = datetime.now()
        return pd.DataFrame({
            'customer_id': ['C001'] * 12,
            'channel': ['sms', 'sms', 'sms', 'whatsapp', 'whatsapp', 'whatsapp',
                       'call', 'call', 'call', 'email', 'email', 'email'],
            'timestamp': [now - timedelta(days=i*5) for i in range(12)],
            'responded': [True, True, False,    # SMS: 67%
                         True, True, True,       # WhatsApp: 100%
                         False, True, False,     # Call: 33%
                         False, False, False],   # Email: 0%
            'blocked': [False] * 12
        })
    
    def test_selects_best_channel(self, selector, sample_history):
        """Test that selector picks highest response rate channel."""
        rec = selector.select('C001', sample_history)
        
        # WhatsApp has 100% response rate
        assert rec.recommended_channel == Channel.WHATSAPP
        assert rec.confidence > 0.5
    
    def test_escalation_prefers_higher_intensity(self, selector, sample_history):
        """Test that escalation level increases channel intensity."""
        rec_normal = selector.select('C001', sample_history, escalation_level=0)
        rec_escalated = selector.select('C001', sample_history, escalation_level=2)
        
        # Escalated should prefer call over SMS/WhatsApp
        intensity_normal = selector.channel_intensity[rec_normal.recommended_channel]
        intensity_escalated = selector.channel_intensity[rec_escalated.recommended_channel]
        
        # Escalation should use same or higher intensity
        assert intensity_escalated.value >= intensity_normal.value
    
    def test_respects_exclusions(self, selector, sample_history):
        """Test that selector respects excluded channels."""
        rec = selector.select(
            'C001', 
            sample_history,
            exclude_channels=[Channel.WHATSAPP, Channel.SMS]
        )
        
        assert rec.recommended_channel not in [Channel.WHATSAPP, Channel.SMS]
    
    def test_batch_selection(self, selector, sample_history):
        """Test batch channel selection."""
        # Add another customer
        now = datetime.now()
        extended_history = pd.concat([
            sample_history,
            pd.DataFrame({
                'customer_id': ['C002'] * 3,
                'channel': ['sms', 'call', 'email'],
                'timestamp': [now - timedelta(days=i*5) for i in range(3)],
                'responded': [True, False, False],
                'blocked': [False, False, False]
            })
        ])
        
        results = selector.select_batch(['C001', 'C002'], extended_history)
        
        assert len(results) == 2
        assert 'recommended_channel' in results.columns
        assert 'confidence' in results.columns


class TestTimingRecommendation:
    """Tests for timing recommendation dataclass."""
    
    def test_recommendation_creation(self):
        """Test creation of timing recommendation."""
        rec = TimingRecommendation(
            customer_id='C001',
            liquidity_pattern=LiquidityPattern.MONTHLY_SALARY,
            current_window=ContactWindow.CONTACT_NOW,
            optimal_contact_date=datetime(2024, 7, 26),
            days_until_window=0,
            confidence=0.85,
            recommendation="Contact now - within 3 days of salary"
        )
        
        assert rec.customer_id == 'C001'
        assert rec.liquidity_pattern == LiquidityPattern.MONTHLY_SALARY
        assert rec.confidence == 0.85


class TestChannelRecommendation:
    """Tests for channel recommendation dataclass."""
    
    def test_recommendation_with_backup(self):
        """Test channel recommendation with backup."""
        rec = ChannelRecommendation(
            customer_id='C001',
            recommended_channel=Channel.WHATSAPP,
            backup_channel=Channel.SMS,
            confidence=0.75,
            reason="WhatsApp has highest response rate (85%)",
            avoid_channels=[Channel.CALL]
        )
        
        assert rec.recommended_channel == Channel.WHATSAPP
        assert rec.backup_channel == Channel.SMS
        assert Channel.CALL in rec.avoid_channels


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
