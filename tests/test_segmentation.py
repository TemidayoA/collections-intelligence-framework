"""
Unit tests for Willingness-Capacity Segmentation
"""

import pytest
import numpy as np
import pandas as pd
from src.segmentation.willingness_capacity import (
    WillingnessCapacityMatrix,
    WillingnessScorer,
    CapacityScorer,
    Segment,
    quick_segment
)


class TestWillingnessScorer:
    """Tests for willingness score calculation."""
    
    def test_high_willingness_signals(self):
        """Test that positive signals produce high willingness score."""
        scorer = WillingnessScorer()
        
        features = {
            'payment_attempts_30d': 3,
            'contact_response_rate': 0.8,
            'proactive_contact': True,
            'ptp_kept_ratio': 0.9,
            'channels_blocked': 0
        }
        
        score = scorer.calculate(features)
        assert score >= 0.7, "High positive signals should yield high willingness"
    
    def test_low_willingness_signals(self):
        """Test that negative signals produce low willingness score."""
        scorer = WillingnessScorer()
        
        features = {
            'payment_attempts_30d': 0,
            'contact_response_rate': 0.1,
            'proactive_contact': False,
            'ptp_kept_ratio': 0.0,
            'channels_blocked': 3
        }
        
        score = scorer.calculate(features)
        assert score <= 0.3, "Negative signals should yield low willingness"
    
    def test_score_bounded(self):
        """Test that scores are bounded between 0 and 1."""
        scorer = WillingnessScorer()
        
        # Extreme positive
        features_high = {
            'payment_attempts_30d': 100,
            'contact_response_rate': 1.0,
            'proactive_contact': True,
            'ptp_kept_ratio': 1.0,
            'channels_blocked': 0
        }
        
        # Extreme negative
        features_low = {
            'payment_attempts_30d': 0,
            'contact_response_rate': 0.0,
            'proactive_contact': False,
            'ptp_kept_ratio': 0.0,
            'channels_blocked': 10
        }
        
        assert 0 <= scorer.calculate(features_high) <= 1
        assert 0 <= scorer.calculate(features_low) <= 1


class TestCapacityScorer:
    """Tests for capacity score calculation."""
    
    def test_high_capacity_signals(self):
        """Test that strong financial signals produce high capacity score."""
        scorer = CapacityScorer()
        
        features = {
            'income_regularity_score': 0.9,
            'transaction_activity_score': 0.8,
            'previous_completion_rate': 1.0,
            'debt_to_income_ratio': 0.2,
            'account_age_months': 24
        }
        
        score = scorer.calculate(features)
        assert score >= 0.6, "Strong financial signals should yield high capacity"
    
    def test_low_capacity_signals(self):
        """Test that weak financial signals produce low capacity score."""
        scorer = CapacityScorer()
        
        features = {
            'income_regularity_score': 0.2,
            'transaction_activity_score': 0.1,
            'previous_completion_rate': 0.3,
            'debt_to_income_ratio': 0.8,
            'account_age_months': 1
        }
        
        score = scorer.calculate(features)
        assert score <= 0.4, "Weak financial signals should yield low capacity"


class TestWillingnessCapacityMatrix:
    """Tests for the main segmentation matrix."""
    
    @pytest.fixture
    def matrix(self):
        return WillingnessCapacityMatrix()
    
    def test_segment_a_high_high(self, matrix):
        """Test Segment A: High willingness, high capacity."""
        features = {
            'payment_attempts_30d': 2,
            'contact_response_rate': 0.8,
            'proactive_contact': True,
            'ptp_kept_ratio': 0.8,
            'channels_blocked': 0,
            'income_regularity_score': 0.9,
            'transaction_activity_score': 0.8,
            'previous_completion_rate': 1.0,
            'debt_to_income_ratio': 0.2,
            'account_age_months': 12
        }
        
        result = matrix.segment('TEST001', features)
        assert result.segment == Segment.A_MONITOR
    
    def test_segment_b_high_low(self, matrix):
        """Test Segment B: High willingness, low capacity."""
        features = {
            'payment_attempts_30d': 3,  # Trying to pay
            'contact_response_rate': 0.9,
            'proactive_contact': True,
            'ptp_kept_ratio': 0.7,
            'channels_blocked': 0,
            'income_regularity_score': 0.2,  # Irregular income
            'transaction_activity_score': 0.2,
            'previous_completion_rate': 0.5,
            'debt_to_income_ratio': 0.7,  # High debt burden
            'account_age_months': 3
        }
        
        result = matrix.segment('TEST002', features)
        assert result.segment == Segment.B_RESTRUCTURE
    
    def test_segment_c_low_high(self, matrix):
        """Test Segment C: Low willingness, high capacity."""
        features = {
            'payment_attempts_30d': 0,  # Not trying
            'contact_response_rate': 0.1,
            'proactive_contact': False,
            'ptp_kept_ratio': 0.1,
            'channels_blocked': 2,
            'income_regularity_score': 0.9,  # Has money
            'transaction_activity_score': 0.8,
            'previous_completion_rate': 0.8,
            'debt_to_income_ratio': 0.1,
            'account_age_months': 18
        }
        
        result = matrix.segment('TEST003', features)
        assert result.segment == Segment.C_ESCALATE
    
    def test_segment_d_low_low(self, matrix):
        """Test Segment D: Low willingness, low capacity."""
        features = {
            'payment_attempts_30d': 0,
            'contact_response_rate': 0.0,
            'proactive_contact': False,
            'ptp_kept_ratio': 0.0,
            'channels_blocked': 3,
            'income_regularity_score': 0.1,
            'transaction_activity_score': 0.1,
            'previous_completion_rate': 0.2,
            'debt_to_income_ratio': 0.9,
            'account_age_months': 1
        }
        
        result = matrix.segment('TEST004', features)
        assert result.segment == Segment.D_DEPRIORITISE
    
    def test_recommended_action_present(self, matrix):
        """Test that segmentation includes recommended action."""
        features = {
            'payment_attempts_30d': 1,
            'contact_response_rate': 0.5,
            'income_regularity_score': 0.5,
            'previous_completion_rate': 0.5
        }
        
        result = matrix.segment('TEST005', features)
        assert result.recommended_action is not None
        assert len(result.recommended_action) > 0
    
    def test_confidence_calculation(self, matrix):
        """Test that confidence is higher for clear-cut cases."""
        # Clear Segment A case
        features_clear = {
            'payment_attempts_30d': 5,
            'contact_response_rate': 1.0,
            'proactive_contact': True,
            'ptp_kept_ratio': 1.0,
            'channels_blocked': 0,
            'income_regularity_score': 1.0,
            'transaction_activity_score': 1.0,
            'previous_completion_rate': 1.0,
            'debt_to_income_ratio': 0.0,
            'account_age_months': 36
        }
        
        # Borderline case
        features_borderline = {
            'payment_attempts_30d': 1,
            'contact_response_rate': 0.5,
            'proactive_contact': False,
            'ptp_kept_ratio': 0.5,
            'channels_blocked': 1,
            'income_regularity_score': 0.5,
            'transaction_activity_score': 0.5,
            'previous_completion_rate': 0.5,
            'debt_to_income_ratio': 0.25,
            'account_age_months': 6
        }
        
        result_clear = matrix.segment('CLEAR', features_clear)
        result_borderline = matrix.segment('BORDER', features_borderline)
        
        assert result_clear.confidence >= result_borderline.confidence


class TestQuickSegment:
    """Tests for the quick segmentation function."""
    
    def test_all_segments(self):
        """Test that quick_segment returns correct segments."""
        assert quick_segment(0.8, 0.8) == 'A_MONITOR'
        assert quick_segment(0.8, 0.2) == 'B_RESTRUCTURE'
        assert quick_segment(0.2, 0.8) == 'C_ESCALATE'
        assert quick_segment(0.2, 0.2) == 'D_DEPRIORITISE'
    
    def test_threshold_boundary(self):
        """Test behaviour at threshold boundaries."""
        # Exactly at threshold should be high
        assert quick_segment(0.5, 0.5) == 'A_MONITOR'
        
        # Just below should be low
        assert quick_segment(0.49, 0.49) == 'D_DEPRIORITISE'


class TestPortfolioSegmentation:
    """Tests for batch portfolio segmentation."""
    
    def test_portfolio_segmentation(self):
        """Test segmentation of a portfolio DataFrame."""
        matrix = WillingnessCapacityMatrix()
        
        portfolio = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003'],
            'payment_attempts_30d': [3, 0, 1],
            'contact_response_rate': [0.9, 0.1, 0.5],
            'income_regularity_score': [0.8, 0.8, 0.3],
            'previous_completion_rate': [1.0, 0.9, 0.4]
        })
        
        results = matrix.segment_portfolio(portfolio)
        
        assert len(results) == 3
        assert 'segment' in results.columns
        assert 'recommended_action' in results.columns
        assert results['customer_id'].tolist() == ['C001', 'C002', 'C003']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
