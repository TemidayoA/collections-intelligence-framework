"""
Willingness-Capacity Matrix for Collections Segmentation

This module implements a two-dimensional segmentation framework that separates
borrower willingness (intent to repay) from capacity (ability to repay).

The framework produces four segments with distinct optimal strategies:
- Segment A (High W, High C): Monitor - minimal intervention needed
- Segment B (High W, Low C): Restructure - willing but constrained
- Segment C (Low W, High C): Escalate - capable but unwilling
- Segment D (Low W, Low C): Deprioritise - minimise resource allocation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum


class Segment(Enum):
    """Collection strategy segments based on willingness-capacity matrix."""
    A_MONITOR = "monitor"
    B_RESTRUCTURE = "restructure"
    C_ESCALATE = "escalate"
    D_DEPRIORITISE = "deprioritise"


@dataclass
class SegmentationResult:
    """Result of segmentation for a single account."""
    customer_id: str
    willingness_score: float
    capacity_score: float
    segment: Segment
    recommended_action: str
    confidence: float


class WillingnessScorer:
    """
    Calculates willingness score based on behavioural signals.
    
    Willingness indicates intent to repay, derived from:
    - Payment attempts (even if failed)
    - Response to contact
    - Proactive communication
    - Promise-to-pay history
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'payment_attempts': 0.30,
            'contact_response': 0.25,
            'proactive_communication': 0.15,
            'ptp_kept_ratio': 0.20,
            'channel_blocking': -0.10  # Negative indicator
        }
    
    def calculate(self, features: Dict) -> float:
        """
        Calculate willingness score from feature dictionary.
        
        Args:
            features: Dictionary containing willingness indicators
            
        Returns:
            Willingness score between 0 and 1
        """
        score = 0.0
        
        # Payment attempts (positive signal)
        if 'payment_attempts_30d' in features:
            attempts = features['payment_attempts_30d']
            attempt_score = min(attempts / 3, 1.0)  # Cap at 3 attempts
            score += self.weights['payment_attempts'] * attempt_score
        
        # Contact response rate
        if 'contact_response_rate' in features:
            score += self.weights['contact_response'] * features['contact_response_rate']
        
        # Proactive communication
        if 'proactive_contact' in features:
            score += self.weights['proactive_communication'] * float(features['proactive_contact'])
        
        # Promise-to-pay history
        if 'ptp_kept_ratio' in features:
            score += self.weights['ptp_kept_ratio'] * features['ptp_kept_ratio']
        
        # Channel blocking (negative indicator)
        if 'channels_blocked' in features:
            blocking_penalty = min(features['channels_blocked'] / 3, 1.0)
            score += self.weights['channel_blocking'] * blocking_penalty
        
        return np.clip(score, 0, 1)


class CapacityScorer:
    """
    Calculates capacity score based on financial signals.
    
    Capacity indicates ability to repay, derived from:
    - Income regularity
    - Transaction patterns
    - Debt burden indicators
    - Historical repayment success
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'income_regularity': 0.30,
            'transaction_activity': 0.20,
            'previous_completion': 0.25,
            'debt_burden': -0.15,  # Negative indicator
            'account_stability': 0.10
        }
    
    def calculate(self, features: Dict) -> float:
        """
        Calculate capacity score from feature dictionary.
        
        Args:
            features: Dictionary containing capacity indicators
            
        Returns:
            Capacity score between 0 and 1
        """
        score = 0.0
        
        # Income regularity (coefficient of variation of inflows)
        if 'income_regularity_score' in features:
            score += self.weights['income_regularity'] * features['income_regularity_score']
        
        # Transaction activity (normalised)
        if 'transaction_activity_score' in features:
            score += self.weights['transaction_activity'] * features['transaction_activity_score']
        
        # Previous loan completion rate
        if 'previous_completion_rate' in features:
            score += self.weights['previous_completion'] * features['previous_completion_rate']
        
        # Debt burden (negative indicator)
        if 'debt_to_income_ratio' in features:
            dti = features['debt_to_income_ratio']
            burden_penalty = min(dti / 0.5, 1.0)  # Penalty increases up to 50% DTI
            score += self.weights['debt_burden'] * burden_penalty
        
        # Account stability
        if 'account_age_months' in features:
            stability = min(features['account_age_months'] / 12, 1.0)
            score += self.weights['account_stability'] * stability
        
        return np.clip(score, 0, 1)


class WillingnessCapacityMatrix:
    """
    Main segmentation class implementing the Willingness-Capacity Matrix.
    
    Example usage:
        matrix = WillingnessCapacityMatrix()
        result = matrix.segment(customer_id='C001', features=feature_dict)
        print(result.segment, result.recommended_action)
    """
    
    def __init__(
        self,
        willingness_threshold: float = 0.5,
        capacity_threshold: float = 0.5,
        willingness_scorer: Optional[WillingnessScorer] = None,
        capacity_scorer: Optional[CapacityScorer] = None
    ):
        self.w_threshold = willingness_threshold
        self.c_threshold = capacity_threshold
        self.willingness_scorer = willingness_scorer or WillingnessScorer()
        self.capacity_scorer = capacity_scorer or CapacityScorer()
        
        self.action_map = {
            Segment.A_MONITOR: "Schedule light-touch reminder aligned to liquidity window",
            Segment.B_RESTRUCTURE: "Offer payment plan, deferral, or restructure options",
            Segment.C_ESCALATE: "Initiate targeted escalation with clear consequences",
            Segment.D_DEPRIORITISE: "Minimise resource allocation; consider write-off path"
        }
    
    def _determine_segment(
        self,
        willingness: float,
        capacity: float
    ) -> Tuple[Segment, float]:
        """
        Determine segment based on scores and calculate confidence.
        
        Confidence is based on distance from thresholds - accounts near
        boundaries have lower confidence.
        """
        # Determine segment
        high_willingness = willingness >= self.w_threshold
        high_capacity = capacity >= self.c_threshold
        
        if high_willingness and high_capacity:
            segment = Segment.A_MONITOR
        elif high_willingness and not high_capacity:
            segment = Segment.B_RESTRUCTURE
        elif not high_willingness and high_capacity:
            segment = Segment.C_ESCALATE
        else:
            segment = Segment.D_DEPRIORITISE
        
        # Calculate confidence based on distance from thresholds
        w_distance = abs(willingness - self.w_threshold)
        c_distance = abs(capacity - self.c_threshold)
        min_distance = min(w_distance, c_distance)
        
        # Confidence scales from 0.5 (at threshold) to 1.0 (far from threshold)
        confidence = 0.5 + min_distance
        confidence = min(confidence, 1.0)
        
        return segment, confidence
    
    def segment(
        self,
        customer_id: str,
        features: Dict,
        willingness_override: Optional[float] = None,
        capacity_override: Optional[float] = None
    ) -> SegmentationResult:
        """
        Segment a single customer based on features.
        
        Args:
            customer_id: Unique customer identifier
            features: Dictionary of feature values
            willingness_override: Optional pre-calculated willingness score
            capacity_override: Optional pre-calculated capacity score
            
        Returns:
            SegmentationResult with segment and recommended action
        """
        # Calculate scores
        willingness = willingness_override or self.willingness_scorer.calculate(features)
        capacity = capacity_override or self.capacity_scorer.calculate(features)
        
        # Determine segment
        segment, confidence = self._determine_segment(willingness, capacity)
        
        return SegmentationResult(
            customer_id=customer_id,
            willingness_score=willingness,
            capacity_score=capacity,
            segment=segment,
            recommended_action=self.action_map[segment],
            confidence=confidence
        )
    
    def segment_portfolio(
        self,
        portfolio_df: pd.DataFrame,
        customer_id_col: str = 'customer_id'
    ) -> pd.DataFrame:
        """
        Segment an entire portfolio.
        
        Args:
            portfolio_df: DataFrame with customer features
            customer_id_col: Name of customer ID column
            
        Returns:
            DataFrame with segmentation results appended
        """
        results = []
        
        for _, row in portfolio_df.iterrows():
            features = row.to_dict()
            customer_id = features.pop(customer_id_col)
            
            result = self.segment(customer_id, features)
            results.append({
                'customer_id': result.customer_id,
                'willingness_score': result.willingness_score,
                'capacity_score': result.capacity_score,
                'segment': result.segment.value,
                'recommended_action': result.recommended_action,
                'confidence': result.confidence
            })
        
        return pd.DataFrame(results)
    
    def get_segment_distribution(self, results_df: pd.DataFrame) -> Dict[str, int]:
        """Get count of customers in each segment."""
        return results_df['segment'].value_counts().to_dict()
    
    def get_high_confidence_actions(
        self,
        results_df: pd.DataFrame,
        min_confidence: float = 0.7
    ) -> pd.DataFrame:
        """Filter to high-confidence segmentation results."""
        return results_df[results_df['confidence'] >= min_confidence]


# Convenience function for quick segmentation
def quick_segment(
    willingness_score: float,
    capacity_score: float,
    w_threshold: float = 0.5,
    c_threshold: float = 0.5
) -> str:
    """
    Quick segmentation without full feature calculation.
    
    Args:
        willingness_score: Pre-calculated willingness (0-1)
        capacity_score: Pre-calculated capacity (0-1)
        w_threshold: Willingness threshold
        c_threshold: Capacity threshold
        
    Returns:
        Segment name as string
    """
    if willingness_score >= w_threshold:
        if capacity_score >= c_threshold:
            return 'A_MONITOR'
        return 'B_RESTRUCTURE'
    else:
        if capacity_score >= c_threshold:
            return 'C_ESCALATE'
        return 'D_DEPRIORITISE'


if __name__ == "__main__":
    # Example usage
    matrix = WillingnessCapacityMatrix()
    
    # Sample customer features
    sample_features = {
        'payment_attempts_30d': 2,
        'contact_response_rate': 0.6,
        'proactive_contact': True,
        'ptp_kept_ratio': 0.5,
        'channels_blocked': 0,
        'income_regularity_score': 0.8,
        'transaction_activity_score': 0.7,
        'previous_completion_rate': 1.0,
        'debt_to_income_ratio': 0.3,
        'account_age_months': 8
    }
    
    result = matrix.segment('CUST001', sample_features)
    
    print(f"Customer: {result.customer_id}")
    print(f"Willingness Score: {result.willingness_score:.2f}")
    print(f"Capacity Score: {result.capacity_score:.2f}")
    print(f"Segment: {result.segment.value}")
    print(f"Action: {result.recommended_action}")
    print(f"Confidence: {result.confidence:.2f}")
