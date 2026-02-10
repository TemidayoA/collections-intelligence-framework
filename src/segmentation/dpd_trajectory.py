"""
DPD Trajectory Analysis

Days Past Due (DPD) is traditionally used as a static bucket (1-7, 8-30, 31-60).
This module treats DPD as a trajectory—the slope of delinquency movement has
greater predictive value than the absolute number.

Two borrowers at 15 DPD can have radically different risk profiles:
- One decelerating (slowing drift, likely to stabilise)
- One accelerating (rapid deterioration, structural default)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta


class TrajectoryType(Enum):
    """Classification of DPD movement patterns."""
    RECOVERING = "recovering"       # Negative slope, improving
    STABLE = "stable"               # Near-zero slope, plateau
    DRIFTING = "drifting"          # Slow positive slope, chronic drift
    FALLING_OFF = "falling_off"    # Steep positive slope, rapid deterioration
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class TrajectoryResult:
    """Result of trajectory analysis for a single account."""
    customer_id: str
    current_dpd: int
    slope: float
    trajectory_type: TrajectoryType
    risk_score: float
    days_to_threshold: Optional[int]  # Estimated days to next DPD bucket
    recommended_strategy: str


class DPDTrajectoryAnalyser:
    """
    Analyses DPD movement patterns to classify borrower trajectories.
    
    Example usage:
        analyser = DPDTrajectoryAnalyser()
        result = analyser.analyse(
            customer_id='C001',
            dpd_history=[0, 3, 5, 8, 10, 12, 14]
        )
        print(result.trajectory_type, result.recommended_strategy)
    """
    
    def __init__(
        self,
        window_size: int = 7,
        slope_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialise trajectory analyser.
        
        Args:
            window_size: Number of days to consider for slope calculation
            slope_thresholds: Custom thresholds for trajectory classification
        """
        self.window_size = window_size
        self.slope_thresholds = slope_thresholds or {
            'recovering': -0.5,      # Slope below this = recovering
            'stable_upper': 0.5,     # Slope below this = stable
            'drifting_upper': 2.0    # Slope below this = drifting, above = falling off
        }
        
        self.strategy_map = {
            TrajectoryType.RECOVERING: "Monitor only - self-cure likely",
            TrajectoryType.STABLE: "Light-touch reminder at liquidity window",
            TrajectoryType.DRIFTING: "Proactive restructure offer or targeted escalation",
            TrajectoryType.FALLING_OFF: "Early deprioritisation or immediate restructure",
            TrajectoryType.INSUFFICIENT_DATA: "Collect more data before action"
        }
    
    def calculate_slope(self, dpd_history: List[int]) -> Tuple[float, bool]:
        """
        Calculate the slope of DPD movement.
        
        Args:
            dpd_history: List of DPD values over time (oldest to newest)
            
        Returns:
            Tuple of (slope, sufficient_data)
        """
        if len(dpd_history) < 2:
            return 0.0, False
        
        # Use most recent window
        history = dpd_history[-self.window_size:]
        
        if len(history) < 2:
            return 0.0, False
        
        # Linear regression
        x = np.arange(len(history))
        y = np.array(history)
        
        # Calculate slope using least squares
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                (n * np.sum(x ** 2) - np.sum(x) ** 2)
        
        sufficient_data = len(history) >= 3
        
        return slope, sufficient_data
    
    def classify_trajectory(self, slope: float, sufficient_data: bool) -> TrajectoryType:
        """
        Classify trajectory type based on slope.
        
        Args:
            slope: Calculated DPD slope
            sufficient_data: Whether enough data points exist
            
        Returns:
            TrajectoryType classification
        """
        if not sufficient_data:
            return TrajectoryType.INSUFFICIENT_DATA
        
        if slope < self.slope_thresholds['recovering']:
            return TrajectoryType.RECOVERING
        elif slope < self.slope_thresholds['stable_upper']:
            return TrajectoryType.STABLE
        elif slope < self.slope_thresholds['drifting_upper']:
            return TrajectoryType.DRIFTING
        else:
            return TrajectoryType.FALLING_OFF
    
    def calculate_risk_score(
        self,
        current_dpd: int,
        slope: float,
        trajectory_type: TrajectoryType
    ) -> float:
        """
        Calculate composite risk score combining position and trajectory.
        
        Args:
            current_dpd: Current days past due
            slope: DPD movement slope
            trajectory_type: Classified trajectory
            
        Returns:
            Risk score between 0 and 1
        """
        # Base risk from current DPD (normalised to 60 days)
        position_risk = min(current_dpd / 60, 1.0)
        
        # Trajectory risk modifier
        trajectory_weights = {
            TrajectoryType.RECOVERING: -0.2,
            TrajectoryType.STABLE: 0.0,
            TrajectoryType.DRIFTING: 0.15,
            TrajectoryType.FALLING_OFF: 0.3,
            TrajectoryType.INSUFFICIENT_DATA: 0.1
        }
        
        trajectory_modifier = trajectory_weights.get(trajectory_type, 0)
        
        # Slope contribution (normalised)
        slope_contribution = np.clip(slope / 5, -0.2, 0.2)
        
        # Composite score
        risk_score = position_risk + trajectory_modifier + slope_contribution
        
        return np.clip(risk_score, 0, 1)
    
    def estimate_days_to_threshold(
        self,
        current_dpd: int,
        slope: float,
        threshold: int = 30
    ) -> Optional[int]:
        """
        Estimate days until DPD reaches a threshold (e.g., 30 DPD bucket).
        
        Args:
            current_dpd: Current days past due
            slope: DPD movement slope
            threshold: Target DPD threshold
            
        Returns:
            Estimated days, or None if not applicable
        """
        if slope <= 0:
            return None  # Not deteriorating
        
        if current_dpd >= threshold:
            return 0  # Already past threshold
        
        days_remaining = threshold - current_dpd
        estimated_days = int(days_remaining / slope)
        
        return max(estimated_days, 1)
    
    def analyse(
        self,
        customer_id: str,
        dpd_history: List[int],
        current_dpd: Optional[int] = None
    ) -> TrajectoryResult:
        """
        Perform complete trajectory analysis for a customer.
        
        Args:
            customer_id: Unique customer identifier
            dpd_history: List of DPD values over time
            current_dpd: Current DPD (defaults to last in history)
            
        Returns:
            TrajectoryResult with full analysis
        """
        current_dpd = current_dpd or dpd_history[-1] if dpd_history else 0
        
        # Calculate slope
        slope, sufficient_data = self.calculate_slope(dpd_history)
        
        # Classify trajectory
        trajectory_type = self.classify_trajectory(slope, sufficient_data)
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(current_dpd, slope, trajectory_type)
        
        # Estimate days to 30 DPD threshold
        days_to_threshold = self.estimate_days_to_threshold(current_dpd, slope, 30)
        
        return TrajectoryResult(
            customer_id=customer_id,
            current_dpd=current_dpd,
            slope=slope,
            trajectory_type=trajectory_type,
            risk_score=risk_score,
            days_to_threshold=days_to_threshold,
            recommended_strategy=self.strategy_map[trajectory_type]
        )
    
    def analyse_portfolio(
        self,
        portfolio_df: pd.DataFrame,
        customer_id_col: str = 'customer_id',
        dpd_history_col: str = 'dpd_history'
    ) -> pd.DataFrame:
        """
        Analyse trajectory for entire portfolio.
        
        Args:
            portfolio_df: DataFrame with customer DPD histories
            customer_id_col: Name of customer ID column
            dpd_history_col: Name of DPD history column (list values)
            
        Returns:
            DataFrame with trajectory analysis results
        """
        results = []
        
        for _, row in portfolio_df.iterrows():
            customer_id = row[customer_id_col]
            dpd_history = row[dpd_history_col]
            
            result = self.analyse(customer_id, dpd_history)
            
            results.append({
                'customer_id': result.customer_id,
                'current_dpd': result.current_dpd,
                'slope': result.slope,
                'trajectory_type': result.trajectory_type.value,
                'risk_score': result.risk_score,
                'days_to_threshold': result.days_to_threshold,
                'recommended_strategy': result.recommended_strategy
            })
        
        return pd.DataFrame(results)


class CohortTrajectoryTracker:
    """
    Tracks trajectory patterns across cohorts for portfolio-level insights.
    
    Identifies:
    - Early stabilisers (quick plateau, high self-cure rate)
    - Late recoveries (delayed but eventual payment)
    - Chronic drifters (slow continuous increase)
    - Rapid falloffs (steep acceleration to write-off)
    """
    
    def __init__(self, observation_window: int = 30):
        self.observation_window = observation_window
        self.cohort_patterns = {}
    
    def track_cohort(
        self,
        cohort_id: str,
        dpd_matrix: np.ndarray
    ) -> Dict:
        """
        Track trajectory patterns for a cohort.
        
        Args:
            cohort_id: Identifier for the cohort (e.g., 'Jan_2024')
            dpd_matrix: Matrix of shape (n_customers, n_days)
            
        Returns:
            Dictionary of cohort trajectory statistics
        """
        analyser = DPDTrajectoryAnalyser()
        
        trajectories = {
            'recovering': 0,
            'stable': 0,
            'drifting': 0,
            'falling_off': 0,
            'insufficient_data': 0
        }
        
        slopes = []
        risk_scores = []
        
        for i in range(dpd_matrix.shape[0]):
            dpd_history = dpd_matrix[i, :].tolist()
            result = analyser.analyse(f'cohort_{i}', dpd_history)
            
            trajectories[result.trajectory_type.value] += 1
            slopes.append(result.slope)
            risk_scores.append(result.risk_score)
        
        pattern = {
            'cohort_id': cohort_id,
            'total_accounts': dpd_matrix.shape[0],
            'trajectory_distribution': trajectories,
            'mean_slope': np.mean(slopes),
            'median_slope': np.median(slopes),
            'mean_risk_score': np.mean(risk_scores),
            'pct_deteriorating': (trajectories['drifting'] + trajectories['falling_off']) / dpd_matrix.shape[0]
        }
        
        self.cohort_patterns[cohort_id] = pattern
        
        return pattern
    
    def compare_cohorts(self) -> pd.DataFrame:
        """Compare trajectory patterns across tracked cohorts."""
        if not self.cohort_patterns:
            return pd.DataFrame()
        
        return pd.DataFrame(self.cohort_patterns.values())


def calculate_acceleration(dpd_history: List[int], window: int = 5) -> float:
    """
    Calculate acceleration (second derivative) of DPD movement.
    
    Positive acceleration = deterioration speeding up
    Negative acceleration = deterioration slowing down
    
    Args:
        dpd_history: List of DPD values
        window: Window for calculation
        
    Returns:
        Acceleration value
    """
    if len(dpd_history) < 3:
        return 0.0
    
    history = dpd_history[-window:]
    
    if len(history) < 3:
        return 0.0
    
    # First differences (velocity)
    velocities = np.diff(history)
    
    # Second differences (acceleration)
    accelerations = np.diff(velocities)
    
    return np.mean(accelerations)


if __name__ == "__main__":
    # Example usage
    analyser = DPDTrajectoryAnalyser()
    
    # Different trajectory patterns
    examples = {
        'early_stabiliser': [0, 3, 5, 6, 6, 6, 6],      # Quick plateau
        'late_recovery': [0, 5, 10, 12, 10, 7, 3],       # Delayed recovery
        'chronic_drifter': [0, 2, 4, 6, 8, 10, 12],      # Slow drift
        'rapid_falloff': [0, 5, 12, 22, 35, 50, 68]      # Steep acceleration
    }
    
    print("DPD Trajectory Analysis Examples\n" + "=" * 50)
    
    for pattern_name, dpd_history in examples.items():
        result = analyser.analyse(pattern_name, dpd_history)
        
        print(f"\nPattern: {pattern_name}")
        print(f"  DPD History: {dpd_history}")
        print(f"  Current DPD: {result.current_dpd}")
        print(f"  Slope: {result.slope:.2f}")
        print(f"  Trajectory: {result.trajectory_type.value}")
        print(f"  Risk Score: {result.risk_score:.2f}")
        print(f"  Strategy: {result.recommended_strategy}")
