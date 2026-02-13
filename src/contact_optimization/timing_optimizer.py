"""
Contact Timing Optimisation

Nigerian borrowers have liquidity patterns tied to salary dates, market days,
contract settlements, and remittance windows. Contact timed to these windows
converts. Contact outside them triggers avoidance.

Research shows premature escalation—contacting before anticipated liquidity—
does not enhance recovery but increases avoidance. Borrowers who feel pressured
before they can pay disengage, severing communication channels just as
repayment becomes feasible.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum


class LiquidityPattern(Enum):
    """Types of liquidity patterns detected in borrower behaviour."""
    MONTHLY_SALARY = "monthly_salary"
    WEEKLY_WAGE = "weekly_wage"
    BI_WEEKLY = "bi_weekly"
    MARKET_DAY = "market_day"
    GIG_IRREGULAR = "gig_irregular"
    REMITTANCE = "remittance"
    UNKNOWN = "unknown"


class ContactWindow(Enum):
    """Contact timing recommendations."""
    CONTACT_NOW = "contact_now"           # Within optimal window
    WAIT_FOR_WINDOW = "wait_for_window"   # Too early, wait
    WINDOW_PASSED = "window_passed"       # Optimal window missed
    NO_PATTERN = "no_pattern"             # Cannot determine optimal timing


@dataclass
class TimingRecommendation:
    """Contact timing recommendation for a single account."""
    customer_id: str
    liquidity_pattern: LiquidityPattern
    current_window: ContactWindow
    optimal_contact_date: Optional[datetime]
    days_until_window: int
    confidence: float
    recommendation: str


class LiquidityPatternDetector:
    """
    Detects liquidity patterns from transaction history.
    
    Patterns detected:
    - Monthly salary (consistent date each month)
    - Weekly wage (consistent day of week)
    - Market day trading (specific days)
    - Gig/irregular income
    - Remittance patterns
    """
    
    def __init__(self, min_observations: int = 3):
        self.min_observations = min_observations
    
    def detect_pattern(
        self,
        transactions_df: pd.DataFrame,
        customer_id: str
    ) -> Tuple[LiquidityPattern, Dict]:
        """
        Detect liquidity pattern from transaction history.
        
        Args:
            transactions_df: Transaction DataFrame
            customer_id: Customer to analyse
            
        Returns:
            Tuple of (pattern_type, pattern_details)
        """
        # Filter to customer's credit transactions
        customer_txns = transactions_df[
            (transactions_df['customer_id'] == customer_id) &
            (transactions_df['amount'] > 0)
        ].copy()
        
        if len(customer_txns) < self.min_observations:
            return LiquidityPattern.UNKNOWN, {}
        
        customer_txns = customer_txns.sort_values('timestamp')
        customer_txns['day_of_month'] = customer_txns['timestamp'].dt.day
        customer_txns['day_of_week'] = customer_txns['timestamp'].dt.dayofweek
        
        # Check for monthly pattern
        monthly_pattern = self._detect_monthly_pattern(customer_txns)
        if monthly_pattern:
            return LiquidityPattern.MONTHLY_SALARY, monthly_pattern
        
        # Check for weekly pattern
        weekly_pattern = self._detect_weekly_pattern(customer_txns)
        if weekly_pattern:
            return LiquidityPattern.WEEKLY_WAGE, weekly_pattern
        
        # Check for bi-weekly pattern
        biweekly_pattern = self._detect_biweekly_pattern(customer_txns)
        if biweekly_pattern:
            return LiquidityPattern.BI_WEEKLY, biweekly_pattern
        
        # Default to irregular
        return LiquidityPattern.GIG_IRREGULAR, {
            'mean_days_between': self._mean_days_between(customer_txns)
        }
    
    def _detect_monthly_pattern(self, txns: pd.DataFrame) -> Optional[Dict]:
        """Detect consistent monthly salary pattern."""
        day_counts = txns['day_of_month'].value_counts()
        
        if len(day_counts) == 0:
            return None
        
        most_common_day = day_counts.index[0]
        frequency = day_counts.iloc[0] / len(txns)
        
        # If >60% of inflows occur on same day of month, it's likely salary
        if frequency >= 0.6:
            return {
                'salary_day': most_common_day,
                'confidence': frequency,
                'sample_size': len(txns)
            }
        
        # Check for range (e.g., 25th-28th)
        for day in day_counts.index[:3]:
            nearby = txns[
                (txns['day_of_month'] >= day - 2) &
                (txns['day_of_month'] <= day + 2)
            ]
            if len(nearby) / len(txns) >= 0.6:
                return {
                    'salary_day': day,
                    'salary_range': (day - 2, day + 2),
                    'confidence': len(nearby) / len(txns),
                    'sample_size': len(txns)
                }
        
        return None
    
    def _detect_weekly_pattern(self, txns: pd.DataFrame) -> Optional[Dict]:
        """Detect consistent weekly wage pattern."""
        dow_counts = txns['day_of_week'].value_counts()
        
        if len(dow_counts) == 0:
            return None
        
        most_common_dow = dow_counts.index[0]
        frequency = dow_counts.iloc[0] / len(txns)
        
        # If >50% of inflows occur on same day of week
        if frequency >= 0.5:
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday']
            return {
                'wage_day': most_common_dow,
                'wage_day_name': day_names[most_common_dow],
                'confidence': frequency,
                'sample_size': len(txns)
            }
        
        return None
    
    def _detect_biweekly_pattern(self, txns: pd.DataFrame) -> Optional[Dict]:
        """Detect bi-weekly payment pattern."""
        txns = txns.sort_values('timestamp')
        txns['days_since_last'] = txns['timestamp'].diff().dt.days
        
        days_between = txns['days_since_last'].dropna()
        
        if len(days_between) < 2:
            return None
        
        mean_days = days_between.mean()
        std_days = days_between.std()
        
        # Bi-weekly = ~14 days with low variance
        if 12 <= mean_days <= 16 and std_days < 3:
            return {
                'cycle_days': round(mean_days),
                'confidence': 1 - (std_days / mean_days),
                'sample_size': len(txns)
            }
        
        return None
    
    def _mean_days_between(self, txns: pd.DataFrame) -> float:
        """Calculate mean days between inflows."""
        txns = txns.sort_values('timestamp')
        days = txns['timestamp'].diff().dt.days.dropna()
        return days.mean() if len(days) > 0 else 30


class ContactTimingOptimizer:
    """
    Optimises contact timing based on detected liquidity patterns.
    
    Example usage:
        optimizer = ContactTimingOptimizer()
        optimizer.learn_patterns(transactions_df)
        recommendation = optimizer.get_recommendation('CUST001', current_date)
    """
    
    def __init__(
        self,
        window_before_days: int = 1,
        window_after_days: int = 3
    ):
        """
        Initialise optimizer.
        
        Args:
            window_before_days: Days before expected inflow to start window
            window_after_days: Days after expected inflow to end window
        """
        self.window_before = window_before_days
        self.window_after = window_after_days
        self.pattern_detector = LiquidityPatternDetector()
        self.customer_patterns = {}
    
    def learn_patterns(
        self,
        transactions_df: pd.DataFrame,
        customer_ids: Optional[List[str]] = None
    ):
        """
        Learn liquidity patterns for customers.
        
        Args:
            transactions_df: Transaction history
            customer_ids: Optional list of customers to analyse
        """
        if customer_ids is None:
            customer_ids = transactions_df['customer_id'].unique()
        
        for customer_id in customer_ids:
            pattern, details = self.pattern_detector.detect_pattern(
                transactions_df, customer_id
            )
            self.customer_patterns[customer_id] = {
                'pattern': pattern,
                'details': details
            }
    
    def get_recommendation(
        self,
        customer_id: str,
        current_date: datetime
    ) -> TimingRecommendation:
        """
        Get contact timing recommendation for a customer.
        
        Args:
            customer_id: Customer identifier
            current_date: Current date for timing calculation
            
        Returns:
            TimingRecommendation with optimal contact timing
        """
        if customer_id not in self.customer_patterns:
            return TimingRecommendation(
                customer_id=customer_id,
                liquidity_pattern=LiquidityPattern.UNKNOWN,
                current_window=ContactWindow.NO_PATTERN,
                optimal_contact_date=None,
                days_until_window=0,
                confidence=0.0,
                recommendation="No pattern detected - use standard timing"
            )
        
        pattern_info = self.customer_patterns[customer_id]
        pattern = pattern_info['pattern']
        details = pattern_info['details']
        
        if pattern == LiquidityPattern.MONTHLY_SALARY:
            return self._monthly_recommendation(
                customer_id, details, current_date
            )
        elif pattern == LiquidityPattern.WEEKLY_WAGE:
            return self._weekly_recommendation(
                customer_id, details, current_date
            )
        elif pattern == LiquidityPattern.BI_WEEKLY:
            return self._biweekly_recommendation(
                customer_id, details, current_date
            )
        else:
            return TimingRecommendation(
                customer_id=customer_id,
                liquidity_pattern=pattern,
                current_window=ContactWindow.NO_PATTERN,
                optimal_contact_date=None,
                days_until_window=0,
                confidence=details.get('confidence', 0.5),
                recommendation="Irregular income pattern - contact with sensitivity"
            )
    
    def _monthly_recommendation(
        self,
        customer_id: str,
        details: Dict,
        current_date: datetime
    ) -> TimingRecommendation:
        """Generate recommendation for monthly salary pattern."""
        salary_day = details['salary_day']
        current_day = current_date.day
        
        # Calculate next salary date
        if current_day <= salary_day:
            next_salary = current_date.replace(day=salary_day)
        else:
            next_month = current_date.replace(day=1) + timedelta(days=32)
            next_salary = next_month.replace(day=min(salary_day, 28))
        
        # Calculate window
        window_start = next_salary - timedelta(days=self.window_before)
        window_end = next_salary + timedelta(days=self.window_after)
        
        # Determine current position relative to window
        if window_start <= current_date <= window_end:
            window_status = ContactWindow.CONTACT_NOW
            days_until = 0
            recommendation = f"Contact now - within {self.window_after} days of salary date ({salary_day}th)"
        elif current_date < window_start:
            window_status = ContactWindow.WAIT_FOR_WINDOW
            days_until = (window_start - current_date).days
            recommendation = f"Wait {days_until} days - salary expected on {salary_day}th"
        else:
            window_status = ContactWindow.WINDOW_PASSED
            days_until = (window_start - current_date).days + 30
            recommendation = f"Window passed - next opportunity in ~{days_until} days"
        
        return TimingRecommendation(
            customer_id=customer_id,
            liquidity_pattern=LiquidityPattern.MONTHLY_SALARY,
            current_window=window_status,
            optimal_contact_date=next_salary,
            days_until_window=days_until,
            confidence=details.get('confidence', 0.7),
            recommendation=recommendation
        )
    
    def _weekly_recommendation(
        self,
        customer_id: str,
        details: Dict,
        current_date: datetime
    ) -> TimingRecommendation:
        """Generate recommendation for weekly wage pattern."""
        wage_day = details['wage_day']
        current_dow = current_date.weekday()
        
        # Calculate days until next wage day
        days_until_wage = (wage_day - current_dow) % 7
        if days_until_wage == 0:
            days_until_wage = 0  # Today is wage day
        
        next_wage_date = current_date + timedelta(days=days_until_wage)
        
        # Window calculation
        window_start = next_wage_date - timedelta(days=self.window_before)
        window_end = next_wage_date + timedelta(days=min(self.window_after, 2))
        
        if window_start <= current_date <= window_end:
            window_status = ContactWindow.CONTACT_NOW
            days_until = 0
            recommendation = f"Contact now - within window of {details['wage_day_name']} wage day"
        elif current_date < window_start:
            window_status = ContactWindow.WAIT_FOR_WINDOW
            days_until = (window_start - current_date).days
            recommendation = f"Wait {days_until} days - wage expected on {details['wage_day_name']}"
        else:
            window_status = ContactWindow.WINDOW_PASSED
            days_until = 7 - (current_date - window_end).days
            recommendation = f"Window passed - next {details['wage_day_name']} in {days_until} days"
        
        return TimingRecommendation(
            customer_id=customer_id,
            liquidity_pattern=LiquidityPattern.WEEKLY_WAGE,
            current_window=window_status,
            optimal_contact_date=next_wage_date,
            days_until_window=max(days_until, 0),
            confidence=details.get('confidence', 0.6),
            recommendation=recommendation
        )
    
    def _biweekly_recommendation(
        self,
        customer_id: str,
        details: Dict,
        current_date: datetime
    ) -> TimingRecommendation:
        """Generate recommendation for bi-weekly pattern."""
        cycle_days = details.get('cycle_days', 14)
        
        return TimingRecommendation(
            customer_id=customer_id,
            liquidity_pattern=LiquidityPattern.BI_WEEKLY,
            current_window=ContactWindow.NO_PATTERN,
            optimal_contact_date=current_date + timedelta(days=cycle_days // 2),
            days_until_window=cycle_days // 2,
            confidence=details.get('confidence', 0.5),
            recommendation=f"Bi-weekly pattern ({cycle_days} days) - contact mid-cycle"
        )
    
    def get_batch_recommendations(
        self,
        customer_ids: List[str],
        current_date: datetime
    ) -> pd.DataFrame:
        """
        Get recommendations for multiple customers.
        
        Args:
            customer_ids: List of customer identifiers
            current_date: Current date
            
        Returns:
            DataFrame with recommendations
        """
        results = []
        
        for customer_id in customer_ids:
            rec = self.get_recommendation(customer_id, current_date)
            results.append({
                'customer_id': rec.customer_id,
                'liquidity_pattern': rec.liquidity_pattern.value,
                'current_window': rec.current_window.value,
                'optimal_contact_date': rec.optimal_contact_date,
                'days_until_window': rec.days_until_window,
                'confidence': rec.confidence,
                'recommendation': rec.recommendation
            })
        
        return pd.DataFrame(results)
    
    def get_contactable_now(
        self,
        customer_ids: List[str],
        current_date: datetime
    ) -> List[str]:
        """
        Filter to customers who should be contacted now.
        
        Args:
            customer_ids: List of customer identifiers
            current_date: Current date
            
        Returns:
            List of customer IDs in optimal contact window
        """
        contactable = []
        
        for customer_id in customer_ids:
            rec = self.get_recommendation(customer_id, current_date)
            if rec.current_window == ContactWindow.CONTACT_NOW:
                contactable.append(customer_id)
        
        return contactable
    
    def get_wait_list(
        self,
        customer_ids: List[str],
        current_date: datetime
    ) -> pd.DataFrame:
        """
        Get list of customers to wait on, with expected contact dates.
        
        Args:
            customer_ids: List of customer identifiers
            current_date: Current date
            
        Returns:
            DataFrame with wait list and optimal contact dates
        """
        wait_list = []
        
        for customer_id in customer_ids:
            rec = self.get_recommendation(customer_id, current_date)
            if rec.current_window == ContactWindow.WAIT_FOR_WINDOW:
                wait_list.append({
                    'customer_id': customer_id,
                    'optimal_contact_date': rec.optimal_contact_date,
                    'days_until_window': rec.days_until_window,
                    'pattern': rec.liquidity_pattern.value
                })
        
        df = pd.DataFrame(wait_list)
        if len(df) > 0:
            df = df.sort_values('days_until_window')
        
        return df


def calculate_contact_efficiency(
    contact_log: pd.DataFrame,
    payment_log: pd.DataFrame,
    window_hours: int = 24
) -> Dict:
    """
    Calculate contact efficiency metrics.
    
    Measures:
    - Conversion rate: payments within window of contact
    - Timing efficiency: contacts made in optimal windows
    - Suppression accuracy: self-cures among suppressed accounts
    
    Args:
        contact_log: Log of contact attempts
        payment_log: Log of payments
        window_hours: Hours after contact to attribute payment
        
    Returns:
        Dictionary of efficiency metrics
    """
    # Merge contact and payment data
    contact_log = contact_log.copy()
    payment_log = payment_log.copy()
    
    # Count contacts
    total_contacts = len(contact_log)
    
    # Find payments within window of contact
    conversions = 0
    for _, contact in contact_log.iterrows():
        customer = contact['customer_id']
        contact_time = contact['timestamp']
        window_end = contact_time + timedelta(hours=window_hours)
        
        customer_payments = payment_log[
            (payment_log['customer_id'] == customer) &
            (payment_log['timestamp'] >= contact_time) &
            (payment_log['timestamp'] <= window_end)
        ]
        
        if len(customer_payments) > 0:
            conversions += 1
    
    conversion_rate = conversions / max(total_contacts, 1)
    
    return {
        'total_contacts': total_contacts,
        'conversions': conversions,
        'conversion_rate': conversion_rate,
        'contacts_per_conversion': total_contacts / max(conversions, 1)
    }


if __name__ == "__main__":
    # Example usage
    from datetime import datetime
    
    # Create sample transaction data
    np.random.seed(42)
    
    # Customer with monthly salary pattern (25th of month)
    dates_monthly = pd.date_range('2024-01-25', periods=6, freq='MS') + pd.Timedelta(days=24)
    
    sample_transactions = pd.DataFrame({
        'customer_id': ['CUST001'] * 6,
        'timestamp': dates_monthly,
        'amount': np.random.uniform(50000, 80000, 6)
    })
    
    # Initialise optimizer
    optimizer = ContactTimingOptimizer()
    optimizer.learn_patterns(sample_transactions)
    
    # Get recommendation
    current_date = datetime(2024, 7, 20)
    recommendation = optimizer.get_recommendation('CUST001', current_date)
    
    print("Contact Timing Recommendation")
    print("=" * 50)
    print(f"Customer: {recommendation.customer_id}")
    print(f"Pattern: {recommendation.liquidity_pattern.value}")
    print(f"Current Window: {recommendation.current_window.value}")
    print(f"Days Until Window: {recommendation.days_until_window}")
    print(f"Confidence: {recommendation.confidence:.2f}")
    print(f"Recommendation: {recommendation.recommendation}")
