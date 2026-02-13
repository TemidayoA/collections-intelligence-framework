"""
Feature Engineering for Collections Intelligence

This module provides feature engineering utilities for propensity modelling
and borrower segmentation. Features are grouped into:

- Payment behaviour signals
- Income and capacity proxies
- Contact response patterns
- Timing and liquidity variables
- Device and engagement signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class FeatureSet:
    """Container for engineered features."""
    customer_id: str
    payment_features: Dict[str, float]
    income_features: Dict[str, float]
    contact_features: Dict[str, float]
    timing_features: Dict[str, float]
    engagement_features: Dict[str, float]
    
    def to_dict(self) -> Dict[str, float]:
        """Flatten all features into single dictionary."""
        result = {'customer_id': self.customer_id}
        result.update(self.payment_features)
        result.update(self.income_features)
        result.update(self.contact_features)
        result.update(self.timing_features)
        result.update(self.engagement_features)
        return result


class PaymentFeatureEngineer:
    """
    Engineers features from payment and transaction history.
    
    Key features:
    - Payment attempt signals (intent indicators)
    - Historical repayment patterns
    - Partial payment behaviour
    - Payment timing relative to due dates
    """
    
    def __init__(self, lookback_days: int = 90):
        self.lookback_days = lookback_days
    
    def engineer(
        self,
        customer_id: str,
        transactions_df: pd.DataFrame,
        loan_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Engineer payment features for a customer.
        
        Args:
            customer_id: Customer identifier
            transactions_df: Transaction history
            loan_df: Loan/account information
            
        Returns:
            Dictionary of payment features
        """
        features = {}
        
        # Filter to customer
        customer_txns = transactions_df[
            transactions_df['customer_id'] == customer_id
        ].copy()
        
        cutoff = datetime.now() - timedelta(days=self.lookback_days)
        recent_txns = customer_txns[customer_txns['timestamp'] >= cutoff]
        
        # Payment attempt features
        payment_attempts = recent_txns[
            recent_txns['transaction_type'] == 'payment_attempt'
        ]
        successful_payments = payment_attempts[
            payment_attempts['status'] == 'success'
        ]
        failed_payments = payment_attempts[
            payment_attempts['status'] == 'failed'
        ]
        
        features['payment_attempts_total'] = len(payment_attempts)
        features['payment_attempts_successful'] = len(successful_payments)
        features['payment_attempts_failed'] = len(failed_payments)
        features['payment_success_rate'] = (
            len(successful_payments) / max(len(payment_attempts), 1)
        )
        
        # Failed attempt is strong intent signal
        features['has_failed_attempt'] = float(len(failed_payments) > 0)
        
        # Recency of last attempt
        if len(payment_attempts) > 0:
            last_attempt = payment_attempts['timestamp'].max()
            features['days_since_last_attempt'] = (
                datetime.now() - last_attempt
            ).days
        else:
            features['days_since_last_attempt'] = 999  # No attempt
        
        # Partial payment behaviour
        if 'amount' in successful_payments.columns and len(successful_payments) > 0:
            loan_info = loan_df[loan_df['customer_id'] == customer_id]
            if len(loan_info) > 0:
                instalment = loan_info['instalment_amount'].iloc[0]
                avg_payment = successful_payments['amount'].mean()
                features['payment_to_instalment_ratio'] = avg_payment / max(instalment, 1)
                features['has_partial_payments'] = float(avg_payment < instalment * 0.95)
            else:
                features['payment_to_instalment_ratio'] = 1.0
                features['has_partial_payments'] = 0.0
        else:
            features['payment_to_instalment_ratio'] = 0.0
            features['has_partial_payments'] = 0.0
        
        # Historical completion rate
        customer_loans = loan_df[loan_df['customer_id'] == customer_id]
        if len(customer_loans) > 0:
            completed = customer_loans[customer_loans['status'] == 'completed']
            features['historical_completion_rate'] = (
                len(completed) / len(customer_loans)
            )
            features['total_previous_loans'] = len(customer_loans) - 1  # Exclude current
        else:
            features['historical_completion_rate'] = 0.5  # No history
            features['total_previous_loans'] = 0
        
        return features


class IncomeFeatureEngineer:
    """
    Engineers features related to income and capacity.
    
    Key features:
    - Income regularity (coefficient of variation)
    - Income pattern detection (salary, weekly, irregular)
    - Debt burden indicators
    - Transaction activity levels
    """
    
    def __init__(self, lookback_days: int = 180):
        self.lookback_days = lookback_days
    
    def engineer(
        self,
        customer_id: str,
        transactions_df: pd.DataFrame,
        loan_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Engineer income/capacity features for a customer.
        
        Args:
            customer_id: Customer identifier
            transactions_df: Transaction history
            loan_df: Loan information
            
        Returns:
            Dictionary of income features
        """
        features = {}
        
        # Filter to customer credits (inflows)
        customer_txns = transactions_df[
            (transactions_df['customer_id'] == customer_id) &
            (transactions_df['amount'] > 0)
        ].copy()
        
        cutoff = datetime.now() - timedelta(days=self.lookback_days)
        recent_inflows = customer_txns[customer_txns['timestamp'] >= cutoff]
        
        if len(recent_inflows) < 3:
            # Insufficient data
            features['income_regularity_score'] = 0.5
            features['income_pattern'] = 'unknown'
            features['mean_monthly_inflow'] = 0.0
            features['inflow_coefficient_of_variation'] = 1.0
        else:
            # Calculate inflow regularity
            recent_inflows = recent_inflows.sort_values('timestamp')
            recent_inflows['days_between'] = (
                recent_inflows['timestamp'].diff().dt.days
            )
            
            days_between = recent_inflows['days_between'].dropna()
            
            if len(days_between) > 0 and days_between.mean() > 0:
                cv = days_between.std() / days_between.mean()
                features['inflow_coefficient_of_variation'] = cv
                features['income_regularity_score'] = 1 / (1 + cv)
            else:
                features['inflow_coefficient_of_variation'] = 1.0
                features['income_regularity_score'] = 0.5
            
            # Detect income pattern
            mean_days = days_between.mean() if len(days_between) > 0 else 30
            
            if mean_days < 10:
                features['income_pattern'] = 'weekly'
            elif 25 <= mean_days <= 35:
                features['income_pattern'] = 'monthly_salary'
            elif 12 <= mean_days <= 16:
                features['income_pattern'] = 'biweekly'
            else:
                features['income_pattern'] = 'irregular'
            
            # Mean monthly inflow
            months = self.lookback_days / 30
            features['mean_monthly_inflow'] = recent_inflows['amount'].sum() / months
        
        # Transaction activity score
        total_txns = len(transactions_df[
            (transactions_df['customer_id'] == customer_id) &
            (transactions_df['timestamp'] >= cutoff)
        ])
        
        # Normalise to 0-1 (assuming 100 txns/period is high activity)
        features['transaction_activity_score'] = min(total_txns / 100, 1.0)
        
        # Debt burden (if data available)
        customer_loans = loan_df[
            (loan_df['customer_id'] == customer_id) &
            (loan_df['status'] == 'active')
        ]
        
        if len(customer_loans) > 0 and features['mean_monthly_inflow'] > 0:
            total_monthly_obligation = customer_loans['instalment_amount'].sum()
            features['debt_to_income_ratio'] = (
                total_monthly_obligation / features['mean_monthly_inflow']
            )
        else:
            features['debt_to_income_ratio'] = 0.3  # Default assumption
        
        # Multiple active loans indicator
        features['active_loan_count'] = len(customer_loans)
        features['has_multiple_loans'] = float(len(customer_loans) > 1)
        
        return features


class ContactFeatureEngineer:
    """
    Engineers features from contact history.
    
    Key features:
    - Response rates by channel
    - Contact fatigue indicators
    - Channel preferences
    - Avoidance signals (blocking)
    """
    
    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
    
    def engineer(
        self,
        customer_id: str,
        contact_history_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Engineer contact response features for a customer.
        
        Args:
            customer_id: Customer identifier
            contact_history_df: Contact attempt history
            
        Returns:
            Dictionary of contact features
        """
        features = {}
        
        # Filter to customer
        customer_contacts = contact_history_df[
            contact_history_df['customer_id'] == customer_id
        ].copy()
        
        cutoff = datetime.now() - timedelta(days=self.lookback_days)
        recent_contacts = customer_contacts[
            customer_contacts['timestamp'] >= cutoff
        ]
        
        if len(recent_contacts) == 0:
            # No contact history
            features['overall_response_rate'] = 0.5
            features['sms_response_rate'] = 0.5
            features['call_response_rate'] = 0.5
            features['whatsapp_response_rate'] = 0.5
            features['preferred_channel'] = 'unknown'
            features['total_contacts'] = 0
            features['contact_fatigue_score'] = 0.0
            features['channels_blocked'] = 0
            features['is_avoiding'] = 0.0
            return features
        
        # Overall response rate
        responded = recent_contacts[recent_contacts['responded'] == True]
        features['overall_response_rate'] = len(responded) / len(recent_contacts)
        features['total_contacts'] = len(recent_contacts)
        
        # Channel-specific response rates
        for channel in ['sms', 'call', 'whatsapp', 'email']:
            channel_contacts = recent_contacts[recent_contacts['channel'] == channel]
            if len(channel_contacts) > 0:
                channel_responded = channel_contacts[channel_contacts['responded'] == True]
                features[f'{channel}_response_rate'] = (
                    len(channel_responded) / len(channel_contacts)
                )
            else:
                features[f'{channel}_response_rate'] = 0.5  # No data
        
        # Preferred channel (highest response rate)
        channel_rates = {
            'sms': features['sms_response_rate'],
            'call': features['call_response_rate'],
            'whatsapp': features['whatsapp_response_rate'],
            'email': features.get('email_response_rate', 0)
        }
        features['preferred_channel'] = max(channel_rates, key=channel_rates.get)
        
        # Contact fatigue (declining response rate over time)
        if len(recent_contacts) >= 5:
            recent_contacts = recent_contacts.sort_values('timestamp')
            first_half = recent_contacts.iloc[:len(recent_contacts)//2]
            second_half = recent_contacts.iloc[len(recent_contacts)//2:]
            
            first_rate = first_half['responded'].mean()
            second_rate = second_half['responded'].mean()
            
            # Fatigue = decline in response rate
            features['contact_fatigue_score'] = max(first_rate - second_rate, 0)
        else:
            features['contact_fatigue_score'] = 0.0
        
        # Channel blocking / avoidance signals
        if 'blocked' in recent_contacts.columns:
            blocked = recent_contacts[recent_contacts['blocked'] == True]
            features['channels_blocked'] = blocked['channel'].nunique()
        else:
            features['channels_blocked'] = 0
        
        # Avoidance indicator (low response + high contact attempts)
        features['is_avoiding'] = float(
            features['overall_response_rate'] < 0.2 and 
            features['total_contacts'] >= 5
        )
        
        return features


class TimingFeatureEngineer:
    """
    Engineers timing-related features.
    
    Key features:
    - Days past due trajectory
    - Position relative to liquidity windows
    - Day of week / month patterns
    - Seasonal indicators
    """
    
    def engineer(
        self,
        customer_id: str,
        loan_df: pd.DataFrame,
        customer_profile: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Engineer timing features for a customer.
        
        Args:
            customer_id: Customer identifier
            loan_df: Loan information with DPD
            customer_profile: Optional profile with salary date etc.
            
        Returns:
            Dictionary of timing features
        """
        features = {}
        current_date = datetime.now()
        
        # Current DPD
        customer_loan = loan_df[
            (loan_df['customer_id'] == customer_id) &
            (loan_df['status'] == 'active')
        ]
        
        if len(customer_loan) > 0:
            features['current_dpd'] = customer_loan['dpd'].iloc[0]
            
            # DPD bucket
            dpd = features['current_dpd']
            if dpd <= 7:
                features['dpd_bucket'] = 1
            elif dpd <= 30:
                features['dpd_bucket'] = 2
            elif dpd <= 60:
                features['dpd_bucket'] = 3
            elif dpd <= 90:
                features['dpd_bucket'] = 4
            else:
                features['dpd_bucket'] = 5
        else:
            features['current_dpd'] = 0
            features['dpd_bucket'] = 0
        
        # Day of week features
        features['day_of_week'] = current_date.weekday()
        features['is_weekend'] = float(current_date.weekday() >= 5)
        features['is_monday'] = float(current_date.weekday() == 0)
        features['is_friday'] = float(current_date.weekday() == 4)
        
        # Day of month features
        features['day_of_month'] = current_date.day
        features['is_month_end'] = float(current_date.day >= 25)
        features['is_month_start'] = float(current_date.day <= 5)
        
        # Days to month end
        next_month = current_date.replace(day=1) + timedelta(days=32)
        month_end = next_month.replace(day=1) - timedelta(days=1)
        features['days_to_month_end'] = (month_end - current_date).days
        
        # Salary proximity (if profile available)
        if customer_profile and 'salary_day' in customer_profile:
            salary_day = customer_profile['salary_day']
            current_day = current_date.day
            
            if current_day <= salary_day:
                days_to_salary = salary_day - current_day
            else:
                days_to_salary = (30 - current_day) + salary_day
            
            features['days_to_salary'] = days_to_salary
            features['days_since_salary'] = 30 - days_to_salary
            features['in_salary_window'] = float(
                features['days_since_salary'] <= 5
            )
        else:
            features['days_to_salary'] = 15  # Default mid-month
            features['days_since_salary'] = 15
            features['in_salary_window'] = 0.0
        
        return features


class EngagementFeatureEngineer:
    """
    Engineers features from app/platform engagement.
    
    Key features:
    - App usage patterns
    - Login recency
    - Feature usage
    - Device signals
    """
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
    
    def engineer(
        self,
        customer_id: str,
        engagement_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Engineer engagement features for a customer.
        
        Args:
            customer_id: Customer identifier
            engagement_df: Optional engagement/app usage data
            
        Returns:
            Dictionary of engagement features
        """
        features = {}
        
        if engagement_df is None or len(engagement_df) == 0:
            # No engagement data available
            features['app_active'] = 0.5
            features['days_since_last_login'] = 30
            features['login_frequency_30d'] = 0
            features['viewed_payment_screen'] = 0.0
            features['app_uninstalled'] = 0.0
            return features
        
        customer_engagement = engagement_df[
            engagement_df['customer_id'] == customer_id
        ]
        
        if len(customer_engagement) == 0:
            features['app_active'] = 0.0
            features['days_since_last_login'] = 999
            features['login_frequency_30d'] = 0
            features['viewed_payment_screen'] = 0.0
            features['app_uninstalled'] = 0.0
            return features
        
        cutoff = datetime.now() - timedelta(days=self.lookback_days)
        recent_engagement = customer_engagement[
            customer_engagement['timestamp'] >= cutoff
        ]
        
        # App activity
        features['app_active'] = float(len(recent_engagement) > 0)
        
        # Login recency
        if len(customer_engagement) > 0:
            last_login = customer_engagement['timestamp'].max()
            features['days_since_last_login'] = (
                datetime.now() - last_login
            ).days
        else:
            features['days_since_last_login'] = 999
        
        # Login frequency
        logins = recent_engagement[recent_engagement['event'] == 'login']
        features['login_frequency_30d'] = len(logins)
        
        # Payment screen views (intent signal)
        if 'event' in recent_engagement.columns:
            payment_views = recent_engagement[
                recent_engagement['event'].str.contains('payment', case=False, na=False)
            ]
            features['viewed_payment_screen'] = float(len(payment_views) > 0)
        else:
            features['viewed_payment_screen'] = 0.0
        
        # App uninstall signal
        if 'event' in customer_engagement.columns:
            uninstalls = customer_engagement[
                customer_engagement['event'] == 'app_uninstall'
            ]
            features['app_uninstalled'] = float(len(uninstalls) > 0)
        else:
            features['app_uninstalled'] = 0.0
        
        return features


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline combining all feature engineers.
    
    Example usage:
        pipeline = FeatureEngineeringPipeline()
        features = pipeline.engineer_features(
            customer_id='C001',
            transactions_df=transactions,
            loan_df=loans,
            contact_history_df=contacts
        )
    """
    
    def __init__(self):
        self.payment_engineer = PaymentFeatureEngineer()
        self.income_engineer = IncomeFeatureEngineer()
        self.contact_engineer = ContactFeatureEngineer()
        self.timing_engineer = TimingFeatureEngineer()
        self.engagement_engineer = EngagementFeatureEngineer()
    
    def engineer_features(
        self,
        customer_id: str,
        transactions_df: pd.DataFrame,
        loan_df: pd.DataFrame,
        contact_history_df: Optional[pd.DataFrame] = None,
        engagement_df: Optional[pd.DataFrame] = None,
        customer_profile: Optional[Dict] = None
    ) -> FeatureSet:
        """
        Engineer all features for a customer.
        
        Args:
            customer_id: Customer identifier
            transactions_df: Transaction history
            loan_df: Loan information
            contact_history_df: Optional contact history
            engagement_df: Optional app engagement data
            customer_profile: Optional customer profile
            
        Returns:
            FeatureSet containing all engineered features
        """
        payment_features = self.payment_engineer.engineer(
            customer_id, transactions_df, loan_df
        )
        
        income_features = self.income_engineer.engineer(
            customer_id, transactions_df, loan_df
        )
        
        if contact_history_df is not None:
            contact_features = self.contact_engineer.engineer(
                customer_id, contact_history_df
            )
        else:
            contact_features = {}
        
        timing_features = self.timing_engineer.engineer(
            customer_id, loan_df, customer_profile
        )
        
        engagement_features = self.engagement_engineer.engineer(
            customer_id, engagement_df
        )
        
        return FeatureSet(
            customer_id=customer_id,
            payment_features=payment_features,
            income_features=income_features,
            contact_features=contact_features,
            timing_features=timing_features,
            engagement_features=engagement_features
        )
    
    def engineer_portfolio(
        self,
        customer_ids: List[str],
        transactions_df: pd.DataFrame,
        loan_df: pd.DataFrame,
        contact_history_df: Optional[pd.DataFrame] = None,
        engagement_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Engineer features for entire portfolio.
        
        Args:
            customer_ids: List of customer identifiers
            transactions_df: Transaction history
            loan_df: Loan information
            contact_history_df: Optional contact history
            engagement_df: Optional engagement data
            
        Returns:
            DataFrame with all features for all customers
        """
        all_features = []
        
        for customer_id in customer_ids:
            feature_set = self.engineer_features(
                customer_id=customer_id,
                transactions_df=transactions_df,
                loan_df=loan_df,
                contact_history_df=contact_history_df,
                engagement_df=engagement_df
            )
            all_features.append(feature_set.to_dict())
        
        return pd.DataFrame(all_features)


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Pipeline")
    print("=" * 50)
    print("\nAvailable feature engineers:")
    print("- PaymentFeatureEngineer: Payment attempts, history, partial payments")
    print("- IncomeFeatureEngineer: Income regularity, patterns, debt burden")
    print("- ContactFeatureEngineer: Response rates, channel preferences, avoidance")
    print("- TimingFeatureEngineer: DPD, day of week/month, salary proximity")
    print("- EngagementFeatureEngineer: App usage, login patterns, intent signals")
    print("\nUse FeatureEngineeringPipeline to combine all engineers.")
