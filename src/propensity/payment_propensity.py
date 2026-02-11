"""
Propensity-to-Pay Modelling

Predicts the probability that a borrower will repay within a specified window,
with or without intervention.

Key insight: The most powerful application is deciding who NOT to contact.
Suppressing unnecessary activity reduces cost, prevents borrower fatigue,
and preserves goodwill without sacrificing recovery.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PropensityScore:
    """Propensity scoring result for a single account."""
    customer_id: str
    p_repay_no_contact: float      # Probability of self-cure
    p_repay_with_contact: float    # Probability if contacted
    uplift: float                  # Incremental value of contact
    recommended_action: str
    suppression_candidate: bool


class FeatureEngineering:
    """
    Feature engineering for propensity modelling.
    
    Features are grouped into:
    - Repayment history signals
    - Payment attempt signals (intent indicators)
    - Income/liquidity proxies
    - Contact response patterns
    - Timing variables
    """
    
    @staticmethod
    def calculate_payment_attempt_features(
        transactions_df: pd.DataFrame,
        customer_id: str,
        window_days: int = 30
    ) -> Dict:
        """
        Calculate features from payment attempts.
        
        Failed attempts signal intent—a borrower who tries to pay twice
        and fails shows willingness despite capacity constraints.
        """
        customer_txns = transactions_df[
            transactions_df['customer_id'] == customer_id
        ].copy()
        
        # Filter to recent window
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=window_days)
        recent = customer_txns[customer_txns['timestamp'] >= cutoff]
        
        # Payment attempts (successful and failed)
        attempts = recent[recent['transaction_type'] == 'payment_attempt']
        successful = attempts[attempts['status'] == 'success']
        failed = attempts[attempts['status'] == 'failed']
        
        features = {
            'payment_attempts_total': len(attempts),
            'payment_attempts_successful': len(successful),
            'payment_attempts_failed': len(failed),
            'attempt_success_rate': len(successful) / max(len(attempts), 1),
            'has_recent_attempt': len(attempts) > 0,
            'days_since_last_attempt': None
        }
        
        if len(attempts) > 0:
            last_attempt = attempts['timestamp'].max()
            features['days_since_last_attempt'] = (
                pd.Timestamp.now() - last_attempt
            ).days
        
        return features
    
    @staticmethod
    def calculate_income_proxy_features(
        transactions_df: pd.DataFrame,
        customer_id: str
    ) -> Dict:
        """
        Calculate income regularity proxies from transaction patterns.
        
        Detects:
        - Regular salary patterns (monthly)
        - Weekly income patterns
        - Irregular/gig income patterns
        """
        customer_txns = transactions_df[
            transactions_df['customer_id'] == customer_id
        ].copy()
        
        # Filter to credits (inflows)
        inflows = customer_txns[customer_txns['amount'] > 0].copy()
        
        if len(inflows) < 3:
            return {
                'income_regularity_score': 0.5,
                'detected_pattern': 'insufficient_data',
                'mean_inflow': 0,
                'inflow_cv': 1.0
            }
        
        # Calculate coefficient of variation of inflow timing
        inflows = inflows.sort_values('timestamp')
        inflows['days_between'] = inflows['timestamp'].diff().dt.days
        
        days_between = inflows['days_between'].dropna()
        
        if len(days_between) > 0:
            cv = days_between.std() / max(days_between.mean(), 1)
        else:
            cv = 1.0
        
        # Detect pattern type
        mean_days = days_between.mean() if len(days_between) > 0 else 30
        
        if mean_days < 10:
            pattern = 'weekly'
        elif 25 <= mean_days <= 35:
            pattern = 'monthly_salary'
        else:
            pattern = 'irregular'
        
        # Regularity score (inverse of CV, normalised)
        regularity_score = 1 / (1 + cv)
        
        return {
            'income_regularity_score': regularity_score,
            'detected_pattern': pattern,
            'mean_inflow': inflows['amount'].mean(),
            'inflow_cv': cv,
            'mean_days_between_inflows': mean_days
        }
    
    @staticmethod
    def calculate_contact_response_features(
        contact_history_df: pd.DataFrame,
        customer_id: str,
        window_days: int = 60
    ) -> Dict:
        """
        Calculate features from contact response history.
        
        Response patterns indicate:
        - Channel preferences
        - Engagement level
        - Avoidance behaviours
        """
        customer_contacts = contact_history_df[
            contact_history_df['customer_id'] == customer_id
        ].copy()
        
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=window_days)
        recent = customer_contacts[customer_contacts['timestamp'] >= cutoff]
        
        if len(recent) == 0:
            return {
                'contact_response_rate': 0.5,
                'sms_response_rate': 0.5,
                'call_answer_rate': 0.5,
                'preferred_channel': 'unknown',
                'total_contacts': 0
            }
        
        # Overall response rate
        responded = recent[recent['responded'] == True]
        response_rate = len(responded) / len(recent)
        
        # Channel-specific rates
        sms = recent[recent['channel'] == 'sms']
        calls = recent[recent['channel'] == 'call']
        
        sms_rate = len(sms[sms['responded']]) / max(len(sms), 1)
        call_rate = len(calls[calls['responded']]) / max(len(calls), 1)
        
        # Determine preferred channel
        channel_rates = {'sms': sms_rate, 'call': call_rate}
        preferred = max(channel_rates, key=channel_rates.get)
        
        return {
            'contact_response_rate': response_rate,
            'sms_response_rate': sms_rate,
            'call_answer_rate': call_rate,
            'preferred_channel': preferred,
            'total_contacts': len(recent)
        }
    
    @staticmethod
    def calculate_timing_features(
        customer_profile: Dict,
        current_date: pd.Timestamp
    ) -> Dict:
        """
        Calculate timing features relative to expected liquidity windows.
        """
        features = {}
        
        # Days since expected salary
        if 'salary_date' in customer_profile:
            salary_day = customer_profile['salary_date']
            current_day = current_date.day
            
            if current_day >= salary_day:
                days_since = current_day - salary_day
            else:
                # Salary was last month
                days_since = current_day + (30 - salary_day)
            
            features['days_since_salary'] = days_since
            features['in_salary_window'] = days_since <= 5
        else:
            features['days_since_salary'] = None
            features['in_salary_window'] = False
        
        # Day of week features
        features['is_weekend'] = current_date.dayofweek >= 5
        features['day_of_week'] = current_date.dayofweek
        features['day_of_month'] = current_date.day
        
        # End of month proximity
        features['days_to_month_end'] = (
            pd.Timestamp(current_date.year, current_date.month, 1) + 
            pd.offsets.MonthEnd(1) - current_date
        ).days
        
        return features


class PropensityModel:
    """
    Propensity-to-pay model.
    
    Predicts:
    - P(repay | no contact): Self-cure probability
    - P(repay | contact): Probability if contacted
    - Uplift: Incremental benefit of contact
    
    Example usage:
        model = PropensityModel()
        model.fit(training_data, labels)
        scores = model.predict(borrower_features)
    """
    
    def __init__(
        self,
        suppression_threshold: float = 0.7,
        contact_threshold: float = 0.3
    ):
        """
        Initialise propensity model.
        
        Args:
            suppression_threshold: P(self-cure) above this = suppress contact
            contact_threshold: P(repay|contact) below this = deprioritise
        """
        self.suppression_threshold = suppression_threshold
        self.contact_threshold = contact_threshold
        
        self.model_no_contact = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model_with_contact = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y_no_contact: np.ndarray,
        y_with_contact: np.ndarray
    ):
        """
        Fit propensity models.
        
        Args:
            X: Feature DataFrame
            y_no_contact: Labels for self-cure (repaid without contact)
            y_with_contact: Labels for repayment after contact
        """
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit both models
        self.model_no_contact.fit(X_scaled, y_no_contact)
        self.model_with_contact.fit(X_scaled, y_with_contact)
        
        self.is_fitted = True
    
    def predict(
        self,
        X: pd.DataFrame,
        customer_ids: Optional[List[str]] = None
    ) -> List[PropensityScore]:
        """
        Predict propensity scores for accounts.
        
        Args:
            X: Feature DataFrame
            customer_ids: Optional list of customer IDs
            
        Returns:
            List of PropensityScore objects
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        p_no_contact = self.model_no_contact.predict_proba(X_scaled)[:, 1]
        p_with_contact = self.model_with_contact.predict_proba(X_scaled)[:, 1]
        
        # Calculate uplift
        uplift = p_with_contact - p_no_contact
        
        results = []
        
        for i in range(len(X)):
            customer_id = customer_ids[i] if customer_ids else f"customer_{i}"
            
            # Determine action
            p_self_cure = p_no_contact[i]
            p_contact = p_with_contact[i]
            
            if p_self_cure >= self.suppression_threshold:
                action = "suppress"
                suppression = True
            elif p_contact < self.contact_threshold:
                action = "deprioritise"
                suppression = True
            elif uplift[i] > 0.1:
                action = "contact_high_value"
                suppression = False
            else:
                action = "contact_standard"
                suppression = False
            
            results.append(PropensityScore(
                customer_id=customer_id,
                p_repay_no_contact=p_self_cure,
                p_repay_with_contact=p_contact,
                uplift=uplift[i],
                recommended_action=action,
                suppression_candidate=suppression
            ))
        
        return results
    
    def get_suppression_list(
        self,
        scores: List[PropensityScore]
    ) -> List[str]:
        """
        Get list of customer IDs to suppress (not contact).
        
        Args:
            scores: List of PropensityScore objects
            
        Returns:
            List of customer IDs to suppress
        """
        return [s.customer_id for s in scores if s.suppression_candidate]
    
    def get_priority_list(
        self,
        scores: List[PropensityScore],
        top_n: Optional[int] = None
    ) -> List[str]:
        """
        Get prioritised list for contact (highest uplift first).
        
        Args:
            scores: List of PropensityScore objects
            top_n: Optional limit on number returned
            
        Returns:
            List of customer IDs sorted by contact priority
        """
        contactable = [s for s in scores if not s.suppression_candidate]
        sorted_scores = sorted(contactable, key=lambda x: x.uplift, reverse=True)
        
        if top_n:
            sorted_scores = sorted_scores[:top_n]
        
        return [s.customer_id for s in sorted_scores]
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test_no_contact: np.ndarray,
        y_test_with_contact: np.ndarray
    ) -> Dict:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary of evaluation metrics
        """
        X_scaled = self.scaler.transform(X_test)
        
        # AUC scores
        from sklearn.metrics import roc_auc_score
        
        p_no_contact = self.model_no_contact.predict_proba(X_scaled)[:, 1]
        p_with_contact = self.model_with_contact.predict_proba(X_scaled)[:, 1]
        
        auc_no_contact = roc_auc_score(y_test_no_contact, p_no_contact)
        auc_with_contact = roc_auc_score(y_test_with_contact, p_with_contact)
        
        return {
            'auc_self_cure_model': auc_no_contact,
            'auc_contact_model': auc_with_contact,
            'mean_predicted_self_cure': np.mean(p_no_contact),
            'mean_predicted_contact': np.mean(p_with_contact),
            'mean_uplift': np.mean(p_with_contact - p_no_contact)
        }


def identify_intent_signals(
    customer_features: Dict
) -> Dict[str, bool]:
    """
    Identify payment intent signals from features.
    
    Intent signals indicate willingness despite potential capacity constraints.
    
    Args:
        customer_features: Dictionary of customer features
        
    Returns:
        Dictionary of intent signal flags
    """
    signals = {}
    
    # Payment attempt without success = strong intent signal
    signals['attempted_payment'] = (
        customer_features.get('payment_attempts_failed', 0) > 0
    )
    
    # Responsive to contact = engagement intent
    signals['responsive'] = (
        customer_features.get('contact_response_rate', 0) > 0.5
    )
    
    # Partial payment made = intent to clear
    signals['partial_payment'] = (
        customer_features.get('partial_payment_made', False)
    )
    
    # Proactive communication = high intent
    signals['proactive_contact'] = (
        customer_features.get('initiated_contact', False)
    )
    
    # No channel blocking = not avoiding
    signals['channels_open'] = (
        customer_features.get('channels_blocked', 0) == 0
    )
    
    # Composite intent score
    signals['intent_score'] = sum(signals.values()) / len(signals)
    
    return signals


if __name__ == "__main__":
    # Example: Demonstrate intent signal identification
    
    sample_features = {
        'payment_attempts_failed': 2,
        'contact_response_rate': 0.7,
        'partial_payment_made': False,
        'initiated_contact': True,
        'channels_blocked': 0
    }
    
    signals = identify_intent_signals(sample_features)
    
    print("Intent Signal Analysis")
    print("=" * 40)
    print(f"Features: {sample_features}")
    print(f"\nSignals detected:")
    for signal, value in signals.items():
        print(f"  {signal}: {value}")
