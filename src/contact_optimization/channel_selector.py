"""
Channel Selection for Collections Contact

Not all borrowers respond to the same channels. This module implements
intelligent channel selection based on historical response patterns,
channel fatigue, and borrower preferences.

Key principles:
- Match channel to borrower preference
- Avoid fatigued channels
- Escalate channel intensity progressively
- Respect channel blocks
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum


class Channel(Enum):
    """Available contact channels."""
    SMS = "sms"
    WHATSAPP = "whatsapp"
    CALL = "call"
    EMAIL = "email"
    PUSH_NOTIFICATION = "push"
    IN_APP = "in_app"


class ChannelIntensity(Enum):
    """Channel intensity levels for progressive escalation."""
    LOW = 1       # Gentle reminder (SMS, push)
    MEDIUM = 2    # Active outreach (WhatsApp, email)
    HIGH = 3      # Direct contact (call)


@dataclass
class ChannelRecommendation:
    """Channel selection recommendation."""
    customer_id: str
    recommended_channel: Channel
    backup_channel: Optional[Channel]
    confidence: float
    reason: str
    avoid_channels: List[Channel]


@dataclass
class ChannelProfile:
    """Channel performance profile for a customer."""
    customer_id: str
    channel_scores: Dict[str, float]
    blocked_channels: List[str]
    fatigued_channels: List[str]
    preferred_channel: str
    last_contact_by_channel: Dict[str, datetime]


class ChannelScorer:
    """
    Scores channels based on historical effectiveness for a customer.
    
    Factors considered:
    - Historical response rate per channel
    - Recency of last contact (fatigue)
    - Channel availability (not blocked)
    - Time of day / week effectiveness
    """
    
    def __init__(
        self,
        fatigue_window_hours: int = 48,
        min_contacts_for_preference: int = 3
    ):
        self.fatigue_window = timedelta(hours=fatigue_window_hours)
        self.min_contacts = min_contacts_for_preference
    
    def score_channels(
        self,
        customer_id: str,
        contact_history: pd.DataFrame
    ) -> ChannelProfile:
        """
        Score all channels for a customer based on history.
        
        Args:
            customer_id: Customer identifier
            contact_history: DataFrame of contact attempts
            
        Returns:
            ChannelProfile with scores and preferences
        """
        customer_contacts = contact_history[
            contact_history['customer_id'] == customer_id
        ].copy()
        
        channel_scores = {}
        last_contact = {}
        blocked = []
        fatigued = []
        
        current_time = datetime.now()
        
        for channel in Channel:
            channel_name = channel.value
            channel_contacts = customer_contacts[
                customer_contacts['channel'] == channel_name
            ]
            
            if len(channel_contacts) == 0:
                # No history - neutral score
                channel_scores[channel_name] = 0.5
                continue
            
            # Response rate
            responded = channel_contacts[channel_contacts['responded'] == True]
            response_rate = len(responded) / len(channel_contacts)
            
            # Recency penalty
            last_attempt = channel_contacts['timestamp'].max()
            last_contact[channel_name] = last_attempt
            
            time_since_last = current_time - last_attempt
            if time_since_last < self.fatigue_window:
                fatigued.append(channel_name)
                recency_penalty = 0.5  # Reduce score if recently contacted
            else:
                recency_penalty = 1.0
            
            # Check for blocking
            if 'blocked' in channel_contacts.columns:
                if channel_contacts['blocked'].any():
                    blocked.append(channel_name)
                    channel_scores[channel_name] = 0.0
                    continue
            
            # Final score
            channel_scores[channel_name] = response_rate * recency_penalty
        
        # Determine preferred channel
        available_channels = {
            k: v for k, v in channel_scores.items() 
            if k not in blocked and v > 0
        }
        
        if available_channels:
            preferred = max(available_channels, key=available_channels.get)
        else:
            preferred = 'sms'  # Default fallback
        
        return ChannelProfile(
            customer_id=customer_id,
            channel_scores=channel_scores,
            blocked_channels=blocked,
            fatigued_channels=fatigued,
            preferred_channel=preferred,
            last_contact_by_channel=last_contact
        )


class ChannelSelector:
    """
    Selects optimal contact channel for a customer.
    
    Selection strategy:
    1. Exclude blocked channels
    2. Penalise fatigued channels
    3. Score by historical response rate
    4. Consider message urgency / escalation level
    
    Example usage:
        selector = ChannelSelector()
        recommendation = selector.select(
            customer_id='C001',
            contact_history=history_df,
            escalation_level=1
        )
    """
    
    def __init__(self):
        self.scorer = ChannelScorer()
        
        # Channel intensity mapping
        self.channel_intensity = {
            Channel.PUSH_NOTIFICATION: ChannelIntensity.LOW,
            Channel.IN_APP: ChannelIntensity.LOW,
            Channel.SMS: ChannelIntensity.LOW,
            Channel.EMAIL: ChannelIntensity.MEDIUM,
            Channel.WHATSAPP: ChannelIntensity.MEDIUM,
            Channel.CALL: ChannelIntensity.HIGH
        }
        
        # Escalation ladder
        self.escalation_sequence = [
            Channel.SMS,
            Channel.WHATSAPP,
            Channel.EMAIL,
            Channel.CALL
        ]
    
    def select(
        self,
        customer_id: str,
        contact_history: pd.DataFrame,
        escalation_level: int = 0,
        required_intensity: Optional[ChannelIntensity] = None,
        exclude_channels: Optional[List[Channel]] = None
    ) -> ChannelRecommendation:
        """
        Select optimal channel for contacting a customer.
        
        Args:
            customer_id: Customer identifier
            contact_history: Contact attempt history
            escalation_level: Current escalation level (0=initial, higher=more urgent)
            required_intensity: Optional minimum intensity required
            exclude_channels: Optional channels to exclude
            
        Returns:
            ChannelRecommendation with selected channel and reasoning
        """
        # Get channel profile
        profile = self.scorer.score_channels(customer_id, contact_history)
        
        # Build exclusion list
        avoid = set(profile.blocked_channels)
        if exclude_channels:
            avoid.update([c.value for c in exclude_channels])
        
        # Filter available channels
        available_scores = {
            k: v for k, v in profile.channel_scores.items()
            if k not in avoid
        }
        
        if not available_scores:
            # No channels available
            return ChannelRecommendation(
                customer_id=customer_id,
                recommended_channel=Channel.SMS,  # Last resort
                backup_channel=None,
                confidence=0.1,
                reason="All preferred channels blocked or unavailable",
                avoid_channels=[Channel(c) for c in avoid if c in [ch.value for ch in Channel]]
            )
        
        # Apply escalation logic
        if escalation_level > 0:
            # Higher escalation = prefer higher intensity channels
            for channel in self.escalation_sequence[escalation_level:]:
                if channel.value in available_scores:
                    selected = channel
                    break
            else:
                # Fall back to best available
                selected = Channel(max(available_scores, key=available_scores.get))
        else:
            # Standard selection - use best performing channel
            selected = Channel(max(available_scores, key=available_scores.get))
        
        # Check intensity requirement
        if required_intensity:
            channel_int = self.channel_intensity[selected]
            if channel_int.value < required_intensity.value:
                # Need higher intensity - find suitable channel
                for channel in self.escalation_sequence:
                    if (self.channel_intensity[channel].value >= required_intensity.value
                        and channel.value in available_scores):
                        selected = channel
                        break
        
        # Apply fatigue penalty
        if selected.value in profile.fatigued_channels:
            # Try to find non-fatigued alternative
            non_fatigued = {
                k: v for k, v in available_scores.items()
                if k not in profile.fatigued_channels
            }
            if non_fatigued:
                backup = Channel(max(non_fatigued, key=non_fatigued.get))
                reason = f"Primary channel {selected.value} fatigued, consider {backup.value}"
            else:
                backup = None
                reason = f"Selected {selected.value} despite fatigue - no alternatives"
        else:
            # Find backup channel
            remaining = {k: v for k, v in available_scores.items() if k != selected.value}
            backup = Channel(max(remaining, key=remaining.get)) if remaining else None
            reason = f"Selected {selected.value} based on {available_scores[selected.value]:.0%} response rate"
        
        # Calculate confidence
        score = available_scores.get(selected.value, 0.5)
        confidence = score * (0.8 if selected.value in profile.fatigued_channels else 1.0)
        
        return ChannelRecommendation(
            customer_id=customer_id,
            recommended_channel=selected,
            backup_channel=backup,
            confidence=confidence,
            reason=reason,
            avoid_channels=[Channel(c) for c in avoid if c in [ch.value for ch in Channel]]
        )
    
    def select_batch(
        self,
        customer_ids: List[str],
        contact_history: pd.DataFrame,
        escalation_levels: Optional[Dict[str, int]] = None
    ) -> pd.DataFrame:
        """
        Select channels for multiple customers.
        
        Args:
            customer_ids: List of customer identifiers
            contact_history: Contact history for all customers
            escalation_levels: Optional dict of customer_id -> escalation_level
            
        Returns:
            DataFrame with channel recommendations
        """
        escalation_levels = escalation_levels or {}
        results = []
        
        for customer_id in customer_ids:
            level = escalation_levels.get(customer_id, 0)
            rec = self.select(customer_id, contact_history, level)
            
            results.append({
                'customer_id': rec.customer_id,
                'recommended_channel': rec.recommended_channel.value,
                'backup_channel': rec.backup_channel.value if rec.backup_channel else None,
                'confidence': rec.confidence,
                'reason': rec.reason
            })
        
        return pd.DataFrame(results)


class MessageTemplateSelector:
    """
    Selects appropriate message template based on channel and context.
    
    Message tones:
    - Reminder: Neutral, informational
    - Nudge: Slightly more urgent
    - Escalation: Firm, consequences mentioned
    - Final: Last chance, clear consequences
    """
    
    def __init__(self):
        self.templates = {
            'sms': {
                'reminder': "Hi {name}, your payment of {amount} is due. Pay now: {link}",
                'nudge': "Hi {name}, your payment is {dpd} days overdue. Avoid late fees: {link}",
                'escalation': "{name}, urgent: {amount} overdue. Pay today to avoid further action: {link}",
                'final': "FINAL NOTICE: {name}, pay {amount} immediately to avoid account restriction."
            },
            'whatsapp': {
                'reminder': "Hello {name}! 👋\n\nJust a quick reminder that your payment of {amount} is due.\n\nPay easily here: {link}",
                'nudge': "Hi {name},\n\nYour payment is now {dpd} days overdue. We're here to help if you're having difficulties.\n\nPay now: {link}",
                'escalation': "Hi {name},\n\nThis is urgent. Your account is {dpd} days overdue with {amount} outstanding.\n\nPlease pay today: {link}\n\nOr reply to discuss payment options.",
                'final': "⚠️ FINAL NOTICE\n\n{name}, your account will be escalated if payment of {amount} is not received today.\n\nPay now: {link}"
            },
            'email': {
                'reminder': "Payment Reminder",
                'nudge': "Your Payment is Overdue",
                'escalation': "Urgent: Payment Required",
                'final': "Final Notice: Immediate Payment Required"
            }
        }
    
    def select_template(
        self,
        channel: Channel,
        escalation_level: int,
        dpd: int
    ) -> Tuple[str, str]:
        """
        Select message template based on context.
        
        Args:
            channel: Selected channel
            escalation_level: Current escalation level
            dpd: Days past due
            
        Returns:
            Tuple of (tone, template_key)
        """
        # Determine tone based on DPD and escalation
        if dpd <= 3 and escalation_level == 0:
            tone = 'reminder'
        elif dpd <= 14 or escalation_level == 1:
            tone = 'nudge'
        elif dpd <= 30 or escalation_level == 2:
            tone = 'escalation'
        else:
            tone = 'final'
        
        return tone, self.templates.get(channel.value, {}).get(tone, '')


class ChannelOptimizer:
    """
    Optimises channel selection across a portfolio for maximum efficiency.
    
    Considers:
    - Overall channel capacity (call center limits)
    - Time of day optimisation
    - Cost per channel
    - Expected response rates
    """
    
    def __init__(
        self,
        call_capacity_per_hour: int = 100,
        sms_cost: float = 0.02,
        call_cost: float = 0.10,
        whatsapp_cost: float = 0.03
    ):
        self.call_capacity = call_capacity_per_hour
        self.channel_costs = {
            'sms': sms_cost,
            'call': call_cost,
            'whatsapp': whatsapp_cost,
            'email': 0.001,
            'push': 0.0
        }
    
    def optimize_portfolio(
        self,
        recommendations: pd.DataFrame,
        budget: Optional[float] = None,
        max_calls: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Optimise channel allocation across portfolio.
        
        Args:
            recommendations: DataFrame with channel recommendations
            budget: Optional contact budget limit
            max_calls: Optional limit on call volume
            
        Returns:
            Optimised recommendations DataFrame
        """
        df = recommendations.copy()
        
        # Apply call capacity constraint
        if max_calls:
            call_recs = df[df['recommended_channel'] == 'call']
            if len(call_recs) > max_calls:
                # Sort by confidence, keep top N for calls
                call_recs = call_recs.sort_values('confidence', ascending=False)
                demote_ids = call_recs.iloc[max_calls:]['customer_id'].tolist()
                
                # Demote excess to backup channel or SMS
                for cid in demote_ids:
                    mask = df['customer_id'] == cid
                    backup = df.loc[mask, 'backup_channel'].iloc[0]
                    df.loc[mask, 'recommended_channel'] = backup or 'sms'
                    df.loc[mask, 'reason'] = 'Demoted from call due to capacity'
        
        # Apply budget constraint
        if budget:
            df['cost'] = df['recommended_channel'].map(self.channel_costs)
            total_cost = df['cost'].sum()
            
            if total_cost > budget:
                # Sort by value (confidence / cost ratio)
                df['value_ratio'] = df['confidence'] / df['cost'].clip(lower=0.001)
                df = df.sort_values('value_ratio', ascending=False)
                
                # Downgrade low-value contacts to cheaper channels
                cumsum = df['cost'].cumsum()
                over_budget = cumsum > budget
                
                df.loc[over_budget, 'recommended_channel'] = 'sms'
                df.loc[over_budget, 'reason'] = 'Downgraded due to budget constraint'
            
            df = df.drop(columns=['cost', 'value_ratio'], errors='ignore')
        
        return df


if __name__ == "__main__":
    # Example usage
    selector = ChannelSelector()
    
    # Create sample contact history
    sample_history = pd.DataFrame({
        'customer_id': ['C001'] * 10,
        'channel': ['sms', 'sms', 'sms', 'whatsapp', 'whatsapp', 
                    'call', 'call', 'sms', 'whatsapp', 'call'],
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='3D'),
        'responded': [True, False, True, True, True, 
                      False, True, True, True, False],
        'blocked': [False] * 10
    })
    
    recommendation = selector.select('C001', sample_history)
    
    print("Channel Selection Example")
    print("=" * 50)
    print(f"Customer: {recommendation.customer_id}")
    print(f"Recommended: {recommendation.recommended_channel.value}")
    print(f"Backup: {recommendation.backup_channel.value if recommendation.backup_channel else 'None'}")
    print(f"Confidence: {recommendation.confidence:.2f}")
    print(f"Reason: {recommendation.reason}")
