# Methodology: Algorithmic Collections Framework

## Overview

This framework replaces coercion-first collections with structured decision-making. The system answers four questions for every delinquent account:

1. **Whom** to contact
2. **When** to contact
3. **How** to contact (channel selection)
4. **Whether** to contact at all

The core premise: repayment behaviour is **predictable**, **heterogeneous**, and **responsive** to timing, channel, and context.

---

## Component 1: Willingness-Capacity Segmentation

### Conceptual Framework

Traditional collections segment on a single dimension (paid/not paid). This framework segments on two:

- **Willingness**: Intent to repay (behavioural signals)
- **Capacity**: Ability to repay (financial signals)

```
                        CAPACITY
                    HIGH         LOW
                ┌───────────┬───────────┐
         HIGH   │  SEGMENT  │  SEGMENT  │
    W           │     A     │     B     │
    I           │  Monitor  │ Restructure│
    L           ├───────────┼───────────┤
    L    LOW    │  SEGMENT  │  SEGMENT  │
    I           │     C     │     D     │
    N           │  Escalate │ Deprioritise│
    G           └───────────┴───────────┘
    N
    E
    S
    S
```

### Segment Definitions

| Segment | Willingness | Capacity | Optimal Strategy |
|---------|-------------|----------|------------------|
| A | High | High | Minimal intervention; timed reminder only |
| B | High | Low | Restructure, defer, or offer payment plan |
| C | Low | High | Targeted escalation with clear consequences |
| D | Low | Low | Early deprioritisation; minimise resource spend |

### Feature Indicators

**Willingness Signals (Positive)**
- Payment attempts (even if failed)
- Responsive to contact
- Proactive communication
- Partial payments
- Promise-to-pay history (kept)

**Willingness Signals (Negative)**
- Channel blocking (calls, SMS)
- App uninstall
- SIM change
- Ignored contact attempts
- Promise-to-pay history (broken without attempt)

**Capacity Signals (Positive)**
- Regular income patterns detected
- Stable device/location signals
- Previous successful repayments
- Low debt-to-income indicators

**Capacity Signals (Negative)**
- Multiple concurrent delinquencies
- Decreasing transaction activity
- Device downgrade signals
- Employment instability indicators

### Implementation

```python
class WillingnessCapacityMatrix:
    def __init__(self, willingness_threshold=0.5, capacity_threshold=0.5):
        self.w_thresh = willingness_threshold
        self.c_thresh = capacity_threshold
    
    def segment(self, willingness_score, capacity_score):
        if willingness_score >= self.w_thresh:
            if capacity_score >= self.c_thresh:
                return 'A'  # Monitor
            return 'B'  # Restructure
        else:
            if capacity_score >= self.c_thresh:
                return 'C'  # Escalate
            return 'D'  # Deprioritise
```

---

## Component 2: Propensity-to-Pay Modelling

### Objective

Predict the probability that a borrower will repay within a specified window, with or without intervention.

### Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT FEATURES                        │
├─────────────────────────────────────────────────────────┤
│  Repayment History    │  Historical payment patterns     │
│  Payment Attempts     │  Failed transactions (intent)    │
│  Income Proxies       │  Transaction regularity          │
│  Contact Response     │  Open rates, response times      │
│  Device Signals       │  App usage, engagement patterns  │
│  Timing Variables     │  Days since salary, market day   │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  PROPENSITY MODEL                        │
│            (Gradient Boosting / Logistic)                │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    OUTPUT SCORES                         │
├─────────────────────────────────────────────────────────┤
│  P(repay | no contact)     │  Self-cure probability     │
│  P(repay | contact)        │  Intervention uplift       │
│  Optimal contact window    │  When to reach out         │
└─────────────────────────────────────────────────────────┘
```

### Key Insight: Contact Suppression

The most valuable model output is identifying accounts to **not contact**:

```python
def identify_suppression_candidates(scores, threshold=0.7):
    """
    High self-cure probability → suppress contact
    Contact adds cost without improving outcome
    """
    return scores[scores['p_repay_no_contact'] > threshold]
```

Benefits of suppression:
- Reduced operational cost
- Prevented borrower fatigue
- Preserved goodwill
- No sacrifice in recovery rate

### Feature Engineering

```python
# Payment attempt signals
df['payment_attempts_7d'] = count_payment_attempts(df, window=7)
df['last_attempt_days'] = days_since_last_attempt(df)
df['attempt_without_success'] = (df['payment_attempts_7d'] > 0) & (df['paid'] == 0)

# Income timing proxies
df['days_since_salary_date'] = calculate_salary_distance(df)
df['is_market_day'] = is_market_day(df['date'])
df['days_to_expected_inflow'] = predict_next_inflow(df)

# Contact response history
df['sms_open_rate_30d'] = calculate_open_rate(df, window=30)
df['call_answer_rate_30d'] = calculate_answer_rate(df, window=30)
df['avg_response_time'] = calculate_response_time(df)
```

---

## Component 3: DPD Trajectory Analysis

### Concept

Days Past Due (DPD) is traditionally used as a static bucket:
- 1-7 DPD
- 8-30 DPD
- 31-60 DPD
- 60+ DPD

This misses critical information. **Trajectory** matters more than position.

### Trajectory Types

```
DPD
 │
30├─────────────────●─────────── Chronic Drifter
 │                 ╱
 │               ╱
15├─────────●───●──────────────── Late Recovery
 │        ╱     ╲
 │      ╱        ╲
 │    ╱           ●────────────── Early Stabiliser
 │  ╱            
 │●───────────────●─────────────  Rapid Falloff
 └────────────────────────────────
   Day 1    Day 7    Day 14   Day 21
```

| Trajectory | Pattern | Risk | Strategy |
|------------|---------|------|----------|
| Early Stabiliser | Quick plateau, self-cure | Low | Monitor only |
| Late Recovery | Delayed but eventual payment | Medium | Wait for liquidity window |
| Chronic Drifter | Slow continuous increase | High | Restructure or escalate |
| Rapid Falloff | Steep acceleration | Very High | Early deprioritisation |

### Implementation

```python
def calculate_dpd_trajectory(dpd_history, window=7):
    """
    Calculate slope of DPD movement over time
    Positive slope = deteriorating
    Negative slope = improving
    Near-zero slope = stable
    """
    if len(dpd_history) < 2:
        return 0
    
    x = np.arange(len(dpd_history[-window:]))
    y = np.array(dpd_history[-window:])
    slope, _ = np.polyfit(x, y, 1)
    
    return slope

def classify_trajectory(slope, dpd_current):
    if slope < -0.5:
        return 'recovering'
    elif slope < 0.5:
        return 'stable'
    elif slope < 2.0:
        return 'drifting'
    else:
        return 'falling_off'
```

---

## Component 4: Contact Timing Optimisation

### Liquidity Windows

Nigerian borrowers exhibit predictable cash flow patterns:

| Borrower Type | Liquidity Window | Optimal Contact Timing |
|---------------|------------------|------------------------|
| Salary earner | Month-end / specific date | 1-3 days after salary |
| Weekly wage | End of week | Friday evening / Saturday |
| Market trader | Market days | Day after market |
| Gig worker | Variable | After detected inflow |
| Remittance recipient | Variable | After remittance arrival |

### Timing Model

```python
class ContactTimingOptimizer:
    def __init__(self):
        self.salary_patterns = {}
        self.inflow_history = {}
    
    def learn_pattern(self, customer_id, transaction_history):
        """
        Detect regular inflow patterns from transaction data
        """
        inflows = transaction_history[transaction_history['amount'] > 0]
        
        # Detect monthly salary pattern
        monthly_pattern = detect_monthly_cycle(inflows)
        if monthly_pattern:
            self.salary_patterns[customer_id] = monthly_pattern
        
        # Detect weekly pattern
        weekly_pattern = detect_weekly_cycle(inflows)
        if weekly_pattern:
            self.inflow_history[customer_id] = weekly_pattern
    
    def optimal_contact_window(self, customer_id, current_date):
        """
        Return optimal contact window based on learned patterns
        """
        if customer_id in self.salary_patterns:
            salary_date = self.salary_patterns[customer_id]
            days_until = days_until_date(current_date, salary_date)
            
            if days_until <= 3:
                return 'contact_now'
            elif days_until <= 7:
                return 'wait_for_salary'
            else:
                return 'too_early'
        
        return 'no_pattern_detected'
```

### Contact Outside Windows

Research shows premature contact (before anticipated liquidity) does not enhance recovery but increases avoidance. Borrowers who feel pressured before they can pay disengage, severing communication channels just as repayment becomes feasible.

**Strategic restraint is a feature, not a bug.**

---

## Component 5: Channel Selection

### Channel Effectiveness Varies

Not all borrowers respond to the same channels:

| Channel | Best For | Avoid When |
|---------|----------|------------|
| SMS | Quick reminders, payment links | Already sent 3+ in cycle |
| WhatsApp | Conversational engagement | Blocked or unresponsive |
| In-app push | Active app users | App uninstalled |
| Voice call | Complex situations, negotiation | Repeatedly declined |
| Email | Documentation, formal notice | Low open rate history |

### Channel Matching Algorithm

```python
def select_channel(customer_profile, contact_history):
    """
    Select optimal channel based on historical response
    """
    # Calculate channel effectiveness scores
    scores = {}
    
    for channel in ['sms', 'whatsapp', 'call', 'push', 'email']:
        response_rate = contact_history.get(f'{channel}_response_rate', 0)
        recency_penalty = calculate_recency_penalty(contact_history, channel)
        fatigue_factor = calculate_fatigue(contact_history, channel)
        
        scores[channel] = response_rate * recency_penalty * fatigue_factor
    
    # Return highest scoring available channel
    return max(scores, key=scores.get)
```

---

## Integration: The Decision Engine

All components integrate into a single decision engine:

```python
class CollectionsDecisionEngine:
    def __init__(self):
        self.segmenter = WillingnessCapacityMatrix()
        self.propensity_model = PropensityModel()
        self.timing_optimizer = ContactTimingOptimizer()
        self.channel_selector = ChannelSelector()
    
    def decide(self, account):
        # Step 1: Segment
        segment = self.segmenter.segment(account)
        
        # Step 2: Score propensity
        propensity = self.propensity_model.predict(account)
        
        # Step 3: Check timing
        timing = self.timing_optimizer.optimal_contact_window(account)
        
        # Step 4: Decide action
        if segment == 'D':
            return Action(type='deprioritise', reason='low_willingness_low_capacity')
        
        if propensity['p_self_cure'] > 0.7:
            return Action(type='suppress', reason='high_self_cure_probability')
        
        if timing == 'too_early':
            return Action(type='wait', until=timing['next_window'])
        
        # Step 5: Select channel
        channel = self.channel_selector.select(account)
        
        return Action(type='contact', channel=channel, message=self.get_message(segment))
```

---

## Evaluation Framework

### Metrics That Matter

| Metric | Definition | Target |
|--------|------------|--------|
| Recovery per Contact | Collections / Contact Attempts | Maximise |
| Contact Efficiency Ratio | Successful Outcomes / Total Contacts | > 15% |
| Self-Cure Rate | Accounts recovered without contact | Track (not target) |
| Segment Migration | Movement between W-C quadrants | Monitor |
| Customer Reactivation | Repeat borrowing after delinquency | > 20% |

### Metrics to Deprecate

| Metric | Problem |
|--------|---------|
| Calls per Agent | Rewards activity, not outcomes |
| Messages Sent | Encourages spam |
| Contact Attempts | Volume ≠ effectiveness |

---

## Summary

The framework shifts collections from **intensity-based** to **intelligence-based**:

| Dimension | Before | After |
|-----------|--------|-------|
| Segmentation | Binary | Willingness-Capacity Matrix |
| Targeting | Everyone | Propensity-scored |
| Timing | Immediate | Liquidity-aligned |
| Channel | Blast all | Response-matched |
| Measurement | Activity | Outcomes |
| Learning | None | Continuous refinement |
