# Willingness-Capacity Matrix

## Overview

The Willingness-Capacity Matrix is a two-dimensional segmentation framework that replaces binary "good payer / defaulter" classification with a nuanced understanding of borrower behaviour.

## The Framework

Traditional collections segment borrowers on a single dimension: **paid** or **not paid**. This collapses a wide spectrum of repayment behaviour into one category and undermines collection effectiveness.

The Willingness-Capacity Matrix segments on two dimensions:

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

## Segment Definitions

### Segment A: High Willingness, High Capacity
**Strategy: Monitor**

These borrowers intend to pay and have the means to do so. Delinquency is typically due to:
- Oversight or forgetfulness
- Minor timing mismatch
- Technical payment issues

**Optimal intervention**: Minimal. A single well-timed reminder is usually sufficient. Over-contact wastes resources and risks irritating a good customer.

**Indicators**:
- Payment attempts (even if failed)
- High response rate to contact
- Strong repayment history
- Regular income patterns

---

### Segment B: High Willingness, Low Capacity
**Strategy: Restructure**

These borrowers want to pay but are genuinely constrained. Common causes:
- Temporary income disruption (job loss, delayed salary)
- Unexpected expenses (medical, emergency)
- Over-indebtedness across multiple lenders
- Seasonal income variation

**Optimal intervention**: Offer flexibility—payment plans, deferrals, reduced settlements. Pressure is counterproductive; it increases stress without improving capacity.

**Indicators**:
- Multiple failed payment attempts (shows intent)
- Proactive communication about difficulties
- Irregular or declining income signals
- High debt-to-income ratio

---

### Segment C: Low Willingness, High Capacity
**Strategy: Escalate**

These borrowers can pay but choose not to. They may be:
- Testing boundaries
- Prioritising other debts
- Disputing the debt
- Experiencing dissatisfaction with the lender

**Optimal intervention**: Targeted escalation with clear consequences. Unlike Segment A/B, these borrowers respond to firmness. However, escalation should be measured—excessive pressure can trigger avoidance.

**Indicators**:
- No payment attempts despite capacity
- Low response rate / ignoring contact
- Active channel blocking
- Strong income signals but no action

---

### Segment D: Low Willingness, Low Capacity
**Strategy: Deprioritise**

These accounts have low recovery probability regardless of intervention. Continued contact:
- Wastes operational resources
- Increases borrower distress
- Risks regulatory complaints

**Optimal intervention**: Early deprioritisation. Accept that some accounts will not recover in the short term. Consider write-off path or long-term passive recovery.

**Indicators**:
- No payment attempts
- No response to contact
- Channels blocked
- Weak or absent income signals
- Multiple concurrent delinquencies

---

## Willingness Signals

### Positive Indicators (High Willingness)

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Payment attempts (failed) | High | Strongest intent signal—tried but couldn't |
| Responsive to contact | Medium | Engaged, not avoiding |
| Proactive communication | High | Reached out before being chased |
| Partial payments | Medium | Contributing what they can |
| Promise-to-pay kept | High | Follows through on commitments |

### Negative Indicators (Low Willingness)

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Channel blocking | High | Actively avoiding contact |
| SIM change / app uninstall | High | Severing communication |
| No payment attempts | Medium | Not even trying |
| Broken promises (no attempt) | High | Commits but doesn't act |
| Ignores all contact | Medium | Passive avoidance |

---

## Capacity Signals

### Positive Indicators (High Capacity)

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Regular income pattern | High | Predictable cash flow |
| Strong transaction activity | Medium | Active financial life |
| Previous loans completed | High | Track record of repayment |
| Low debt-to-income ratio | Medium | Not over-leveraged |
| Stable account tenure | Low | Established relationship |

### Negative Indicators (Low Capacity)

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Irregular/declining income | High | Unstable cash flow |
| High debt-to-income ratio | High | Over-leveraged |
| Multiple concurrent defaults | High | Systemic inability |
| Declining transaction activity | Medium | Financial distress |
| Recent negative shocks | Medium | Job loss, medical emergency |

---

## Implementation

### Scoring Approach

```python
willingness_score = weighted_sum(willingness_signals)  # 0 to 1
capacity_score = weighted_sum(capacity_signals)        # 0 to 1

if willingness_score >= 0.5 and capacity_score >= 0.5:
    segment = 'A'  # Monitor
elif willingness_score >= 0.5 and capacity_score < 0.5:
    segment = 'B'  # Restructure
elif willingness_score < 0.5 and capacity_score >= 0.5:
    segment = 'C'  # Escalate
else:
    segment = 'D'  # Deprioritise
```

### Confidence Scoring

Accounts near threshold boundaries (e.g., willingness = 0.48) have lower confidence than accounts clearly in a segment (e.g., willingness = 0.85). Low-confidence accounts may warrant human review.

### Dynamic Re-segmentation

Segments are not permanent. As new data arrives (payment attempts, contact responses, income signals), accounts should be re-scored and may migrate between segments.

---

## Resource Allocation

The matrix enables efficient resource allocation:

| Segment | % of Delinquent Portfolio (typical) | Resource Allocation |
|---------|-------------------------------------|---------------------|
| A | 15-25% | Minimal (automated reminders) |
| B | 20-30% | Moderate (restructuring team) |
| C | 15-25% | Higher (skilled escalation) |
| D | 25-35% | Minimal (passive monitoring) |

Most collection effort should focus on Segments B and C, where intervention has the highest marginal impact.

---

## Validation

To validate segmentation effectiveness:

1. **Recovery rate by segment**: A > B > C > D expected
2. **Self-cure rate**: Segment A should have high self-cure without intervention
3. **Restructure success**: Segment B should respond well to flexibility
4. **Escalation response**: Segment C should respond to firmness
5. **Write-off rate**: Segment D should have highest eventual write-off

If these patterns don't hold, recalibrate feature weights or thresholds.

---

## References

- `src/segmentation/willingness_capacity.py` — Implementation
- `notebooks/02_segmentation_analysis.ipynb` — Analysis and validation
- `docs/methodology.md` — Full framework methodology
