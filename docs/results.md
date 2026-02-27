# Results: Production Performance

## Overview

This document summarises results from production deployment of the Collections Intelligence Framework in Nigerian digital lending portfolios.

---

## Experiment 1: Behavioural Segmentation for SMS Campaigns

### Setup

- **Portfolio**: Consumer lending (digital)
- **Intervention**: SMS collection campaigns
- **Treatment**: Segmented by prior repayment behaviour
- **Control**: Untargeted mass campaigns (all delinquent accounts)
- **Message content**: Identical across treatment and control

### Results

| Metric | Control (Mass) | Treatment (Segmented) | Improvement |
|--------|----------------|----------------------|-------------|
| Conversion Rate | Baseline | 2x+ Baseline | **>100%** |
| Messages Sent | High volume | Reduced volume | Lower cost |
| Borrower Complaints | Higher | Lower | Better experience |

### Key Finding

**Targeting alone—without changing message content—more than doubled conversion rates.**

The improvement came entirely from reaching the right borrowers at the right time, not from better copywriting or stronger language.

---

## Experiment 2: Willingness-Capacity Segmentation

### Setup

Analysis of broken payment arrangements (borrowers who made a promise-to-pay but did not fulfil it).

### Finding

Approximately **1 in 4 customers** with broken arrangements had actually **attempted repayment but failed**.

These borrowers were being treated as low-willingness (broken promise) when they were actually high-willingness, low-capacity (payment attempt failed due to insufficient funds or technical issues).

### Intervention

Reclassified this segment and applied:
- Lower-friction contact (simple reminder vs escalation)
- Timing aligned to next expected liquidity window
- Payment retry facilitation

### Results

| Metric | Generic Escalation | Targeted Approach | Improvement |
|--------|-------------------|-------------------|-------------|
| Same-day Response Rate | Baseline | 8x Baseline | **8x** |

### Key Finding

**Correctly identifying intent signals (payment attempts) and responding appropriately produced 8x improvement in same-day response.**

---

## Experiment 3: Contact Timing Optimisation

### Setup

Aligned contact attempts with borrowers' historical payment windows (detected from transaction data patterns).

### Methodology

1. Analysed historical transaction data to detect inflow patterns
2. Identified likely salary dates, market days, remittance windows
3. Scheduled contact attempts within 1-3 days of expected liquidity

### Results

| Metric | Random Timing | Optimised Timing | Improvement |
|--------|---------------|------------------|-------------|
| Promise-to-Pay Fulfilment | Baseline | +5-15% | **5-15%** |

### Key Finding

**Timing contact to liquidity windows improved promise-to-pay fulfilment by 5-15%.**

Contact outside liquidity windows not only failed to improve recovery—it often triggered avoidance behaviours that reduced future recoverability.

---

## Operational Impact

### Contact Volume

| Metric | Before | After |
|--------|--------|-------|
| Contact Attempts per Account | High | Significantly Reduced |
| Recovery per Contact | Low | Higher |

Algorithmic triage reserves contact for moments when it is likely to work. Collections transformed from a volume game into a precision exercise.

### Collections Team Performance

| Metric | Before | After |
|--------|--------|-------|
| Agent Time Allocation | Chasing noise | Resolving viable cases |
| Team Attrition | Higher | Reduced |
| Agent Morale | Fatigue from pressure | Improved (effort → outcomes) |

Harassment-based models depend on relentless pressure—high call quotas, repetitive scripts, constant escalation. This creates emotional fatigue, high attrition, and declining quality of engagement.

When algorithms filter and prioritise accounts, agents spend more time on cases with realistic recovery chances.

### Borrower Experience

| Metric | Before | After |
|--------|--------|-------|
| Customer Hostility | Higher | Declined |
| Communication Tone | Confrontational | Proportionate |
| Reactivation Rate | Low | Viable |

Borrowers experienced communication aligned to their circumstances:
- Well-timed reminders
- Clearer payment pathways
- Fewer confrontational messages

Trust preserved during delinquency makes reactivation and repeat borrowing viable rather than exceptional.

---

## Compliance Impact

### Regulatory Alignment

The framework provides:

| Requirement | How Framework Addresses |
|-------------|------------------------|
| No excessive messaging | Contact suppression for high self-cure accounts |
| No harassment | Proportionate, timed contact only |
| No unauthorised third-party contact | Eliminated from strategy |
| Audit trail | Documented decision logic for every action |

### Risk Reduction

Algorithmic collections provide:
- Clear rules for escalation
- Documented decision logic
- Explainable actions (why contact was made or withheld)
- Audit trails for regulatory review

---

## Summary of Results

| Component | Metric | Result |
|-----------|--------|--------|
| Behavioural Segmentation | Conversion Rate | **>100% improvement** |
| Willingness-Capacity Matrix | Same-day Response | **8x improvement** |
| Contact Timing | PTP Fulfilment | **5-15% improvement** |
| Overall | Contact Volume | **Significant reduction** |
| Overall | Team Attrition | **Decreased** |
| Overall | Borrower Hostility | **Declined** |

---

## Conclusion

The results demonstrate that **restraint—algorithmically informed—outperforms pressure**.

Intelligence-based collections:
- Improve recovery rates
- Reduce operational costs
- Preserve customer relationships
- Lower compliance risk

The framework validates the core thesis: **harassment feels decisive but teaches nothing; algorithms compound insight with every cycle.**
