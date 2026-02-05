# Collections Intelligence Framework

**Algorithmic decision systems for debt recovery in consumer lending**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements a data-driven collections framework that replaces harassment-based recovery tactics with algorithmic decision-making. The system determines **whom to contact**, **when to contact**, **how to contact**, and critically, **when not to act at all**.

Built from production experience in Nigerian digital lending, where coercion-first collections dominate despite evidence of their inefficiency. The framework demonstrates that restraint, algorithmically informed, often outperform, s pressure.

## The Problem

Traditional collections systems treat all delinquent borrowers identically:

- **Day 1 past due** → Aggressive SMS, calls, escalation
- **Binary classification** → "Good payer" vs "Defaulter"
- **Activity metrics** → Calls made, messages sent (not outcomes achieved)
- **No learning** → Each cycle extracts value without improving future performance

This approach:
- Misclassifies **liquidity stress** as moral failure
- Triggers **avoidance behaviours** (SIM changes, app deletion, number blocking)
- Destroys **customer lifetime value** (repeat borrowing, referrals)
- Exposes lenders to **regulatory risk**

## The Solution

### Willingness-Capacity Matrix

Segment borrowers on two dimensions rather than one:

```
                    HIGH CAPACITY    LOW CAPACITY
                   ┌────────────────┬────────────────┐
    HIGH           │   Minimal      │   Restructure  │
    WILLINGNESS    │   Intervention │   or Defer     │
                   ├────────────────┼────────────────┤
    LOW            │   Targeted     │   Deprioritise │
    WILLINGNESS    │   Escalation   │   Early        │
                   └────────────────┴────────────────┘
```

Resources deploy where **marginal recovery probability is highest**.

### Propensity-to-Pay Modelling

Predict repayment likelihood using:
- Repayment history and payment attempt signals
- Income regularity proxies
- Response to prior contact
- Device and behavioural signals
- Timing variables aligned to cash inflow patterns

**Key insight**: The most powerful application is deciding **who not to contact**. Suppressing unnecessary activity reduces cost, prevents borrower fatigue, and preserves goodwill.

### DPD as Trajectory

Days Past Due (DPD) is not a status, it's a trajectory.

Two borrowers at 15 DPD can have radically different risk profiles:
- One **decelerating** (slowing drift, likely to stabilise)
- One **accelerating** (rapid deterioration, structural default)

The slope of delinquency has greater predictive value than the absolute number.

### Contact Timing Optimisation

Nigerian borrowers have liquidity patterns tied to:
- Salary dates
- Market days
- Contract settlements
- Remittance windows

Contact timed to these windows converts. Contact outside them triggers avoidance.

## Results

From production deployment:

| Metric | Improvement |
|--------|-------------|
| SMS campaign conversion | **>100%** vs untargeted campaigns |
| Same-day response (targeted segment) | **8x** vs generic escalation |
| Promise-to-pay fulfilment | **5-15%** improvement |
| Contact volume | **Significant reduction** |
| Collections team attrition | **Decreased** |

## Repository Structure

```
collections-intelligence-framework/
├── README.md
├── LICENSE
├── requirements.txt
├── docs/
│   ├── problem_statement.md
│   ├── methodology.md
│   ├── willingness_capacity_matrix.md
│   └── results.md
├── notebooks/
│   ├── 01_delinquency_patterns.ipynb
│   ├── 02_segmentation_analysis.ipynb
│   ├── 03_propensity_modelling.ipynb
│   ├── 04_contact_timing.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── segmentation/
│   │   ├── willingness_capacity.py
│   │   └── dpd_trajectory.py
│   ├── propensity/
│   │   ├── payment_propensity.py
│   │   └── feature_engineering.py
│   └── contact_optimization/
│       ├── timing_optimizer.py
│       └── channel_selector.py
├── tests/
│   ├── test_segmentation.py
│   ├── test_propensity.py
│   └── test_timing.py
└── data/
    └── synthetic/
        └── README.md
```

## Installation

```bash
git clone https://github.com/[username]/collections-intelligence-framework.git
cd collections-intelligence-framework
pip install -r requirements.txt
```

## Quick Start

```python
from src.segmentation.willingness_capacity import WillingnessCapacityMatrix
from src.propensity.payment_propensity import PropensityModel

# Segment portfolio
matrix = WillingnessCapacityMatrix()
segments = matrix.segment(portfolio_df)

# Score propensity
model = PropensityModel()
scores = model.predict(borrower_features)

# Identify suppression candidates (high propensity, no contact needed)
suppress = scores[scores['propensity'] > 0.7]['customer_id']
```

## Key Concepts

### Temporary vs Structural Default

| Type | Characteristics | Optimal Response |
|------|-----------------|------------------|
| **Temporary** | Intent to repay, liquidity-constrained (delayed salary, inventory turnover) | Wait for liquidity window, minimal contact |
| **Structural** | Inability or unwillingness, over-indebtedness, income instability | Early deprioritisation or restructure |

### Contact Suppression

Not contacting is a strategic choice. Suppression candidates:
- High propensity to self-cure
- Recently contacted (fatigue risk)
- Outside liquidity window
- Showing payment intent signals (failed attempts)

### Friction Signals

Failed payment attempts are data, not failures:
- **Technical failure** → Retry facilitation
- **Insufficient funds** → Wait for liquidity
- **No attempt + responsive** → Timing issue
- **No attempt + blocking** → Avoidance (deprioritise)

## Regulatory Context

This framework aligns with consumer protection requirements, including Nigeria's FCCPC Digital Lending Regulations which prohibit:
- Harassment and excessive messaging
- Unauthorised third-party contact
- Privacy violations in debt recovery

Algorithmic collections provide audit trails and documented decision logic for compliance.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Related Reading

- [Why Nigerian Lenders Should Replace Harassment with Algorithms](#) ,  Article explaining the business case
- FCCPC Digital, Electronic, Online or Non-Traditional Consumer Lending Regulations, 2025

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Temidayo Akindahunsi**  
Data Professional specialising in collections intelligence and credit risk modelling for fintech and digital lending.
