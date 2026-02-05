# Problem Statement: The Failure of Coercion-First Collections

## Industry Context

Nigeria's digital lending sector represents one of the most technologically advanced segments of the financial services ecosystem—offering real-time credit decisioning, instant disbursement, and mobile-first platforms. Yet collections architecture remains primitive.

When loans become delinquent, most lenders default to:
- Mass SMS and WhatsApp messaging (multiple times daily)
- Aggressive voice calls
- Contact-list harvesting (messaging borrowers' friends, family, colleagues)
- Public shaming tactics

An industry capable of building real-time underwriting engines enforces repayment with methods that predate data science.

## The Binary Classification Problem

Collections systems typically segment borrowers into two categories:

```
┌─────────────────────────────────────────────────────────┐
│                    TRADITIONAL VIEW                      │
├────────────────────────┬────────────────────────────────┤
│      GOOD PAYER        │          DEFAULTER             │
│    (pays on time)      │    (everyone else)             │
└────────────────────────┴────────────────────────────────┘
```

This collapses a spectrum of repayment behaviours into a single "bad" category:

| Borrower Type | Actual Situation | Treatment Under Binary Model |
|---------------|------------------|------------------------------|
| Salary earner paid late | Temporary liquidity constraint | Harassed from Day 1 |
| Trader awaiting inventory turnover | Timing mismatch | Harassed from Day 1 |
| Gig worker with irregular income | Predictable cash flow pattern | Harassed from Day 1 |
| Over-indebted borrower | Structural inability to repay | Harassed from Day 1 |
| Borrower with no intent to repay | True default risk | Harassed from Day 1 |

All five receive identical treatment despite requiring fundamentally different interventions.

## The Misattribution Problem

Coercion-based collections appear effective because some borrowers pay following aggressive contact. This creates a dangerous misattribution:

```
Observation:  Borrower received 10 calls → Borrower paid
Attribution:  Calls caused payment
Reality:      Borrower's salary arrived → Borrower paid
              (Calls merely coincided with liquidity event)
```

A significant proportion of delinquent borrowers will self-cure without any intervention once liquidity constraints ease. Harassment often takes credit for recoveries it did not cause.

## The Avoidance Response

Early aggression triggers defensive behaviours, particularly among digitally savvy borrowers:

1. **SIM card changes** — Primary contact number abandoned
2. **App deletion** — Platform access severed
3. **Number blocking** — Future contact prevented
4. **Social media blocking** — Alternative channels closed

Once disengagement occurs, traceability is lost. What might have been recoverable short-term delinquency becomes permanent write-off.

Research on consumer credit behaviour indicates that borrower responsiveness decreases sharply after repeated, non-targeted contact attempts. Each additional message yields diminishing marginal recovery.

## The Lifetime Value Destruction

Repeat borrowing, cross-sell opportunities, and referrals form a substantial portion of long-term portfolio profitability. Harassment-based collections sever these relationships irreversibly.

```
┌─────────────────────────────────────────────────────────┐
│              CUSTOMER JOURNEY UNDER HARASSMENT           │
├─────────────────────────────────────────────────────────┤
│  Loan 1 → Temporary Delinquency → Harassment →          │
│  Repayment Under Duress → NEVER RETURNS                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│           CUSTOMER JOURNEY UNDER INTELLIGENCE            │
├─────────────────────────────────────────────────────────┤
│  Loan 1 → Temporary Delinquency → Timed Reminder →      │
│  Self-Cure → Loan 2 → Loan 3 → Referrals → LTV          │
└─────────────────────────────────────────────────────────┘
```

## The Regulatory Shift

Nigeria's Federal Competition and Consumer Protection Commission (FCCPC) introduced the Digital, Electronic, Online or Non-Traditional Consumer Lending Regulations specifically targeting:

- Harassment and excessive messaging
- Unauthorised third-party contact (contact-list abuse)
- Privacy violations in debt recovery
- Unfair collection practices

As of January 2026, over 100 loan apps face sanctions including fines up to ₦100 million for non-compliance. The regulatory environment has shifted from tolerance to active enforcement.

## The No-Learning Problem

The most fundamental failure of coercion-based collections is structural: **it does not learn**.

| Characteristic | Harassment-Based | Algorithmic |
|----------------|------------------|-------------|
| Feedback loop | None | Continuous |
| Model improvement | Static | Each cycle refines predictions |
| Segmentation | Binary (good/bad) | Multi-dimensional |
| Timing | Immediate escalation | Aligned to liquidity windows |
| Outcome measurement | Activity (calls made) | Effectiveness (recovery per contact) |

Each harassment cycle extracts value without improving future outcomes. Algorithmic systems compound insight with every repayment cycle, continuously refining predictions and interventions.

## Problem Summary

The industry requires a transition from:

> **Intensity-based collections** (maximise pressure, measure activity)

To:

> **Intelligence-based collections** (optimise timing, measure outcomes)

This repository provides the analytical framework and implementation tools to make that transition.
