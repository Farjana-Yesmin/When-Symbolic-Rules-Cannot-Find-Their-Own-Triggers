# When Symbolic Rules Cannot Find Their Own Triggers

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange.svg)](https://pytorch.org/)
[![EIML@ICML 2026](https://img.shields.io/badge/EIML%40ICML-2026-purple.svg)](https://icml.cc)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official implementation of:

> *When Symbolic Rules Cannot Find Their Own Triggers: An Epistemic Diagnosis
> of Predicate Grounding Collapse in Neuro-Symbolic Fraud Detection.*
> Under review, EIML@ICML 2026 (2nd Workshop on Epistemic Intelligence in ML).

---

## This Is a Negative Result

The central finding is a **predicate grounding collapse**: RuleNet's symbolic
rules never activate on realistic fraud data. Rule firing rate = **0% across
all 5 seeds**. RuleNet is **statistically indistinguishable from a plain DNN**
(F1: 0.9826±0.0052 vs 0.9835±0.0052, p=0.068 — not significant).

This is an honest negative result that cautions the neuro-symbolic community
about predicate activation conditions on standardised continuous tabular data.

---

## Key Finding

**The model has an epistemic blind spot.** It cannot signal whether a prediction
is grounded in symbolic domain rules or opaque pattern matching — the failure
mode neuro-symbolic integration is supposed to prevent.

Three interacting causes:

1. **Product t-norm collapse:** With standardised features, sigmoid outputs ≈ 0.5.
   Two-body conjunction: 0.5 × 0.5 = 0.25 < 0.5 firing threshold.
   Rules cannot fire at initialisation.

2. **Gradient dominance:** A strong DNN (AUROC ≈ 0.9998) produces large
   classification gradients that dwarf the semantic loss — rules receive
   near-zero training signal.

3. **Random initialisation:** No predicate has any prior toward its correct
   activation region.

---

## Results

### Main Comparison (mean ± std, 5 seeds)

| Model | Accuracy | F1 | ROC-AUC | Fire Rate | Infer. |
|---|---|---|---|---|---|
| **RuleNet (ours)** | 0.9965±0.0010 | 0.9826±0.0052 | 0.9998±0.0001 | **0.0%** | 1.14 µs |
| PureDNN | 0.9967±0.0010 | 0.9835±0.0052 | 0.9998±0.0001 | — | 1.17 µs |
| LTN | 0.9963±0.0011 | 0.9815±0.0055 | 0.9998±0.0001 | — | — |
| RuleFit | 0.9688±0.0020 | 0.8592±0.0090 | 0.9937±0.0021 | — | — |
| Decision Tree | 0.9899±0.0024 | 0.9509±0.0111 | 0.9832±0.0047 | — | — |

RuleNet vs PureDNN: t=−1.86, p=0.068 — **not significant**.

### Ablation — Rules Are Inert

| Configuration | F1 | Fire Rate |
|---|---|---|
| RuleNet λ=0.5 | 0.9826 | 0.0% |
| λ=10 (rule-dominant) | 0.9826 | 0.0% |
| Correct rules | 0.9776 | 0.0% |
| **Irrelevant rules** | **0.9825** | **0.0%** |

Wrong rules slightly outperform correct ones (+0.0049 F1) — rule content
is entirely irrelevant because no rule ever fires.

### Rule Count Ablation

| n rules | F1 | Fire Rate |
|---|---|---|
| 0 | 0.9809 | 0.0% |
| 1 | 0.9818 | 0.0% |
| 3 | 0.9801 | 0.0% |
| 10 | 0.9825 | 0.0% |

Adding more rules has no effect.

---

## Proposed Remedies

**Remedy 1 — Calibrated predicate initialisation:**
Set predicate biases so ϕ(x̄_fraud) = 0.5, where x̄_fraud is the
class-conditional mean for fraud instances.

**Remedy 2 — T-norm substitution:**
Replace product t-norm with Łukasiewicz or geometric mean.
Geometric mean gives two-body conjunction ≈ 0.5 at initialisation
instead of 0.25 — already at the firing threshold.

**Remedy 3 — Firing-rate regularisation:**
Add auxiliary loss: L_fire = max(0, r* − mean(c_ij))
where r* is a target firing rate (e.g., 0.2).

---

## Three Symbolic Fraud Rules

```python
L1: highAmount ∧ lowBalance   → isFraud
L2: highVelocity ∧ oddHour    → isFraud
L3: highRisk ∧ crossBorder    → isFraud
```

Each predicate is grounded on semantically relevant features only
via an independent linear module with sigmoid output.

---

## Dataset

Synthetic fraud benchmark (n=40,000, 10% fraud, 8 features).
Substantial class overlap, calibrated Gaussian noise (σ=0.6) prevents
trivial separability. All features standardised before model input.

5 seeds: {42, 123, 456, 789, 1011}. Train/val/test: 70/15/15.

---

## Quick Start

```bash
pip install torch scikit-learn pandas numpy matplotlib
```

```python
# Run the notebook
jupyter notebook when-symbolic-rules-cannot-find-their-own-triggers.ipynb
```

**Key hyperparameters:**
```python
config = {
    'lambda': 0.5,          # semantic loss weight (grid: 1e-4 to 10)
    'hidden_dim': 128,
    'lr': 1e-3,
    'threshold': 0.5,       # rule firing threshold
    'n_seeds': 5,
    'patience': 12,         # early stopping on val F1
}
```

---

## Epistemic Transparency Metrics

Two metrics introduced in this paper for diagnosing symbolic integration:

- **Rule firing rate:** fraction of test instances where any rule body
  conjunction exceeds 0.5. Ours: **0.0%** (complete collapse).
- **Rule alignment rate:** among fired instances, fraction where rule head
  agrees with ground truth. Ours: 16.7% (near-random, computed on
  marginal conjunction values).

**We call on the neuro-symbolic community to report rule firing rates
alongside accuracy, so that symbolic interpretability claims can be
verified rather than assumed.**

---

## Citation

```bibtex
@inproceedings{yesmin et al 2026rulenet,
  title  = {When Symbolic Rules Cannot Find Their Own Triggers:
            An Epistemic Diagnosis of Predicate Grounding Collapse
            in Neuro-Symbolic Fraud Detection},
  author = {Farjana Yesmin, Nusrat Shirmin},
  note   = {Under review, EIML@ICML 2026},
  year   = {2026}
}
```

---

**Author:** Farjana Yesmin ·
[farjana-yesmin.github.io](https://farjana-yesmin.github.io) · Apache 2.0
