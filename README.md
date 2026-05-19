# When Symbolic Rules Cannot Find Their Own Triggers

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange.svg)](https://pytorch.org/)
[![EIML@ICML 2026](https://img.shields.io/badge/EIML%40ICML-2026-purple.svg)](https://sites.google.com/view/eimlicml2026)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://www.kaggle.com/code/farjanayesmin/rulenet)

**Official implementation of:**

> Nusrat Shirmin and Farjana Yesmin.
> *When Symbolic Rules Cannot Find Their Own Triggers:
> An Epistemic Diagnosis of Predicate Grounding Collapse
> in Neuro-Symbolic Fraud Detection.*
> **Accepted, EIML@ICML 2026** — 2nd Workshop on Epistemic
> Intelligence in Machine Learning, Seoul, South Korea.

---

## This Is a Negative Result — and That Is the Point

The central finding is **predicate grounding collapse**: RuleNet's symbolic
rules fire on only **≈0.1% of synthetic instances** and **≈0.0% of real
instances**. RuleNet is **statistically indistinguishable from a plain DNN**
on both datasets:

| Dataset | RuleNet F1 | PureDNN F1 | p-value |
|---|---|---|---|
| Synthetic (n=40,000) | 0.9826 ± 0.0052 | 0.9835 ± 0.0052 | 0.068 (n.s.) |
| ULB Credit Card Fraud (n=284,807) | 0.5939 ± 0.0564 | 0.5868 ± 0.0566 | 0.752 (n.s.) |

This is an honest, rigorous negative result that cautions the neuro-symbolic
community about predicate activation conditions on standardised continuous
tabular data.

---

## The Epistemic Blind Spot

The model **cannot signal whether a prediction is grounded in symbolic domain
rules or opaque pattern matching** — the failure mode neuro-symbolic
integration is supposed to prevent.

We identify three interacting structural causes:

**1. Product t-norm collapse**
With standardised features, sigmoid outputs ≈ 0.5.
A two-body conjunction: 0.5 × 0.5 = 0.25 — well below the 0.5 firing
threshold. Rules structurally cannot fire at initialisation.

**2. Gradient dominance**
A DNN with AUROC ≈ 0.9998 produces large classification gradients that
dwarf the semantic loss. When the DNN already satisfies the rule head,
the semantic gradient vanishes. The λ grid (10⁻⁴ to 10) produces flat
validation F1 — direct evidence the rule loss has no effect.

**3. No initialisation prior**
Random N(0, 0.1) initialisation gives predicates no pull toward their
correct activation region. `highAmount` has no prior that large
transactions should score near 1.

---

## Key Results

### Main Comparison — Synthetic Benchmark (5 seeds, mean ± std)

| Model | Accuracy | F1 | ROC-AUC | Fire Rate |
|---|---|---|---|---|
| **RuleNet (ours)** | 0.9965 ± 0.0010 | 0.9826 ± 0.0052 | 0.9998 ± 0.0001 | **≈0.1%** |
| PureDNN | 0.9967 ± 0.0010 | 0.9835 ± 0.0052 | 0.9998 ± 0.0001 | — |
| LTN (Badreddine et al., 2022) | 0.9963 ± 0.0011 | 0.9815 ± 0.0055 | 0.9998 ± 0.0001 | — |
| RuleFit (Friedman & Popescu, 2008) | 0.9688 ± 0.0020 | 0.8592 ± 0.0090 | 0.9937 ± 0.0021 | — |
| Decision Tree | 0.9899 ± 0.0024 | 0.9509 ± 0.0111 | 0.9832 ± 0.0047 | — |

RuleNet vs PureDNN: t = −1.86, **p = 0.068 — not significant**.

### Critical Nuance: When Rules DO Fire, They Are Correct

| Metric | Value |
|---|---|
| Rule firing rate (Fire%) | ≈0.1% |
| Align-T: alignment when fired | **80.0%** |
| Align-M: marginal head quality | 15.6% |

When a rule fires, it agrees with the ground truth 80% of the time.
The grounding mechanism has real signal — the conjunction threshold
is simply almost never cleared.

### Ablation — Rules Are Inert

| Configuration | F1 | Fire% |
|---|---|---|
| Pure DNN (λ=0) | 0.9809 | 0.0% |
| RuleNet (λ=0.5) | 0.9801 | ≈0.1% |
| Rule-dominant (λ=10) | 0.9801 | ≈0.1% |
| Correct rules | 0.9776 | ≈0.1% |
| **Irrelevant rules** | **0.9825** | **≈0.1%** |

Wrong rules slightly outperform correct ones (+0.0049 F1) — rule
**content** is irrelevant because rules almost never fire.

### Rule Count — No Effect

| n rules | F1 | Fire% |
|---|---|---|
| 0 | 0.9809 | 0.0% |
| 1 | 0.9818 | 0.0% |
| 3 | 0.9801 | 0.0% |
| 5 | 0.9816 | 0.0% |
| 10 | 0.9825 | 0.0% |

### Weak Backbone — Collapse Is Structural

| Model | F1 | Fire% |
|---|---|---|
| WeakDNN (1 layer, 16 units) | 0.9745 | — |
| WeakRuleNet (random init) | 0.9745 | 0.0% |
| WeakRuleNet (calibrated) | 0.9745 | 0.0% |

Collapse persists with a 16-unit backbone. It is not caused by an
overly strong classifier — it is structural.

### Remedy Validation

| Configuration | F1 | Fire% | Align-T | Align-M |
|---|---|---|---|---|
| Baseline (product, random) | 0.9825 | 0.1% | 80.0% | 15.6% |
| Remedy A: calibrated + product | 0.9759 | **0.4%** | **90.0%** | 22.3% |
| Remedy B: calibrated + Łukasiewicz | 0.9809 | 0.0% | 0.0% | **68.9%** |

Calibrated initialisation raises firing **4×** with 90% alignment quality.
Łukasiewicz t-norm improves marginal head quality but imposes a stricter
threshold (requires each body predicate > 0.75 to clear 0.5 conjunction).

---

## The Three Symbolic Fraud Rules

```
L1: highAmount  ∧ lowBalance   → isFraud
L2: highVelocity ∧ oddHour    → isFraud
L3: highRisk    ∧ crossBorder  → isFraud
```

Each predicate is grounded as an independent linear sigmoid module that
observes only its semantically relevant features.

---

## Three Proposed Remedies

**Remedy A — Calibrated initialisation (empirically validated):**
Set predicate biases so that σ(w·x̄_fraud + b) = 0.5, giving each
predicate a correct starting point in feature space.

```python
# After fitting the scaler, compute the fraud-class mean
fraud_mean = X_train_scaled[y_train == 1, feat_idx].mean(axis=0)
# Set weight positive, bias to centre activation at fraud mean
nn.init.constant_(predicate.linear.weight, 0.5)
nn.init.constant_(predicate.linear.bias,
                  -0.5 * float(np.mean(fraud_mean)))
```

**Remedy B — Geometric-mean t-norm (preferred over Łukasiewicz):**
Replace the product with the geometric mean to preserve the 0.5 threshold
at initialisation:

```python
# Product t-norm (current — collapses to 0.25 at init)
c = body_vals.prod(dim=1, keepdim=True)

# Geometric-mean t-norm (recommended — stays at 0.5 at init)
n_body = body_vals.shape[1]
c = body_vals.prod(dim=1, keepdim=True).pow(1.0 / n_body)
```

**Remedy C — Firing-rate regularisation (proposed, not yet validated):**
Add an auxiliary loss that encourages body conjunctions toward a
target firing rate:

```python
def firing_rate_loss(rules, x, r_star=0.3):
    conj = torch.cat([r.conjunction(x) for r in rules], 1)
    return torch.clamp(r_star - conj.mean(), min=0.0)
```

---

## Datasets

### Synthetic Fraud Benchmark

- n = 40,000 transactions, 10% fraud
- 8 features: `amount`, `balance_org`, `balance_dest`, `hour`,
  `merchant_risk`, `velocity`, `cross_border`, `high_risk_country`
- Overlapping class distributions; calibrated Gaussian noise (σ=0.6)
- All features standardised (zero mean, unit variance) before model input
- Fully reproducible — generated in the notebook

### ULB Credit Card Fraud (Real Dataset)

> **Dataset:** Transactions by European cardholders, September 2013.
> 284,807 transactions, 492 frauds (0.172%).
> 28 PCA-transformed features (V1–V28) + Time + Amount.
> Due to confidentiality, original features are not available.

**Source:** [Kaggle — mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Please cite:**
> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca
> Bontempi. *Calibrating Probability with Undersampling for Unbalanced
> Classification.* IEEE Symposium on Computational Intelligence and Data
> Mining (CIDM), 2015.

The dataset was collected through a research collaboration between
**Worldline** and the **Machine Learning Group** at Université Libre de
Bruxelles (ULB). See also the
[Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/)
for additional methodology and a transaction simulator.

We sub-sample 60,000 transactions (stratified) and select 8 features:
`Amount`, `V4`, `V11`, `V12`, `V14`, `V17`, `Time_hour` (derived), `V10`.
These PCA components are strongly fraud-associated per the original paper.

---

## Epistemic Transparency Metrics

Two new metrics introduced in this paper for diagnosing symbolic integration:

**Rule firing rate (Fire%):**
Fraction of test instances where any rule body conjunction exceeds τ=0.5.
A low rate means the model operates entirely outside its symbolic
region of competence.

**Rule alignment rate — threshold (Align-T):**
Among fired instances, the fraction where the rule head agrees with the
ground-truth label. Measures *correctness* of activated rules.

**Rule alignment rate — marginal (Align-M):**
Across *all* instances, the fraction where the head predicate output > 0.5
agrees with the label. Measures head predicate quality regardless of whether
the conjunction threshold is cleared.

> **We call on the neuro-symbolic community to report rule firing rates
> alongside accuracy. Symbolic interpretability cannot be assumed from the
> presence of a semantic loss alone.**

---

## Quick Start

```bash
pip install torch scikit-learn pandas numpy matplotlib scipy
```

Open and run the Kaggle notebook:
[kaggle.com/code/farjanayesmin/rulenet](https://www.kaggle.com/code/farjanayesmin/rulenet)

Or clone and run locally:

```bash
git clone https://github.com/Farjana-Yesmin/When-Symbolic-Rules-Cannot-Find-Their-Own-Triggers
cd When-Symbolic-Rules-Cannot-Find-Their-Own-Triggers
jupyter notebook rulenet.ipynb
```

**To add the ULB dataset on Kaggle:**
Notebook → Data → Add Data → search `creditcardfraud` (mlg-ulb)
It will appear at `/kaggle/input/creditcardfraud/creditcard.csv`.

**Key hyperparameters:**
```python
config = {
    'lambda': 0.5,       # semantic loss weight (best of 8-value grid)
    'hidden_dim': 64,    # DNN layer sizes: d→64→32→1
    'lr': 1e-3,
    'threshold': 0.5,    # rule firing threshold τ
    'n_seeds': 5,        # seeds: {42, 123, 456, 789, 1011}
    'patience': 12,      # early stopping on validation F1
    'batch_size': 256,
}
```

---

## Repository Structure

```
.
├── rulenet.ipynb               # Full experiment notebook
├── README.md                   # This file
├── main.tex                    # Camera-ready paper (EIML 2026)
├── references.bib              # Bibliography
└── LICENSE                     # Apache 2.0
```

---

## Citation

```bibtex
@inproceedings{shirmin2026symbolic,
  title     = {When Symbolic Rules Cannot Find Their Own Triggers:
               An Epistemic Diagnosis of Predicate Grounding Collapse
               in Neuro-Symbolic Fraud Detection},
  author    = {Shirmin, Nusrat and Yesmin, Farjana},
  booktitle = {2nd Workshop on Epistemic Intelligence in Machine
               Learning (EIML@ICML 2026)},
  year      = {2026},
  note      = {Seoul, South Korea}
}
```

---

**Authors:** Nusrat Shirmin · Farjana Yesmin | Apache 2.0 License
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
@inproceedings{yesmin et al 2026 When Symbolic Rules Cannot Find Their Own Triggers,
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
