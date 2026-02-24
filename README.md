# Wage-Simulator
Full-stack causal inference engine for simulating wage policy impacts on municipal workforces


# Wage Policy Impact Simulation Engine

A full-stack data science application that applies causal inference methods 
to simulate the fiscal and employment effects of wage policy decisions on 
municipal workforces. Built with a Python/Flask REST API performing real 
statistical computation and an interactive web dashboard for policy exploration.

## What it does
Analysts input policy parameters (wage floor, phase-in period, sector coverage) 
and the engine estimates causal effects using three independent methods, runs 
Monte Carlo simulations to quantify fiscal uncertainty, and outputs confidence 
intervals and risk scenarios across a range of assumptions. Results are displayed 
in a live dashboard with event study charts, fiscal cost distributions, and a 
sensitivity analysis curve across labor demand elasticity scenarios.

## Methods & Technical Implementation
- **Difference-in-Differences (TWFE)** — Two-way fixed effects panel regression 
  with unit and time fixed effects; standard errors clustered at the bargaining-unit 
  level using HC sandwich estimator via statsmodels
- **Synthetic Control** — Constrained optimization (scipy.optimize) to find donor 
  pool weights minimizing pre-treatment MSE; inference via placebo permutation tests
- **Propensity Score Matching** — Logistic regression PS estimation, nearest-neighbor 
  1:1 matching, bootstrapped standard errors (200 replications)
- **Inverse-Variance Weighted Ensemble** — Precision-weighted combination of all 
  three estimators to minimize combined estimation error under model uncertainty
- **Event Study Regression** — Relative-time indicators with pre-trend diagnostics 
  to validate parallel trends assumption
- **Monte Carlo Simulation** — Up to 5,000 parameter draws using numpy random 
  sampling across ATT estimation error, elasticity uncertainty, and take-up rate 
  variability to produce P5/P25/P50/P75/P95 fiscal risk quantiles

## Stack
Python · Flask · statsmodels · linearmodels · scipy · numpy · pandas · Chart.js
