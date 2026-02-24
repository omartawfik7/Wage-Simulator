# Wage Policy Impact Simulation Engine

A full-stack causal inference tool for analyzing wage policy impacts on municipal workforces.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python · Flask |
| Causal Inference | statsmodels · linearmodels |
| Simulation | scipy · numpy |
| Data | pandas |
| Frontend | HTML · CSS · JavaScript · Chart.js |

## Statistical Methods

- **Difference-in-Differences (TWFE)** — Two-way fixed effects panel regression with clustered standard errors
- **Synthetic Control** — Constrained optimization to find donor pool weights minimizing pre-treatment MSE; inference via placebo permutation
- **Propensity Score Matching** — Logistic regression PS estimation, nearest-neighbor matching, bootstrap SEs (200 reps)
- **Ensemble** — Inverse-variance weighted combination of all three estimators
- **Event Study** — Relative-time regression with pre-trend diagnostics
- **Monte Carlo** — 1,000–5,000 parameter draws to quantify fiscal uncertainty (P5/P25/P50/P75/P95)

## Setup

```bash
# 1. Clone / download project
cd wage_simulator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python app.py

# 4. Open browser
# http://localhost:5000
```

## Project Structure

```
wage_simulator/
├── app.py              # Flask backend — all causal inference logic
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Frontend dashboard
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/api/simulate` | POST | Run full simulation |
| `/api/health` | GET | Server health check |

### POST /api/simulate

```json
{
  "method": "did",
  "wage": 18.0,
  "phase": 3,
  "elast": -0.20,
  "mc": 1000,
  "spill": 1.15,
  "scenarios": {
    "floor": true,
    "phased": false,
    "spillover": false,
    "exempt": false
  }
}
```
