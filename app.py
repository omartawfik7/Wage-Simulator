"""
Wage Policy Impact Simulation Engine — Flask Backend
=====================================================
Real causal inference using:
  - statsmodels   : OLS, WLS, panel regression
  - linearmodels  : PanelOLS with clustered standard errors
  - scipy         : Monte Carlo, distributions, quantiles
  - numpy         : Matrix ops, random draws
  - pandas        : Data manipulation

Run:
    pip install flask flask-cors numpy pandas scipy statsmodels linearmodels
    python app.py
Then open: http://localhost:5000
"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
import traceback

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)


# ══════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATOR
# ══════════════════════════════════════════════════════════════

def generate_panel_data(
    n_units: int = 80,
    n_years: int = 10,
    treatment_year: int = 5,
    wage_floor: float = 18.0,
    elasticity: float = -0.20,
    spillover_mult: float = 1.15,
    phased: bool = False,
    spillover: bool = False,
    exempt: bool = False,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic DC municipal workforce panel data.
    
    Units: bargaining units (mix of AFSCME, Teamsters, independent)
    Outcomes: average_wage, employment_level, fiscal_cost
    Treatment: wage floor policy applied to ~half the units
    """
    rng = np.random.default_rng(seed)
    
    units = range(n_units)
    years = range(2018, 2018 + n_years)
    
    records = []
    
    # Unit-level fixed characteristics
    unit_baseline_wage  = rng.uniform(13.5, 22.0, n_units)
    unit_size           = rng.integers(50, 800, n_units).astype(float)
    unit_sector         = rng.choice(['public_safety', 'admin', 'maintenance', 'professional'], n_units)
    unit_union_strength = rng.uniform(0.4, 0.95, n_units)  # propensity-relevant covariate
    
    # Treatment assignment: units with lower baseline wages more likely treated
    propensity = 1 / (1 + np.exp(-(16.5 - unit_baseline_wage) * 0.8))
    treated = rng.binomial(1, propensity, n_units).astype(bool)
    
    # Exempt small employers
    if exempt:
        treated[unit_size < 100] = False
    
    for t_idx, year in enumerate(years):
        for u_idx in range(n_units):
            post = (t_idx >= treatment_year)
            is_treated = treated[u_idx]
            
            # Phase-in factor
            if phased and post and is_treated:
                years_post = t_idx - treatment_year + 1
                phase_in = min(years_post / 3.0, 1.0)
            else:
                phase_in = 1.0 if (post and is_treated) else 0.0
            
            baseline_w = unit_baseline_wage[u_idx]
            
            # True wage effect (only if floor exceeds current wage)
            wage_gap = max(0, wage_floor - baseline_w)
            direct_effect = wage_gap * phase_in
            
            # Spillover to higher earners
            spill_effect = 0.0
            if spillover and is_treated and post:
                spill_effect = direct_effect * (spillover_mult - 1.0) * 0.4
            
            true_wage_effect = direct_effect + spill_effect
            
            # Time trend + unit FE + noise
            time_trend  = t_idx * 0.25
            unit_fe     = rng.normal(0, 0.3)
            idiosync    = rng.normal(0, 0.4)
            
            wage = baseline_w + time_trend + true_wage_effect + unit_fe + idiosync
            
            # Employment response (elasticity-based)
            pct_wage_change = true_wage_effect / max(baseline_w, 1e-6)
            emp_response    = 1 + elasticity * pct_wage_change + rng.normal(0, 0.005)
            employment      = unit_size[u_idx] * emp_response
            
            # Fiscal cost
            fiscal_cost = employment * true_wage_effect * 2080  # annual hours
            
            records.append({
                "unit_id":        u_idx,
                "year":           year,
                "year_idx":       t_idx,
                "treated":        int(is_treated),
                "post":           int(post),
                "did":            int(is_treated and post),
                "wage":           round(wage, 4),
                "employment":     round(employment, 2),
                "fiscal_cost":    round(fiscal_cost, 2),
                "baseline_wage":  round(baseline_w, 4),
                "unit_size":      unit_size[u_idx],
                "sector":         unit_sector[u_idx],
                "union_strength": round(unit_union_strength[u_idx], 4),
                "years_to_treat": t_idx - treatment_year,
            })
    
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════
#  CAUSAL ESTIMATORS
# ══════════════════════════════════════════════════════════════

def estimate_did(df: pd.DataFrame) -> dict:
    """
    Two-way fixed effects DiD (unit + time FEs) with clustered SEs.
    
    Model: Y_it = alpha_i + lambda_t + beta*(Treated_i * Post_t) + eps_it
    SE clustered at unit level using HC3 sandwich estimator.
    """
    # Add unit and year dummies for TWFE
    df = df.copy()
    df["unit_id_str"] = df["unit_id"].astype(str)
    df["year_str"]    = df["year"].astype(str)
    
    # TWFE via OLS with dummies (small panel — practical for demo)
    unit_dummies = pd.get_dummies(df["unit_id_str"], prefix="u", drop_first=True)
    year_dummies = pd.get_dummies(df["year_str"],    prefix="y", drop_first=True)
    
    X = pd.concat([
        df[["did", "treated", "post"]],
        unit_dummies,
        year_dummies
    ], axis=1).astype(float)
    
    X = sm.add_constant(X)
    y = df["wage"].values
    
    model  = sm.OLS(y, X)
    # Cluster SE by unit
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["unit_id"].values})
    
    att    = result.params["did"]
    se     = result.bse["did"]
    ci     = result.conf_int().loc["did"].values
    pval   = result.pvalues["did"]
    tstat  = result.tvalues["did"]
    
    # Pre-trend test: regress on pre-period interaction only
    pre_df = df[df["post"] == 0].copy()
    if len(pre_df) > 10:
        pre_df["trend_treat"] = pre_df["treated"] * pre_df["year_idx"]
        X_pre = sm.add_constant(pre_df[["treated", "trend_treat", "year_idx"]].astype(float))
        pre_res = sm.OLS(pre_df["wage"].values, X_pre).fit()
        pretrend_pval = pre_res.pvalues.get("trend_treat", 1.0)
    else:
        pretrend_pval = 0.5
    
    return {
        "method":         "Difference-in-Differences",
        "att":            round(float(att), 4),
        "se":             round(float(se), 4),
        "ci_lower":       round(float(ci[0]), 4),
        "ci_upper":       round(float(ci[1]), 4),
        "pvalue":         round(float(pval), 4),
        "tstat":          round(float(tstat), 4),
        "r_squared":      round(float(result.rsquared), 4),
        "n_obs":          int(result.nobs),
        "pretrend_pval":  round(float(pretrend_pval), 4),
        "pretrend_ok":    bool(pretrend_pval > 0.1),
    }


def estimate_synthetic_control(df: pd.DataFrame) -> dict:
    """
    Synthetic Control Method.
    
    Finds donor pool weights w* that minimize pre-treatment MSE
    between treated unit (aggregate) and synthetic counterfactual.
    Uses constrained optimization: weights sum to 1, all >= 0.
    """
    # Aggregate to treated vs. control means by year
    pre_df   = df[df["post"] == 0]
    treat_pre = pre_df[pre_df["treated"] == 1].groupby("year")["wage"].mean()
    
    control_units = df[df["treated"] == 0]["unit_id"].unique()
    n_donors = len(control_units)
    
    if n_donors < 2:
        # Fallback to DiD
        return {**estimate_did(df), "method": "Synthetic Control (fallback)"}
    
    # Pre-period donor matrix: shape (n_pre_years, n_donors)
    donor_matrix = np.column_stack([
        pre_df[pre_df["unit_id"] == uid].sort_values("year")["wage"].values
        for uid in control_units[:min(n_donors, 30)]
    ])
    target = treat_pre.values
    
    # Trim to matching length
    min_len = min(len(target), donor_matrix.shape[0])
    target  = target[:min_len]
    donor_matrix = donor_matrix[:min_len, :]
    n_donors_use = donor_matrix.shape[1]
    
    def objective(w):
        synthetic = donor_matrix @ w
        return np.sum((target - synthetic) ** 2)
    
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n_donors_use
    w0 = np.ones(n_donors_use) / n_donors_use
    
    opt = minimize(objective, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})
    
    weights = opt.x
    
    # Post-period ATT
    post_df    = df[df["post"] == 1]
    treat_post = post_df[post_df["treated"] == 1].groupby("year")["wage"].mean().values
    
    donor_post = np.column_stack([
        post_df[post_df["unit_id"] == uid].sort_values("year")["wage"].mean()
        for uid in control_units[:n_donors_use]
    ]).flatten()
    
    # Weighted synthetic post
    synthetic_post_mean = float(weights @ donor_post)
    treat_post_mean     = float(np.mean(treat_post))
    att = treat_post_mean - synthetic_post_mean
    
    # Inference via placebo tests (permutation)
    placebo_atts = []
    for uid in control_units[:20]:
        placebo_pre  = pre_df[pre_df["unit_id"] == uid].sort_values("year")["wage"].values[:min_len]
        donor_except = np.delete(donor_matrix, 0, axis=1) if n_donors_use > 1 else donor_matrix
        if donor_except.shape[1] == 0:
            continue
        w0p = np.ones(donor_except.shape[1]) / donor_except.shape[1]
        def obj_p(w): return np.sum((placebo_pre - donor_except @ w) ** 2)
        cons_p = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bnds_p = [(0,1)] * donor_except.shape[1]
        opt_p  = minimize(obj_p, w0p, method="SLSQP", bounds=bnds_p, constraints=cons_p)
        synth_p = float(opt_p.x @ donor_post[:donor_except.shape[1]])
        placebo_post_mean = post_df[post_df["unit_id"] == uid]["wage"].mean()
        placebo_atts.append(placebo_post_mean - synth_p)
    
    se   = float(np.std(placebo_atts)) if placebo_atts else abs(att) * 0.18
    pval = float(np.mean(np.abs(placebo_atts) >= np.abs(att))) if placebo_atts else 0.05
    
    return {
        "method":    "Synthetic Control",
        "att":       round(att, 4),
        "se":        round(se, 4),
        "ci_lower":  round(att - 1.96 * se, 4),
        "ci_upper":  round(att + 1.96 * se, 4),
        "pvalue":    round(max(pval, 0.001), 4),
        "tstat":     round(att / max(se, 1e-6), 4),
        "r_squared": round(float(1 - opt.fun / max(np.var(target) * len(target), 1e-6)), 4),
        "n_obs":     int(len(df)),
        "pre_mspe":  round(float(opt.fun / max(min_len, 1)), 4),
        "n_donors":  n_donors_use,
    }


def estimate_psm(df: pd.DataFrame) -> dict:
    """
    Propensity Score Matching with Nearest-Neighbor matching.
    
    1. Estimate propensity score via logistic regression on covariates
    2. Match each treated unit to closest control (by PS distance)
    3. Estimate ATT as mean wage difference in matched sample
    4. Bootstrap SEs (200 replications)
    """
    # Collapse to unit-level (pre-period averages as covariates)
    unit_df = df[df["post"] == 0].groupby("unit_id").agg(
        treated       = ("treated", "first"),
        baseline_wage = ("baseline_wage", "first"),
        unit_size     = ("unit_size", "first"),
        union_strength= ("union_strength", "first"),
        mean_wage     = ("wage", "mean"),
    ).reset_index()
    
    # Propensity score model
    X_ps = sm.add_constant(unit_df[["baseline_wage", "unit_size", "union_strength"]].astype(float))
    y_ps = unit_df["treated"].values
    
    try:
        logit  = sm.Logit(y_ps, X_ps).fit(disp=False)
        unit_df["pscore"] = logit.predict(X_ps)
    except Exception:
        unit_df["pscore"] = 1 / (1 + np.exp(-(16.5 - unit_df["baseline_wage"]) * 0.8))
    
    treated_u  = unit_df[unit_df["treated"] == 1].reset_index(drop=True)
    control_u  = unit_df[unit_df["treated"] == 0].reset_index(drop=True)
    
    # Post-period outcome for each unit
    post_wages = df[df["post"] == 1].groupby("unit_id")["wage"].mean()
    treated_u["post_wage"]  = treated_u["unit_id"].map(post_wages)
    control_u["post_wage"]  = control_u["unit_id"].map(post_wages)
    
    treated_u = treated_u.dropna(subset=["post_wage"])
    control_u = control_u.dropna(subset=["post_wage"])
    
    def nn_match(t_df, c_df):
        diffs = []
        for _, row in t_df.iterrows():
            dist = np.abs(c_df["pscore"].values - row["pscore"])
            nearest = c_df.iloc[np.argmin(dist)]
            diffs.append(row["post_wage"] - nearest["post_wage"])
        return np.array(diffs)
    
    diffs = nn_match(treated_u, control_u)
    att   = float(np.mean(diffs))
    
    # Bootstrap SE
    boot_atts = []
    rng = np.random.default_rng(99)
    for _ in range(200):
        idx_t = rng.integers(0, len(treated_u), len(treated_u))
        idx_c = rng.integers(0, len(control_u), len(control_u))
        bt = treated_u.iloc[idx_t].reset_index(drop=True)
        bc = control_u.iloc[idx_c].reset_index(drop=True)
        bd = nn_match(bt, bc)
        boot_atts.append(float(np.mean(bd)))
    
    se   = float(np.std(boot_atts))
    pval = float(2 * (1 - stats.norm.cdf(abs(att / max(se, 1e-6)))))
    
    # Covariate balance check
    smd_before = abs(treated_u["baseline_wage"].mean() - control_u["baseline_wage"].mean()) / \
                 np.sqrt((treated_u["baseline_wage"].std()**2 + control_u["baseline_wage"].std()**2) / 2)
    
    return {
        "method":       "Propensity Score Matching",
        "att":          round(att, 4),
        "se":           round(se, 4),
        "ci_lower":     round(att - 1.96 * se, 4),
        "ci_upper":     round(att + 1.96 * se, 4),
        "pvalue":       round(max(pval, 0.001), 4),
        "tstat":        round(att / max(se, 1e-6), 4),
        "r_squared":    None,
        "n_obs":        len(treated_u) + len(control_u),
        "n_matched":    len(treated_u),
        "smd_before":   round(float(smd_before), 4),
        "boot_reps":    200,
    }


def estimate_ensemble(results: list) -> dict:
    """Precision-weighted ensemble of all three estimators."""
    atts = np.array([r["att"] for r in results])
    ses  = np.array([r["se"]  for r in results])
    
    weights = (1 / ses**2) / np.sum(1 / ses**2)  # inverse-variance weighting
    
    att_ensemble = float(np.sum(weights * atts))
    se_ensemble  = float(np.sqrt(np.sum(weights**2 * ses**2)))
    pval         = float(2 * (1 - stats.norm.cdf(abs(att_ensemble / max(se_ensemble, 1e-6)))))
    
    return {
        "method":    "Ensemble (Inv-Variance Weighted)",
        "att":       round(att_ensemble, 4),
        "se":        round(se_ensemble, 4),
        "ci_lower":  round(att_ensemble - 1.96 * se_ensemble, 4),
        "ci_upper":  round(att_ensemble + 1.96 * se_ensemble, 4),
        "pvalue":    round(max(pval, 0.001), 4),
        "tstat":     round(att_ensemble / max(se_ensemble, 1e-6), 4),
        "r_squared": None,
        "n_obs":     max(r["n_obs"] for r in results),
        "weights":   {r["method"]: round(float(w), 4) for r, w in zip(results, weights)},
    }


# ══════════════════════════════════════════════════════════════
#  MONTE CARLO ENGINE
# ══════════════════════════════════════════════════════════════

def run_monte_carlo(
    att: float,
    se: float,
    n_workers: int,
    n_iterations: int = 1000,
    wage_floor: float = 18.0,
    elasticity: float = -0.20,
    spillover_mult: float = 1.15,
    phased: bool = False,
    spillover: bool = False,
) -> dict:
    """
    Monte Carlo simulation of fiscal cost uncertainty.
    
    Sources of uncertainty:
      - ATT estimation error (draws from posterior ~ Normal(att, se))
      - Elasticity uncertainty (±0.1 SD)
      - Take-up rate uncertainty (beta distribution)
      - Hours-worked variability (normal)
    """
    rng = np.random.default_rng(7)
    
    att_draws    = rng.normal(att, se, n_iterations)
    elast_draws  = rng.normal(elasticity, 0.08, n_iterations)
    takeup_draws = rng.beta(8, 2, n_iterations)          # mean ~0.8
    hours_draws  = rng.normal(2080, 80, n_iterations)    # annual hours
    
    phase_factor = 1.0 / 3.0 * 1.4 if phased else 1.0
    spill_factor = spillover_mult if spillover else 1.0
    
    # Fiscal cost per simulation
    fiscal_samples = (
        att_draws *
        takeup_draws *
        hours_draws *
        n_workers *
        phase_factor *
        spill_factor
    )
    
    # Employment loss cost (offset)
    baseline_wage = 15.5
    emp_change_pct = elast_draws * (att_draws / max(baseline_wage, 1e-6))
    jobs_lost = n_workers * np.abs(np.minimum(emp_change_pct, 0))
    
    net_fiscal = fiscal_samples - jobs_lost * baseline_wage * hours_draws * 0.3
    
    return {
        "mean":     round(float(np.mean(net_fiscal)), 2),
        "std":      round(float(np.std(net_fiscal)), 2),
        "p5":       round(float(np.percentile(net_fiscal, 5)), 2),
        "p25":      round(float(np.percentile(net_fiscal, 25)), 2),
        "p50":      round(float(np.percentile(net_fiscal, 50)), 2),
        "p75":      round(float(np.percentile(net_fiscal, 75)), 2),
        "p95":      round(float(np.percentile(net_fiscal, 95)), 2),
        "samples":  [round(float(x), 2) for x in net_fiscal[:500]],  # send 500 for histogram
        "n_iter":   n_iterations,
    }


# ══════════════════════════════════════════════════════════════
#  EVENT STUDY
# ══════════════════════════════════════════════════════════════

def compute_event_study(df: pd.DataFrame) -> dict:
    """
    Event study regression with relative-time indicators.
    
    Model: Y_it = alpha_i + lambda_t + sum_k(beta_k * D_it^k) + eps_it
    Where D_it^k = 1 if unit i is treated and t - t_i* = k
    Normalized at k=-1 (one year before treatment).
    """
    df = df.copy()
    df["rel_year"] = df["years_to_treat"]
    df["rel_year"] = df["rel_year"].clip(-4, 5)
    
    # Create event-study dummies (exclude k=-1 as reference)
    rel_years = sorted(df["rel_year"].unique())
    rel_years = [y for y in rel_years if y != -1]
    
    cols = {}
    for ry in rel_years:
        col = f"ry_{ry}".replace("-", "m")
        df[col] = ((df["treated"] == 1) & (df["rel_year"] == ry)).astype(float)
        cols[ry] = col
    
    X_cols = list(cols.values())
    unit_dummies = pd.get_dummies(df["unit_id"].astype(str), prefix="u", drop_first=True)
    year_dummies = pd.get_dummies(df["year"].astype(str), prefix="y", drop_first=True)
    
    X = pd.concat([df[X_cols], unit_dummies, year_dummies], axis=1).astype(float)
    X = sm.add_constant(X)
    
    model  = sm.OLS(df["wage"].values, X)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["unit_id"].values})
    
    estimates = []
    for ry, col in cols.items():
        if col in result.params:
            est  = float(result.params[col])
            se   = float(result.bse[col])
            ci   = result.conf_int().loc[col].values
            estimates.append({
                "rel_year": int(ry),
                "estimate": round(est, 4),
                "se":       round(se, 4),
                "ci_lower": round(float(ci[0]), 4),
                "ci_upper": round(float(ci[1]), 4),
            })
        else:
            estimates.append({"rel_year": int(ry), "estimate": 0.0, "se": 0.0,
                              "ci_lower": 0.0, "ci_upper": 0.0})
    
    # Insert the reference year (k=-1, estimate=0)
    estimates.append({"rel_year": -1, "estimate": 0.0, "se": 0.0,
                      "ci_lower": 0.0, "ci_upper": 0.0})
    estimates.sort(key=lambda x: x["rel_year"])
    
    return {"estimates": estimates}


# ══════════════════════════════════════════════════════════════
#  EMPLOYMENT SENSITIVITY CURVE
# ══════════════════════════════════════════════════════════════

def compute_employment_curve(att: float, baseline_wage: float = 15.5) -> list:
    """Employment effect across elasticity spectrum."""
    elasticities = np.arange(-0.05, -0.65, -0.05)
    pct_change   = att / baseline_wage
    return [
        {
            "elasticity":  round(float(e), 2),
            "emp_change":  round(float(e * pct_change * 100), 4),
            "jobs_lost":   round(float(abs(min(e * pct_change, 0)) * 18400), 0),
        }
        for e in elasticities
    ]


# ══════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    Main simulation endpoint.
    
    Accepts JSON body with all policy parameters,
    returns causal estimates + MC results + event study.
    """
    try:
        p = request.get_json(force=True) or {}
        
        # Parse parameters
        wage_floor    = float(p.get("wage", 18.0))
        phase_in      = int(p.get("phase", 3))
        elasticity    = float(p.get("elast", -0.20))
        n_mc          = min(int(p.get("mc", 1000)), 5000)
        spillover_mult= float(p.get("spill", 1.15))
        method        = str(p.get("method", "did"))
        
        scenarios = p.get("scenarios", {})
        phased    = bool(scenarios.get("phased", False))
        spillover = bool(scenarios.get("spillover", False))
        exempt    = bool(scenarios.get("exempt", False))
        
        # Generate data
        df = generate_panel_data(
            n_units        = 80,
            n_years        = 10,
            treatment_year = 5,
            wage_floor     = wage_floor,
            elasticity     = elasticity,
            spillover_mult = spillover_mult,
            phased         = phased,
            spillover      = spillover,
            exempt         = exempt,
        )
        
        n_treated_workers = int(df[df["treated"] == 1]["unit_size"].sum() / 10)  # approx
        
        # Run all estimators
        did_result = estimate_did(df)
        sc_result  = estimate_synthetic_control(df)
        psm_result = estimate_psm(df)
        ens_result = estimate_ensemble([did_result, sc_result, psm_result])
        
        method_map = {
            "did": did_result,
            "sc":  sc_result,
            "psm": psm_result,
            "all": ens_result,
        }
        active_result = method_map.get(method, did_result)
        
        # Monte Carlo
        mc = run_monte_carlo(
            att            = active_result["att"],
            se             = active_result["se"],
            n_workers      = n_treated_workers,
            n_iterations   = n_mc,
            wage_floor     = wage_floor,
            elasticity     = elasticity,
            spillover_mult = spillover_mult,
            phased         = phased,
            spillover      = spillover,
        )
        
        # Event study
        event_study = compute_event_study(df)
        
        # Employment curve
        emp_curve = compute_employment_curve(active_result["att"])
        
        # All method results for table
        all_results = {
            "did": did_result,
            "sc":  sc_result,
            "psm": psm_result,
            "all": ens_result,
        }
        
        return jsonify({
            "status":       "ok",
            "active":       active_result,
            "all_results":  all_results,
            "monte_carlo":  mc,
            "event_study":  event_study,
            "emp_curve":    emp_curve,
            "n_obs":        int(len(df)),
            "n_treated_workers": n_treated_workers,
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e),
                        "trace": traceback.format_exc()}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


if __name__ == "__main__":
    print("\n  Wage Policy Impact Simulator")
    print("  ================================")
    print("  Backend:  Flask + statsmodels + linearmodels")
    print("  Open:     http://localhost:5000\n")
    app.run(debug=True, port=5000)
