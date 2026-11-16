"""Run the out-of-sample VaR backtest and write results/.

  results/metrics.json      everything the README reports
  results/var_forecasts.csv daily portfolio return and the 99% VaR of every model
  results/figures/*.png     VaR against returns, rolling exception counts
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src import metrics, walkforward as wf  # noqa: E402

WEIGHTS = {"SPY": 0.35, "QQQ": 0.30, "EFA": 0.15, "TLT": 0.10}   # renormalised to sum to one
PERIODS = [("2001-2006", "2001", "2006"), ("2007-2009", "2007", "2009"), ("2010-2019", "2010", "2019"),
           ("2020", "2020", "2020"), ("2021-2026", "2021", "2026")]
STRESS = ("2008-01-02", "2009-06-30")


def rounded(x):
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(f"{float(x):.6g}")
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def component_var(R: pd.DataFrame, w: pd.Series, z: float) -> dict:
    """Delta-normal component VaR shares for one covariance matrix."""
    cov = R.cov().to_numpy()
    sigma = float(np.sqrt(w.to_numpy() @ cov @ w.to_numpy()))
    comp = w.to_numpy() * (cov @ w.to_numpy()) / sigma * z
    return {"var": float(comp.sum()), "share": {a: float(c / comp.sum()) for a, c in zip(R.columns, comp)}}


def figures(r: pd.Series, var: pd.DataFrame, hits: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
    colors = {"HS": "#8a8980", "GARCH-N": "#eb6834", "GJR-EVT": "#2a78d6"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "xtick.color": muted, "ytick.color": muted,
                         "text.color": ink, "figure.facecolor": surface, "axes.facecolor": surface,
                         "savefig.facecolor": surface, "axes.spines.top": False, "axes.spines.right": False})
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    roll = hits[list(colors)].astype(float).rolling(250).sum()
    ax.axhspan(0, 4.5, color="#1baf7a", alpha=0.07, linewidth=0)
    ax.axhspan(4.5, 9.5, color="#e0a800", alpha=0.07, linewidth=0)
    ax.axhspan(9.5, max(12, roll.max().max() + 1), color="#d64545", alpha=0.07, linewidth=0)
    for n, c in colors.items():
        ax.plot(roll.index, roll[n], color=c, linewidth=1.4, label=n)
    ax.set_xlim(roll.index[250], roll.index[-1])
    ax.set_ylim(0, max(12, roll.max().max() + 1))
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_title("99% VaR exceptions in the last 250 days; bands are the Basel green, yellow and red zones",
                 loc="left", fontsize=10.5)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "rolling_exceptions.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = wf.Config()
    R = pd.read_csv(ROOT / "data" / "snapshot" / "asset_returns.csv", index_col="date", parse_dates=True)
    w = pd.Series(WEIGHTS)
    w = w / w.sum()
    r = (R[w.index] * w).sum(axis=1)
    var, es = wf.walk_forward(r, cfg, progress=True)
    ret = r.reindex(var.index)
    hits = var.lt(-ret, axis=0)
    n = len(ret)

    models = {}
    for m in wf.MODELS:
        c = metrics.christoffersen_test(ret, var[m], cfg.alpha)
        row = {"violations": int(c["n_violations"]), "rate": float(c["violation_rate"]), "p_pof": float(c["p_POF"]),
               "p_ind": float(c["p_IND"]), "p_cc": float(c["p_CC"]),
               "status": metrics.backtest_status(c["p_POF"], c["p_IND"], c["p_CC"]),
               "tick_loss": float(metrics._tick_loss(ret.to_numpy(), var[m].to_numpy(), cfg.alpha).mean()),
               "mean_var": float(var[m].mean()), "zones": wf.basel_zones(hits[m])}
        if m in wf.ES_MODELS:
            e = metrics.mcneil_frey_es_test(ret, var[m], es[m])
            row["es_t"], row["es_p"] = float(e["t_stat"]), float(e["p_val"])
        models[m] = row
    loss = pd.DataFrame({m: metrics._tick_loss(ret.to_numpy(), var[m].to_numpy(), cfg.alpha) for m in wf.MODELS},
                        index=ret.index)
    mcs = metrics.model_confidence_set(loss, alpha=0.10, B=1000)
    for m in wf.MODELS:
        models[m]["in_mcs"] = bool(str(mcs.loc[m, "in_MCS"]).strip() != "")

    periods = {lab: {"days": int(len(ret.loc[a:b])),
                     "rate": {m: float(hits[m].loc[a:b].mean()) for m in wf.MODELS}} for lab, a, b in PERIODS}

    # the in-sample shortcut, on the same days
    once = wf.fit_once(r, cfg).reindex(var.index)
    shortcut = {}
    for m in once.columns:
        c = metrics.christoffersen_test(ret, once[m], cfg.alpha)
        shortcut[m] = {"violations": int(c["n_violations"]), "rate": float(c["violation_rate"]),
                       "p_pof": float(c["p_POF"]), "p_cc": float(c["p_CC"]),
                       "status": metrics.backtest_status(c["p_POF"], c["p_IND"], c["p_CC"])}

    z = -wf.var_es_multipliers(cfg.alpha, "normal")[0]
    Rb = R[w.index].loc[cfg.first_day:]
    comp = {"full": component_var(Rb, w, z), "stress": component_var(Rb.loc[STRESS[0]:STRESS[1]], w, z),
            "stress_window": list(STRESS), "stress_days": int(len(Rb.loc[STRESS[0]:STRESS[1]]))}

    x = ret.to_numpy()
    payload = rounded({
        "sample": {"data_first": str(r.index[0].date()), "first_day": str(ret.index[0].date()),
                   "last_day": str(ret.index[-1].date()), "n_days": n, "expected_violations": n * cfg.alpha,
                   "refit_days": cfg.refit, "n_refits": int(np.ceil(n / cfg.refit)),
                   "training_days_first": int(r.index.searchsorted(ret.index[0])), "alpha": cfg.alpha,
                   "ann_vol": float(x.std() * np.sqrt(252)), "skew": float(pd.Series(x).skew()),
                   "excess_kurtosis": float(pd.Series(x).kurt())},
        "weights": {k: float(v) for k, v in w.items()},
        "models": models, "periods": periods, "in_sample_shortcut": shortcut, "component_var": comp,
    })
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    pd.concat([ret.rename("portfolio_return"), var], axis=1).round(6).to_csv(out / "var_forecasts.csv",
                                                                           index_label="date")
    figures(ret, var, hits, out / "figures")
    t = pd.DataFrame(models).T[["violations", "rate", "p_pof", "p_ind", "p_cc", "status", "in_mcs", "es_t"]]
    print(t.to_string())
    print(pd.DataFrame(shortcut).T.to_string())
    print(pd.DataFrame({k: v["rate"] for k, v in periods.items()}).round(4).to_string())
    print({m: models[m]["zones"] for m in ("HS", "GARCH-N", "GJR-EVT")})


if __name__ == "__main__":
    main()
