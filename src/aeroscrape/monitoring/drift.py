"""Input drift: are recent model inputs distributed like the training data?

For each checked feature the Population Stability Index (PSI) compares the share of rows
per bin: deciles of the training values for numbers, the values themselves for categories,
and missing values as a bin of their own (a feature that suddenly goes missing is drift
too). PSI below 0.1 is stable, 0.1-0.25 a moderate shift, above 0.25 a large one. For
numbers the Kolmogorov-Smirnov p-value is reported as well; with thousands of rows it
flags even tiny shifts, so the decision uses PSI.

This is a small, testable replacement for Evidently, which would add dozens of
dependencies to the Airflow image for the same handful of numbers.
"""

import html
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPSILON = 1e-4  # share given to empty bins, so the log ratio stays finite
MISSING = "(missing)"
MAX_CATEGORIES = 20  # rarer categories are pooled as "(other)"


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    kind: str  # numerical or categorical
    psi: float
    ks_pvalue: float | None
    reference_rows: int
    current_rows: int
    drifted: bool
    shares: pd.DataFrame  # bin, reference share, current share


def _psi(ref_share: np.ndarray, cur_share: np.ndarray) -> float:
    ref = np.clip(ref_share, EPSILON, None)
    cur = np.clip(cur_share, EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _shares(ref_bins: pd.Series, cur_bins: pd.Series) -> pd.DataFrame:
    ref = ref_bins.value_counts(normalize=True)
    cur = cur_bins.value_counts(normalize=True)
    table = pd.concat([ref.rename("reference"), cur.rename("current")], axis=1).fillna(0.0)
    return table.rename_axis("bin").reset_index()


def _numeric_bins(reference: pd.Series, values: pd.Series, bins: int) -> pd.Series:
    known = reference.dropna()
    edges = np.unique(np.quantile(known, np.linspace(0, 1, bins + 1))) if len(known) else np.array([])
    if len(edges) < 2:
        return values.astype(object).where(values.notna(), MISSING).astype(str)
    edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])
    labelled = pd.cut(values, edges, include_lowest=True).astype(str)
    return labelled.where(values.notna(), MISSING)


def _category_bins(reference: pd.Series, values: pd.Series) -> pd.Series:
    common = set(reference.dropna().astype(str).value_counts().index[:MAX_CATEGORIES])
    as_text = values.astype(str).where(values.notna(), MISSING)
    return as_text.where(as_text.isin(common | {MISSING}), "(other)")


def feature_drift(
    reference: pd.Series, current: pd.Series, numerical: bool, threshold: float, bins: int = 10
) -> FeatureDrift:
    if numerical:
        reference = pd.to_numeric(reference, errors="coerce")
        current = pd.to_numeric(current, errors="coerce")
        if reference.nunique() <= bins:  # a flag or a count: compare the values themselves
            ref_bins, cur_bins = _category_bins(reference, reference), _category_bins(reference, current)
        else:
            ref_bins, cur_bins = _numeric_bins(reference, reference, bins), _numeric_bins(reference, current, bins)
        both = reference.dropna(), current.dropna()
        ks = float(ks_2samp(*both).pvalue) if all(len(s) for s in both) else None
    else:
        ref_bins, cur_bins = _category_bins(reference, reference), _category_bins(reference, current)
        ks = None
    shares = _shares(ref_bins, cur_bins)
    psi = _psi(shares["reference"].to_numpy(), shares["current"].to_numpy())
    return FeatureDrift(
        feature=str(reference.name),
        kind="numerical" if numerical else "categorical",
        psi=psi,
        ks_pvalue=ks,
        reference_rows=len(reference),
        current_rows=len(current),
        drifted=psi >= threshold,
        shares=shares,
    )


def detect_drift(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str], numerical: set[str], threshold: float
) -> list[FeatureDrift]:
    """Drift of each feature present in both frames."""
    present = [f for f in features if f in reference.columns and f in current.columns]
    return [feature_drift(reference[f].rename(f), current[f].rename(f), f in numerical, threshold) for f in present]


def drift_html(results: list[FeatureDrift], title: str, threshold: float) -> str:
    """A self-contained HTML report: a summary table and each feature's bin shares."""
    esc = html.escape

    def bar(share: float, color: str) -> str:
        return f'<span class="bar" style="width:{share * 200:.0f}px;background:{color}"></span>'

    summary = "".join(
        f"<tr class={'drift' if r.drifted else 'ok'}><td>{esc(r.feature)}</td><td>{r.kind}</td>"
        f"<td>{r.psi:.3f}</td><td>{'' if r.ks_pvalue is None else f'{r.ks_pvalue:.3g}'}</td>"
        f"<td>{r.reference_rows:,}</td><td>{r.current_rows:,}</td>"
        f"<td>{'drift' if r.drifted else 'stable'}</td></tr>"
        for r in results
    )
    details = ""
    for r in results:
        rows = "".join(
            f"<tr><td>{esc(str(b))}</td><td>{ref:.1%} {bar(ref, '#8a8f98')}</td>"
            f"<td>{cur:.1%} {bar(cur, '#c2410c' if r.drifted else '#2563eb')}</td></tr>"
            for b, ref, cur in r.shares.itertuples(index=False)
        )
        details += (
            f"<h2>{esc(r.feature)} <small>PSI {r.psi:.3f}</small></h2>"
            f"<table><tr><th>Bin</th><th>Training</th><th>Recent</th></tr>{rows}</table>"
        )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(title)}</title><style>
body{{font:14px system-ui,sans-serif;margin:24px;color:#1f2328;background:#fff}}
table{{border-collapse:collapse;margin-bottom:16px}}td,th{{padding:4px 10px;border-bottom:1px solid #ddd;text-align:left}}
tr.drift td{{color:#c2410c;font-weight:600}}.bar{{display:inline-block;height:9px;vertical-align:middle}}
small{{color:#666;font-weight:normal}}</style></head><body>
<h1>{esc(title)}</h1><p>PSI of {threshold} or more counts as drift. Bins are deciles of the training values
for numbers and the most common values for categories; missing values are a bin of their own.</p>
<table><tr><th>Feature</th><th>Kind</th><th>PSI</th><th>KS p-value</th><th>Training rows</th>
<th>Recent rows</th><th></th></tr>{summary}</table>{details}</body></html>"""
