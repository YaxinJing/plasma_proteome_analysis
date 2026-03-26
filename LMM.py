#!/usr/bin/env python3
"""

Usage:
  python LMM.py \
    --data cohort1_ART_raw_long.csv \
    --timepoints base D1 D2 D3 D4 \
    --outdir results_lmm_filtered \
    --contaminants crap_contaminants.tsv \
    --detection_patterns detection_results/detection_patterns.csv \
    --stable_only \
    --min_intensity 0 \
    --impute half_min_protein \
    --normalize quantile

  # manually provide a protein whitelist
  python LMM.py \
    --data cohort1_protein.csv \
    --timepoints base D1 D2 D3 D4 \
    --outdir results_lmm \
    --protein_whitelist stable_protein_ids.txt \
    --min_intensity 0 \
    --impute half_min_protein \
    --normalize quantile
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import mixedlm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# PROTEIN FILTERING (contaminants, detection patterns, whitelist)
# ============================================================================

def load_contaminants(contam_path: str) -> set:
    """Load contaminant protein IDs from TSV/CSV."""
    sep = '\t' if contam_path.endswith('.tsv') else ','
    try:
        df = pd.read_csv(contam_path, sep=sep)
    except Exception as e:
        log.warning(f"  Initial parse failed ({type(e).__name__}: {e}), retrying with tab separator")
        df = pd.read_csv(contam_path, sep='\t')

    # Find accession column
    acc_col = None
    for candidate in ['UniProt_Accession', 'uniprot_accession', 'UniProt', 'uniprot',
                      'Accession', 'accession', 'protein', 'Protein', 'Entry', 'From', 'ID']:
        if candidate in df.columns:
            acc_col = candidate
            break
    if acc_col is None:
        acc_col = df.columns[0]

    contam_ids = set(df[acc_col].astype(str).str.strip().tolist())
    return contam_ids


def load_detection_patterns(det_path: str, keep_patterns: list = None) -> set:
    """
    Load detection patterns CSV and return protein IDs matching specified patterns.
    Default: keep only "Stable detected" proteins.
    """
    df = pd.read_csv(det_path)

    if 'pattern' not in df.columns:
        raise ValueError(f"detection_patterns.csv must have 'pattern' column. "
                         f"Found: {list(df.columns)}")

    if keep_patterns is None:
        keep_patterns = ['Stable detected']

    kept = df[df['pattern'].isin(keep_patterns)]
    protein_col = 'protein' if 'protein' in df.columns else df.columns[0]
    protein_ids = set(kept[protein_col].astype(str).str.strip().tolist())

    print(f"  Detection patterns loaded: {len(df)} proteins total")
    pattern_counts = df['pattern'].value_counts()
    for pattern, count in pattern_counts.items():
        marker = " <-- KEPT" if pattern in keep_patterns else ""
        print(f"    {pattern}: {count}{marker}")
    print(f"  Proteins kept: {len(protein_ids)}")

    return protein_ids


def load_protein_whitelist(whitelist_path: str) -> set:
    """Load a simple text/CSV file with one protein ID per line."""
    try:
        df = pd.read_csv(whitelist_path)
        # Use first column
        ids = set(df.iloc[:, 0].astype(str).str.strip().tolist())
    except Exception:
        with open(whitelist_path) as f:
            ids = set(line.strip() for line in f if line.strip())
    return ids


def exclude_contaminant_proteins(df: pd.DataFrame, contam_ids: set,
                                  protein_col: str = 'protein') -> pd.DataFrame:
    """Remove contaminant proteins, handling isoform suffixes."""
    n_before = df[protein_col].nunique()

    def is_contaminant(pid):
        pid_str = str(pid).strip()
        base = pid_str.split('-')[0]  # handle isoforms
        return pid_str in contam_ids or base in contam_ids

    mask = ~df[protein_col].apply(is_contaminant)
    df_clean = df[mask].copy()
    n_after = df_clean[protein_col].nunique()
    print(f"  Contaminants removed: {n_before - n_after} proteins ({n_before} -> {n_after})")
    return df_clean


def convert_zeros_to_na(df: pd.DataFrame, min_intensity: float = 0,
                        intensity_col: str = 'raw_intensity') -> pd.DataFrame:
    """Convert values <= min_intensity to NaN (for 0-filled data)."""
    df = df.copy()
    n_zeros = (df[intensity_col] <= min_intensity).sum()
    df.loc[df[intensity_col] <= min_intensity, intensity_col] = np.nan
    pct = n_zeros / len(df) * 100 if len(df) > 0 else 0
    print(f"  Converted {n_zeros:,} values <= {min_intensity} to NaN ({pct:.1f}% of data)")
    return df


# ============================================================================
# UTILITIES
# ============================================================================

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(pvals, dtype=float)
    # FIX #5: handle empty input
    if p.size == 0:
        return p.copy()
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    bh = ranked * m / (np.arange(1, m + 1))
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)
    out = np.empty_like(pv)
    out[order] = bh
    q[valid] = out
    return q


def effect_size_label(r_squared: float) -> str:
    if r_squared < 0.01:
        return "negligible"
    if r_squared < 0.09:
        return "small"
    if r_squared < 0.25:
        return "medium"
    return "large"


def add_missing_flags(df: pd.DataFrame, min_intensity: float = None) -> pd.DataFrame:
    df = df.copy()
    df["raw_intensity"] = pd.to_numeric(df["raw_intensity"], errors="coerce")
    if min_intensity is not None and min_intensity > 0:
        df["is_missing"] = df["raw_intensity"].isna() | (df["raw_intensity"] <= min_intensity)
    else:
        df["is_missing"] = df["raw_intensity"].isna() | (df["raw_intensity"] <= 0)
    return df


def protein_passes_missing_filter(
        df: pd.DataFrame,
        protein: str,
        timepoint_order: list[str],
        max_missing_frac: float = 0.7,
        min_timepoints: int = 2,
) -> bool:
    sub = df[df["protein"] == protein]
    frac_missing = sub.groupby("timepoint")["is_missing"].mean()
    frac_missing = frac_missing.reindex(timepoint_order)
    n_good = int((frac_missing <= max_missing_frac).sum(skipna=True))
    return n_good >= min_timepoints


# ============================================================================
# IMPUTATION
# ============================================================================

def impute_minimum_value(df: pd.DataFrame, method: str = "half_min_protein") -> pd.DataFrame:
    df = df.copy()
    print(f"\nImputation method: {method}")
    before_missing = df["log_intensity"].isna().sum()
    print(f"Missing values before imputation: {before_missing:,} ({before_missing / len(df) * 100:.1f}%)")

    if method == "half_min_protein":
        def impute_protein(group):
            min_val = group["log_intensity"].min()
            if pd.isna(min_val):
                min_val = df["log_intensity"].min()
            # FIX #3: warn if global min is also NaN
            if pd.isna(min_val):
                log.warning(f"  WARNING: protein {group.name} has all missing values "
                            f"and global min is also NaN — cannot impute")
                return group
            group["log_intensity"] = group["log_intensity"].fillna(min_val * 0.5)
            return group
        df = df.groupby("protein", group_keys=False).apply(impute_protein)

    elif method == "min_protein":
        protein_min = df.groupby("protein")["log_intensity"].transform("min")
        df["log_intensity"] = df["log_intensity"].fillna(protein_min)

    elif method == "half_min_global":
        global_min = df["log_intensity"].min()
        if pd.isna(global_min):
            log.warning("  WARNING: global min is NaN — half_min_global cannot impute")
        else:
            df["log_intensity"] = df["log_intensity"].fillna(global_min * 0.5)

    elif method == "percentile":
        def impute_protein(group):
            p5 = group["log_intensity"].quantile(0.05)
            if pd.isna(p5):
                p5 = df["log_intensity"].quantile(0.05)
            if pd.isna(p5):
                log.warning(f"  WARNING: protein {group.name} — cannot compute 5th percentile")
                return group
            group["log_intensity"] = group["log_intensity"].fillna(p5 * 0.5)
            return group
        df = df.groupby("protein", group_keys=False).apply(impute_protein)

    after_missing = df["log_intensity"].isna().sum()
    print(f"Missing values after imputation: {after_missing:,}")
    print(f"Values imputed: {before_missing - after_missing:,}")
    return df


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_quantile(df: pd.DataFrame, timepoint_order: list[str]) -> pd.DataFrame:
    df = df.copy()
    print("\nApplying quantile normalization per timepoint...")

    # Compute a single reference distribution from all non-missing values
    all_values = df["log_intensity"].dropna().values
    if len(all_values) == 0:
        print("  WARNING: no non-missing values for quantile normalization")
        return df

    for tp in timepoint_order:
        mask = df["timepoint"] == tp
        if mask.sum() == 0:
            continue
        values = df.loc[mask, "log_intensity"].values
        if np.isnan(values).all():
            continue
        is_missing = np.isnan(values)
        if not is_missing.all():
            non_missing = values[~is_missing]
            ranks = stats.rankdata(non_missing, method='average')
            # Map ranks onto the global reference distribution
            target_quantiles = np.percentile(all_values, np.linspace(0, 100, len(non_missing)))
            normalized = np.interp(ranks, np.arange(1, len(ranks) + 1), target_quantiles)
            values_normalized = values.copy()
            values_normalized[~is_missing] = normalized
            df.loc[mask, "log_intensity"] = values_normalized
    print("Quantile normalization complete.")
    return df


def normalize_median_center(df: pd.DataFrame, timepoint_order: list[str]) -> pd.DataFrame:
    df = df.copy()
    print("\nApplying median centering per timepoint...")
    for tp in timepoint_order:
        mask = df["timepoint"] == tp
        tp_median = df.loc[mask, "log_intensity"].median()
        if not pd.isna(tp_median):
            df.loc[mask, "log_intensity"] -= tp_median
            print(f"  {tp}: median = {tp_median:.3f}")
    print("Median centering complete.")
    return df


def normalize_zscore_protein(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("\nApplying z-score normalization per protein...")
    def zscore_protein(group):
        mean = group["log_intensity"].mean()
        std = group["log_intensity"].std()
        if std > 0:
            group["log_intensity"] = (group["log_intensity"] - mean) / std
        return group
    df = df.groupby("protein", group_keys=False).apply(zscore_protein)
    print("Z-score normalization complete.")
    return df


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

def load_and_prepare(
        data_file: Path,
        timepoint_order: list[str],
        impute_method: str = "none",
        normalize_method: str = "none",
        dedup_agg: str = "median",
        covariate_columns: list[str] = None,
        contaminant_ids: set = None,
        keep_protein_ids: set = None,
        min_intensity: float = None,
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(data_file)
    df.columns = [c.strip() for c in df.columns]

    required = {"protein", "cluster", "timepoint", "patient", "raw_intensity"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    counts = {}
    counts['Total in data'] = df['protein'].nunique()
    print(f"Loaded {len(df):,} rows, {counts['Total in data']:,} proteins")

    # Exclude contaminants
    if contaminant_ids:
        print("\nExcluding contaminant proteins...")
        df = exclude_contaminant_proteins(df, contaminant_ids)
        counts['After contaminant removal'] = df['protein'].nunique()

    # Filter to stable/whitelisted proteins
    if keep_protein_ids:
        n_before = df['protein'].nunique()
        df = df[df['protein'].astype(str).str.strip().isin(keep_protein_ids)].copy()
        n_after = df['protein'].nunique()
        print(f"\nProtein filter applied: {n_before} -> {n_after} proteins")
        counts['After detection filter'] = n_after

    # Convert 0-filled values to NaN
    if min_intensity is not None:
        print(f"\nHandling 0-filled data (min_intensity={min_intensity})...")
        df = convert_zeros_to_na(df, min_intensity, 'raw_intensity')

    # Check for covariates
    if covariate_columns:
        missing_covariates = set(covariate_columns) - set(df.columns)
        if missing_covariates:
            print(f"WARNING: Covariates not found in data: {missing_covariates}")
            covariate_columns = [c for c in covariate_columns if c in df.columns]
        if covariate_columns:
            print(f"Using covariates: {covariate_columns}")

    df["timepoint"] = df["timepoint"].astype(str).str.strip()
    standardization_map = {
        "Baseline": "base", "baseline": "base", "BASE": "base",
        "Day1": "D1", "day1": "D1",
        "Day2": "D2", "day2": "D2",
        "Day3": "D3", "day3": "D3",
        "Day4": "D4", "day4": "D4",
        "PreOp": "PREOP", "preop": "PREOP", "pre-op": "PREOP",
    }
    df["timepoint"] = df["timepoint"].replace(standardization_map)
    df = df[df["timepoint"].isin(timepoint_order)].copy()

    if len(df) == 0:
        raise ValueError(f"No data found for specified timepoints: {timepoint_order}")

    tp_to_idx = {tp: i for i, tp in enumerate(timepoint_order)}
    df["tp_idx"] = df["timepoint"].map(tp_to_idx).astype(int)

    # FIX #1: pass min_intensity to add_missing_flags so threshold is consistent
    df = add_missing_flags(df, min_intensity=min_intensity)
    df.loc[df["raw_intensity"] <= 0, "raw_intensity"] = np.nan
    df["log_intensity"] = np.log2(df["raw_intensity"])

    if impute_method != "none":
        df = impute_minimum_value(df, method=impute_method)

    if normalize_method == "quantile":
        df = normalize_quantile(df, timepoint_order)
    elif normalize_method == "median_center":
        df = normalize_median_center(df, timepoint_order)
    elif normalize_method == "zscore_protein":
        df = normalize_zscore_protein(df)

    before = len(df)
    agg_dict = {
        "raw_intensity": (np.nanmedian if dedup_agg == "median" else np.nanmean),
        "log_intensity": (np.nanmedian if dedup_agg == "median" else np.nanmean),
        "is_missing": "max"
    }

    group_cols = ["protein", "cluster", "patient", "timepoint", "tp_idx"]
    if covariate_columns:
        for cov in covariate_columns:
            if cov in df.columns:
                agg_dict[cov] = "first"

    df = (df.groupby(group_cols, as_index=False)
          .agg(**{k: pd.NamedAgg(column=k, aggfunc=v) for k, v in agg_dict.items()}))

    after = len(df)
    if after < before:
        print(f"Deduplicated: {before:,} -> {after:,} rows (agg={dedup_agg})")

    print(f"Proteins: {df['protein'].nunique():,}")
    print(f"Patients: {df['patient'].nunique():,}")
    print(f"Timepoints: {sorted(df['timepoint'].unique(), key=lambda x: tp_to_idx[x])}")

    total = len(df)
    missing = df['log_intensity'].isna().sum()
    print(f"Missing data: {missing:,} / {total:,} ({missing/total*100:.1f}%)")

    return df, counts


# ============================================================================
# LINEAR MIXED MODEL
# ============================================================================

def fit_protein_lmm(
        protein_data: pd.DataFrame,
        timepoint_order: list[str],
        min_patients: int = 5,
        time_as_continuous: bool = False,
        covariate_columns: list[str] = None,
        reml: bool = True
) -> dict | None:
    """
    Fit linear mixed model for a single protein.
    Model: log_intensity ~ timepoint + covariates + (1|patient)
    Returns dict with results or None if model fails.
    """
    protein = protein_data["protein"].iloc[0]
    cluster = protein_data["cluster"].iloc[0]

    # Remove missing values
    model_data = protein_data[protein_data["log_intensity"].notna()].copy()

    if len(model_data) < min_patients:
        return None

    n_patients = model_data["patient"].nunique()
    if n_patients < min_patients:
        return None

    n_timepoints = model_data["timepoint"].nunique()
    if n_timepoints < 2:
        return None

    # Build formula
    if time_as_continuous:
        formula = "log_intensity ~ tp_idx"
    else:
        formula = "log_intensity ~ C(timepoint)"

    if covariate_columns:
        for cov in covariate_columns:
            if cov in model_data.columns and model_data[cov].notna().sum() > 0:
                if pd.api.types.is_numeric_dtype(model_data[cov]):
                    formula += f" + {cov}"
                else:
                    formula += f" + C({cov})"

    try:
        model = mixedlm(
            formula=formula,
            data=model_data,
            groups=model_data["patient"],
            re_formula="1"
        )
        result = model.fit(reml=reml, method='bfgs')

        # Extract overall timepoint effect p-value
        if time_as_continuous:
            timepoint_pval = result.pvalues.get('tp_idx', np.nan)
        else:
            timepoint_params = [p for p in result.params.index if 'timepoint' in p]
            if len(timepoint_params) > 0:
                try:
                    wald_test = result.wald_test_terms(skip_single=False)
                    if hasattr(wald_test, 'loc'):
                        timepoint_pval = wald_test.loc['C(timepoint)', 'pvalue']
                    elif hasattr(wald_test, 'table'):
                        timepoint_pval = wald_test.table.loc['C(timepoint)', 'pvalue']
                    else:
                        raise AttributeError(
                            f"wald_test_terms returned unexpected type: {type(wald_test)}")
                except Exception as e:
                    log.debug(f"  wald_test_terms failed for {protein} "
                              f"({type(e).__name__}: {e}), using first timepoint param p-value")
                    timepoint_pval = result.pvalues.get(timepoint_params[0], np.nan)
            else:
                timepoint_pval = np.nan

        # R-squared (Nakagawa & Schielzeth approximation)
        fixed_var = result.fittedvalues.var()
        random_var = float(result.cov_re.iloc[0, 0])
        residual_var = result.scale
        total_var = fixed_var + random_var + residual_var

        r_squared_marginal = fixed_var / total_var if total_var > 0 else 0
        r_squared_marginal = float(np.clip(r_squared_marginal, 0, 1))

        r_squared_conditional = (fixed_var + random_var) / total_var if total_var > 0 else 0
        r_squared_conditional = float(np.clip(r_squared_conditional, 0, 1))

        # ── Fold-changes ──────────────────────────────────────────────
        # compute per-timepoint levels (simple medians + model estimates)
        # then derive dynamic range (highest - lowest across all timepoints)
        # which captures the full swing regardless of which timepoint is the baseline.
        # keep per-timepoint FCs from base and endpoint FC for reference.
        first_tp = timepoint_order[0]
        last_tp = timepoint_order[-1]

        # --- Simple (data-driven): per-timepoint medians ---
        simple_medians = {}  # {tp: median log2 intensity}
        for tp in timepoint_order:
            tp_vals = model_data[model_data["timepoint"] == tp]["log_intensity"]
            simple_medians[tp] = float(tp_vals.median()) if len(tp_vals) > 0 else np.nan

        # Per-timepoint FC from baseline (for reference columns)
        baseline_median = simple_medians.get(first_tp, np.nan)
        simple_fc_from_base = {}
        for tp in timepoint_order:
            if not np.isnan(simple_medians[tp]) and not np.isnan(baseline_median):
                simple_fc_from_base[tp] = simple_medians[tp] - baseline_median
            else:
                simple_fc_from_base[tp] = np.nan

        # Dynamic range (simple): highest - lowest across ALL timepoints
        valid_simple = {tp: v for tp, v in simple_medians.items() if not np.isnan(v)}
        if len(valid_simple) >= 2:
            high_tp_simple = max(valid_simple, key=lambda t: valid_simple[t])
            low_tp_simple = min(valid_simple, key=lambda t: valid_simple[t])
            log2_fc_range_simple = valid_simple[high_tp_simple] - valid_simple[low_tp_simple]
            fc_range_simple = 2 ** log2_fc_range_simple
            # Signed: +ve if high comes after low temporally, -ve otherwise
            high_idx_s = timepoint_order.index(high_tp_simple)
            low_idx_s = timepoint_order.index(low_tp_simple)
            if high_idx_s > low_idx_s:
                log2_fc_range_simple_signed = log2_fc_range_simple
            else:
                log2_fc_range_simple_signed = -log2_fc_range_simple
        else:
            high_tp_simple = "unknown"
            low_tp_simple = "unknown"
            log2_fc_range_simple = np.nan
            fc_range_simple = np.nan
            log2_fc_range_simple_signed = np.nan

        # Endpoint FC (simple, base vs last)
        log2_fc_endpoint_simple = simple_fc_from_base.get(last_tp, np.nan)
        fc_endpoint_simple = 2 ** log2_fc_endpoint_simple if not np.isnan(log2_fc_endpoint_simple) else np.nan

        # --- Model-based: per-timepoint estimated means ---
        # The intercept is the estimated mean at the reference level (first_tp).
        # Coefficients give shifts from that reference.
        intercept = result.params.get('Intercept', np.nan)
        model_means = {}  # {tp: estimated mean log2 intensity}
        model_fc_from_base = {}  # {tp: log2FC vs baseline}

        if time_as_continuous:
            time_coef = result.params.get('tp_idx', np.nan)
            for tp in timepoint_order:
                tp_idx = timepoint_order.index(tp)
                if not np.isnan(time_coef) and not np.isnan(intercept):
                    model_means[tp] = intercept + time_coef * tp_idx
                    model_fc_from_base[tp] = time_coef * tp_idx
                else:
                    model_means[tp] = np.nan
                    model_fc_from_base[tp] = np.nan
        else:
            model_means[first_tp] = intercept if not np.isnan(intercept) else np.nan
            model_fc_from_base[first_tp] = 0.0
            for tp in timepoint_order:
                if tp == first_tp:
                    continue
                param_name = f'C(timepoint)[T.{tp}]'
                if param_name in result.params.index:
                    coef = float(result.params[param_name])
                    model_fc_from_base[tp] = coef
                    model_means[tp] = (intercept + coef) if not np.isnan(intercept) else np.nan
                else:
                    model_fc_from_base[tp] = np.nan
                    model_means[tp] = np.nan

        # Per-timepoint columns (model FC from baseline)
        model_fold_changes = {f'{tp}_log2fc_model': model_fc_from_base.get(tp, np.nan)
                              for tp in timepoint_order}

        # Dynamic range (model): highest - lowest estimated mean
        valid_model = {tp: v for tp, v in model_means.items() if not np.isnan(v)}
        if len(valid_model) >= 2:
            high_tp_model = max(valid_model, key=lambda t: valid_model[t])
            low_tp_model = min(valid_model, key=lambda t: valid_model[t])
            log2_fc_range_model = valid_model[high_tp_model] - valid_model[low_tp_model]
            fc_range_model = 2 ** log2_fc_range_model
        else:
            high_tp_model = "unknown"
            low_tp_model = "unknown"
            log2_fc_range_model = np.nan
            fc_range_model = np.nan

        # Endpoint FC (model, base vs last)
        log2_fc_endpoint_model = model_fc_from_base.get(last_tp, np.nan)
        fc_endpoint_model = 2 ** log2_fc_endpoint_model if not np.isnan(log2_fc_endpoint_model) else np.nan

        # Direction: based on temporal order of high vs low
        # high comes after low → "up" (protein rises); low comes after high → "down"
        # Note: when log2_fc_range_model > 0, high_tp != low_tp so high_idx != low_idx
        if not np.isnan(log2_fc_range_model) and log2_fc_range_model > 0:
            high_idx = timepoint_order.index(high_tp_model)
            low_idx = timepoint_order.index(low_tp_model)
            if high_idx > low_idx:
                fold_change_direction = "up"
                log2_fc_range_signed = log2_fc_range_model
            else:
                fold_change_direction = "down"
                log2_fc_range_signed = -log2_fc_range_model
        else:
            fold_change_direction = "no_change"
            log2_fc_range_signed = 0.0

        # Timepoint descriptives
        desc = {}
        for tp in timepoint_order:
            vals = model_data[model_data["timepoint"] == tp]["log_intensity"]
            desc[tp] = {
                "n": len(vals),
                "mean": float(vals.mean()) if len(vals) > 0 else np.nan,
                "median": float(vals.median()) if len(vals) > 0 else np.nan,
                "std": float(vals.std()) if len(vals) > 0 else np.nan,
            }

        out = {
            "protein": protein,
            "cluster": cluster,
            "model": "LMM",
            "converged": result.converged,
            "p_value": float(timepoint_pval),
            "r_squared_marginal": r_squared_marginal,
            "r_squared_conditional": r_squared_conditional,
            "effect_size": effect_size_label(r_squared_marginal),
            "n_observations": int(result.nobs),
            "n_patients": n_patients,
            "n_timepoints": n_timepoints,
            "timepoints_tested": ",".join(sorted(model_data["timepoint"].unique())),
            "random_intercept_var": float(random_var),
            "residual_var": float(residual_var),
            "AIC": float(result.aic),
            "BIC": float(result.bic),
            "log2_fc_range_simple": float(log2_fc_range_simple) if not np.isnan(log2_fc_range_simple) else np.nan,
            "fc_range_simple": float(fc_range_simple) if not np.isnan(fc_range_simple) else np.nan,
            "high_timepoint_simple": high_tp_simple,
            "low_timepoint_simple": low_tp_simple,
            "log2_fc_range_simple_signed": float(log2_fc_range_simple_signed),
            "log2_fc_range_model": float(log2_fc_range_model) if not np.isnan(log2_fc_range_model) else np.nan,
            "fc_range_model": float(fc_range_model) if not np.isnan(fc_range_model) else np.nan,
            "high_timepoint_model": high_tp_model,
            "low_timepoint_model": low_tp_model,
            "log2_fc_range_signed": float(log2_fc_range_signed) if not np.isnan(log2_fc_range_signed) else np.nan,
            "log2_fc_endpoint_simple": float(log2_fc_endpoint_simple) if not np.isnan(log2_fc_endpoint_simple) else np.nan,
            "fc_endpoint_simple": float(fc_endpoint_simple) if not np.isnan(fc_endpoint_simple) else np.nan,
            "log2_fc_endpoint_model": float(log2_fc_endpoint_model) if not np.isnan(log2_fc_endpoint_model) else np.nan,
            "fc_endpoint_model": float(fc_endpoint_model) if not np.isnan(fc_endpoint_model) else np.nan,
            "fold_change_direction": fold_change_direction,
            "comparison_range_simple": f"{low_tp_simple}_vs_{high_tp_simple}",
            "comparison_range_model": f"{low_tp_model}_vs_{high_tp_model}",
            "comparison_endpoint": f"{first_tp}_vs_{last_tp}",
        }

        out.update(model_fold_changes)

        # Per-timepoint simple FCs from baseline
        for tp in timepoint_order:
            out[f'{tp}_log2fc_simple'] = simple_fc_from_base.get(tp, np.nan)

        for tp in timepoint_order:
            for metric in ["n", "mean", "median", "std"]:
                out[f"{tp}_{metric}"] = desc[tp][metric]

        if covariate_columns:
            for cov in covariate_columns:
                if cov in result.params.index:
                    out[f"{cov}_coef"] = float(result.params[cov])
                    out[f"{cov}_pval"] = float(result.pvalues[cov])

        return out

    except Exception as e:
        return None


# ============================================================================
# PLOTS
# ============================================================================

def plot_significance_vs_effect(results_df: pd.DataFrame, figdir: Path,
                                timepoint_str: str, alpha: float = 0.05):
    df = results_df.copy()
    df["neg_log10_q"] = -np.log10(df["q_value"].clip(1e-50))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if q < alpha else 'gray' for q in df['q_value']]
    ax.scatter(df["r_squared_marginal"], df["neg_log10_q"], c=colors, alpha=0.6, s=25)
    ax.axhline(-np.log10(alpha), linestyle="--", color="red", alpha=0.5, label=f"q={alpha}")
    ax.set_xlabel("Marginal R² (Effect Size)")
    ax.set_ylabel("-log10(q-value)")
    ax.set_title(f"Effect Size vs Significance: {timepoint_str}")
    ax.grid(alpha=0.3)
    ax.legend()

    top = df.nsmallest(10, "q_value")
    for _, r in top.iterrows():
        ax.annotate(str(r["protein"]), (r["r_squared_marginal"], r["neg_log10_q"]),
                    fontsize=7, xytext=(5, 2), textcoords="offset points", alpha=0.8)

    plt.tight_layout()
    plt.savefig(figdir / "significance_vs_effect.png", dpi=160)
    plt.close()


def plot_filtering_funnel(counts: dict, figdir: Path):
    """Bar chart showing protein counts at each filtering stage."""
    labels = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#3498db'] * len(labels)
    if len(colors) >= 2:
        colors[-1] = '#e74c3c'
        colors[-2] = '#e67e22'

    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='black', alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Proteins')
    ax.set_title('LMM Analysis: Protein Filtering Summary')
    ax.grid(axis='x', alpha=0.3)

    # Add count labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(figdir / 'filtering_funnel.png', dpi=160)
    plt.close()


def plot_model_convergence(results_df: pd.DataFrame, figdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    converged = results_df['converged'].value_counts()
    ax.pie(converged.values,
           labels=['Converged' if k else 'Failed' for k in converged.index],
           autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
    ax.set_title('Model Convergence')

    ax = axes[1]
    ax.hist(results_df['r_squared_marginal'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(results_df['r_squared_marginal'].median(), color='red', linestyle='--',
               label=f"Median: {results_df['r_squared_marginal'].median():.3f}")
    ax.set_xlabel('Marginal R-squared')
    ax.set_ylabel('Number of Proteins')
    ax.set_title('Effect Size Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figdir / 'model_diagnostics.png', dpi=160)
    plt.close()


def plot_effect_size_comparison(results_df: pd.DataFrame, figdir: Path, alpha: float = 0.05):
    sig = results_df[results_df["q_value"] < alpha].copy()
    if sig.empty:
        return

    effect_counts = sig["effect_size"].value_counts().reindex(
        ["negligible", "small", "medium", "large"], fill_value=0)
    colors = ["#d3d3d3", "#90EE90", "#FFD700", "#FF6347"]

    fig, ax = plt.subplots(figsize=(8, 5))
    effect_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="black")
    ax.set_xlabel("Effect Size Category (Marginal R-squared)")
    ax.set_ylabel("Number of Significant Proteins")
    ax.set_title(f"Significant Proteins by Effect Size (q < {alpha})")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(effect_counts):
        if v > 0:
            ax.text(i, v + 0.5, str(int(v)), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(figdir / "effect_sizes.png", dpi=160)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Linear Mixed Model analysis with protein filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--timepoints", nargs="+", required=True)
    ap.add_argument("--outdir", required=True, type=Path)

    # Filtering options
    ap.add_argument("--contaminants", default=None, type=str,
                    help="Contaminant protein list TSV/CSV (e.g., cRAP list)")
    ap.add_argument("--detection_patterns", default=None, type=str,
                    help="detection_patterns.csv from protein_pattern_detection.py")
    ap.add_argument("--stable_only", action="store_true",
                    help="Keep only 'Stable detected' proteins (requires --detection_patterns)")
    ap.add_argument("--keep_patterns", nargs="*", default=None,
                    help="Which detection patterns to keep. Takes precedence over --stable_only "
                         "if both are provided. "
                         "Example: --keep_patterns 'Stable detected' 'Fluctuating'")
    ap.add_argument("--protein_whitelist", default=None, type=str,
                    help="Text/CSV file with protein IDs to keep (one per line)")
    ap.add_argument("--min_intensity", type=float, default=None,
                    help="Treat raw_intensity values <= this as missing. "
                         "Use 0 if NAs were filled with 0.")

    # Model options
    ap.add_argument("--time_as_continuous", action="store_true")
    ap.add_argument("--covariates", nargs="*", default=[])
    ap.add_argument("--reml", action="store_true", default=True)
    ap.add_argument("--no-reml", dest="reml", action="store_false")

    # Data processing
    ap.add_argument("--impute",
                    choices=["none", "half_min_protein", "min_protein",
                             "half_min_global", "percentile"],
                    default="half_min_protein")
    ap.add_argument("--normalize",
                    choices=["none", "quantile", "median_center", "zscore_protein"],
                    default="quantile")
    ap.add_argument("--dedup_agg", choices=["median", "mean"], default="median")
    ap.add_argument("--max_missing_frac", type=float, default=0.7)
    ap.add_argument("--min_good_timepoints", type=int, default=None)
    ap.add_argument("--min_patients", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fc_threshold", type=float, default=1.0,
                    help="log2 fold-change threshold for the FC-filtered output file. "
                         "Default 1.0 (= 2-fold change). Proteins with q<alpha AND "
                         "|log2FC_range| >= this value are saved to lmm_results_significant_fc.csv")

    args = ap.parse_args()


    if args.stable_only and not args.detection_patterns:
        ap.error("--stable_only requires --detection_patterns")


    if args.keep_patterns and args.stable_only:
        print("NOTE: --keep_patterns takes precedence over --stable_only")

    if args.min_good_timepoints is None:
        args.min_good_timepoints = max(2, len(args.timepoints) - 1)

    args.outdir.mkdir(parents=True, exist_ok=True)
    figdir = args.outdir / "figures"
    figdir.mkdir(exist_ok=True)

    timepoint_str = " -> ".join(args.timepoints)

    print("=" * 70)
    print("LINEAR MIXED MODEL - PROTEIN TIMEPOINT ANALYSIS (FILTERED)")
    print("=" * 70)
    print(f"Timepoints       : {timepoint_str}")
    print(f"Time as continuous: {args.time_as_continuous}")
    print(f"Covariates       : {args.covariates if args.covariates else 'None'}")
    print(f"Estimation method: {'REML' if args.reml else 'ML'}")
    print(f"Imputation       : {args.impute}")
    print(f"Normalization    : {args.normalize}")
    print(f"Min patients     : {args.min_patients}")
    print(f"Significance     : q < {args.alpha}")
    print(f"FC threshold     : |log2FC| >= {args.fc_threshold} (= {2**args.fc_threshold:.1f}-fold)")
    if args.contaminants:
        print(f"Contaminants     : {args.contaminants}")
    if args.detection_patterns:
        print(f"Detection filter : {args.detection_patterns}")
        print(f"Stable only      : {args.stable_only}")
    if args.protein_whitelist:
        print(f"Protein whitelist: {args.protein_whitelist}")
    if args.min_intensity is not None:
        print(f"Min intensity    : {args.min_intensity} (values <= this -> NaN)")
    print("=" * 70)

    # Load filtering data
    contaminant_ids = None
    if args.contaminants:
        print("\nLoading contaminant list...")
        contaminant_ids = load_contaminants(args.contaminants)
        print(f"  Contaminant proteins: {len(contaminant_ids)}")

    keep_protein_ids = None
    if args.detection_patterns:
        print("\nLoading detection patterns...")
        if args.keep_patterns:
            patterns = args.keep_patterns
        elif args.stable_only:
            patterns = ['Stable detected']
        else:
            patterns = None  # Keep all
        keep_protein_ids = load_detection_patterns(args.detection_patterns, patterns)

    elif args.protein_whitelist:
        print(f"\nLoading protein whitelist: {args.protein_whitelist}")
        keep_protein_ids = load_protein_whitelist(args.protein_whitelist)
        print(f"  Whitelist proteins: {len(keep_protein_ids)}")

    # Load data (with filtering)
    df, funnel_counts = load_and_prepare(
        args.data,
        args.timepoints,
        impute_method=args.impute,
        normalize_method=args.normalize,
        dedup_agg=args.dedup_agg,
        covariate_columns=args.covariates,
        contaminant_ids=contaminant_ids,
        keep_protein_ids=keep_protein_ids,
        min_intensity=args.min_intensity,
    )

    # QC filter
    all_proteins = df["protein"].unique()
    keep = [
        p for p in all_proteins
        if protein_passes_missing_filter(
            df, p, args.timepoints,
            max_missing_frac=args.max_missing_frac,
            min_timepoints=args.min_good_timepoints
        )
    ]
    df = df[df["protein"].isin(keep)].copy()

    print(f"\nMissingness QC filter:")
    print(f"  Proteins before : {len(all_proteins):,}")
    print(f"  Proteins kept   : {len(keep):,}")
    print(f"  Proteins removed: {len(all_proteins) - len(keep):,}")
    funnel_counts['After missingness QC'] = len(keep)

    proteins = df["protein"].unique()
    print(f"\nFitting LMM for {len(proteins):,} proteins...")
    print("(This may take several minutes...)")

    # Fit models
    rows = []
    failed = 0
    for i, prot in enumerate(proteins, 1):
        res = fit_protein_lmm(
            df[df["protein"] == prot],
            args.timepoints,
            min_patients=args.min_patients,
            time_as_continuous=args.time_as_continuous,
            covariate_columns=args.covariates,
            reml=args.reml
        )
        if res is not None:
            rows.append(res)
        else:
            failed += 1

        if i % 100 == 0:
            print(f"  {i}/{len(proteins)} processed ({failed} failed)")

    if not rows:
        print("ERROR: No proteins successfully fitted. Check data quality.")
        return

    print(f"\nModel fitting complete:")
    print(f"  Successful: {len(rows):,}")
    print(f"  Failed: {failed:,}")

    results_df = pd.DataFrame(rows)


    if len(results_df) == 0:
        print("ERROR: results_df is empty after model fitting.")
        return

    results_df["q_value"] = bh_fdr(results_df["p_value"].to_numpy())
    results_df = results_df.sort_values("q_value")

    # Save results
    out_all = args.outdir / "lmm_results_all_proteins.csv"
    results_df.to_csv(out_all, index=False)
    print(f"\nSaved: {out_all}")

    sig = results_df[results_df["q_value"] < args.alpha].copy()
    out_sig = args.outdir / "lmm_results_significant.csv"
    sig.to_csv(out_sig, index=False)
    print(f"Saved: {out_sig} ({len(sig)} significant proteins)")

    # FC-threshold filtered output (use simple/median-based range — less shrinkage than model)
    fc_col = "log2_fc_range_simple"
    if fc_col not in sig.columns:
        fc_col = "log2_fc_range_model"

    if fc_col in sig.columns:
        sig_fc = sig[sig[fc_col].abs() >= args.fc_threshold].copy()
        out_sig_fc = args.outdir / "lmm_results_significant_fc.csv"
        sig_fc.to_csv(out_sig_fc, index=False)
        print(f"Saved: {out_sig_fc} ({len(sig_fc)} proteins: q<{args.alpha} & |log2FC|>={args.fc_threshold})")
    else:
        sig_fc = pd.DataFrame()
        print("WARNING: no fold-change column available for FC filtering")

    # Complete funnel counts
    funnel_counts['LMM fitted'] = len(rows)
    funnel_counts[f'Significant (q<{args.alpha})'] = len(sig)
    funnel_counts[f'Sig + FC (|log2FC|>={args.fc_threshold})'] = len(sig_fc)

    # Save filtering info
    filter_info = {
        'total_in_data': funnel_counts.get('Total in data', 0),
        'contaminants_excluded': len(contaminant_ids) if contaminant_ids else 0,
        'detection_filter': args.detection_patterns if args.detection_patterns else 'None',
        'stable_only': args.stable_only,
        'min_intensity': args.min_intensity,
        'fc_threshold_log2': args.fc_threshold,
        'proteins_after_filter': len(proteins),
        'proteins_fitted': len(rows),
        'proteins_significant': len(sig),
        'proteins_significant_fc': len(sig_fc),
    }
    pd.DataFrame([filter_info]).to_csv(args.outdir / "filtering_info.csv", index=False)

    # Plots
    print("\nGenerating plots...")
    plot_filtering_funnel(funnel_counts, figdir)
    plot_significance_vs_effect(results_df, figdir, timepoint_str, alpha=args.alpha)
    plot_model_convergence(results_df, figdir)
    plot_effect_size_comparison(results_df, figdir, alpha=args.alpha)

    # ── Summary ───────────────────────────────────────────────────────
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("LMM ANALYSIS SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append("Parameters:")
    summary_lines.append(f"  Data file        : {args.data}")
    summary_lines.append(f"  Timepoints       : {timepoint_str}")
    summary_lines.append(f"  Imputation       : {args.impute}")
    summary_lines.append(f"  Normalization    : {args.normalize}")
    summary_lines.append(f"  Estimation       : {'REML' if args.reml else 'ML'}")
    summary_lines.append(f"  Min patients     : {args.min_patients}")
    summary_lines.append(f"  Alpha            : {args.alpha}")
    summary_lines.append(f"  FC threshold     : |log2FC| >= {args.fc_threshold} ({2**args.fc_threshold:.1f}-fold)")
    if args.contaminants:
        summary_lines.append(f"  Contaminants     : {args.contaminants}")
    if args.detection_patterns:
        summary_lines.append(f"  Detection filter : {args.detection_patterns} (stable_only={args.stable_only})")
    if args.covariates:
        summary_lines.append(f"  Covariates       : {args.covariates}")
    summary_lines.append("")

    summary_lines.append("Protein filtering funnel:")
    for stage, count in funnel_counts.items():
        summary_lines.append(f"  {stage:40s}: {count:,}")
    summary_lines.append("")

    summary_lines.append(f"Model fitting:")
    summary_lines.append(f"  Models converged : {results_df['converged'].sum():,} / {len(results_df):,} "
                         f"({results_df['converged'].sum() / len(results_df) * 100:.1f}%)")
    summary_lines.append(f"  Models failed    : {failed:,}")
    summary_lines.append("")

    if not sig.empty:
        summary_lines.append(f"Effect size distribution (q<{args.alpha}):")
        for eff in ["large", "medium", "small", "negligible"]:
            n = (sig["effect_size"] == eff).sum()
            summary_lines.append(f"  {eff:12s}: {n:4d} ({n / len(sig) * 100:5.1f}%)")
        summary_lines.append(f"  Median marginal R²   : {sig['r_squared_marginal'].median():.3f}")
        summary_lines.append(f"  Mean obs per protein  : {sig['n_observations'].mean():.1f}")
        summary_lines.append("")

        if "high_timepoint_simple" in sig.columns:
            summary_lines.append(f"High timepoint (q<{args.alpha}):")
            for tp, cnt in sig["high_timepoint_simple"].value_counts().items():
                summary_lines.append(f"  {tp:6s}: {cnt:4d} ({cnt / len(sig) * 100:5.1f}%)")
            summary_lines.append(f"Low timepoint (q<{args.alpha}):")
            for tp, cnt in sig["low_timepoint_simple"].value_counts().items():
                summary_lines.append(f"  {tp:6s}: {cnt:4d} ({cnt / len(sig) * 100:5.1f}%)")
            summary_lines.append("")

    summary_lines.append("Top 10 proteins by q-value:")
    cols = ["protein", "cluster", "p_value", "q_value", "r_squared_marginal",
            "effect_size", "log2_fc_range_simple", "high_timepoint_simple",
            "low_timepoint_simple", "n_observations"]
    top10 = results_df[[c for c in cols if c in results_df.columns]].head(10).to_string(index=False)
    summary_lines.append(top10)
    summary_lines.append("")

    summary_lines.append("Output files:")
    summary_lines.append(f"  {args.outdir / 'lmm_results_all_proteins.csv'}")
    summary_lines.append(f"  {args.outdir / 'lmm_results_significant.csv'}")
    summary_lines.append(f"  {args.outdir / 'lmm_results_significant_fc.csv'}")
    summary_lines.append(f"  {args.outdir / 'filtering_info.csv'}")
    summary_lines.append(f"  {figdir / 'filtering_funnel.png'}")
    summary_lines.append(f"  {figdir / 'significance_vs_effect.png'}")
    summary_lines.append(f"  {figdir / 'model_diagnostics.png'}")
    summary_lines.append(f"  {figdir / 'effect_sizes.png'}")
    summary_lines.append("")
    summary_lines.append("=" * 70)

    summary_text = "\n".join(summary_lines)

    # Print to console
    print("\n" + summary_text)

    # Save summary file
    summary_path = args.outdir / "lmm_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text + "\n")
    print(f"\nSaved: {summary_path}")

    print("\nDone.")
    print(f"Output directory: {args.outdir}")


if __name__ == "__main__":
    main()