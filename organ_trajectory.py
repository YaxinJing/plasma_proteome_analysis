#!/usr/bin/env python3
"""
Usage:
  python organ_trajectory.py \
    --data cohort1_ART_raw_long.csv \
    --timepoints base D1 D2 D3 D4 \
    --hpa_summary proteinatlas.tsv \
    --hpa_organ_map normal_ihc_tissues.tsv \
    --contaminants crap_contaminants.tsv \
    --clusters patient_clusters.csv \
    --min_intensity 0 \
    --outdir organ_trajectory_results
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────
# 1. PROTEIN-TO-ORGAN ASSIGNMENT (from HPA)
# ──────────────────────────────────────────────────────────────────────

def load_hpa_tissue_assignments(hpa_summary_path, hpa_rna_path=None, organ_map_path=None,
                                 specificity_filter=None):
    """
    Assign proteins to their primary organ using HPA data.

    Uses proteinatlas.tsv columns:
      - Gene, Uniprot
      - RNA tissue specificity (Tissue enriched, Group enriched, etc.)
      - RNA tissue specific nTPM (e.g., "liver: 1585.8")

    Only keeps tissue-enriched and group-enriched proteins (not ubiquitous).

    Returns: DataFrame with protein, gene, organ_system, top_tissue, specificity
    """
    if specificity_filter is None:
        specificity_filter = ['Tissue enriched', 'Group enriched', 'Tissue enhanced']

    # Load proteinatlas.tsv (main summary)
    print(f"  Loading HPA summary: {hpa_summary_path}")
    # Only load columns we need (file is very wide)
    needed_cols = ['Gene', 'Uniprot', 'RNA tissue specificity',
                   'RNA tissue specific nTPM', 'RNA tissue distribution']
    try:
        hpa = pd.read_csv(hpa_summary_path, sep='\t',
                          usecols=lambda c: c.strip() in needed_cols)
    except Exception:
        hpa = pd.read_csv(hpa_summary_path, sep='\t')

    # Normalize column names
    hpa.columns = [c.strip() for c in hpa.columns]
    print(f"  HPA entries: {len(hpa)}")
    print(f"  Columns found: {list(hpa.columns)}")

    # Filter to tissue-specific proteins
    spec_col = 'RNA tissue specificity'
    if spec_col not in hpa.columns:
        # Try alternatives
        for alt in ['RNA tissue specificity', 'Tissue specificity']:
            if alt in hpa.columns:
                spec_col = alt
                break

    if spec_col in hpa.columns:
        spec_counts = hpa[spec_col].value_counts()
        print(f"\n  RNA tissue specificity distribution:")
        for spec, count in spec_counts.items():
            marker = " <-- KEPT" if spec in specificity_filter else ""
            print(f"    {spec}: {count}{marker}")

        hpa_specific = hpa[hpa[spec_col].isin(specificity_filter)].copy()
        print(f"\n  Tissue-specific proteins: {len(hpa_specific)}")
    else:
        print(f"  WARNING: '{spec_col}' column not found, using all proteins")
        hpa_specific = hpa.copy()

    # Parse tissue assignments from "RNA tissue specific nTPM" column
    # Format: "liver: 1585.8" or "bone marrow: 196.5;lung: 83.0;lymphoid tissue: 136.4"
    ntpm_col = 'RNA tissue specific nTPM'
    if ntpm_col not in hpa_specific.columns:
        for alt in ['RNA tissue specific nTPM', 'Tissue specific nTPM']:
            if alt in hpa_specific.columns:
                ntpm_col = alt
                break

    # Load organ map
    organ_map = load_organ_map(organ_map_path)

    assignments = []
    for _, row in hpa_specific.iterrows():
        gene = str(row.get('Gene', '')).strip()
        uniprot = str(row.get('Uniprot', '')).strip()
        specificity = str(row.get(spec_col, '')).strip()

        if not gene or gene == 'nan' or not uniprot or uniprot == 'nan':
            continue

        # Parse tissue:nTPM pairs
        ntpm_str = str(row.get(ntpm_col, ''))
        if ntpm_str and ntpm_str != 'nan':
            # Parse "liver: 1585.8" or "bone marrow: 196.5;lung: 83.0"
            tissues = []
            for pair in ntpm_str.split(';'):
                pair = pair.strip()
                if ':' in pair:
                    tissue_name = pair.rsplit(':', 1)[0].strip().lower()
                    try:
                        ntpm_val = float(pair.rsplit(':', 1)[1].strip())
                    except ValueError:
                        ntpm_val = 0
                    tissues.append((tissue_name, ntpm_val))

            if tissues:
                # Sort by nTPM, take the highest as primary tissue
                tissues.sort(key=lambda x: x[1], reverse=True)
                primary_tissue = tissues[0][0]
                primary_ntpm = tissues[0][1]
                all_tissues = ';'.join(f"{t}:{n:.1f}" for t, n in tissues)

                # Map to organ system
                organ = organ_map.get(primary_tissue, None)
                # Try partial match
                if organ is None:
                    for key, sys in organ_map.items():
                        if key in primary_tissue or primary_tissue in key:
                            organ = sys
                            break
                if organ is None:
                    organ = 'Other'

                assignments.append({
                    'gene': gene,
                    'uniprot': uniprot,
                    'specificity': specificity,
                    'primary_tissue': primary_tissue,
                    'primary_ntpm': primary_ntpm,
                    'organ_system': organ,
                    'all_tissues_ntpm': all_tissues,
                })

    assign_df = pd.DataFrame(assignments)
    print(f"\n  Proteins with organ assignments: {len(assign_df)}")

    if not assign_df.empty:
        organ_counts = assign_df['organ_system'].value_counts()
        print(f"\n  Proteins per organ system:")
        for organ, count in organ_counts.items():
            print(f"    {organ}: {count}")

    # Also use HPA RNA file for broader coverage if provided
    if hpa_rna_path and os.path.exists(hpa_rna_path):
        assign_df = enrich_assignments_from_rna(assign_df, hpa_rna_path, organ_map, hpa,
                                                 specificity_filter=specificity_filter)

    return assign_df


def enrich_assignments_from_rna(assign_df, hpa_rna_path, organ_map, hpa_summary,
                                specificity_filter=None):
    """
    For proteins in HPA summary that have tissue specificity but no nTPM string,
    use the RNA consensus file to find their primary tissue.
    """
    if specificity_filter is None:
        specificity_filter = ['Tissue enriched', 'Group enriched', 'Tissue enhanced']

    already_assigned = set(assign_df['uniprot'].tolist())

    # Find tissue-specific proteins not yet assigned
    spec_col = 'RNA tissue specificity'
    if spec_col not in hpa_summary.columns:
        return assign_df

    missing = hpa_summary[
        (hpa_summary[spec_col].isin(specificity_filter)) &
        (~hpa_summary['Uniprot'].isin(already_assigned))
    ]

    if missing.empty:
        return assign_df

    print(f"\n  Enriching {len(missing)} additional proteins from RNA consensus file...")
    rna = pd.read_csv(hpa_rna_path, sep='\t')
    rna.columns = [c.lower().strip() for c in rna.columns]

    gene_col = 'gene name' if 'gene name' in rna.columns else 'gene'
    tpm_col = 'ntpm' if 'ntpm' in rna.columns else 'tpm'

    extra = []
    for _, row in missing.iterrows():
        gene = str(row['Gene']).strip()
        uniprot = str(row['Uniprot']).strip()

        gene_data = rna[rna[gene_col] == gene]
        if gene_data.empty:
            continue

        top = gene_data.loc[gene_data[tpm_col].idxmax()]
        tissue = str(top['tissue']).lower().strip()
        ntpm = float(top[tpm_col])

        organ = organ_map.get(tissue, None)
        if organ is None:
            for key, sys in organ_map.items():
                if key in tissue or tissue in key:
                    organ = sys
                    break
        if organ is None:
            organ = 'Other'

        extra.append({
            'gene': gene, 'uniprot': uniprot,
            'specificity': row[spec_col],
            'primary_tissue': tissue, 'primary_ntpm': ntpm,
            'organ_system': organ, 'all_tissues_ntpm': f"{tissue}:{ntpm:.1f}",
        })

    if extra:
        extra_df = pd.DataFrame(extra)
        assign_df = pd.concat([assign_df, extra_df], ignore_index=True)
        print(f"  Added {len(extra)} proteins from RNA file")
        print(f"  Total assigned: {len(assign_df)}")

    return assign_df


def load_organ_map(organ_map_path=None):
    """Load tissue -> organ system mapping."""
    DEFAULT_MAP = {
        'adipose tissue': 'Connective & Soft tissue', 'adrenal gland': 'Endocrine tissues',
        'appendix': 'Bone marrow & Lymphoid tissues', 'bone marrow': 'Bone marrow & Lymphoid tissues',
        'breast': 'Breast & female reproductive system', 'bronchus': 'Respiratory system',
        'brain': 'Brain', 'caudate': 'Brain', 'cerebellum': 'Brain', 'cerebral cortex': 'Brain',
        'colon': 'Gastrointestinal tract', 'duodenum': 'Gastrointestinal tract',
        'esophagus': 'Proximal digestive tract', 'gallbladder': 'Liver & Gallbladder',
        'heart muscle': 'Muscle & Vascular tissue', 'hippocampus': 'Brain',
        'hippocampal formation': 'Brain', 'midbrain': 'Brain',
        'kidney': 'Kidney & Urinary bladder', 'liver': 'Liver & Gallbladder',
        'lung': 'Respiratory system', 'lymph node': 'Bone marrow & Lymphoid tissues',
        'lymphoid tissue': 'Bone marrow & Lymphoid tissues',
        'pancreas': 'Pancreas', 'parathyroid gland': 'Endocrine tissues',
        'placenta': 'Breast & female reproductive system',
        'rectum': 'Gastrointestinal tract', 'retina': 'Eye',
        'salivary gland': 'Proximal digestive tract',
        'seminal vesicle': 'Male reproductive system',
        'skeletal muscle': 'Muscle & Vascular tissue', 'skin': 'Skin',
        'small intestine': 'Gastrointestinal tract', 'smooth muscle': 'Muscle & Vascular tissue',
        'spleen': 'Bone marrow & Lymphoid tissues', 'stomach': 'Proximal digestive tract',
        'testis': 'Male reproductive system', 'thyroid gland': 'Endocrine tissues',
        'tonsil': 'Bone marrow & Lymphoid tissues',
        'amygdala': 'Brain', 'basal ganglia': 'Brain', 'choroid plexus': 'Brain',
        'hypothalamus': 'Brain', 'pituitary gland': 'Endocrine tissues',
        'substantia nigra': 'Brain', 'thalamus': 'Brain', 'spinal cord': 'Brain',
        'white matter': 'Brain', 'eye': 'Eye', 'blood': 'Blood', 'blood vessel': 'Muscle & Vascular tissue',
        'thymus': 'Bone marrow & Lymphoid tissues', 'tongue': 'Proximal digestive tract',
        'prostate': 'Male reproductive system', 'epididymis': 'Male reproductive system',
        'urinary bladder': 'Kidney & Urinary bladder',
        'cervix': 'Breast & female reproductive system',
        'endometrium': 'Breast & female reproductive system',
        'fallopian tube': 'Breast & female reproductive system',
        'ovary': 'Breast & female reproductive system',
        'vagina': 'Breast & female reproductive system',
        'nasopharynx': 'Respiratory system', 'oral mucosa': 'Proximal digestive tract',
        'soft tissue': 'Connective & Soft tissue',
    }

    if organ_map_path is None:
        return DEFAULT_MAP

    df = pd.read_csv(organ_map_path, sep='\t')
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)
    if 'tissue' in df.columns and 'organ' in df.columns:
        file_map = dict(zip(df['tissue'].str.lower().str.strip(), df['organ'].str.strip()))
    else:
        file_map = dict(zip(df.iloc[:, 0].str.lower().str.strip(), df.iloc[:, 1].str.strip()))

    merged = DEFAULT_MAP.copy()
    merged.update(file_map)
    return merged


# ──────────────────────────────────────────────────────────────────────
# 1c. PROTEIN-TO-ORGAN ASSIGNMENT (from GLS - Malmström et al. 2025)
# ──────────────────────────────────────────────────────────────────────

# GLS tissue label → organ system mapping
GLS_ORGAN_MAP = {
    # Tissues → organ systems
    'brain': 'Brain', 'nerve': 'Brain & Nerve',
    'heart': 'Heart', 'heart.muscle': 'Heart & Muscle',
    'liver': 'Liver', 'kidney': 'Kidney', 'lung': 'Lung',
    'pancreas': 'Pancreas', 'spleen': 'Spleen',
    'muscle': 'Muscle', 'skin': 'Skin',
    'colon': 'GI tract', 'stomach': 'GI tract', 'esophagus': 'GI tract',
    'artery': 'Vasculature', 'adiposetissue': 'Adipose',
    'ovary': 'Ovary', 'prostate': 'Prostate', 'bladder': 'Bladder',
    'adrenalgland': 'Adrenal gland', 'thyroid': 'Thyroid',
    'bonemarrow': 'Bone marrow',
    # Cell types → cell-based labels
    'neutrophils': 'Neutrophils', 'macrophages': 'Macrophages',
    'monocytes': 'Monocytes', 'platelets': 'Platelets',
    'erythrocytes': 'Erythrocytes', 'bcell': 'B cells',
    'tcellcd4': 'T cells CD4', 'tcellcd8': 'T cells CD8',
    # Common multi-labels
    'bonemarrow.muscle': 'Bone marrow', 'bonemarrow.spleen': 'Bone marrow',
    'brain.muscle': 'Brain', 'liver.pancreas': 'Liver',
    'monocytes.neutrophils': 'Monocytes & Neutrophils',
    'macrophages.neutrophils': 'Macrophages & Neutrophils',
    'macrophages.platelets': 'Macrophages & Platelets',
    'macrophages.monocytes': 'Macrophages & Monocytes',
    'erythrocytes.platelets': 'Erythrocytes & Platelets',
    'tcellcd4.tcellcd8': 'T cells', 'bcell.macrophages': 'B cells & Macrophages',
    'neutrophils.platelets': 'Neutrophils & Platelets',
    'neutrophils.tcellcd4': 'Neutrophils & T cells',
    'erythrocytes.macrophages': 'Erythrocytes & Macrophages',
}


def load_gls_assignments(gls_path, min_gls=1.0, include_cells=False,
                          include_multilabel=True):
    """
    Load protein-to-organ assignments from the Global Label Score (GLS) atlas.

    Malmström et al. 2025, Cell 188, 2810-2822.
    Supplementary Table S11 (mmc11.xlsx).

    Parameters
    ----------
    gls_path : str
        Path to mmc11.xlsx
    min_gls : float
        Minimum GLS to include (default 1.0). Higher = more confident:
          1.0 = all assigned (broadest, ~14k proteins)
          2.0 = moderate confidence (~5k)
          3.0 = high confidence (~2k)
          4.0 = highest confidence (~365)
    include_cells : bool
        If True, include cell-type assignments (neutrophils, macrophages, etc.)
    include_multilabel : bool
        If True, include multi-tissue labels (e.g. heart.muscle).
        Multi-label proteins are mapped to their primary organ via GLS_ORGAN_MAP.

    Returns
    -------
    DataFrame with columns: uniprot, gene, organ_system, gls, global_label,
                            specificity, primary_tissue, top_tissue
    """
    import openpyxl  # already available from pandas excel support

    print(f"  Loading GLS atlas: {gls_path}")
    gls = pd.read_excel(gls_path)
    print(f"  Total entries: {len(gls)}")

    # Standardize columns
    gls.columns = [c.strip() for c in gls.columns]

    # Find key columns (robust to typos and version differences)
    gls_col_candidates = [c for c in gls.columns if 'global' in c.lower() and 'score' in c.lower()]
    if not gls_col_candidates:
        # Fallback: the original file has a typo "Gobal"
        gls_col_candidates = [c for c in gls.columns if 'obal' in c.lower() and 'score' in c.lower()]
    if not gls_col_candidates:
        raise ValueError(f"Cannot find GLS score column. Available: {list(gls.columns)}")
    gls_col = gls_col_candidates[0]

    label_col_candidates = [c for c in gls.columns if c.strip().lower() == 'global label']
    if not label_col_candidates:
        label_col_candidates = [c for c in gls.columns if 'global' in c.lower() and 'label' in c.lower()
                                and 'score' not in c.lower()]
    if not label_col_candidates:
        raise ValueError(f"Cannot find Global label column. Available: {list(gls.columns)}")
    label_col = label_col_candidates[0]

    print(f"  GLS score column: '{gls_col}', Label column: '{label_col}'")

    # Filter by GLS
    gls_filtered = gls[gls[gls_col] >= min_gls].copy()
    print(f"  After GLS >= {min_gls}: {len(gls_filtered)}")

    # Remove 'common' and 'plasma prot.' labels
    gls_filtered = gls_filtered[~gls_filtered[label_col].isin(['common', 'plasma prot.', None])]
    gls_filtered = gls_filtered[gls_filtered[label_col].notna()]
    print(f"  After removing common/plasma: {len(gls_filtered)}")

    # Classify as tissue or cell
    cell_labels = {'neutrophils', 'macrophages', 'monocytes', 'platelets',
                   'erythrocytes', 'bcell', 'tcellcd4', 'tcellcd8'}

    def is_cell_label(label):
        """Check if label is cell-type (including multi-cell)."""
        parts = str(label).split('.')
        return all(p in cell_labels for p in parts)

    def is_multilabel(label):
        return '.' in str(label)

    # Apply filters
    if not include_cells:
        mask = ~gls_filtered[label_col].apply(is_cell_label)
        n_removed = (~mask).sum()
        gls_filtered = gls_filtered[mask]
        print(f"  Removed {n_removed} cell-type entries (use --gls_include_cells to keep)")

    if not include_multilabel:
        mask = ~gls_filtered[label_col].apply(is_multilabel)
        n_removed = (~mask).sum()
        gls_filtered = gls_filtered[mask]
        print(f"  Removed {n_removed} multi-label entries")

    # Map to organ systems
    assignments = []
    unmapped_labels = set()

    for _, row in gls_filtered.iterrows():
        label = str(row[label_col]).strip()
        organ = GLS_ORGAN_MAP.get(label)

        if organ is None:
            # Try first part of multi-label
            if '.' in label:
                first = label.split('.')[0]
                organ = GLS_ORGAN_MAP.get(first, first.capitalize())
            else:
                organ = label.capitalize()
            if label not in GLS_ORGAN_MAP:
                unmapped_labels.add(label)

        # Determine specificity equivalent based on GLS
        gls_score = row[gls_col]
        if gls_score >= 4:
            specificity = 'Tissue enriched (GLS4)'
        elif gls_score >= 3:
            specificity = 'Tissue enriched (GLS3)'
        elif gls_score >= 2:
            specificity = 'Group enriched (GLS2)'
        else:
            specificity = 'Tissue enhanced (GLS1)'

        assignments.append({
            'uniprot': str(row['uniprot']).strip(),
            'gene': str(row['name']).strip() if pd.notna(row.get('name')) else '',
            'organ_system': organ,
            'gls': gls_score,
            'global_label': label,
            'specificity': specificity,
            'primary_tissue': label.split('.')[0] if '.' in label else label,
            'top_tissue': label,
        })

    assign_df = pd.DataFrame(assignments)
    print(f"\n  Proteins with organ assignments: {len(assign_df)}")

    if unmapped_labels:
        print(f"  Note: {len(unmapped_labels)} labels mapped by fallback: "
              f"{', '.join(sorted(unmapped_labels)[:10])}")

    # Summary
    organ_counts = assign_df['organ_system'].value_counts()
    print(f"  Organ systems: {len(organ_counts)}")
    for organ in organ_counts.head(15).index:
        gls_range = assign_df[assign_df['organ_system'] == organ]['gls']
        print(f"    {organ}: {organ_counts[organ]} proteins "
              f"(GLS {gls_range.min():.1f}-{gls_range.max():.1f})")
    if len(organ_counts) > 15:
        print(f"    ... and {len(organ_counts) - 15} more organs")

    return assign_df


# ──────────────────────────────────────────────────────────────────────
# 1b. IMPUTATION
# ──────────────────────────────────────────────────────────────────────

def impute_missing(df, method='none', intensity_col='log_intensity', protein_col='protein'):
    """
    Impute missing intensity values.
    Methods:
      none:              no imputation (default)
      half_min_protein:  half of the minimum observed value per protein
      min_protein:       minimum observed value per protein
      half_min_global:   half of the global minimum
      percentile:        half of the 5th percentile per protein
    """
    if method == 'none':
        return df

    df = df.copy()
    before = df[intensity_col].isna().sum()
    print(f"  Imputation ({method}): {before:,} missing values")

    if method == 'half_min_protein':
        def _impute(group):
            min_val = group[intensity_col].min()
            if pd.isna(min_val):
                min_val = df[intensity_col].min()
            group[intensity_col] = group[intensity_col].fillna(min_val * 0.5)
            return group
        df = df.groupby(protein_col, group_keys=False).apply(_impute)

    elif method == 'min_protein':
        protein_min = df.groupby(protein_col)[intensity_col].transform('min')
        df[intensity_col] = df[intensity_col].fillna(protein_min)

    elif method == 'half_min_global':
        global_min = df[intensity_col].min()
        df[intensity_col] = df[intensity_col].fillna(global_min * 0.5)

    elif method == 'percentile':
        def _impute(group):
            p5 = group[intensity_col].quantile(0.05)
            if pd.isna(p5):
                p5 = df[intensity_col].quantile(0.05)
            group[intensity_col] = group[intensity_col].fillna(p5 * 0.5)
            return group
        df = df.groupby(protein_col, group_keys=False).apply(_impute)

    after = df[intensity_col].isna().sum()
    print(f"  Imputed: {before - after:,} values, remaining NaN: {after:,}")
    return df


# ──────────────────────────────────────────────────────────────────────
# 2. COMPUTE ORGAN SCORES
# ──────────────────────────────────────────────────────────────────────

def compute_organ_scores(data_df, assignments, timepoints, protein_col='protein',
                          patient_col='patient', intensity_col='log_intensity',
                          min_proteins_per_organ=3):
    """
    For each patient x timepoint, compute organ scores.

    organ_score = mean(log2_intensity of all proteins assigned to that organ)

    Also computes fold-change relative to first timepoint.
    """
    # Map protein -> organ
    protein_to_organ = dict(zip(assignments['uniprot'], assignments['organ_system']))
    protein_to_gene = dict(zip(assignments['uniprot'], assignments['gene']))

    # Filter data to assigned proteins
    data_df = data_df.copy()
    data_df['organ_system'] = data_df[protein_col].map(protein_to_organ)
    data_df['gene'] = data_df[protein_col].map(protein_to_gene)
    matched = data_df[data_df['organ_system'].notna()].copy()

    n_matched = matched[protein_col].nunique()
    n_total = data_df[protein_col].nunique()
    print(f"  Proteins matched to organs: {n_matched} / {n_total}")

    # Check which organs have enough proteins
    organ_protein_counts = matched.groupby('organ_system')[protein_col].nunique()
    valid_organs = organ_protein_counts[organ_protein_counts >= min_proteins_per_organ].index.tolist()
    print(f"  Organs with >= {min_proteins_per_organ} proteins: {len(valid_organs)}")
    for organ in sorted(valid_organs):
        print(f"    {organ}: {organ_protein_counts[organ]} proteins")

    excluded_organs = organ_protein_counts[organ_protein_counts < min_proteins_per_organ]
    if not excluded_organs.empty:
        print(f"  Excluded (too few proteins):")
        for organ, count in excluded_organs.items():
            print(f"    {organ}: {count} proteins")

    matched = matched[matched['organ_system'].isin(valid_organs)]

    # Compute organ scores per patient per timepoint per organ
    scores = matched.groupby([patient_col, 'timepoint', 'organ_system']).agg(
        organ_score_mean=(intensity_col, 'mean'),
        organ_score_median=(intensity_col, 'median'),
        organ_score_sum=(intensity_col, 'sum'),
        n_proteins_detected=(protein_col, 'nunique'),
        organ_score_std=(intensity_col, 'std'),
    ).reset_index()

    # Keep backward compatibility
    scores['organ_score'] = scores['organ_score_mean']

    # Add timepoint order
    tp_order = {tp: i for i, tp in enumerate(timepoints)}
    scores['tp_idx'] = scores['timepoint'].map(tp_order)
    scores = scores.sort_values([patient_col, 'organ_system', 'tp_idx'])

    # Compute fold-change relative to first timepoint (per patient per organ)
    base_tp = timepoints[0]
    base_scores = scores[scores['timepoint'] == base_tp][
        [patient_col, 'organ_system', 'organ_score']].rename(
        columns={'organ_score': 'base_score'})

    scores = scores.merge(base_scores, on=[patient_col, 'organ_system'], how='left')
    scores['organ_fc'] = scores['organ_score'] - scores['base_score']  # log2 FC (mean-based)

    # Sum-based fold-change
    base_sums = scores[scores['timepoint'] == base_tp][
        [patient_col, 'organ_system', 'organ_score_sum']].rename(
        columns={'organ_score_sum': 'base_sum'})
    scores = scores.merge(base_sums, on=[patient_col, 'organ_system'], how='left')
    scores['organ_sum_fc'] = scores['organ_score_sum'] - scores['base_sum']

    return scores, matched


def compute_organ_score_summary(scores, timepoints, patient_col='patient'):
    """Compute group-level statistics per organ per timepoint."""
    summary = scores.groupby(['organ_system', 'timepoint']).agg(
        mean_score=('organ_score', 'mean'),
        median_score=('organ_score', 'median'),
        std_score=('organ_score', 'std'),
        mean_sum=('organ_score_sum', 'mean'),
        median_sum=('organ_score_sum', 'median'),
        mean_fc=('organ_fc', 'mean'),
        median_fc=('organ_fc', 'median'),
        std_fc=('organ_fc', 'std'),
        mean_sum_fc=('organ_sum_fc', 'mean'),
        n_patients=(patient_col, 'nunique'),
    ).reset_index()

    tp_order = {tp: i for i, tp in enumerate(timepoints)}
    summary['tp_idx'] = summary['timepoint'].map(tp_order)
    summary = summary.sort_values(['organ_system', 'tp_idx'])

    return summary


# ──────────────────────────────────────────────────────────────────────
# 3. STATISTICAL TESTS
# ──────────────────────────────────────────────────────────────────────

def test_organ_temporal_changes(scores, timepoints, patient_col='patient'):
    """
    For each organ system, test if organ score changes significantly over time.

    Strategy (priority order):
      1. LMM: organ_score ~ C(timepoint) + (1|patient)  [handles missing, uses all data]
      2. Friedman: patients with ALL timepoints (paired non-parametric)
      3. Kruskal-Wallis: unpaired comparison across timepoints (fallback)
      4. Pairwise Wilcoxon: base vs each later timepoint (supplementary)
    """
    # Try importing statsmodels for LMM
    try:
        from statsmodels.formula.api import mixedlm
        has_lmm = True
    except ImportError:
        print("  WARNING: statsmodels not available, using non-parametric tests only")
        has_lmm = False

    organs = sorted(scores['organ_system'].unique())
    results = []
    tp_order = {tp: i for i, tp in enumerate(timepoints)}

    for organ in organs:
        organ_data = scores[scores['organ_system'] == organ].copy()

        # Build pivot for non-parametric tests
        pivot = organ_data.pivot_table(
            index=patient_col, columns='timepoint',
            values='organ_score', aggfunc='mean')

        n_per_tp = {tp: pivot[tp].notna().sum() for tp in timepoints if tp in pivot.columns}

        # --- Test 1: LMM (primary) ---
        lmm_p, lmm_stat = np.nan, np.nan
        lmm_converged = False
        lmm_r2 = np.nan
        tp_coefficients = {}

        if has_lmm:
            lmm_data = organ_data[[patient_col, 'timepoint', 'organ_score']].dropna().copy()
            lmm_data['tp_idx'] = lmm_data['timepoint'].map(tp_order)
            if (lmm_data['timepoint'].nunique() >= 2 and
                lmm_data[patient_col].nunique() >= 5 and
                len(lmm_data) >= 10):
                try:
                    model = mixedlm(
                        formula="organ_score ~ C(timepoint)",
                        data=lmm_data,
                        groups=lmm_data[patient_col],
                        re_formula="1")
                    result = model.fit(reml=True, method='bfgs')
                    lmm_converged = result.converged

                    try:
                        wald = result.wald_test_terms(skip_single=False)
                        lmm_p = wald.loc['C(timepoint)', 'pvalue']
                        lmm_stat = wald.loc['C(timepoint)', 'statistic']
                    except Exception:
                        tp_params = [p for p in result.pvalues.index if 'timepoint' in p]
                        if tp_params:
                            lmm_p = result.pvalues[tp_params].min()

                    for tp in timepoints[1:]:
                        param = f'C(timepoint)[T.{tp}]'
                        if param in result.params.index:
                            tp_coefficients[f'lmm_coef_{tp}'] = float(result.params[param])
                            tp_coefficients[f'lmm_pval_{tp}'] = float(result.pvalues[param])

                    fixed_var = result.fittedvalues.var()
                    random_var = float(result.cov_re.iloc[0, 0])
                    resid_var = result.scale
                    total = fixed_var + random_var + resid_var
                    lmm_r2 = float(np.clip(fixed_var / total, 0, 1)) if total > 0 else 0
                except Exception:
                    pass

        # --- Test 2: Friedman (patients with ALL timepoints) ---
        friedman_stat, friedman_p = np.nan, np.nan
        n_complete = 0
        available_tps = [tp for tp in timepoints if tp in pivot.columns]
        if len(available_tps) >= 3:
            pivot_complete = pivot[available_tps].dropna()
            n_complete = len(pivot_complete)
            if n_complete >= 5:
                try:
                    groups = [pivot_complete[tp].values for tp in available_tps]
                    friedman_stat, friedman_p = stats.friedmanchisquare(*groups)
                except Exception:
                    pass

        # --- Test 3: Kruskal-Wallis (unpaired, all data) ---
        kw_stat, kw_p = np.nan, np.nan
        groups_kw = []
        for tp in timepoints:
            if tp in pivot.columns:
                vals = pivot[tp].dropna().values
                if len(vals) >= 3:
                    groups_kw.append(vals)
        if len(groups_kw) >= 2:
            try:
                kw_stat, kw_p = stats.kruskal(*groups_kw)
            except Exception:
                pass

        # --- Test 4: Pairwise Wilcoxon ---
        base_tp = timepoints[0]
        pairwise = {}
        best_pairwise_p = np.nan
        for tp in timepoints[1:]:
            if base_tp in pivot.columns and tp in pivot.columns:
                paired = pivot[[base_tp, tp]].dropna()
                if len(paired) >= 5:
                    try:
                        w_stat, w_p = stats.wilcoxon(paired[base_tp].values, paired[tp].values)
                        fc = paired[tp].mean() - paired[base_tp].mean()
                        pairwise[f'p_{base_tp}_vs_{tp}'] = w_p
                        pairwise[f'fc_{base_tp}_vs_{tp}'] = fc
                        pairwise[f'n_{base_tp}_vs_{tp}'] = len(paired)
                        if np.isnan(best_pairwise_p) or w_p < best_pairwise_p:
                            best_pairwise_p = w_p
                    except Exception:
                        pairwise[f'p_{base_tp}_vs_{tp}'] = np.nan
                        pairwise[f'fc_{base_tp}_vs_{tp}'] = np.nan

        # --- Choose primary p-value (LMM preferred) ---
        if not np.isnan(lmm_p) and lmm_converged:
            primary_p = lmm_p
            primary_test = 'LMM'
        elif not np.isnan(friedman_p):
            primary_p = friedman_p
            primary_test = 'Friedman'
        elif not np.isnan(kw_p):
            primary_p = kw_p
            primary_test = 'Kruskal-Wallis'
        else:
            primary_p = best_pairwise_p
            primary_test = 'Wilcoxon_best_pair'

        # Effect size
        kendall_w = np.nan
        if not np.isnan(friedman_stat) and n_complete > 0:
            k = len(available_tps)
            kendall_w = friedman_stat / (n_complete * (k - 1))

        # Direction
        first_vals = pivot[timepoints[0]].dropna() if timepoints[0] in pivot.columns else pd.Series()
        last_tp_list = [tp for tp in reversed(timepoints) if tp in pivot.columns]
        last_vals = pivot[last_tp_list[0]].dropna() if last_tp_list else pd.Series()
        if len(first_vals) > 0 and len(last_vals) > 0:
            fc = last_vals.mean() - first_vals.mean()
            direction = 'increasing' if fc > 0.01 else 'decreasing' if fc < -0.01 else 'stable'
        else:
            fc = np.nan
            direction = 'unknown'

        results.append({
            'organ_system': organ,
            'primary_test': primary_test,
            'primary_p': primary_p,
            'lmm_p': lmm_p,
            'lmm_converged': lmm_converged,
            'lmm_r2_marginal': lmm_r2,
            'friedman_stat': friedman_stat,
            'friedman_p': friedman_p,
            'friedman_n_complete': n_complete,
            'kruskal_wallis_stat': kw_stat,
            'kruskal_wallis_p': kw_p,
            'best_pairwise_p': best_pairwise_p,
            'kendall_w': kendall_w,
            'direction': direction,
            'overall_fc': fc,
            'base_mean': first_vals.mean() if len(first_vals) > 0 else np.nan,
            'final_mean': last_vals.mean() if len(last_vals) > 0 else np.nan,
            **{f'n_patients_{tp}': n_per_tp.get(tp, 0) for tp in timepoints},
            **tp_coefficients,
            **pairwise,
        })

    result_df = pd.DataFrame(results)

    # FDR correction on primary p-value
    if not result_df.empty:
        valid = result_df['primary_p'].notna()
        if valid.sum() > 0:
            from statsmodels.stats.multitest import multipletests
            pvals = result_df.loc[valid, 'primary_p'].values
            _, qvals, _, _ = multipletests(pvals, method='fdr_bh')
            result_df.loc[valid, 'primary_q'] = qvals

    return result_df


def test_organ_cluster_interaction(scores, timepoints, patient_col='patient',
                                    fc_col='organ_sum_fc'):
    """
    For each organ, test if temporal trajectory differs between patient clusters.
    Uses Kruskal-Wallis on fold-changes per timepoint.

    fc_col: 'organ_fc' (mean-based) or 'organ_sum_fc' (sum-based, default).
      Sum-based captures both intensity changes and detection rate changes,
      making it more sensitive for cluster comparisons.
    """
    if 'cluster' not in scores.columns:
        return pd.DataFrame()

    # Fallback if requested column doesn't exist
    if fc_col not in scores.columns:
        fc_col = 'organ_fc'

    organs = sorted(scores['organ_system'].unique())
    clusters = sorted(scores['cluster'].dropna().unique())

    if len(clusters) < 2:
        return pd.DataFrame()

    results = []
    for organ in organs:
        organ_data = scores[scores['organ_system'] == organ]

        for tp in timepoints[1:]:  # Skip baseline (FC=0)
            tp_data = organ_data[organ_data['timepoint'] == tp]
            groups = [tp_data[tp_data['cluster'] == cl][fc_col].dropna().values
                     for cl in clusters]
            groups = [g for g in groups if len(g) >= 2]

            if len(groups) < 2:
                continue

            try:
                h_stat, kw_p = stats.kruskal(*groups)
            except Exception:
                continue

            results.append({
                'organ_system': organ,
                'timepoint': tp,
                'kruskal_h': h_stat,
                'kruskal_p': kw_p,
                'fc_metric': fc_col,
                **{f'cluster_{cl}_mean_fc': tp_data[tp_data['cluster'] == cl][fc_col].mean()
                   for cl in clusters},
            })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────
# 4. VISUALIZATION
# ──────────────────────────────────────────────────────────────────────

def plot_organ_trajectories(summary, test_results, timepoints, outpath):
    """Line plot: organ scores over time, one line per organ system."""
    organs = sorted(summary['organ_system'].unique())
    n_organs = len(organs)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, n_organs))

    for i, organ in enumerate(organs):
        organ_data = summary[summary['organ_system'] == organ].sort_values('tp_idx')

        # Check significance
        if not test_results.empty:
            organ_test = test_results[test_results['organ_system'] == organ]
            is_sig = not organ_test.empty and organ_test.iloc[0].get('primary_q', 1) < 0.05
        else:
            is_sig = False

        lw = 2.5 if is_sig else 1.2
        alpha = 1.0 if is_sig else 0.5
        marker = 'o' if is_sig else '.'
        label = f"{organ} *" if is_sig else organ

        ax.plot(organ_data['tp_idx'], organ_data['mean_fc'],
                color=colors[i], linewidth=lw, alpha=alpha, marker=marker,
                markersize=6 if is_sig else 3, label=label)

        # Error bars for significant organs
        if is_sig:
            ax.fill_between(organ_data['tp_idx'],
                           organ_data['mean_fc'] - organ_data['std_fc'] / np.sqrt(organ_data['n_patients']),
                           organ_data['mean_fc'] + organ_data['std_fc'] / np.sqrt(organ_data['n_patients']),
                           color=colors[i], alpha=0.15)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(range(len(timepoints)))
    ax.set_xticklabels(timepoints, fontsize=11)
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Mean log2 fold-change from baseline', fontsize=12)
    ax.set_title('Organ-Level Protein Trajectories After Cardiac Arrest', fontsize=14,
                fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_organ_heatmap(summary, timepoints, outpath):
    """Heatmap: organs x timepoints, values = mean fold-change."""
    pivot = summary.pivot_table(index='organ_system', columns='timepoint',
                                 values='mean_fc', aggfunc='mean')
    # Reorder columns by timepoint
    pivot = pivot[[tp for tp in timepoints if tp in pivot.columns]]

    fig, ax = plt.subplots(figsize=(max(8, len(timepoints) * 2), max(6, len(pivot) * 0.5)))
    sns.heatmap(pivot, cmap='RdBu_r', center=0, ax=ax, linewidths=0.5,
               annot=True, fmt='.3f', annot_kws={'fontsize': 9},
               cbar_kws={'label': 'Mean log2 FC from baseline'})
    ax.set_title('Organ Score Changes Over Time (Mean)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_organ_sum_heatmap(summary, timepoints, outpath):
    """Heatmap: organs x timepoints, values = mean sum log2 intensity."""
    pivot = summary.pivot_table(index='organ_system', columns='timepoint',
                                 values='mean_sum', aggfunc='mean')
    pivot = pivot[[tp for tp in timepoints if tp in pivot.columns]]

    fig, ax = plt.subplots(figsize=(max(8, len(timepoints) * 2), max(6, len(pivot) * 0.5)))
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.5,
               annot=True, fmt='.0f', annot_kws={'fontsize': 9},
               cbar_kws={'label': 'Mean sum log2 intensity'})
    ax.set_title('Organ Total Protein Signal Over Time', fontsize=13, fontweight='bold')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_organ_sum_heatmap_fc(summary, timepoints, outpath):
    """Heatmap: organs x timepoints, values = mean sum fold-change."""
    pivot = summary.pivot_table(index='organ_system', columns='timepoint',
                                 values='mean_sum_fc', aggfunc='mean')
    pivot = pivot[[tp for tp in timepoints if tp in pivot.columns]]

    fig, ax = plt.subplots(figsize=(max(8, len(timepoints) * 2), max(6, len(pivot) * 0.5)))
    sns.heatmap(pivot, cmap='RdBu_r', center=0, ax=ax, linewidths=0.5,
               annot=True, fmt='.1f', annot_kws={'fontsize': 9},
               cbar_kws={'label': 'Mean sum log2 FC from baseline'})
    ax.set_title('Organ Total Signal Change Over Time', fontsize=13, fontweight='bold')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_organ_sum_trajectories(summary, timepoints, outpath):
    """Line plot: sum log2 intensity over time per organ."""
    organs = sorted(summary['organ_system'].unique())
    n_organs = len(organs)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, n_organs))

    for i, organ in enumerate(organs):
        organ_data = summary[summary['organ_system'] == organ].sort_values('tp_idx')
        ax.plot(organ_data['tp_idx'], organ_data['mean_sum'],
                color=colors[i], linewidth=2, marker='o', markersize=5, label=organ)

    ax.set_xticks(range(len(timepoints)))
    ax.set_xticklabels(timepoints, fontsize=11)
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Mean sum log2 intensity', fontsize=12)
    ax.set_title('Organ Total Protein Signal Over Time', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_organ_trajectories_by_cluster(scores, timepoints, outpath, top_organs=6):
    """Faceted plot: one panel per organ, lines colored by cluster."""
    if 'cluster' not in scores.columns:
        return

    clusters = sorted(scores['cluster'].dropna().unique())

    # Select top organs (by variance of FC)
    organ_var = scores.groupby('organ_system')['organ_fc'].var().sort_values(ascending=False)
    top = organ_var.head(top_organs).index.tolist()

    n_cols = min(3, len(top))
    n_rows = (len(top) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    if len(top) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    tp_order = {tp: i for i, tp in enumerate(timepoints)}
    colors = plt.cm.Set1(np.linspace(0, 1, len(clusters)))

    for idx, organ in enumerate(top):
        ax = axes[idx]
        organ_data = scores[scores['organ_system'] == organ]

        for ci, cl in enumerate(clusters):
            cl_data = organ_data[organ_data['cluster'] == cl]
            cl_summary = cl_data.groupby('timepoint').agg(
                mean_fc=('organ_fc', 'mean'),
                se_fc=('organ_fc', lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0),
                n=('organ_fc', 'count')
            ).reset_index()
            cl_summary['tp_idx'] = cl_summary['timepoint'].map(tp_order)
            cl_summary = cl_summary.sort_values('tp_idx')

            ax.plot(cl_summary['tp_idx'], cl_summary['mean_fc'],
                   color=colors[ci], linewidth=2, marker='o', markersize=5,
                   label=f'Cl {int(cl)} (n={cl_data["patient"].nunique()})')
            ax.fill_between(cl_summary['tp_idx'],
                           cl_summary['mean_fc'] - cl_summary['se_fc'],
                           cl_summary['mean_fc'] + cl_summary['se_fc'],
                           color=colors[ci], alpha=0.15)

        ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xticks(range(len(timepoints)))
        ax.set_xticklabels(timepoints, fontsize=9)
        ax.set_title(organ, fontsize=11, fontweight='bold')
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7)

    for i in range(len(top), len(axes)):
        axes[i].set_visible(False)

    fig.text(0.04, 0.5, 'Mean log2 FC from baseline', va='center', rotation='vertical', fontsize=12)
    plt.suptitle('Organ Trajectories by Patient Cluster', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_individual_patient_organs(scores, timepoints, outdir, patient_col='patient'):
    """
    Heatmap per patient: 3 panels
      1. Mean log2 FC from baseline
      2. Sum log2 intensity (absolute)
      3. Sum log2 intensity FC from baseline
    """
    pat_dir = os.path.join(outdir, 'individual_patients')
    os.makedirs(pat_dir, exist_ok=True)

    all_patients = sorted(scores[patient_col].unique())
    print(f"  Generating individual patient heatmaps ({len(all_patients)} patients)...")

    base_tp = timepoints[0]

    for pat in all_patients:
        pat_data = scores[scores[patient_col] == pat]

        # Panel 1: Mean FC
        pivot_fc = pat_data.pivot_table(index='organ_system', columns='timepoint',
                                         values='organ_fc', aggfunc='mean')
        pivot_fc = pivot_fc[[tp for tp in timepoints if tp in pivot_fc.columns]]

        if pivot_fc.empty:
            continue

        # Panel 2: Sum intensity (absolute)
        pivot_sum = pat_data.pivot_table(index='organ_system', columns='timepoint',
                                          values='organ_score_sum', aggfunc='mean')
        pivot_sum = pivot_sum[[tp for tp in timepoints if tp in pivot_sum.columns]]

        # Panel 3: Sum intensity FC from baseline
        pivot_sum_fc = pivot_sum.copy()
        if base_tp in pivot_sum_fc.columns:
            for tp in pivot_sum_fc.columns:
                if tp != base_tp:
                    pivot_sum_fc[tp] = pivot_sum_fc[tp] - pivot_sum_fc[base_tp]
            pivot_sum_fc[base_tp] = 0

        # Plot 3 panels
        fig_width = max(6, len(timepoints) * 1.5) * 3
        fig_height = max(4, len(pivot_fc) * 0.4)
        fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))

        # Panel 1: Mean FC
        ax = axes[0]
        sns.heatmap(pivot_fc, cmap='RdBu_r', center=0, ax=ax, linewidths=0.5,
                   annot=True, fmt='.2f', annot_kws={'fontsize': 7},
                   cbar_kws={'label': 'log2 FC', 'shrink': 0.8})
        ax.set_title('Mean log2 FC', fontsize=11, fontweight='bold')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('')

        # Panel 2: Sum intensity
        ax = axes[1]
        sns.heatmap(pivot_sum, cmap='YlOrRd', ax=ax, linewidths=0.5,
                   annot=True, fmt='.0f', annot_kws={'fontsize': 7},
                   cbar_kws={'label': 'Sum log2 intensity', 'shrink': 0.8})
        ax.set_title('Sum Intensity', fontsize=11, fontweight='bold')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('')

        # Panel 3: Sum FC
        ax = axes[2]
        max_abs = max(pivot_sum_fc.abs().max().max(), 1)
        sns.heatmap(pivot_sum_fc, cmap='RdBu_r', center=0, ax=ax, linewidths=0.5,
                   annot=True, fmt='.0f', annot_kws={'fontsize': 7},
                   vmin=-max_abs, vmax=max_abs,
                   cbar_kws={'label': 'Sum FC', 'shrink': 0.8})
        ax.set_title('Sum Intensity Change', fontsize=11, fontweight='bold')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('')

        plt.suptitle(f'Patient {pat}: Organ Trajectories', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(pat_dir, f'patient_{pat}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(all_patients)} patient heatmaps in: {pat_dir}/")


def plot_individual_patient_protein_counts(count_data_all, count_data_stable,
                                            timepoints, outdir,
                                            protein_col='protein', patient_col='patient'):
    """
    Per-patient heatmap: number of proteins detected per organ per timepoint.
    4 panels: All proteins (count + change) | Stable proteins (count + change)
    If count_data_stable is None, falls back to 2-panel (all only).
    """
    pat_dir = os.path.join(outdir, 'individual_patients_counts')
    os.makedirs(pat_dir, exist_ok=True)

    def add_detected(df):
        df = df.copy()
        int_col = 'log_intensity' if 'log_intensity' in df.columns else 'raw_intensity'
        if int_col in df.columns:
            df['detected'] = df[int_col].notna().astype(int)
        else:
            df['detected'] = 1
        return df

    count_data_all = add_detected(count_data_all)
    has_stable = count_data_stable is not None
    if has_stable:
        count_data_stable = add_detected(count_data_stable)

    all_patients = sorted(count_data_all[patient_col].unique())
    n_all_prots = count_data_all[protein_col].nunique()
    n_stable_prots = count_data_stable[protein_col].nunique() if has_stable else 0
    print(f"  Generating protein count heatmaps ({len(all_patients)} patients)...")
    print(f"    All proteins: {n_all_prots}")
    if has_stable:
        print(f"    Stable proteins: {n_stable_prots}")

    base_tp = timepoints[0]
    n_cols = 4 if has_stable else 2

    for pat in all_patients:
        pat_all = count_data_all[count_data_all[patient_col] == pat]

        # All proteins: count per organ per timepoint
        counts_all = pat_all.groupby(['timepoint', 'organ_system'])['detected'].sum().reset_index()
        pivot_all = counts_all.pivot_table(index='organ_system', columns='timepoint',
                                            values='detected', aggfunc='sum')
        pivot_all = pivot_all[[tp for tp in timepoints if tp in pivot_all.columns]]
        pivot_all = pivot_all.fillna(0).astype(int)

        if pivot_all.empty:
            continue

        # All proteins: change from baseline
        pivot_all_fc = pivot_all.copy().astype(float)
        if base_tp in pivot_all_fc.columns:
            for tp in pivot_all_fc.columns:
                if tp != base_tp:
                    pivot_all_fc[tp] = pivot_all_fc[tp] - pivot_all_fc[base_tp]
            pivot_all_fc[base_tp] = 0

        # Stable proteins
        pivot_stable = None
        pivot_stable_fc = None
        if has_stable:
            pat_stable = count_data_stable[count_data_stable[patient_col] == pat]
            if not pat_stable.empty:
                counts_stable = pat_stable.groupby(['timepoint', 'organ_system'])['detected'].sum().reset_index()
                pivot_stable = counts_stable.pivot_table(index='organ_system', columns='timepoint',
                                                          values='detected', aggfunc='sum')
                pivot_stable = pivot_stable[[tp for tp in timepoints if tp in pivot_stable.columns]]
                pivot_stable = pivot_stable.fillna(0).astype(int)

                pivot_stable_fc = pivot_stable.copy().astype(float)
                if base_tp in pivot_stable_fc.columns:
                    for tp in pivot_stable_fc.columns:
                        if tp != base_tp:
                            pivot_stable_fc[tp] = pivot_stable_fc[tp] - pivot_stable_fc[base_tp]
                    pivot_stable_fc[base_tp] = 0

        # Plot
        fig_height = max(5, len(pivot_all) * 0.4)
        fig_width = max(6, len(timepoints) * 1.5) * n_cols
        fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height))
        if n_cols == 1:
            axes = [axes]

        # Panel 1: All proteins — absolute counts
        ax = axes[0]
        sns.heatmap(pivot_all, cmap='YlOrRd', ax=ax, linewidths=0.5,
                   annot=True, fmt='d', annot_kws={'fontsize': 7},
                   cbar_kws={'label': 'Count', 'shrink': 0.8})
        ax.set_title(f'All Proteins Detected\n({n_all_prots} total)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('')

        # Panel 2: All proteins — change from baseline
        ax = axes[1]
        max_abs = max(pivot_all_fc.abs().max().max(), 1)
        sns.heatmap(pivot_all_fc, cmap='RdBu_r', center=0, ax=ax, linewidths=0.5,
                   annot=True, fmt='.0f', annot_kws={'fontsize': 7},
                   vmin=-max_abs, vmax=max_abs,
                   cbar_kws={'label': 'Change', 'shrink': 0.8})
        ax.set_title('All: Change from Baseline', fontsize=10, fontweight='bold')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('')

        # Panel 3 & 4: Stable proteins
        if has_stable and pivot_stable is not None and not pivot_stable.empty:
            ax = axes[2]
            sns.heatmap(pivot_stable, cmap='YlOrRd', ax=ax, linewidths=0.5,
                       annot=True, fmt='d', annot_kws={'fontsize': 7},
                       cbar_kws={'label': 'Count', 'shrink': 0.8})
            ax.set_title(f'Stable Proteins Detected\n({n_stable_prots} total)', fontsize=10, fontweight='bold')
            ax.set_xlabel('Timepoint')
            ax.set_ylabel('')

            ax = axes[3]
            max_abs_s = max(pivot_stable_fc.abs().max().max(), 1)
            sns.heatmap(pivot_stable_fc, cmap='RdBu_r', center=0, ax=ax, linewidths=0.5,
                       annot=True, fmt='.0f', annot_kws={'fontsize': 7},
                       vmin=-max_abs_s, vmax=max_abs_s,
                       cbar_kws={'label': 'Change', 'shrink': 0.8})
            ax.set_title('Stable: Change from Baseline', fontsize=10, fontweight='bold')
            ax.set_xlabel('Timepoint')
            ax.set_ylabel('')
        elif has_stable:
            for i in [2, 3]:
                axes[i].set_visible(False)

        plt.suptitle(f'Patient {pat}: Organ Protein Counts', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(pat_dir, f'patient_{pat}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(all_patients)} count heatmaps in: {pat_dir}/")


def save_protein_level_detail(matched_data, outdir, protein_col='protein',
                               patient_col='patient', filename='protein_organ_detail.csv'):
    """Save full protein-level table with organ annotations."""
    detail_cols = [patient_col, 'timepoint', protein_col, 'gene',
                   'organ_system', 'primary_tissue']
    if 'specificity' in matched_data.columns:
        detail_cols.append('specificity')
    if 'raw_intensity' in matched_data.columns:
        detail_cols.append('raw_intensity')
    if 'log_intensity' in matched_data.columns:
        detail_cols.append('log_intensity')

    available = [c for c in detail_cols if c in matched_data.columns]
    detail = matched_data[available].sort_values(
        [patient_col, 'timepoint', 'organ_system', protein_col])
    if 'log_intensity' in detail.columns:
        detail['detected'] = detail['log_intensity'].notna().astype(int)
    elif 'raw_intensity' in detail.columns:
        detail['detected'] = detail['raw_intensity'].notna().astype(int)
    else:
        detail['detected'] = 1

    outpath = os.path.join(outdir, filename)
    detail.to_csv(outpath, index=False)
    print(f"  Saved: {filename} ({len(detail):,} rows, "
          f"{detail[protein_col].nunique()} proteins, {detail[patient_col].nunique()} patients)")

    # Also save group-level count summary
    counts = detail.groupby(['timepoint', 'organ_system']).agg(
        mean_detected=('detected', 'mean'),
        total_detected=('detected', 'sum'),
        n_measurements=('detected', 'count'),
    ).reset_index()
    counts.to_csv(os.path.join(outdir, 'organ_protein_counts_summary.csv'), index=False)
    print(f"  Saved: organ_protein_counts_summary.csv")

    # Group-level mean count heatmap
    mean_counts = detail.groupby([patient_col, 'timepoint', 'organ_system'])['detected'].sum().reset_index()
    mean_per_tp = mean_counts.groupby(['timepoint', 'organ_system'])['detected'].mean().reset_index()
    return detail


# ──────────────────────────────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Organ trajectory analysis: track organ-level protein signatures over time")

    parser.add_argument('--data', required=True,
                        help='Long-format proteomics data CSV')
    parser.add_argument('--timepoints', nargs='+', required=True,
                        help='Ordered timepoint labels')
    parser.add_argument('--hpa_summary', default=None,
                        help='proteinatlas.tsv (main HPA summary with tissue specificity)')
    parser.add_argument('--gls', default=None,
                        help='GLS atlas Excel file (mmc11.xlsx from Malmström et al. 2025). '
                             'Alternative to --hpa_summary for protein-organ assignments.')
    parser.add_argument('--min_gls', type=float, default=1.0,
                        help='Minimum GLS score (1-4, default: 1.0). '
                             'Higher = more confident: 2=moderate, 3=high, 4=highest.')
    parser.add_argument('--gls_include_cells', action='store_true',
                        help='Include cell-type labels (neutrophils, macrophages, etc.) from GLS')
    parser.add_argument('--hpa_rna', default=None,
                        help='rna_tissue_consensus.tsv (optional, for broader coverage)')
    parser.add_argument('--hpa_organ_map', default=None,
                        help='Tissue->organ mapping TSV')
    parser.add_argument('--contaminants', default=None,
                        help='Contaminant protein list')
    parser.add_argument('--stable_proteins', default=None,
                        help='detection_patterns.csv from detection analysis. '
                             'Filters to "Stable detected" proteins only.')
    parser.add_argument('--protein_whitelist', default=None,
                        help='Text file with protein IDs to include (one per line). '
                             'Use output from classify_proteins.py, e.g., '
                             'leakage_all_proteins.txt or organ_enriched_proteins.txt. '
                             'Applied AFTER --stable_proteins filter.')
    parser.add_argument('--clusters', default=None,
                        help='patient_clusters.csv')
    parser.add_argument('--protein_col', default='protein')
    parser.add_argument('--patient_col', default='patient')
    parser.add_argument('--intensity_col', default='log_intensity')
    parser.add_argument('--min_intensity', type=float, default=0,
                        help='Values <= this treated as not detected (default: 0)')
    parser.add_argument('--impute', choices=['none', 'half_min_protein', 'min_protein',
                                              'half_min_global', 'percentile'],
                        default='none',
                        help='Imputation method for missing values. '
                             'Applied after 0->NaN conversion, before organ score computation. '
                             'Options: none (default), half_min_protein, min_protein, '
                             'half_min_global, percentile')
    parser.add_argument('--min_proteins_per_organ', type=int, default=3,
                        help='Min proteins to compute organ score (default: 3)')
    parser.add_argument('--specificity', nargs='+',
                        default=['Tissue enriched', 'Group enriched', 'Tissue enhanced'],
                        help='HPA specificity categories to include')
    parser.add_argument('--organ_filter', nargs='*', default=None,
                        help='Run additional organ-specific analysis for these organs. '
                             'Uses only "Tissue enriched" proteins for that organ. '
                             'Example: --organ_filter "Liver & Gallbladder" "Brain" "Kidney & Urinary bladder"')
    parser.add_argument('--organ_specific_all', action='store_true',
                        help='Run organ-specific analysis for ALL organs (strict tissue-enriched only)')
    parser.add_argument('--count_specificity', nargs='*', default=None,
                        help='HPA specificity for count heatmaps (overrides --specificity for counts). '
                             'Example: --count_specificity "Tissue enriched" "Group enriched" "Tissue enhanced". '
                             'If not set, uses all three categories.')
    parser.add_argument('--outdir', default='organ_trajectory_results')

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Validate: need either --hpa_summary or --gls
    if args.hpa_summary is None and args.gls is None:
        parser.error("Provide either --hpa_summary or --gls for protein-organ assignments")
    use_gls = args.gls is not None

    print("=" * 70)
    print("ORGAN TRAJECTORY ANALYSIS")
    print("Track organ-level protein signatures over time")
    if use_gls:
        print(f"  Source: GLS atlas (Malmström et al. 2025), min_gls={args.min_gls}")
    else:
        print(f"  Source: HPA ({', '.join(args.specificity)})")
    print("=" * 70)

    # ── Load protein-to-organ assignments ─────────────────────────────
    if use_gls:
        print(f"\n[1] Assigning proteins to organ systems (GLS, min_gls={args.min_gls})...")
        assignments = load_gls_assignments(
            args.gls, min_gls=args.min_gls,
            include_cells=args.gls_include_cells,
            include_multilabel=True)
    else:
        print("\n[1] Assigning proteins to organ systems (HPA)...")
        assignments = load_hpa_tissue_assignments(
            args.hpa_summary, args.hpa_rna, args.hpa_organ_map, args.specificity)
    assignments.to_csv(os.path.join(args.outdir, 'protein_organ_assignments.csv'), index=False)

    if assignments.empty:
        print("ERROR: No proteins assigned to organs. Check HPA files.")
        return

    # ── Load proteomics data ──────────────────────────────────────────
    print("\n[2] Loading proteomics data...")
    data = pd.read_csv(args.data)
    print(f"  Rows: {len(data):,}, Proteins: {data[args.protein_col].nunique()}")
    print(f"  Data columns: {list(data.columns)}")

    # Auto-detect intensity column
    intensity_col = args.intensity_col
    if intensity_col not in data.columns:
        # Try common alternatives
        for alt in ['log_intensity', 'log2_intensity', 'LogIntensity',
                    'raw_intensity', 'intensity', 'Intensity', 'value']:
            if alt in data.columns:
                intensity_col = alt
                print(f"  Auto-detected intensity column: '{intensity_col}'")
                break

    # If we have raw_intensity but not log_intensity, create it
    if intensity_col == 'raw_intensity' or (intensity_col not in data.columns and 'raw_intensity' in data.columns):
        intensity_col = 'raw_intensity'
        print(f"  Creating log_intensity from raw_intensity...")
        data['raw_intensity'] = pd.to_numeric(data['raw_intensity'], errors='coerce')

        # Handle 0-filled data BEFORE log transform
        if args.min_intensity is not None:
            n_zeros = (data['raw_intensity'] <= args.min_intensity).sum()
            data.loc[data['raw_intensity'] <= args.min_intensity, 'raw_intensity'] = np.nan
            print(f"  Converted {n_zeros:,} values <= {args.min_intensity} to NaN")

        data['log_intensity'] = np.log2(data['raw_intensity'])
        intensity_col = 'log_intensity'
        print(f"  log_intensity created ({data['log_intensity'].notna().sum():,} valid values)")
    else:
        # Handle 0-filled data for log intensity columns
        if args.min_intensity is not None and intensity_col in data.columns:
            n_zeros = (data[intensity_col] <= args.min_intensity).sum()
            data.loc[data[intensity_col] <= args.min_intensity, intensity_col] = np.nan
            print(f"  Converted {n_zeros:,} values <= {args.min_intensity} to NaN")

    # Update intensity_col for downstream use
    args.intensity_col = intensity_col
    print(f"  Using intensity column: '{intensity_col}'")

    # Exclude contaminants
    if args.contaminants:
        sep = '\t' if args.contaminants.endswith('.tsv') else ','
        try:
            contam = pd.read_csv(args.contaminants, sep=sep)
        except:
            contam = pd.read_csv(args.contaminants, sep='\t')
        acc_col = contam.columns[0]
        for c in ['UniProt_Accession', 'uniprot', 'protein', 'Accession']:
            if c in contam.columns:
                acc_col = c
                break
        contam_ids = set(contam[acc_col].astype(str).str.strip())
        n_before = data[args.protein_col].nunique()
        data = data[~data[args.protein_col].isin(contam_ids)]
        print(f"  Contaminants removed: {n_before - data[args.protein_col].nunique()}")

    # Filter to stable detected proteins
    if args.stable_proteins:
        print(f"\n  Filtering to stable detected proteins...")
        det_patterns = pd.read_csv(args.stable_proteins)
        stable = det_patterns[det_patterns['pattern'] == 'Stable detected']
        stable_ids = set(stable['protein'].astype(str).str.strip())
        n_before = data[args.protein_col].nunique()
        data = data[data[args.protein_col].astype(str).str.strip().isin(stable_ids)]
        n_after = data[args.protein_col].nunique()
        print(f"  Stable detected proteins: {len(stable_ids)}")
        print(f"  Proteins in data after filter: {n_before} -> {n_after}")

    # Filter to protein whitelist (from classify_proteins.py)
    if args.protein_whitelist:
        print(f"\n  Applying protein whitelist: {args.protein_whitelist}")
        try:
            wl = pd.read_csv(args.protein_whitelist, header=None)
            whitelist_ids = set(wl.iloc[:, 0].astype(str).str.strip())
        except:
            with open(args.protein_whitelist) as f:
                whitelist_ids = set(line.strip() for line in f if line.strip())
        n_before = data[args.protein_col].nunique()
        data = data[data[args.protein_col].astype(str).str.strip().isin(whitelist_ids)]
        n_after = data[args.protein_col].nunique()
        print(f"  Whitelist proteins: {len(whitelist_ids)}")
        print(f"  Proteins in data after whitelist: {n_before} -> {n_after}")

    # Filter to valid timepoints
    data = data[data['timepoint'].isin(args.timepoints)]

    # Load clusters if provided
    if args.clusters:
        print("\n  Loading patient clusters...")
        clusters = pd.read_csv(args.clusters)
        print(f"  Cluster file columns: {list(clusters.columns)}")
        print(f"  Cluster file rows: {len(clusters)}")

        # Find cluster column
        cl_col = None
        for candidate in ['cluster', 'Cluster', 'CLUSTER']:
            if candidate in clusters.columns:
                cl_col = candidate
                break
        if cl_col is None:
            cl_cols = [c for c in clusters.columns if 'cluster' in c.lower()]
            cl_col = cl_cols[0] if cl_cols else clusters.columns[-1]

        # Find patient column in cluster file
        pat_col_cl = None
        for candidate in ['patient', 'Patient', 'patient_id', 'Patient_ID', 'ID', 'id', 'sample']:
            if candidate in clusters.columns:
                pat_col_cl = candidate
                break
        if pat_col_cl is None:
            pat_col_cl = clusters.columns[0]

        print(f"  Using patient column: '{pat_col_cl}', cluster column: '{cl_col}'")

        # Drop existing 'cluster' column from data if present (from previous analysis)
        if 'cluster' in data.columns:
            data = data.drop('cluster', axis=1)
            print(f"  Dropped existing 'cluster' column from data")

        # Ensure patient ID types match
        clusters[pat_col_cl] = clusters[pat_col_cl].astype(str).str.strip()
        data[args.patient_col] = data[args.patient_col].astype(str).str.strip()

        clusters_clean = clusters[[pat_col_cl, cl_col]].copy()
        clusters_clean = clusters_clean.rename(
            columns={pat_col_cl: args.patient_col, cl_col: 'cluster'})

        data = data.merge(clusters_clean, on=args.patient_col, how='left')

        if 'cluster' in data.columns:
            n_matched = data['cluster'].notna().sum()
            if n_matched == 0:
                print(f"  WARNING: No patients matched!")
                print(f"    Data patients (first 5): {list(data[args.patient_col].unique()[:5])}")
                print(f"    Cluster patients (first 5): {list(clusters_clean[args.patient_col].unique()[:5])}")
                data.drop('cluster', axis=1, inplace=True)
            else:
                print(f"  Matched: {n_matched}/{len(data)} rows")
                print(f"  Clusters: {data['cluster'].value_counts().to_dict()}")
        else:
            print(f"  WARNING: Merge failed to produce 'cluster' column")

    # ── Imputation ─────────────────────────────────────────────────────
    if args.impute != 'none':
        print(f"\n[2b] Imputing missing values ({args.impute})...")
        data = impute_missing(data, method=args.impute,
                              intensity_col=args.intensity_col,
                              protein_col=args.protein_col)

    # ── Compute organ scores ──────────────────────────────────────────
    print("\n[3] Computing organ scores per patient per timepoint...")
    scores, matched_data = compute_organ_scores(
        data, assignments, args.timepoints,
        protein_col=args.protein_col, patient_col=args.patient_col,
        intensity_col=args.intensity_col,
        min_proteins_per_organ=args.min_proteins_per_organ)

    scores.to_csv(os.path.join(args.outdir, 'organ_scores_per_patient.csv'), index=False)
    print(f"  Total organ score entries: {len(scores)}")

    # Add cluster to scores if available
    if 'cluster' in data.columns:
        cl_map = data[[args.patient_col, 'cluster']].drop_duplicates()
        scores = scores.merge(cl_map, on=args.patient_col, how='left')

    # Group-level summary
    summary = compute_organ_score_summary(scores, args.timepoints, args.patient_col)
    summary.to_csv(os.path.join(args.outdir, 'organ_score_summary.csv'), index=False)

    # ── Statistical tests ─────────────────────────────────────────────
    print("\n[4] Testing temporal changes per organ (Friedman test)...")
    test_results = test_organ_temporal_changes(scores, args.timepoints, args.patient_col)
    test_results.to_csv(os.path.join(args.outdir, 'organ_temporal_tests.csv'), index=False)

    if not test_results.empty:
        print(f"\n  Organ temporal change results:")
        for _, row in test_results.sort_values('primary_p').iterrows():
            q = row.get('primary_q', row['primary_p'])
            sig = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"
            print(f"    {row['organ_system']}: test={row['primary_test']}, p={row['primary_p']:.4e}, q={q:.4e}, "
                  f"{row['direction']} (FC={row['overall_fc']:.3f}) {sig}")

    # Cluster interaction test
    if 'cluster' in scores.columns:
        print("\n[4b] Testing organ x cluster interactions...")
        interaction_results = test_organ_cluster_interaction(
            scores, args.timepoints, args.patient_col)
        if not interaction_results.empty:
            interaction_results.to_csv(
                os.path.join(args.outdir, 'organ_cluster_interaction_tests.csv'), index=False)
            sig_interactions = interaction_results[interaction_results['kruskal_p'] < 0.05]
            print(f"  Significant organ x cluster interactions: {len(sig_interactions)}")

    # ── Visualization ─────────────────────────────────────────────────
    print("\n[5] Generating plots...")

    plot_organ_trajectories(summary, test_results, args.timepoints,
                            os.path.join(args.outdir, 'organ_trajectories.png'))

    plot_organ_heatmap(summary, args.timepoints,
                       os.path.join(args.outdir, 'organ_trajectory_heatmap.png'))

    # Sum-based plots
    plot_organ_sum_heatmap(summary, args.timepoints,
                           os.path.join(args.outdir, 'organ_sum_intensity_heatmap.png'))

    plot_organ_sum_heatmap_fc(summary, args.timepoints,
                              os.path.join(args.outdir, 'organ_sum_fc_heatmap.png'))

    plot_organ_sum_trajectories(summary, args.timepoints,
                                os.path.join(args.outdir, 'organ_sum_trajectories.png'))

    if 'cluster' in scores.columns:
        plot_organ_trajectories_by_cluster(scores, args.timepoints,
                                            os.path.join(args.outdir, 'organ_trajectories_by_cluster.png'))

    plot_individual_patient_organs(scores, args.timepoints, args.outdir,
                                   patient_col=args.patient_col)

    # Save sum intensity table (wide format: patients x organs x timepoints)
    sum_wide = scores.pivot_table(
        index=[args.patient_col, 'timepoint'],
        columns='organ_system',
        values='organ_score_sum',
        aggfunc='mean'
    ).reset_index()
    sum_wide.to_csv(os.path.join(args.outdir, 'organ_sum_intensity_wide.csv'), index=False)
    print(f"  Saved: organ_sum_intensity_wide.csv")

    # ── Protein-level detail table & count heatmaps ───────────────────
    print("\n[5b] Saving protein-level detail and count heatmaps...")

    # Add extra annotation columns to matched_data from assignments
    assign_lookup = assignments.set_index('uniprot')
    if 'primary_tissue' not in matched_data.columns and 'primary_tissue' in assign_lookup.columns:
        pt_map = assign_lookup['primary_tissue'].to_dict()
        matched_data['primary_tissue'] = matched_data[args.protein_col].astype(str).str.strip().map(pt_map)
    if 'specificity' not in matched_data.columns and 'specificity' in assign_lookup.columns:
        sp_map = assign_lookup['specificity'].to_dict()
        matched_data['specificity'] = matched_data[args.protein_col].astype(str).str.strip().map(sp_map)

    # Always save FC-matched protein detail
    save_protein_level_detail(matched_data, args.outdir,
                              protein_col=args.protein_col, patient_col=args.patient_col)

    # --- Build count_data_all: ALL proteins (reload raw, no stable filter) ---
    print("  Building count data: ALL proteins...")
    if use_gls:
        count_assignments = load_gls_assignments(
            args.gls, min_gls=1.0,
            include_cells=args.gls_include_cells,
            include_multilabel=True)
    else:
        count_specificity = args.count_specificity if args.count_specificity else \
            ['Tissue enriched', 'Group enriched', 'Tissue enhanced']
        count_assignments = load_hpa_tissue_assignments(
            args.hpa_summary, args.hpa_rna, args.hpa_organ_map,
            specificity_filter=count_specificity)
    count_protein_to_organ = dict(zip(count_assignments['uniprot'].astype(str).str.strip(),
                                      count_assignments['organ_system']))
    count_protein_to_gene = dict(zip(count_assignments['uniprot'].astype(str).str.strip(),
                                      count_assignments['gene']))

    count_data_all = pd.read_csv(args.data)
    count_data_all[args.patient_col] = count_data_all[args.patient_col].astype(str).str.strip()
    # Auto-detect intensity
    if 'log_intensity' not in count_data_all.columns and 'raw_intensity' in count_data_all.columns:
        count_data_all['raw_intensity'] = pd.to_numeric(count_data_all['raw_intensity'], errors='coerce')
        if args.min_intensity is not None:
            count_data_all.loc[count_data_all['raw_intensity'] <= args.min_intensity, 'raw_intensity'] = np.nan
        count_data_all['log_intensity'] = np.log2(count_data_all['raw_intensity'])
    elif 'log_intensity' in count_data_all.columns and args.min_intensity is not None:
        count_data_all.loc[count_data_all['log_intensity'] <= args.min_intensity, 'log_intensity'] = np.nan
    # Remove contaminants
    if args.contaminants:
        sep = '\t' if args.contaminants.endswith('.tsv') else ','
        try:
            contam_df = pd.read_csv(args.contaminants, sep=sep)
        except:
            contam_df = pd.read_csv(args.contaminants, sep='\t')
        acc_col = contam_df.columns[0]
        for c in ['UniProt_Accession', 'uniprot', 'protein', 'Accession']:
            if c in contam_df.columns:
                acc_col = c
                break
        contam_ids_count = set(contam_df[acc_col].astype(str).str.strip())
        count_data_all = count_data_all[~count_data_all[args.protein_col].isin(contam_ids_count)]
    count_data_all = count_data_all[count_data_all['timepoint'].isin(args.timepoints)]
    count_data_all['organ_system'] = count_data_all[args.protein_col].astype(str).str.strip().map(count_protein_to_organ)
    count_data_all['gene'] = count_data_all[args.protein_col].astype(str).str.strip().map(count_protein_to_gene)
    count_data_all = count_data_all[count_data_all['organ_system'].notna()]
    print(f"    All proteins with organ annotation: {count_data_all[args.protein_col].nunique()}")

    # Add specificity and tissue annotation to count_data_all
    count_assign_lookup = count_assignments.set_index('uniprot')
    if 'primary_tissue' in count_assign_lookup.columns:
        count_data_all['primary_tissue'] = count_data_all[args.protein_col].astype(str).str.strip().map(
            count_assign_lookup['primary_tissue'].to_dict())
    if 'specificity' in count_assign_lookup.columns:
        count_data_all['specificity'] = count_data_all[args.protein_col].astype(str).str.strip().map(
            count_assign_lookup['specificity'].to_dict())

    # Save all-proteins detail table
    save_protein_level_detail(count_data_all, args.outdir,
                              protein_col=args.protein_col, patient_col=args.patient_col,
                              filename='protein_organ_detail_all.csv')

    # --- Build count_data_stable: STABLE proteins only ---
    count_data_stable = None
    if args.stable_proteins:
        print("  Building count data: STABLE proteins...")
        det_patterns = pd.read_csv(args.stable_proteins)
        stable_ids_count = set(det_patterns[det_patterns['pattern'] == 'Stable detected']['protein'].astype(str).str.strip())
        count_data_stable = count_data_all[
            count_data_all[args.protein_col].astype(str).str.strip().isin(stable_ids_count)].copy()
        print(f"    Stable proteins with organ annotation: {count_data_stable[args.protein_col].nunique()}")

    plot_individual_patient_protein_counts(count_data_all, count_data_stable,
                                           args.timepoints, args.outdir,
                                           protein_col=args.protein_col,
                                           patient_col=args.patient_col)

    # ── Organ-specific analysis (strict tissue-enriched only) ─────────
    organs_to_analyze = []
    if args.organ_specific_all:
        organs_to_analyze = sorted(scores['organ_system'].unique())
        print(f"\n[6] Running organ-specific analysis for ALL organs...")
    elif args.organ_filter:
        organs_to_analyze = args.organ_filter
        print(f"\n[6] Running organ-specific analysis for: {organs_to_analyze}")

    if organs_to_analyze:
        os.makedirs(os.path.join(args.outdir, 'organ_specific'), exist_ok=True)

        # Load strict tissue-enriched assignments
        print("  Loading strict tissue-enriched assignments...")
        if use_gls:
            strict_assignments = load_gls_assignments(
                args.gls, min_gls=3.0,
                include_cells=False, include_multilabel=False)
        else:
            strict_assignments = load_hpa_tissue_assignments(
                args.hpa_summary, args.hpa_rna, args.hpa_organ_map,
                specificity_filter=['Tissue enriched'])

        for target_organ in organs_to_analyze:
            organ_assignments = strict_assignments[
                strict_assignments['organ_system'] == target_organ]

            if len(organ_assignments) < 3:
                print(f"\n  {target_organ}: only {len(organ_assignments)} tissue-enriched proteins, skipping")
                continue

            organ_safe = target_organ.replace(' ', '_').replace('&', 'and').replace('/', '_')
            organ_dir = os.path.join(args.outdir, 'organ_specific', organ_safe)
            os.makedirs(organ_dir, exist_ok=True)

            print(f"\n  {target_organ}: {len(organ_assignments)} tissue-enriched proteins")

            # Save protein list
            organ_assignments.to_csv(os.path.join(organ_dir, 'proteins.csv'), index=False)

            # Compute scores using only these organ-specific proteins
            organ_scores, organ_matched = compute_organ_scores(
                data, organ_assignments, args.timepoints,
                protein_col=args.protein_col, patient_col=args.patient_col,
                intensity_col=args.intensity_col,
                min_proteins_per_organ=2)

            if organ_scores.empty:
                print(f"    No scores computed, skipping plots")
                continue

            # Add cluster if available
            if 'cluster' in data.columns:
                cl_map = data[[args.patient_col, 'cluster']].drop_duplicates()
                organ_scores = organ_scores.merge(cl_map, on=args.patient_col, how='left')

            organ_scores.to_csv(os.path.join(organ_dir, 'scores.csv'), index=False)

            # Summary
            organ_summary = compute_organ_score_summary(
                organ_scores, args.timepoints, args.patient_col)
            organ_summary.to_csv(os.path.join(organ_dir, 'summary.csv'), index=False)

            # Statistical test
            organ_test = test_organ_temporal_changes(
                organ_scores, args.timepoints, args.patient_col)
            if not organ_test.empty:
                organ_test.to_csv(os.path.join(organ_dir, 'temporal_test.csv'), index=False)
                for _, row in organ_test.iterrows():
                    p = row.get('primary_p', np.nan)
                    sig = "*" if p < 0.05 else "ns"
                    print(f"    {row.get('primary_test','')}: p={p:.4e}, "
                          f"direction={row['direction']}, FC={row['overall_fc']:.3f} {sig}")

            # Per-patient trajectory for this organ
            if target_organ in organ_scores['organ_system'].values:
                organ_only = organ_scores[organ_scores['organ_system'] == target_organ]

                # Line plot: all patients
                fig, ax = plt.subplots(figsize=(10, 6))
                tp_order = {tp: i for i, tp in enumerate(args.timepoints)}

                # Color by cluster if available
                if 'cluster' in organ_only.columns:
                    clusters_list = sorted(organ_only['cluster'].dropna().unique())
                    cl_colors = plt.cm.Set1(np.linspace(0, 1, max(len(clusters_list), 1)))
                    for ci, cl in enumerate(clusters_list):
                        cl_pats = organ_only[organ_only['cluster'] == cl][args.patient_col].unique()
                        for pat in cl_pats:
                            pat_data = organ_only[organ_only[args.patient_col] == pat].sort_values(
                                'timepoint', key=lambda x: x.map(tp_order))
                            ax.plot(pat_data['timepoint'].map(tp_order), pat_data['organ_score_sum'],
                                   color=cl_colors[ci], alpha=0.3, linewidth=0.8)
                        # Cluster mean
                        cl_data = organ_only[organ_only['cluster'] == cl]
                        cl_mean = cl_data.groupby('timepoint')['organ_score_sum'].mean()
                        cl_mean = cl_mean.reindex(args.timepoints)
                        ax.plot(range(len(args.timepoints)), cl_mean.values,
                               color=cl_colors[ci], linewidth=3, marker='o',
                               label=f'Cluster {int(cl)}')
                else:
                    patients = organ_only[args.patient_col].unique()
                    for pat in patients:
                        pat_data = organ_only[organ_only[args.patient_col] == pat].sort_values(
                            'timepoint', key=lambda x: x.map(tp_order))
                        ax.plot(pat_data['timepoint'].map(tp_order), pat_data['organ_score_sum'],
                               color='gray', alpha=0.3, linewidth=0.8)
                    # Overall mean
                    overall_mean = organ_only.groupby('timepoint')['organ_score_sum'].mean()
                    overall_mean = overall_mean.reindex(args.timepoints)
                    ax.plot(range(len(args.timepoints)), overall_mean.values,
                           color='red', linewidth=3, marker='o', label='Mean')

                ax.set_xticks(range(len(args.timepoints)))
                ax.set_xticklabels(args.timepoints)
                ax.set_xlabel('Timepoint', fontsize=11)
                ax.set_ylabel('Sum log2 intensity', fontsize=11)
                ax.set_title(f'{target_organ}: Patient Trajectories (Tissue-Enriched Proteins Only)',
                           fontsize=12, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(alpha=0.2)
                plt.tight_layout()
                plt.savefig(os.path.join(organ_dir, 'patient_trajectories.png'),
                           dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    Saved: patient_trajectories.png")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary_lines = []
    summary_lines.append("ORGAN TRAJECTORY ANALYSIS SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Proteins assigned to organs: {len(assignments)}")
    summary_lines.append(f"Proteins matched in data: {matched_data[args.protein_col].nunique()}")
    summary_lines.append(f"Organ systems analyzed: {scores['organ_system'].nunique()}")
    summary_lines.append(f"Patients: {scores[args.patient_col].nunique()}")
    summary_lines.append(f"Timepoints: {' -> '.join(args.timepoints)}")
    summary_lines.append("")

    if not test_results.empty:
        if 'primary_q' in test_results.columns:
            sig = test_results[test_results['primary_q'] < 0.05]
        else:
            sig = test_results[test_results['primary_p'] < 0.05]
        summary_lines.append(f"Organs with significant temporal changes (q<0.05): {len(sig)}")
        for _, row in sig.iterrows():
            q = row.get('primary_q', row['primary_p'])
            summary_lines.append(f"  {row['organ_system']}: {row['direction']} "
                                f"(FC={row['overall_fc']:.3f}, test={row['primary_test']}, q={q:.4e})")

    summary_lines.append("")
    summary_lines.append("OUTPUT FILES:")
    for f in sorted(os.listdir(args.outdir)):
        fpath = os.path.join(args.outdir, f)
        if os.path.isfile(fpath):
            summary_lines.append(f"  {f} ({os.path.getsize(fpath):,} bytes)")

    summary_text = '\n'.join(summary_lines)
    print(summary_text)

    with open(os.path.join(args.outdir, 'organ_trajectory_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f"\nAll outputs in: {args.outdir}/")
    print("Done!")


if __name__ == '__main__':
    main()