<p align="center">
  <h1 align="center">Longitudinal Proteomics Analysis </h1>
  <p align="center">
    Tools for analysing temporal changes in plasma proteome data from longitudinal clinical cohorts (updating)
    <br />
  </p>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/badge/statsmodels-LMM-orange" />
  <img src="https://img.shields.io/badge/atlas-HPA%20%7C%20GLS-purple" />
</p>




This repository contains Python scripts for complementary aspects of longitudinal proteomics analysis:
ScriptPurposeLMM.pyPer-protein linear mixed-effects modelling with full preprocessing pipelineorgan_trajectory.pyOrgan-level trajectory analysis using tissue atlas annotations (HPA / GLS)
Both scripts accept long-format proteomics data (one row per protein × patient × timepoint) and produce tabular results, diagnostic plots, and human-readable summaries.

LMM.py — Linear Mixed Model Analysis
Fits a linear mixed model to each protein independently, testing whether protein abundance changes significantly across timepoints while accounting for within-patient correlation.

Model: log2(intensity) ~ C(timepoint) + covariates + (1 | patient)


organ_trajectory.py — Organ-Level Trajectory Analysis
Aggregates individual protein measurements into organ-level scores using tissue atlas annotations, then tests whether organ scores change significantly over time.
Tissue Atlas Support
Supports two complementary protein-to-organ mapping strategies:
AtlasDescriptionSourceHPARNA tissue specificity annotations (tissue-enriched, group-enriched, tissue-enhanced) with nTPM-based primary tissue assignmentHuman Protein AtlasGLSMulti-atlas consensus scoring (0–4) across tissues and cell types, with organ-adaptive confidence thresholdsMalmström et al. 2025, Cell
Statistical Testing (per organ)
Uses a priority cascade, selecting the most powerful applicable method:

Linear mixed model (primary) — organ_score ~ C(timepoint) + (1 | patient) — handles unbalanced/missing data
Friedman test — paired non-parametric for patients with complete data
Kruskal–Wallis — unpaired fallback when paired data is insufficient
Pairwise Wilcoxon — baseline vs each subsequent timepoint (supplementary)

All primary p-values are FDR-corrected (Benjamini–Hochberg).
 Additional Analyses

Cluster × organ interaction — tests whether pre-defined patient clusters show divergent organ trajectories
Organ-specific deep dives — tissue-enriched-only analysis with per-patient trajectory plots for organs of interest
Detection count heatmaps — tracks how many organ-specific proteins are detected per patient per timepoint


Input Data Format
Both scripts expect a long-format CSV:
csvprotein,cluster,timepoint,patient,raw_intensity
P12345,1,base,patient_01,1523400
P12345,1,D1,patient_01,1876200
P12345,1,D2,patient_01,1245800
ColumnDescriptionproteinUniProt accession (isoform suffixes handled automatically)clusterProtein group or cluster IDtimepointMust match values passed via --timepointspatientPatient/sample identifierraw_intensityUntransformed protein intensity (log₂-transformed internally)
