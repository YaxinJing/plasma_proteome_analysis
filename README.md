# plasma_proteome_analysis



#organ_trajectory

Groups plasma proteins by their tissue of origin (HPA), computes per-patient per-timepoint organ scores, and tracks how each organ's protein signature changes after cardiac arrest.

Concept:
  1. Use HPA tissue specificity to assign proteins to primary organs (only tissue-enriched and group-enriched — not ubiquitous proteins)
  2. For each patient x timepoint: organ_score = mean(log2_intensity of that organ's proteins)
  3. Track organ trajectories over D1->D2->D3->D4
  4. Test which organs show significant temporal changes (LMM or repeated measures)
  5. Compare organ trajectories across patient clusters
