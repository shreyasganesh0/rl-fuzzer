# Differential Fuzzing Analysis Summary

## Bug Pair: xml005

### Coverage Trajectory

- Buggy final coverage: 5371.0 (+/- 25.7) edges across 3 seeds
- Fixed final coverage: 5165.0 (+/- 9.4) edges across 3 seeds
- **Divergence point**: 3,486 executions
- Final coverage difference: 206.0 edges (buggy > fixed)

### Differential Edges (at final timepoint)

- Buggy-only edges: 3342
- Fixed-only edges: 3078
- Shared edges: 1975
- Buggy total: 5317, Fixed total: 5053

### Most Differentially Effective Mutations

Mutations with the largest buggy/fixed effectiveness ratio near the divergence point:

1. **ARITH_SUB4LE**: buggy gain rate = 0.001521, fixed gain rate = 0.000686, ratio = 2.217
2. **HAVOC_ARITH16BE**: buggy gain rate = 0.000928, fixed gain rate = 0.000450, ratio = 2.061
3. **INT_2BE**: buggy gain rate = 0.000885, fixed gain rate = 0.000453, ratio = 1.956
4. **HAVOC_ARITH32_**: buggy gain rate = 0.001676, fixed gain rate = 0.000898, ratio = 1.868
5. **ARITH_ADD2BE**: buggy gain rate = 0.000504, fixed gain rate = 0.001411, ratio = 0.357

### Feature Importance (Mann-Whitney U)

- Bonferroni-corrected alpha: 0.0007692307692307692

Top discriminative features (by mean |A12 - 0.5| across timepoints):

- **total_edges**: mean |A12 - 0.5| = 0.500
- **cold_edges**: mean |A12 - 0.5| = 0.322
- **cool_edges**: mean |A12 - 0.5| = 0.256
- **avg_exec_time_us**: mean |A12 - 0.5| = 0.256
- **warm_edges**: mean |A12 - 0.5| = 0.256

## Bug Pair: xml017

### Coverage Trajectory

- Buggy final coverage: 5784.0 (+/- 31.6) edges across 3 seeds
- Fixed final coverage: 5488.0 (+/- 41.2) edges across 3 seeds
- **Divergence point**: 238,453 executions
- Final coverage difference: 296.0 edges (buggy > fixed)

### Most Differentially Effective Mutations

Mutations with the largest buggy/fixed effectiveness ratio near the divergence point:

1. **HAVOC_INT32**: buggy gain rate = 0.000765, fixed gain rate = 0.000430, ratio = 1.780
2. **HAVOC_INT16BE**: buggy gain rate = 0.000807, fixed gain rate = 0.000459, ratio = 1.756
3. **FLIP_1BYTE**: buggy gain rate = 0.000366, fixed gain rate = 0.001068, ratio = 0.342
4. **FLIP_2BITS**: buggy gain rate = 0.001440, fixed gain rate = 0.000922, ratio = 1.562
5. **ARITH_SUB4BE**: buggy gain rate = 0.000914, fixed gain rate = 0.001887, ratio = 0.484

### Feature Importance (Mann-Whitney U)

- Bonferroni-corrected alpha: 0.0007692307692307692

Top discriminative features (by mean |A12 - 0.5| across timepoints):

- **hot_edges**: mean |A12 - 0.5| = 0.300
- **corpus_size**: mean |A12 - 0.5| = 0.300
- **total_edges**: mean |A12 - 0.5| = 0.278
- **edge_hit_mean**: mean |A12 - 0.5| = 0.233
- **cool_edges**: mean |A12 - 0.5| = 0.211
