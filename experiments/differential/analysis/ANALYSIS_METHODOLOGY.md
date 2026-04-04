# Differential Analysis Methodology

## Data Sources

### Files Loaded

- `coverage_dynamics_xml005_buggy_seed1.csv`
- `coverage_dynamics_xml005_buggy_seed2.csv`
- `coverage_dynamics_xml005_buggy_seed3.csv`
- `coverage_dynamics_xml005_fixed_seed1.csv`
- `coverage_dynamics_xml005_fixed_seed2.csv`
- `coverage_dynamics_xml005_fixed_seed3.csv`
- `coverage_dynamics_xml017_buggy_seed1.csv`
- `coverage_dynamics_xml017_buggy_seed2.csv`
- `coverage_dynamics_xml017_buggy_seed3.csv`
- `coverage_dynamics_xml017_fixed_seed1.csv`
- `coverage_dynamics_xml017_fixed_seed2.csv`
- `coverage_dynamics_xml017_fixed_seed3.csv`
- `mutation_attribution_xml005_buggy_seed1.csv`
- `mutation_attribution_xml005_buggy_seed2.csv`
- `mutation_attribution_xml005_buggy_seed3.csv`
- `mutation_attribution_xml005_fixed_seed1.csv`
- `mutation_attribution_xml005_fixed_seed2.csv`
- `mutation_attribution_xml005_fixed_seed3.csv`
- `mutation_attribution_xml017_buggy_seed1.csv`
- `mutation_attribution_xml017_buggy_seed2.csv`
- `mutation_attribution_xml017_buggy_seed3.csv`
- `mutation_attribution_xml017_fixed_seed1.csv`
- `mutation_attribution_xml017_fixed_seed2.csv`
- `mutation_attribution_xml017_fixed_seed3.csv`

### Bug Pairs Analyzed

- **xml005**: `xml005_buggy` vs `xml005_fixed`
- **xml017**: `xml017_buggy` vs `xml017_fixed`

### Seeds: 1, 2, 3

## Statistical Tests

### Mann-Whitney U Test

- Non-parametric test for comparing two independent samples
- Null hypothesis: the distributions of buggy and fixed values are equal
- Alternative: two-sided
- Nominal alpha = 0.05
- Bonferroni correction applied: alpha_corrected = 0.05 / (n_features x n_timepoints)

### Vargha-Delaney A12 Effect Size

- Measures the probability that a randomly chosen value from group A
  exceeds a randomly chosen value from group B
- A12 = 0.5: no effect
- |A12 - 0.5| >= 0.06: small effect
- |A12 - 0.5| >= 0.14: medium effect
- |A12 - 0.5| >= 0.21: large effect

## Divergence Detection Algorithm

1. Interpolate coverage curves from all seeds to a common execution axis
   (500 equally spaced points across the common range).
2. Compute mean and standard deviation across seeds for each variant.
3. Compute pooled standard deviation at each point:
   `pooled_std = sqrt(((n_b-1)*std_b^2 + (n_f-1)*std_f^2) / (n_b+n_f-2))`
4. A point is considered diverged when `|mean_buggy - mean_fixed| > pooled_std`.
5. The divergence point is the start of the first run of >= 5 consecutive
   diverged points. If no such run exists, the first single diverged point
   is reported.

## Feature Computation Formulas

### Coverage Velocity
First difference of total_edges with respect to total_execs:
`velocity[i] = (edges[i] - edges[i-1]) / (execs[i] - execs[i-1])`

### Edge Heat Ratios
From cumulative bitmap snapshots (65536-byte maps):
- **hot**: `count(map[i] > 128) / count(map[i] > 0)`
- **warm**: `count(8 <= map[i] <= 128) / count(map[i] > 0)`
- **cool**: `count(1 <= map[i] <= 7) / count(map[i] > 0)`
- **cold**: `count(map[i] == 0) / MAP_SIZE`

Note: ratios from CSV use all 65536 entries as denominator (hot+warm+cool+cold).

### Edge Entropy
Shannon entropy over 8 power-of-2 hit-count bins (1, 2, 4, 8, 16, 32, 64, 128+):
`entropy = -sum(p_i * log2(p_i))` where `p_i = bin_count_i / nonzero_edges`

### Mutation Effectiveness (Coverage Gain Rate)
`gain_rate = sum(new_edges) / sum(count)` for each mutation across a time window.

### Differential Edges
At matched exec counts, load bitmap snapshots from all available seeds.
Union the non-zero bytes across seeds. Then:
- `buggy_only = count(buggy_union & ~fixed_union)`
- `fixed_only = count(fixed_union & ~buggy_union)`
- `shared = count(buggy_union & fixed_union)`
