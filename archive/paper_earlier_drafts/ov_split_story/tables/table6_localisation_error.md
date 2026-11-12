**Sensitivity to map-localisation error, clear, density 0.05**

| policy | lateral err m | heading err rad | n | success | 95% CI |
|---|---|---|---|---|---|
| ov_split (ours) | 0.0 | 0.0 | 40 | 0.925 | [0.80, 0.97] |
| ov_split (ours) | 0.25 | 0.02 | 40 | 0.900 | [0.77, 0.96] |
| ov_split (ours) | 0.5 | 0.05 | 40 | 0.900 | [0.77, 0.96] |
| ov_split (ours) | 1.0 | 0.1 | 40 | 0.700 | [0.55, 0.82] |
| ov_split (ours) | 2.0 | 0.2 | 40 | 0.275 | [0.16, 0.43] |
| Repo BC, sparse-traffic | 0.0 | 0.0 | 40 | 0.775 | [0.62, 0.88] |
| Repo BC, sparse-traffic | 0.25 | 0.02 | 40 | 0.600 | [0.45, 0.74] |
| Repo BC, sparse-traffic | 0.5 | 0.05 | 40 | 0.550 | [0.40, 0.69] |
| Repo BC, sparse-traffic | 1.0 | 0.1 | 40 | 0.225 | [0.12, 0.38] |
| Repo BC, sparse-traffic | 2.0 | 0.2 | 40 | 0.150 | [0.07, 0.29] |
| DAgger, GitHub BEV | 0.0 | 0.0 | 40 | 0.875 | [0.74, 0.95] |
| DAgger, GitHub BEV | 0.25 | 0.02 | 40 | 0.750 | [0.60, 0.86] |
| DAgger, GitHub BEV | 0.5 | 0.05 | 40 | 0.500 | [0.35, 0.65] |
| DAgger, GitHub BEV | 1.0 | 0.1 | 40 | 0.275 | [0.16, 0.43] |
| DAgger, GitHub BEV | 2.0 | 0.2 | 40 | 0.300 | [0.18, 0.45] |
