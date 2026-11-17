# Paired A vs C evaluation (20 paired seeds 5200-5219, pilot scale)

## Clear (n = 20 paired seeds 5200-5219)

| Metric | A (non-adaptive) | C (adaptive) |
|---|---|---|
| Success | 20/20 (0.84-1.00) | 17/20 (0.64-0.95) |
| Outcomes (succ/crash/off-road/timeout) | 20/0/0/0 | 17/1/0/2 |
| Collisions + off-road | 0 | 1 |
| Route completion | 0.989 | 0.906 |
| Mean speed (m/s) | 7.58 | 7.13 |
| Duration (s) | 53.4 | 59.9 |
| Min bumper gap (m, privileged log) | 7.9 | 5.9 |
| Hard-braking share of steps | 0.035 | 0.017 |
| Wh/km, both-success pairs (n=17) | 106.0 | 97.0 |
| Wh/km, all episodes | 110.7 | 109.8 |
| Estimated caution (mean, true = 0.00) | n/a | 0.00 |

McNemar (A-only success 3, C-only success 0): exact p = 0.250. Paired Wh/km difference C-A on both-success pairs: -9.0 (95% bootstrap CI -24.8 to 5.2), -8.5%.

## Fog 1.0 (n = 20 paired seeds 5200-5219)

| Metric | A (non-adaptive) | C (adaptive) |
|---|---|---|
| Success | 20/20 (0.84-1.00) | 20/20 (0.84-1.00) |
| Outcomes (succ/crash/off-road/timeout) | 20/0/0/0 | 20/0/0/0 |
| Collisions + off-road | 0 | 0 |
| Route completion | 0.989 | 0.989 |
| Mean speed (m/s) | 7.82 | 5.57 |
| Duration (s) | 50.8 | 71.2 |
| Min bumper gap (m, privileged log) | 4.2 | 9.5 |
| Hard-braking share of steps | 0.027 | 0.003 |
| Wh/km, both-success pairs (n=20) | 106.8 | 54.5 |
| Wh/km, all episodes | 106.8 | 54.5 |
| Estimated caution (mean, true = 1.00) | n/a | 0.96 |

McNemar (A-only success 0, C-only success 0): exact p = 1.000. Paired Wh/km difference C-A on both-success pairs: -52.3 (95% bootstrap CI -65.5 to -41.1), -48.9%.

## Rain 1.0 (n = 20 paired seeds 5200-5219)

| Metric | A (non-adaptive) | C (adaptive) |
|---|---|---|
| Success | 20/20 (0.84-1.00) | 20/20 (0.84-1.00) |
| Outcomes (succ/crash/off-road/timeout) | 20/0/0/0 | 20/0/0/0 |
| Collisions + off-road | 0 | 0 |
| Route completion | 0.989 | 0.988 |
| Mean speed (m/s) | 7.82 | 5.50 |
| Duration (s) | 50.8 | 72.1 |
| Min bumper gap (m, privileged log) | 6.5 | 10.4 |
| Hard-braking share of steps | 0.027 | 0.002 |
| Wh/km, both-success pairs (n=20) | 103.4 | 53.1 |
| Wh/km, all episodes | 103.4 | 53.1 |
| Estimated caution (mean, true = 1.00) | n/a | 0.99 |

McNemar (A-only success 0, C-only success 0): exact p = 1.000. Paired Wh/km difference C-A on both-success pairs: -50.3 (95% bootstrap CI -61.1 to -40.4), -48.7%.
