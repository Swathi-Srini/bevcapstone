**Weather sensitivity. Identical failed-seed sets across weathers mean the failures are scenario-driven, not weather-driven.**

| model | clear | fog 1.0 | rain 0.6 | rain 1.0 | range (max-min) | weathers whose failed-seed set equals clear's |
|---|---|---|---|---|---|---|
| Repo BC (sparse traffic) | 0.775 | 0.800 | 0.800 | 0.800 | 0.025 | 0/3 |
| Repo BC (clear, no traffic) | 0.875 | 0.875 | 0.875 | 0.875 | 0.000 | 3/3 |
| ov_split (ours) | 0.925 | 0.925 | 0.925 | 0.950 | 0.025 | 2/3 |
| ctrl_nocaut (round-0 BC) | 0.975 | 0.975 | 0.975 | 0.975 | 0.000 | 3/3 |
| caut_c0 (caution off) | 0.900 | 0.900 | 0.900 | 0.900 | 0.000 | 3/3 |
| caut_auto (caution = weather) | 0.900 | 0.925 | 0.900 | 0.850 | 0.075 | 0/3 |
