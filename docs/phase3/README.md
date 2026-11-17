# Phase 3: weather-adaptive learned driving under fog and rain

Snapshot written on **17 November 2026** (branch `phase3-severity-2026-11-17`). Nothing on `main` was changed by this branch; it only adds files.

| Read this | For |
|---|---|
| [STATUS.md](STATUS.md) | where the project is right now, what is still running, what is open |
| [ARCHITECTURE.md](ARCHITECTURE.md) | the final system (diagram, modules, models A / B / C) |
| [WORK_DONE.md](WORK_DONE.md) | what has been done so far, in order |
| [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) | the completed 20-seed results, with the honest limitations |
| [REPRODUCE.md](REPRODUCE.md) | commands that regenerate the models, estimator, evaluation and figures |
| [FILE_MAP.md](FILE_MAP.md) | where every phase-3 file is, and what is deliberately NOT in Git |

One-paragraph summary. A camera-driven MetaDrive policy estimates how severe the (synthetic) fog or rain is **from the camera image alone**, turns that into a caution score, and uses it to drive more slowly and keep larger gaps. In severe fog and rain the adaptive policy (Model C) succeeded in 20/20 paired episodes, drove about 30 % slower and used about 49 % less traction energy than the non-adaptive policy (Model A). In clear weather it failed 3 of 20 episodes (A: 0 of 20); the cause is not yet determined. The study is pilot-scale, simulation-only and uses synthetic weather.
