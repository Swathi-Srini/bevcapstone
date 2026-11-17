# report/ : what is tracked and what is local-only

## Tracked on GitHub (sources and current PDFs)
| File | Purpose |
|---|---|
| `latest_14th_Nov_26.tex/.pdf` | complete technical record up to 14 Nov 2026 |
| `Final_Methodology_and_Approach.tex/.pdf` | final methodology and schedule |
| `Five_Model_Comparison.tex/.pdf` | the 200-episode five-model study (simulator-grid evaluation) |
| `Effective_Range_Measurement_Protocol.tex/.pdf` | protocol for an image-to-range estimator (set aside) |
| `Live_Caution_Sensor_and_Episode_Walkthrough.tex/.pdf` | the archived image-statistics sensor and the episode walkthrough |
| `Architecture_Live_Caution_Branch*.drawio` | architecture diagram with the caution branch (open at app.diagrams.net) |
| `GitHub_vs_This_Work.tex/.pdf` | **partly stale** (names a deleted branch and old numbers) |
| `figures/`, `tables/`, `make_*.py` | figures, tables and the scripts that generate them for the five-model report |

## Local-only (git-ignored; not on GitHub)
* `local_only/`: `Directors_Research_Guide` and `Complete_Technical_Research_Report` (LaTeX sources, PDFs, build files).
* `videos/`, `videos_adaptive/`, `videos_share_raw/` (686 MB of `.mp4`) and `caution_model_detector_table_images/` (257 MB): deliverables and generated images.
* LaTeX build artefacts (`*.aux`, `*.log`, `*.out`, `*.toc`).
All of these are included in the dated local backup (`docs/BACKUP_STRATEGY.md`).

## Rebuilding a PDF
Install MiKTeX (or any LaTeX with `amsmath`, `booktabs`, `longtable`, `hyperref`, `enumitem`). From this folder run `pdflatex -interaction=nonstopmode <name>.tex` **twice** (the second pass resolves references). `build_documents.ps1` and `finish_five_model_document.sh` rebuild the older reports; they contain machine-specific paths that are scheduled to be generalised after the experiments (`docs/CLEANUP_LOG.md`, step B1).
