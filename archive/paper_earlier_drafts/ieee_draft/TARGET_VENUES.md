# Target conferences / workshops (compiled 2026-11-10)

**How to read this:** every date below was found by web search on 2026-11-10. "Aggregator" means a third-party listing, NOT the official call; confirm on the official site before relying on a date. Deadlines move between years.

| # | Venue | Dates / location | Paper deadline | Verification | Fit for this paper |
|---|---|---|---|---|---|
| 1 | **IEEE Intelligent Vehicles Symposium (IV 2027)**, 38th | 15-18 Jun 2027, Perth, Australia | **15 Nov 2026** (notification 15 Jan 2027, camera-ready 1 Feb 2027) | Aggregator ([beri.net](https://www.beri.net/events/ieee-iv-2027), [ITSS Korea](https://ieee-itss-korea.github.io/web/events/2027-06-iv-2027)); official site not seen | On-topic (driving policy, BEV). **Only 5 days away** and the evidence is thin (no isolated contribution, simulation only). Not recommended unless the team accepts a likely reject. IV 2026 used an Oct 2025 deadline, so dates vary by year. |
| 2 | **IEEE ITSC 2027** (Intelligent Transportation Systems Conference) | 21 Sep 2027, Boston, MA | **1 Mar 2027** (notification 1 May 2027) | Aggregator ([myhuiban](https://www.myhuiban.com/conference/5848)); ITSC 2026 also used 1 Mar; official site not seen | Best peer-reviewed target. Leaves ~3.5 months to add the safe-teacher student, an overlay/scalar/DAgger ablation at n>=200, several training seeds, and a degradation regime that actually hurts a reference. |
| 3 | **IEEE/RSJ IROS 2027** | 26 Sep - 1 Oct 2027, Florence, Italy | **1 Mar 2027** (one source: 2 Mar 2027 07:59:59 UTC) | IEEE RAS deadlines calendar ([ieee-ras.org](https://www.ieee-ras.org/events/category/deadlines/)); [mldeadlines](https://mldeadlines.com/conference/iros/) | Competitive robotics main track; current scope is too narrow. Consider only with the stronger experiments above. |
| 4 | **ICBINB-style negative-results workshop** ("I Can't Believe It's Not Better") | ICLR 2026 edition was 26-27 Apr 2026 | 2026 edition: 31 Jan 2026 (**past**). **No 2027 call found.** | [icbinb-2026](https://sites.google.com/view/icbinb-2026/submit) (past edition only) | **Best thematic fit** (null results with analysis of why: caution, pose conditioning, PPO, the 0.975 artefact). Watch for the next call; the date pattern is Jan-Feb but that is an inference, not a confirmed date. |
| 5 | **IEEE CONECCT 2027** (Bangalore) | likely Mar 2027 (aggregator) | One aggregator says **15 Dec 2026**; earlier editions had hard deadlines 31 Mar 2025 and 11 Apr 2024 | **Unverified, possibly stale.** No official 2027 call found ([2026 site](https://2026.connect.ieeebangalore.org/)) | Realistic regional IEEE venue and close to the university. Check the IEEE Bangalore Section page for the real dates. |
| 6 | IEEE ICRA 2027 | (not confirmed) | (not read; ICRA calls typically close in Sep, so likely **past**) | [call page](https://2027.ieee-icra.org/announcements/call-for-technical-papers/) found but not read | Verify; probably closed. |
| 7 | IEEE INDICON 2026 | 18-20 Dec 2026, Chennai | submission 31 Aug 2026 (**closed**); camera-ready 16 Nov 2026 | [ieeeindiacouncil.org](https://ieeeindiacouncil.org/ieee-india-council-international-conference-indicon-2026/) | Closed; also no robotics / AV track visible. Look for the 2027 edition. |

## Recommended plan
1. **Primary: IEEE ITSC 2027 (1 Mar 2027).** Time to strengthen the evidence (see "Gaps" below).
2. **Secondary: a negative-results workshop** when the next ICBINB-type call appears, using the same data reframed as null results with analysis.
3. **Fallback for the review deck:** list IV 2027 (Nov 15) as "considered, not targeted: insufficient time", CONECCT 2027 (dates to be confirmed) and ITSC 2027 as targets.
4. Do **not** claim a submission date you have not verified on the official site.

## Gaps that would most improve acceptance odds (in order of value)
1. Evaluate the safe-teacher student at n=200 (trained; evaluation queued behind the weather job).
2. Attribution ablation at n>=200: DAgger without overlay vs overlay vs overlay+scalar split.
3. Several training seeds per configuration (currently one).
4. A regime where perception loss demonstrably hurts a reference policy (otherwise no robustness claim).
5. Fair localisation test (identical perturbation for every model).
6. Small camera-in-the-loop check of the dropout proxy (needs the team's go-ahead to run the camera pipeline).

## Sources
- IV 2027: https://www.beri.net/events/ieee-iv-2027 ; https://ieee-itss-korea.github.io/web/events/2027-06-iv-2027
- ITSC 2027: https://www.myhuiban.com/conference/5848
- IROS 2027: https://www.ieee-ras.org/events/category/deadlines/ ; https://mldeadlines.com/conference/iros/
- ICBINB: https://sites.google.com/view/icbinb-2026/submit
- CONECCT: https://2026.connect.ieeebangalore.org/ ; https://www.myhuiban.com/conference/4809.md
- INDICON 2026: https://ieeeindiacouncil.org/ieee-india-council-international-conference-indicon-2026/
- ICRA 2027: https://2027.ieee-icra.org/announcements/call-for-technical-papers/
