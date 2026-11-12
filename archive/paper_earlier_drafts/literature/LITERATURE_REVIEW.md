# Literature review (living document)

**Status: preliminary.** Built from web-search summaries on 2026-11-02. Every item must be read in full and verified before it is cited. No novelty claim is made yet.

## Found so far (verify before citing)
- Sakaridis, Dai, Van Gool, *Semantic Foggy Scene Understanding with Synthetic Data*, IJCV 2018 (arXiv 1708.07819): fog synthesis with the Koschmieder optical model and depth-based transmission; Foggy Cityscapes. Basis for the form of our fog model.
- Hahner et al., purely synthetic foggy scenes (arXiv 1910.03997).
- Xie et al., *Benchmarking Bird's Eye View Detection Robustness to Real-World Corruptions* (RoboBEV / nuScenes-C, ICLR-W 2023); related BEV robustness benchmark (arXiv 2405.17426). Perception only, no control.
- Li et al., MetaDrive (arXiv 2109.12674): simulator, procedural generation, domain randomisation.
- Snow-induced lane keeping with robust RL (arXiv 2512.12987): camera-based RL under simulated snow; not BEV, not fog/rain-conditioned.
- SCPL, salience-invariant consistent policy learning (arXiv 2502.08336): visual RL generalisation to CARLA weather shifts.
- Prompt-driven domain adaptation via in-context RL (arXiv 2511.12755): CARLA adaptation to rain/snow/fog.
- COLA, self-adaptive driving in nonstationary environments (arXiv 2210.03209): clear-trained policy fails in rain.
- Multi-weather BEV planning with MPC (Biomimetic Intelligence & Robotics): BEV plus weather, but not RL.
- Review of RL for driving in CARLA (arXiv 2509.08221).

## Scalar-state / BEV input design (added 2026-11-04; verification level marked per item)
Design under review: 6-D scalar [speed, route progress, lateral offset, heading error, curvature ahead, distance to goal] from simulator lane/navigation data (the spec treats this as HD-map localisation) + a 1-channel camera BEV (GitHub encoding: -1 unknown, 0 free, 0.9 ego, 1.0 detected vehicle; route 0.5 defined but not drawn).

**Verified against the paper text / official docs**
- MetaDrive default observation (official docs, https://metadrive-simulator.readthedocs.io/en/latest/obs.html): ego state (steering, heading, velocity, relative distance to boundaries), navigation (next route checkpoints in ego coordinates, road direction and curvature), lidar. SUPPORTS: our scalar resembles the simulator's own default ego+navigation information.
- Moghadam, Alizadeh, Tekin, Elkaim, *An End-to-end Deep RL Approach for the Long-term Short-term Planning on the Frenet Space* (arXiv 2011.13098): Frenet-frame state (longitudinal s, lateral d, surrounding vehicles' offsets). SUPPORTS lateral-offset-style state; does not cover heading error or curvature.
- Zhang et al., *End-to-End Urban Driving by Imitating a RL Coach* (Roach, ICCV 2021, arXiv 2108.08265), Sec. 3.1: BEV rendered from ground-truth simulator state with channels drivable area, desired route, lane boundaries, vehicles, pedestrians, lights/stops; a 6-D measurement vector holds only ego state NOT in the BEV (steering, throttle, brake, gear, lateral and horizontal speed). Lateral offset and heading error are NOT inputs (reward terms only). CHALLENGES our split: the closest hybrid design puts road/route/lane in the BEV and keeps the vector for ego dynamics, as the team spec (route 0.5, boundary 0.8) describes; the GitHub BEV does not draw them yet. Main reviewer risk.
- de Haan, Jayaraman, Levine, *Causal Confusion in Imitation Learning* (NeurIPS 2019, arXiv 1905.11979): cloned policies can get worse with more input information (causal misidentification). Consistent with our scalar ablation (removing curvature or lateral offset improved a plain MLP clone); supports the interpretation, does not prove it for our setting.

**From search summaries only (abstract level; read before citing)**
- Chen, Zhou, Koltun, Krahenbuhl, *Learning by Cheating* (CoRL 2019, arXiv 1912.12294): privileged teacher with ground-truth layout and traffic trains a vision-only student. Supports the privileged-teacher then student logic; exact inputs not confirmed.
- Bansal, Krizhevsky, Ogale, *ChauffeurNet* (RSS 2019, arXiv 1812.03079): plain behaviour cloning insufficient for complex driving; perturbation and extra losses. Consistent with our need for DAgger/RL; input details not confirmed.

**Found but NOT checked (do not cite yet)**
- Frenet/lateral-error/heading state in RL lane-keeping and racing: arXiv 2411.13885; APSIPA 2025 P239 (DDPG lane following with lateral deviation, heading error); arXiv 2607.20973 (racing, state (s, d, delta-psi)). Titles/claims come from search snippets only.

**HD-map localisation realism and weather (searched 2026-11-04; SNIPPET LEVEL ONLY - verify before citing)**
- Perception and Sensing for Autonomous Vehicles Under Adverse Weather Conditions: A Survey (arXiv 2112.08936): snippet says fog/rain/snow can cut LiDAR detection range (~25% reported) and reduce points per object. Relevance: sensors used for map-matched localisation degrade in weather.
- Robust Vehicle Localization and Tracking in Rain using Street Maps (arXiv 2409.01038): snippet reports 6.05 m vs 2.46 m error (rain vs clear, 150 m route) for a map-based fusion method. Relevance: lane-level localisation is NOT guaranteed in rain; supports adding weather-scaled noise to lateral/heading inputs.
- Lane-Level Localization and Map Matching for Connected and Automated Vehicle Applications (USDOT, rosap.ntl.gov/view/dot/67203): snippet says 17 cm horizontal map accuracy met the lane-level map requirement (max error < 30 cm); this is MAP accuracy, not full localisation error. Low-cost GNSS struggles at lane level (UMich snippet).
- Liu et al. 2019 LiDAR localisation (cited inside a review; <10 cm at 95% over 1000 km) - secondary citation, not verified.
- Conclusion for the paper: HD-map localisation at ~0.1-0.3 m is achievable with LiDAR/camera map matching but degrades in adverse weather; the clean-state assumption should be stated and tested with noise (approved item 2). No source found for a formal lane-level accuracy standard.

**Gaps**
- No source found that justifies this exact 6-D vector; the design is defensible as a composite of MetaDrive's default observation and Frenet-style RL practice, not as an established standard.
- Localisation-realism evidence above is snippet-level; read the primary papers before citing.

## Preliminary gap (hypothesis, not a finding)
The searches did not surface work combining (i) an RL control policy, (ii) a camera-derived BEV with explicit unknown regions, (iii) explicit weather type+intensity conditioning, and (iv) controlled ablations of conditioning vs mixed-weather training. A systematic search is required before this is stated in a paper.

## TODO
Systematic search: RL+BEV, weather/context-conditioned RL, FiLM-style conditioning, latent-context RL, physically-based rain rendering, MetaDrive weather; read primary sources; build a comparison table.

## Novelty check, 2026-11-05 (4 web searches + 2 abstract-level reads; NOT a systematic review)
Questions: (Q1) is "overlay-BEV + scalar split + DAgger from an IDM teacher in MetaDrive" new? (Q2) is conditioning a driving policy on a perception-reliability / caution signal new? (Q3) weather-intensity-conditioned driving policy? (Q4) visibility/uncertainty -> speed adaptation?
- Q1: searches found MetaDrive imitation work (EGPO arXiv 2110.06831; HG-DAgger 1810.02890; active adaptive expert involvement 2303.02073; robot-gated IIL 2506.09176) but none using an IDM teacher + map-overlay BEV; the one found reports MetaDrive's expert was SAC and DAgger gave no advantage over BC there. Components are prior art (Roach BEV split, verified earlier; DAgger, Ross 2011). Not found != absent. Treat the COMBINATION as engineering, not a contribution.
- Q2: no paper found that feeds a measured detector-reliability value to a driving policy. Closest (abstract-read only): Manchingal et al., *Uncertainty-Aware Autonomous Vehicles: Predicting the Road Ahead*, arXiv 2510.22680 (Oct 2025): classifier uncertainty modulates speed on a racing stack; uncertainty from road-layout classification, not weather/detectors; mapping not stated in abstract. Also a patent application on feeding perception uncertainty into a safety driving model (rule-based). 
- Q3: weather-vector-conditioned policies exist per a search snippet (a 3-number weather vector concatenated with image input in an actor-critic) but I could NOT identify the paper: the snippet's link, arXiv 1907.05246 (Makantasis et al., SUMO), was read and has NO weather conditioning. => that snippet attribution was wrong; do not cite. Related, unread: COLA 2210.03209, snow lane-keeping 2512.12987, in-context RL 2511.12755 (all adaptation, none uses measured detector retention).
- Q4: visibility-dependent speed limits are known (rule-based, safety driving models; limited-visibility planning 2018); a LEARNED mapping from measured detection retention to speed was not found.
- Verdict: no exact precedent found for "caution = 1 - measured detector retention(weather) as a policy input + caution-aware teacher", but (a) searches were shallow, (b) our own sparse-traffic data currently shows no weather effect and caution is not beating its control, so there is nothing yet to claim. NEXT: read 2510.22680 and COLA in full; search "perception-aware planning", "confidence-conditioned", "ODD-aware speed" before any novelty sentence is written.

## Novelty check, full-text pass (2026-11-05, later). PDFs in downloaded_pdfs/ (untrusted data; text via pdftotext). Reading depth stated per item.
**Corrections to the earlier section**
- The "weather-vector actor-critic" snippet was **COLA** (Pan et al., arXiv 2210.03209), not Makantasis 1907.05246. Read: vision-based CARLA lane keeping; weather = hidden mode of 3 parameters (cloudiness, rain, puddles) modelled as an HM-MDP; online meta-RL adaptation. Weather is a latent mode adapted online; no measured detector reliability, no BEV, no teacher.
- Do not cite Makantasis 1907.05246 for weather (SUMO traffic-flow study).

**Read (text extracted, relevant sections read; not a line-by-line review)**
- Zhang et al., Roach, arXiv 2108.08265 (ICCV 2021): RL coach on a BEV rendered from ground-truth simulator state with drivable area, desired ROUTE and LANE BOUNDARIES channels plus a measurement vector for ego states not in the BEV; IL students trained with DAgger (DA-RB baseline) under the coach. => our `ov_split` (map-overlay BEV + scalar split + DAgger) is this recipe with an IDM teacher in MetaDrive. NOT NOVEL; cite Roach as the origin of the design.
- Manchingal et al., arXiv 2510.22680 (Oct 2025): Formula Student racing car (ROS2, ZED stereo + LiDAR, YOLOv11n cones). Random-Set NN classifies upcoming track curvature (7 classes); predicted ENTROPY feeds a hand-coded five-tier speed-scaling rule (e.g. entropy 2.3-2.4 -> 80%, 2.4-2.6 -> 60%, >=2.6 -> stop). Uncertainty-to-speed is thus a fixed rule on layout-classifier uncertainty; no weather, no learned policy, no detector retention. Closest prior art for "uncertainty slows the vehicle"; differs in signal, setting and that ours is learned/conditioned.
- Taylor, Picon, Dhakal, *Weather Adaptive Autonomous Driving*, arXiv 2611.04305 (3 Nov 2026; abstract, intro and related-work opening read): camera detectors lose recall in rain/fog/snow/sand; per-weather conformal-risk-control confidence thresholds on frozen YOLOv8x keep miss fraction within a 20% budget in each DAWN weather bin, a single global threshold fails in fog. Perception-side only; supports our premise (weather-dependent recall) and is a citable alternative to our measured retention table.
**Seen only at abstract/snippet level (unread)**: R2SE (arXiv 2506.09800: GRPO refinement for E2E driving under degraded perception, 'perception loosely assessed via normalised loss'); snow lane-keeping robust RL 2512.12987; DCARL (Nature MI 2023); CARIL 2503.00783.

**Verdict for the paper (what can be said honestly)**
1. ov_split: Roach-style input + DAgger; present as the experimental base, never as a contribution.
2. Caution conditioning (caution = 1 - measured detector retention(weather) as a policy input with a caution-aware teacher): no exact precedent found in ~8 searches + the papers above; nearest are Manchingal (rule on layout-classifier entropy), COLA (latent weather mode, online meta-RL), R2SE (unread). Claim at most "to our knowledge, not found in a limited search" and only once a result supports it.
3. Current data: weather has no effect on sparse-traffic success for any policy (n=40), so there is no demonstrated benefit to claim.
