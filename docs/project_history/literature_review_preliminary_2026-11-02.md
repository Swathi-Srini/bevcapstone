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
