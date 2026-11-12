# Brief for the weather-module owner (and their AI agent): make the weather module physically grounded

Written 2026-11-05 by the policy-side agent. Audience: the teammate who owns `YOLO+weather module/` (camera, YOLO detector, fog/rain image augmentation) and their coding agent.

**The policy side will NOT implement any rain/fog physics. This document asks you to.** Everything marked *[verify]* comes from memory or search snippets and must be checked against the primary source before it is used or cited.

---

## 0. Why this matters (one paragraph)

Our paper studies a driving policy that must stay safe and energy-efficient when weather degrades what the vehicle can perceive. Today "weather" in our pipeline is an **uncalibrated 0-1 slider** (fog 0.15...1.0, rain 0.1...1.0) applied to camera images. We then *measure* how many YOLOv8 detections survive (`retention`). That is honest, but it is not physics: nobody can say what fog 0.6 means in metres of visibility, or what rain 0.6 means in mm/h.

**If you can replace the slider with physically parameterised models (visibility V in metres, rain rate R in mm/h), validate them against real adverse-weather data, and give us the resulting detection and localisation degradation as functions of those physical units, that can become a main contribution of the paper.** It is a very big "if". The contribution exists only if the physical model is (a) correct, (b) validated against real data, and (c) shown to change what the policy should do. If it does not validate, we report that honestly instead.

---

## 1. The whole project (read this; use it to search the literature)

**Working title:** Weather-aware, energy-efficient driving from a BEV representation: coupling measured perception reliability to a learned control policy.

**Pipeline.** MetaDrive simulator -> camera + YOLOv8 detector (weather augmentation) -> 64x64 single-channel BEV grid (`bev_state/`, GitHub version, frozen: ego, free/unknown space, detected vehicles) -> **policy** (CNN on the BEV + small MLP on scalars, optional caution input) -> steering/throttle. Energy comes from a traction model (rolling resistance + aero + inertia), reported in Wh/km.

**Policy side (done by the policy agent):**
- Imitation by DAgger from a privileged IDM teacher. This is a Roach-style design (Zhang et al., ICCV 2021, arXiv 2108.08265): route and lane boundary are drawn into the BEV from the HD map, and ego dynamics stay in scalars. **This part is NOT novel.**
- Detector failure is simulated by dropping each simulator vehicle from the BEV with probability 1 - keep_prob(weather). keep_prob comes from **your module's measured retention table** (`retention_table.json`).
- A "caution" input c = 1 - keep_prob(weather) feeds the policy. The teacher slows (30 -> 20 km/h) and widens its gap (10 -> 25 m) as c rises. This mapping is hand-set, not physics.

**Measured so far (honest; n=40 fresh seeds unless stated):**
1. Detector retention vs your slider (20 paired scenes, 222 clean detections; these are COUNTS, not recall; the 3 m position check failed even on clean frames, so no recall claims). Fog 0.15/0.3/0.45/0.6/0.8/1.0 keeps 104/113/110/102/71/27 %. Rain 0.1/0.2/0.3/0.45/0.6/1.0 keeps 95/84/71/51/16/4 %.
2. In **sparse traffic (density 0.05)** this detector dropout does NOT change task success for any policy (0.78-0.95 success, flat across weather). There are few vehicles, so little to miss. We report this honestly.
3. A degradation that DOES hurt is **ego-pose (HD-map localisation) error**. The overlay-BEV policy holds 0.90 success up to 0.5 m / 0.05 rad of pose error, then 0.70 at 1 m and 0.28 at 2 m (mostly off-road). Scalar-input policies fall to about 0.50 at 0.5 m. Search snippets say map-based localisation degrades in rain (6.05 m vs 2.46 m error in one study, arXiv 2409.01038, *[verify]*). **A physically grounded link from weather to localisation error is the second thing we need from you** (section 5).
4. Caution input in sparse traffic: success unchanged, energy about -30 % (85 -> ~57 Wh/km), more timeouts (0.08-0.15). That is an energy/time trade, not robustness.

**What is NOT shown and must not be claimed:** that weather hurts the policy through detection in sparse traffic; that caution helps; that anything is novel before the literature check below.

**Key literature already read (full text):**
- Roach (arXiv 2108.08265).
- Manchingal et al. (arXiv 2510.22680): classifier entropy -> hand-coded speed scaling, racing car, no weather.
- COLA (arXiv 2210.03209): weather as a hidden mode, online meta-RL, CARLA lane keeping.
- Weather Adaptive Autonomous Driving (arXiv 2611.04305): per-weather conformal detector thresholds, perception only.
- Sakaridis et al. 2018 (Foggy Cityscapes, Koschmieder fog model).

**Search these, and tell us what you actually read (full text vs abstract):** "adverse weather perception-aware planning"; "visibility-limited safe speed"; "weather-dependent detector recall vs visibility"; "physically based rain fog simulation camera LiDAR"; "localisation degradation rain map matching"; "weather-conditioned reinforcement learning driving"; "operational design domain weather speed adaptation"; "sight-distance-based speed control autonomous vehicle fog".

---

## 2. What we ask you to build (deliverables)

- **D1. `weather_physics.py`** in your module. Pure functions with units in each docstring: `beta_from_visibility`, `fog_render`, `rain_dsd`, `rain_extinction`, `rain_render`, `lidar_max_range` (optional), `loc_error_sigma` (section 5). No hidden magic constants: every constant cites a source or is flagged as a fitted parameter.
- **D2. A mapping from our 0-1 levels to physical units**, as a table. Suggested grid (adjust with reasons): fog V = {1000, 500, 200, 100, 50, 25} m; rain R = {2.5 (light), 10 (moderate), 25, 50 (heavy), 100 (extreme)} mm/h. Keep the old slider working for backward compatibility.
- **D3. `retention_table_physical.json`**: detector retention vs physical units, same schema as today (`"fog:V=100": {"retention": x, "ci": [lo, hi]}`). Use at least 100 paired scenes with traffic present, and report TRUE recall and precision (fix the 3 m position check, or use simulator ground-truth boxes) instead of raw counts.
- **D4. Validation report** (this decides whether it is a contribution). Compare your synthetic fog/rain against real data: YOLO AP/recall vs visibility on real adverse-weather sets (ACDC, DAWN, Seeing Through Fog / Bijelic et al., nuScenes rain subsets; *[verify availability]*; Foggy Cityscapes is synthetic and does not count as real). Plot the synthetic and real curves together and report the gap. Also estimate beta from your rendered images (dark-channel or transmission fit) and check it equals the beta you set.
- **D5. Do not edit** `bev_state/` or any `bev_rl/adaptive/` policy file. Deliver files into your module plus a short CHANGELOG. If your change alters an interface the BEV assembler uses (for example `visibility_range_m`), say so explicitly.

---

## 3. Fog physics (camera and LiDAR)

**Koschmieder / atmospheric scattering model** (homogeneous fog; Koschmieder 1924; used by Sakaridis et al. 2018):

```
I(x) = J(x) t(x) + A (1 - t(x)),     t(x) = exp(-beta d(x))
```
J = clear-scene radiance, A = atmospheric light, d = scene depth (MetaDrive provides depth), beta = extinction coefficient [1/m].

**Visibility to extinction.** The apparent contrast of an object at distance d is C(d) = C0 exp(-beta d). Meteorological visibility V is the distance at which contrast falls to the threshold C_th = 0.02:

```
V = -ln(C_th) / beta = 3.912 / beta        (C_th = 0.05 gives 3.0 / beta)
=>  beta = 3.912 / V
```

**Microphysics (for reference).** Fog droplets have radius r ~ 1-20 um, much larger than the wavelength, so the extinction efficiency is Q_ext ~ 2 (non-selective at visible and 905 nm):

```
beta = integral pi r^2 Q_ext n(r) dr,     n(r) = a r^alpha exp(-b r^gamma)   (modified gamma; Shettle & Fenn 1979 [verify])
beta ~ 144.7 * LWC^0.88   [1/km, LWC in g/m^3]                                (Kunkel 1984 [verify])
```

**Detector-relevant quantity: sight distance.** A detector needs apparent contrast above its own threshold C_min (estimate it empirically with YOLO on rendered fog):

```
d_det = ln(C0 / C_min) / beta = (V / 3.912) * ln(C0 / C_min)
```

**LiDAR in fog** (Beer-Lambert two-way attenuation, extended Lambertian target):

```
P_r(R) = P_t * rho * A_r * eta / (pi R^2) * exp(-2 alpha R),      alpha ~ beta (905 nm)
alpha(lambda) = alpha(550 nm) * (lambda / 550 nm)^(-q),   q = 1.6 (V > 50 km), 1.3 (6-50 km), 0.585 V_km^(1/3) (V < 6 km)   [Kim/Kruse; verify]
R_max solves P_r(R_max) = P_min   <=>   R^2 exp(2 alpha R) = rho P_t A_r eta / (pi P_min)
```
Fog also adds near-field backscatter (volume returns) that can mask targets; see Hahner et al., "Fog simulation on real LiDAR point clouds", ICCV 2021 *[verify]*.

**Optional extension:** spatially varying beta(x) (patchy fog). Legitimate, not required.

---

## 4. Rain physics (camera and LiDAR)

**Drop size distribution (Marshall-Palmer 1948):**

```
N(D) = N0 exp(-Lambda D),     N0 = 8000 m^-3 mm^-1,     Lambda = 4.1 R^(-0.21) mm^-1       (R in mm/h)
```

**Terminal fall speed** (Gunn-Kinzer; Atlas et al. 1973 fit *[verify]*):

```
v(D) = 9.65 - 10.3 exp(-0.6 D)     [m/s, D in mm]
```

**Consistency check you must run.** The rain rate recomputed from the distribution must return the R you started with. With D in mm, v in m/s and N in m^-3 mm^-1, the standard form is

```
R = 6 pi 1e-4 * integral v(D) D^3 N(D) dD        [mm/h]       (the pi/6 volume factor is already inside the 6 pi 1e-4)
```
If it does not return your input R, your units are wrong (check this before using any other formula in this section).

**Optical extinction by rain** (geometric optics, Q_ext = 2):

```
alpha_rain = integral (pi D^2 / 4) * 2 * N(D) dD = (pi/2) * (2 N0 / Lambda^3) = pi N0 / Lambda^3
=>  alpha_rain ~ 0.365 * R^0.63   [1/km, R in mm/h]            (the commonly quoted law [verify])
```
The rain visibility V_rain = 3.912 / alpha_rain is about 0.9 km at 50 mm/h. So rain barely lowers *visibility* at 20-50 m. Its strong effect on cameras comes from **streaks, lens/windshield water and veiling**, not from extinction. State this clearly in the paper: it is the physical reason rain 0.6 destroys YOLO detection counts while fog 0.6 does not.

**Camera streaks** (Garg & Nayar, "Vision and Rain", IJCV 2007 *[verify]*). For a drop of diameter D at depth z, exposure time T and focal length f in pixels:

```
streak length  l  ~ (f / z) * v(D) * T
streak width   w  ~ f D / z
intensity gain dI ~ (tau / T) * (E_drop - E_bg),     tau = time a drop spends inside a pixel footprint ~ pixel_size / ((f / z) * v)
```
Drops closer than a depth limit are blurred out of focus (depth of field). Far drops merge into a **veil**:

```
O = T_r * (B + sum_i S_i) + (1 - T_r) * A_r,     T_r = exp(-alpha_eff d)      (Li et al., CVPR 2019 [verify])
```
Implement streaks by sampling drops from N(D) inside the camera frustum with these geometric rules, not by pasting texture overlays.

**Water on lens/windshield:** model as spatially localised defocus plus refraction (Roser & Geiger; Halimeh & Roser 2009; search "raindrop detection on windshield" *[verify]*). Optional, but it dominates real-camera rain degradation.

**LiDAR in rain:** attenuation exp(-2 alpha_rain R) plus random volume returns and dropout. See Rasshofer et al. 2011; Goodin et al. 2019 (Electronics 8(1):89); Heinzler et al. 2020; Kilic et al. 2021 (LISA) *[verify all]*.

**Road surface (vehicle dynamics; OUTSIDE this brief's scope unless the team re-approves).** An earlier team decision dropped rain-friction coupling in MetaDrive. If it is ever reinstated: stopping distance d_stop = v t_r + v^2 / (2 mu g), and Burckhardt tyre friction mu(s) = c1 (1 - exp(-c2 s)) - c3 s, with dry asphalt (c1, c2, c3) = (1.2801, 23.99, 0.52) and wet asphalt (0.857, 33.822, 0.347) *[verify]*. Do NOT enable this without the team's explicit OK.

---

## 5. Weather -> localisation error (second request; probably the real hook for our policy)

Our policy fails mainly because of **map-alignment error**, not detection loss in sparse traffic. We need a defensible model sigma_loc(weather) for the ego pose against the HD map: a lateral sigma_y [m] and a heading sigma_psi [rad], as functions of V and R. One possible form:

```
sigma_y = sqrt( sigma0^2 + (k_s * g(beta * d_ref))^2 ),     g(x) = 1 - exp(-x)
```
d_ref = distance to the farthest useful landmark; sigma0 ~ 0.1-0.3 m is clear-weather map matching; k_s is fitted to published rain/fog localisation studies.

Candidate sources *[verify]*: arXiv 2409.01038 (rain, street maps; snippet says 6.05 m vs 2.46 m); the 17 cm lane-level localisation requirement (USDOT); LiDAR map-matching in rain/snow papers.

**Deliver sigma_y(V, R) and sigma_psi(V, R), with a source for every number.** We will plug them into our pose-error experiments (we already sweep 0-2 m and 0-0.2 rad).

---

## 6. The link to the policy (what makes it a paper, not just a renderer)

The vehicle must be able to stop within what it can reliably see. With sight distance d_sight = min(V, d_det, d_map):

```
d_sight = v t_r + v^2 / (2 a_b)
=>  v_safe = a_b * ( -t_r + sqrt( t_r^2 + 2 d_sight / a_b ) )
```
t_r is the reaction time (about 0.5-1.0 s) and a_b is the comfortable braking deceleration (about 3 m/s^2; mu*g in the wet).

Our hand-set caution mapping (30 -> 20 km/h) should become **c = 1 - v_safe(d_sight) / v_free**, with d_sight computed from YOUR physical quantities. If the physics-derived caution reproduces or beats the hand-set one, and the policy trained on it is safer or more efficient under weather, THAT is the contribution. You supply d_det(V, R) and sigma_loc(V, R). The policy side supplies the learned policy and the evaluation.

---

## 7. Acceptance criteria (be strict; we would rather report a null result than a false positive)

1. Unit tests: beta = 3.912 / V exactly; the Marshall-Palmer rain-rate integral recovers R within 2 %; alpha_rain matches 0.365 R^0.63 km^-1 within 5 %; a transmission fit on a rendered image recovers beta within 10 %.
2. Retention curves are monotone in V (fog) and R (rain), with bootstrap confidence intervals, on at least 100 paired scenes, using true recall and precision.
3. Validation against at least one real adverse-weather dataset per weather type. Synthetic and real curves are plotted together, the gap is reported, and nothing is tuned on the validation split.
4. Every constant is sourced or flagged as fitted. A `LIMITATIONS.md` lists the assumptions (homogeneous fog, no lens water if skipped, simulator-only, and so on).
5. A one-page note answering: does physically parameterised weather change the detector-retention picture we already have (fog 0.6 harmless, rain 0.6 near-total loss)? If not, say so.

## 8. Things you must NOT do

- Do not tune the physics so retention matches what would make our policy look good.
- Do not claim novelty until you have read the primary papers, and list what you read in full vs abstract only.
- Do not edit policy or BEV assembler code.
- Do not fabricate citations; mark unread items as unread.

## 9. Files on our side you can read

- `bev_rl/adaptive/results/retention_table.json`: the current measured retention.
- `bev_rl/adaptive/caution.py`: `keep_prob`, `c_auto`, `expert_params`.
- `paper/PAPER_OUTLINE.md` and `paper/literature/LITERATURE_REVIEW.md`: what we read and at what depth.
- `bev_rl/adaptive/results/localisation_error_sensitivity_*.json`: policy success vs pose error.
- `.claude/PROJECT_STATE.md`: the decision log.
