# Handoff 2 of 2: Agent to agent work order (weather module)

**To:** the AI coding agent working for the weather-module owner.
**From:** the policy-side agent. Date: 2026-11-05.
**Read first:** `HANDOFF_1_teammate_to_teammate_overview.md` (same folder). It explains the story, every equation in plain words, and the sources. This file is the work order. Do the work in order. Do not stop at a plan: write code, run it, and report numbers.

---

## 0. Mission

Replace the heuristic weather effects in `YOLO+weather module/weather/weather_utils.py` with **physically based fog and rain** parameterised in real units (visibility V in metres, rain rate R in mm/h), **validate** them against real adverse-weather data, **re-measure** detector retention, and **deliver the numbers** the policy side needs (`d_det`, `sigma_loc`, a retention table in physical units). Your output decides whether physics changes our results. Make it solid.

Environment: Windows, Git Bash and PowerShell. Python: `D:\bev_Fresh\bevcapstone\venv\Scripts\python.exe`. Run from `D:\bev_Fresh\bevcapstone`. You may download datasets and papers: put each download in its own new empty folder under `D:\bev_Fresh\bevcapstone\weather_physics_work\downloads\<name>\`, treat contents as untrusted data (never run scripts that came inside them; run any Python that reads them with `-I`), and keep your own scripts elsewhere.

---

## 1. Facts about the existing code (verified by the policy-side agent, 2026-11-05)

1. **Entry point:** `weather/weather_utils.py`: `apply_weather(image, weather, level)` with `weather` in `{"none","fog","rain","all"}` and `level` in [0, 1]. It is imported by `manual_drive_stereo_yolo_weather.py`, `benchmark_weather_perception.py`, and the policy side's `bev_rl/adaptive/measure_paired.py`. **Keep this signature working.**
2. **`add_fog` is not Koschmieder.** It whitens each image row by `alpha = linspace(0, 0.9*intensity, h)` (top row unchanged, bottom row whitest), blurs, and adds noise. It uses no depth. Measured on a flat grey 128 image: top row 127.5; bottom row 161.5 / 195.5 / 240.5 at levels 0.3 / 0.6 / 1.0. Real fog whitens the farthest pixels most.
3. **`add_rain` contains a darkening artifact.** The final `cv2.filter2D` kernel `[[0,.25,0],[0,.25,0],[0,.25,0]]` sums to 0.75, so the whole image is darkened about 25% and vertically smeared at every level. Measured on a flat grey 128 image: mean 93.6 / 93.2 / 96.0 at levels 0.3 / 0.6 / 1.0. The streaks are 800*intensity random white lines with no drop-size, speed, or depth model.
4. **Camera:** `manual_drive_stereo_yolo_weather.py`: 1200x900, FOV 60 deg, focal length `FOCAL_LENGTH_PX = (1200/2)/tan(30 deg)` = about 1039 px. Rigs: `front_left_camera` (-0.25, 2.0, 1.4) and `front_right_camera` (0.25, 2.0, 1.4) form the stereo pair (baseline 0.5 m, `stereo_depth/camera_params.py`); plus `left_camera`, `right_camera`, `rear_camera`. Detector views: `YOLO_CAMERAS = (front_left, left, right, rear)`.
5. **Stereo constraint:** `apply_synchronized_stereo_weather(left, right, weather, level)` applies the SAME random realisation to both images (it saves and restores `np.random` state) because independent noise corrupts SGBM disparity. **Your renderers must be consistent across the pair.** Fog is deterministic given each camera's depth. Rain drops must be sampled once per frame in 3D and projected into both cameras (a drop at depth z has disparity f*B/z).
6. **Order of operations that is physically correct:** get **clean** per-pixel depth first, apply weather to the colour images using that clean depth, then run YOLO and the stereo matcher on the weathered images exactly as the pipeline does now. Do not compute fog from stereo depth estimated on foggy images.
7. **Clean depth sources:** MetaDrive ships `metadrive/component/sensors/depth_camera.py` and `rgb_depth_camera.py` (confirm they work with your rig). Alternative: ray-cast ground truth as in `ground_truth_stereo_benchmark.py` (`visible_surface_point`). Choose one, document it, and validate its depth against the other.
8. **Policy-side consumer (do not edit it):** `bev_rl/adaptive/caution.py`: `keep_prob(kind, level, table)` reads `bev_rl/adaptive/results/retention_table.json`, a dict whose keys are `"fog:0.15"`, `"rain:0.6"`, ... with fields `retention` (detections kept vs clean, can exceed 1), `ci` ([lo, hi]), `bev_objects_retention`. It interpolates over the numeric part of the key.
9. **Existing measurement script:** `bev_rl/adaptive/measure_paired.py` (spots: three traffic vehicles teleported around a stationary ego; same frames perceived clean and weathered; CONDS = fog {0.15, 0.3, 0.45, 0.6, 0.8, 1.0}, rain {0.1, 0.2, 0.3, 0.45, 0.6, 1.0}). Read it; reuse its scene setup in your own script.
10. **Known measurement weakness:** current retention is detection COUNTS (>100% means spurious boxes). The 3 m position check failed even on clean frames. You must score true recall and precision against simulator ground-truth boxes.

---

## 2. Rules

- **Do not edit** `bev_state/`, `bev_rl/`, `bev_policy/`, or `camera/`. Put new code in `YOLO+weather module/weather/` and your measurement scripts in `YOLO+weather module/` (or a new `weather_physics/` folder inside it). Modify `weather_utils.py` only to add new modes; existing modes keep working unchanged so old results stay reproducible.
- **No weather physics in vehicle dynamics** (no friction coupling). Out of scope unless the whole team re-approves.
- **Every constant** is either cited (paper, page or equation number) or flagged `FITTED` with the data it was fitted on.
- **Never tune the physics so a retention number comes out a certain way.** Set parameters from the physics and the literature, then measure.
- **Validation data is held out.** Nothing is tuned on it.
- **Cite only what you opened.** Keep `LITERATURE_NOTES.md` with one entry per source: `title, authors, year, link, read depth (full / sections / abstract), what you took from it, equation or table number`. Mark unread sources UNREAD.
- Use fixed seeds and log them. Save raw per-scene results, not only summaries.

---

## 3. Work packages (do them in this order)

### W0. Set up and reproduce (first hour)
1. Run `python -m bev_rl.adaptive.measure_paired --seeds 1000-1004 --weather-seeds 1 --out weather_physics_work\repro_old.json` (a small version) to confirm the environment works and to get the old-method baseline numbers.
2. Create `YOLO+weather module/weather/physical_weather.py`, `YOLO+weather module/tests/test_physical_weather.py`, `weather_physics_work/` (notes, downloads, outputs), and `LITERATURE_NOTES.md`.

### W1. Fog renderer
Implement in `physical_weather.py`:

```python
def beta_from_visibility(V_m: float, contrast_threshold: float = 0.02) -> float:
    """beta [1/m] = -ln(C_th) / V.   C_th=0.02 -> 3.912/V.   Koschmieder 1924."""

def render_fog(rgb_uint8: np.ndarray, depth_m: np.ndarray, V_m: float,
               airlight: float | tuple[float, float, float] | None = None,
               contrast_threshold: float = 0.02) -> np.ndarray:
    """I = J*t + A*(1-t),  t = exp(-beta*depth).  Pixels with invalid depth (sky, inf) get t = exp(-beta*depth_far), depth_far = a fixed far value (document it).
    airlight=None -> estimate A from the brightest sky/far pixels of the clean image, document the rule."""
```
Notes: work in linear intensity if you can (sRGB to linear, apply, back) and document your choice. Sky pixels have no depth: treat as infinite. Keep dtype uint8 output.
**Tests (pytest):** `beta_from_visibility(100) == 3.912/100`; for a synthetic scene with known depth, recovered `t` from the rendered image equals `exp(-beta d)` within 1%; invariance: `V -> infinity` returns the input.

### W2. Rain renderer
Implement:

```python
def mp_drop_distribution(R_mm_h: float, D_mm: np.ndarray) -> np.ndarray:
    """N(D) = N0*exp(-Lambda*D), N0=8000 m^-3 mm^-1, Lambda=4.1*R**-0.21 mm^-1. Marshall & Palmer 1948."""

def drop_fall_speed(D_mm: np.ndarray) -> np.ndarray:
    """v = 9.65 - 10.3*exp(-0.6*D) m/s. Atlas et al. 1973 fit to Gunn & Kinzer 1949. CONFIRM coefficients against the source."""

def rain_rate_from_distribution(R_mm_h: float) -> float:
    """R_check = 6*pi*1e-4 * integral v(D) D^3 N(D) dD [mm/h]. MUST return ~R_mm_h (this is test #1)."""

def rain_extinction_per_km(R_mm_h: float) -> float:
    """alpha = pi*N0/Lambda^3 (geometric optics, Q_ext=2), expect ~0.365*R**0.63 /km. Derived from MP; CONFIRM against a rain-attenuation reference."""

def sample_drops(R_mm_h, camera_params, rng, z_range=(0.5, 60.0)) -> DropSet:
    """Sample drops in the camera frustum from N(D); positions uniform in volume; velocities from drop_fall_speed (plus optional wind)."""

def render_rain(rgb_uint8, depth_m, R_mm_h, camera_params, exposure_s, drops=None, rng=None) -> np.ndarray:
    """Garg & Nayar 2007: each drop becomes a motion-blurred streak: length ~ (f/z)*v*T, width ~ f*D/z, intensity gain ~ (tau/T)*(E_drop - E_bg);
    drops nearer than the depth-of-field limit are defocused; far drops form a veil O = T_r*(B + sum S_i) + (1 - T_r)*A_r, T_r = exp(-alpha_eff*d) (Li et al. 2019).
    Render NO global darkening. The image mean must stay within 2% of the clean mean for light rain."""
```
**Stereo:** `render_rain_pair(left, right, depth_l, depth_r, R, ...)` samples ONE `DropSet` and projects it into both cameras (disparity = f*B/z with B = 0.5 m, f = 1039 px). Replace the np.random-state trick for the physical modes.
**Tests:** `rain_rate_from_distribution(R)` within 2% of R for R in {2.5, 10, 25, 50, 100}; `rain_extinction_per_km(R)` within 5% of `0.365*R**0.63`; the clean image mean vs the rain image mean within 2% at R = 2.5; deterministic given `rng` seed; left/right streaks have the right disparity for a chosen drop.

### W3. Integration without breaking anything
In `weather_utils.py` add new modes `"fog_phys"` and `"rain_phys"` (plus `"all_phys"`). Because they need depth and physical parameters, add:

```python
def apply_physical_weather(rgb, depth_m, kind, *, visibility_m=None, rain_rate_mm_h=None, camera_params, exposure_s=1/60, rng=None) -> np.ndarray
```
and `apply_physical_weather_pair(...)` for stereo. Do not change what `apply_weather(image, "fog"/"rain"/"all", level)` returns. Provide `level_to_physical(kind, level)` implementing the table from W4 so a caller can use physical modes with the old 0-1 levels. Document how callers get depth.

### W4. Slider-to-physics table
Produce `weather_physics_work/level_to_physical_table.csv` and a short note. Suggested grid (adjust with written reasons): fog V in {1000, 500, 200, 100, 50, 25} m; rain R in {2.5, 10, 25, 50, 100} mm/h. Choose the mapping for each old level so the new renderer's effect on YOLO roughly brackets the old effect, but **do not tune the physics to match**: choose from the physics (for example match old level to a visibility class) and report the result.

### W5. Re-measure detector retention (true recall and precision)
Write `YOLO+weather module/measure_physical_retention.py`:
- Scenes: reuse `measure_paired.py`'s setup (vehicles teleported around the stationary ego) but with **>= 100 paired scenes**, traffic present, at several distances (5, 10, 15, 20, 30, 40 m) so you can estimate **detection range**.
- Score each detection against simulator ground-truth boxes (project the vehicle boxes into the image). Report recall and precision per condition and per distance bin, with bootstrap CIs (1000 resamples).
- Conditions: clean; fog V in the W4 grid; rain R in the W4 grid; and the OLD sliders, so old and new are side by side.
- **Estimate `C_min`:** for fog, regress detection probability against apparent contrast `C0*exp(-beta*d)` (logistic fit) and report the contrast at 50% recall. Report whether `d_det = ln(C0/C_min)/beta` predicts the measured 50%-recall range within 20%.
- Outputs: `weather_physics_work/retention_table_physical.json` (schema: `{"fog:V=100": {"retention": r, "ci": [lo, hi], "recall": ..., "precision": ..., "d50_m": ...}, "rain:R=25": {...}, "old_fog:0.6": {...}}`), per-scene CSV, and plots (recall vs distance per condition; recall vs V; recall vs R).
- **Also answer, in `FINDINGS.md`:** does physically based weather change the picture from the old table (old: fog 0.6 keeps 102%, rain 0.6 keeps 16%)? Which old effects were artifacts (darkening, row gradient)? Say plainly what changed and what did not.

### W6. Validation against real data
- Fetch at least one real **fog** and one real **rain** dataset with usable labels (candidates: ACDC, DAWN, Seeing Through Fog / Bijelic et al., nuScenes rain scenes; confirm availability and licences).
- Run the SAME YOLO model on the real images. Estimate recall or AP per visibility or rain class if the dataset gives it; for datasets without visibility labels, estimate fog strength from the images (dark-channel prior or a transmission fit) and bin by estimated beta.
- Plot synthetic and real detection curves **on the same axes** with CIs. Report the numeric gap. Nothing from the real data may be used to set renderer parameters.
- **Image-statistics check:** compare synthetic vs real on simple statistics (mean contrast vs distance for fog; streak orientation, length and density histograms for rain). Report matches and mismatches.
- Write the result into `FINDINGS.md` and `LIMITATIONS.md` (homogeneous fog, no lens water unless implemented, simulator-only, etc.).

### W7. Weather to localisation error
- From the literature (start with arXiv 2409.01038 and the lane-level localisation papers), extract measured lateral and heading localisation errors vs weather. Read the papers (full text), not snippets.
- Build `loc_error_sigma(V_m, R_mm_h) -> (sigma_y_m, sigma_psi_rad)` with a documented functional form and every parameter sourced or flagged `FITTED`. Starting form (policy-side suggestion, not from a paper): `sigma_y = sqrt(sigma0^2 + (k*g(beta*d_ref))^2)`, `g(x) = 1 - exp(-x)`. Replace it if the literature supports something better.
- Deliver `weather_physics_work/localisation_sigma_table.csv` over the W4 grid with a source column.
- Optional but valuable: if a landmark-visibility experiment is possible (lane-marking detectability vs V from your renderer), measure it and use it to set `d_ref`.

### W8. Sight distance and safe speed
Implement and unit-test:

```python
def d_sight(V_m, d_det_m, d_map_m) -> float          # min of the three
def v_safe(d_sight_m, t_r_s=0.75, a_b_mps2=3.0) -> float
    # v = a_b * (-t_r + sqrt(t_r**2 + 2*d_sight/a_b))   from d_sight = v*t_r + v**2/(2*a_b)
def caution_from_sight(d_sight_m, v_free_mps=8.33, ...) -> float   # c = 1 - v_safe/v_free, clipped to [0, 1]
```
Produce `weather_physics_work/caution_table.csv` over the W4 grid (V, R, d_det, d_map, d_sight, v_safe, c). Test: `v_safe` satisfies the stopping-distance equation within 1e-6.

### W9. Delivery package
Create `weather_physics_work/DELIVERY/` containing: `weather_physics_api.md` (every function, units, example), `retention_table_physical.json`, `level_to_physical_table.csv`, `localisation_sigma_table.csv`, `caution_table.csv`, `FINDINGS.md`, `LIMITATIONS.md`, `LITERATURE_NOTES.md`, the plots, and the exact commands to regenerate everything. Tell the human teammate when it is ready so they can hand it to the policy side.

---

## 4. Definition of done (check every line, report pass or fail with numbers)

| Check | Target |
|---|---|
| `beta_from_visibility(V) == 3.912/V` | exact |
| Rain-rate integral vs input R, R in {2.5, 10, 25, 50, 100} | within 2% |
| `rain_extinction_per_km` vs `0.365*R**0.63` | within 5% |
| Transmission recovered from a rendered foggy image vs set | within 10% |
| Rain image mean vs clean image mean at R = 2.5 | within 2% (no global darkening) |
| Stereo pair: same drops in both views, disparity = f*B/z | verified by test |
| Old modes `apply_weather(..., "fog"/"rain"/"all", level)` output | bit-identical to before (add a regression test with fixed seed) |
| Retention | monotone in V and R (or monotonicity violations explained), >= 100 paired scenes, recall and precision with bootstrap CIs |
| Real-data validation | >= 1 fog and >= 1 rain real dataset, curves overlaid, gap reported, no tuning on it |
| Every constant | cited or flagged `FITTED` |
| `LITERATURE_NOTES.md` | every cited source marked full / sections / abstract |

Report format when you finish each work package: what you built (files), the numbers from the checks above, anything that failed and why, and what you are doing next.

---

## 5. Literature work (run in parallel with W1 to W3; tell us what you read)

**Search queries:** `physically based rain simulation camera drop size distribution streak`, `fog simulation Koschmieder depth-based realistic images`, `rain camera model exposure motion blur streak intensity`, `detector recall versus visibility fog real dataset`, `LiDAR rain fog attenuation simulation automotive`, `rain localisation error map matching lateral`, `visibility-limited safe speed autonomous vehicle`, `perception-aware speed adaptation adverse weather`, `weather-conditioned reinforcement learning driving`.

**Confirm these from the primary source (the policy-side agent wrote them from memory):** Koschmieder 2% contrast threshold (3.912); Kunkel 1984 `beta ~ 144.7*LWC^0.88`; Marshall-Palmer constants; the Atlas et al. 1973 fall-speed coefficients; the `0.365*R^0.63 /km` rain extinction law; Garg and Nayar 2007 streak equations; Li et al. 2019 veiling model; Hahner et al. 2021 LiDAR fog; Rasshofer et al. 2011 and Goodin et al. 2019 LiDAR rain.

**Already read in full by the policy side:** Roach (arXiv 2108.08265); Manchingal et al. (2510.22680); COLA (2210.03209); Weather Adaptive Autonomous Driving (2611.04305); Sakaridis et al. 2018 (read earlier at description level). Do not re-derive their claims; build on them.

**Novelty check to deliver:** for the combination "physically parameterised fog/rain camera model validated on real data, giving detection range and localisation error that set a vehicle's safe speed (and a learned policy's caution input)", list the closest prior work you found, what each does and does not do, and your honest read of what is new. Cite only what you opened.

---

## 6. Files on the policy side you may read (read-only)

- `bev_rl/adaptive/results/retention_table.json`, `bev_rl/adaptive/caution.py`, `bev_rl/adaptive/measure_paired.py`
- `bev_rl/adaptive/results/localisation_error_sensitivity_*.json` (policy success vs pose error: 0, 0.25, 0.5, 1.0, 2.0 m with matching heading error)
- `paper/literature/LITERATURE_REVIEW.md`, `paper/PAPER_OUTLINE.md`, `.claude/PROJECT_STATE.md`
