# Pre-registration amendment 1 — image-estimated weather severity as the caution signal (2026-11-16)

Authorised by the user on 2026-11-16. This is a NEW document; `PREREGISTRATION_model_c_live_caution.md` and `PREREGISTRATION_e2e.md` are not edited. Where this amendment conflicts with them, this amendment governs for the severity-based system; the older text is kept as the record of the retired retention-based plan.

## 1. What changes and why
* **Retired:** detector-retention (`retention_table.json`, `c_auto`, old sensor weights) as the caution target. Reason: it is a property of the detector, not of the corruption, and cannot be measured without the expensive full pipeline.
* **New caution signal:** camera image → seven existing image features (`visibility_sensor.features`) → weather-kind classifier (clear/fog/rain) + separate fog and rain intensity regressors → caution `c`:
  clear → 0; fog → predicted fog level; rain → predicted rain level (identity mapping). Fog and rain levels are normalised inside their own generators; equal numbers do **not** mean equal visibility or danger. This is a design assumption, not a physical claim.
* **Unchanged:** weather transformations (`weather_utils.py`), live five-camera path, density 0.05, teacher relationship (target speed 30·(1−c/3) km/h, gap 10+15·c m), SafeTeacher + curve guard 4.0, OV-Split architecture, `bev_rl/energy.py`, Model A and all historical checkpoints.

## 2. Models
* **A** existing camera OV-Split student (`camera_dagger_final/e2e_ov_split_final/best.pt`), no caution input. Not modified.
* **B** oracle-caution reference: same architecture as the caution student, trained by camera DAgger (`camera_dagger_severity.py`, rounds 0–3) with the true identity caution as both teacher parameter and student input.
* **C** proposed system: the **same trained network as B**, with the image-estimated filtered caution at inference (plug-in). No second DAgger run is launched unless the plug-in shows a meaningful degradation relative to B (criterion in §6); in that case in-loop training is the corrective step and will be reported as a deviation.

## 3. Estimator protocol
* Training data: (i) whole-route frames recorded during the Model B DAgger rounds (every step, density 0.05, seeds 2000–2999 scenarios, weather levels clear, fog 0.3/0.8/1.0, rain 0.2/0.6/1.0); (ii) the existing 28 retention-pass episodes (seeds 9000–9027), non-probe conditions only, as *preliminary* material (limited scene diversity, stated as such).
* Held-out: streams of seeds 9040–9049 (whole route, clear every step, all 10 weather conditions at even steps, collected independently by `measure_retention_camera collect`, Teacher A, no action noise). Never used for fitting or tuning.
* Probes never enter any fit: fog 0.45, rain 0.3 (old), fog 0.6, rain 0.45 (policy-evaluation probes).
* Hyper-parameters fixed a priori (softmax L2 λ=1e-2, ridge λ=1, balanced class weights, live EMA α_p=α_c=0.15). No tuning on held-out data.
* Reported on held-out: kind accuracy and macro recall, fog and rain MAE separately, per-level bias (calibration), probe-level error, clear-weather false caution rate, and the same quantities after the live filter.
* **Stability requirement for live control:** clear-weather false-caution frame rate (c_hat > 0.05) reported; filtered clear c_hat mean reported.

## 4. Model B training (documented deviation from the original recipe)
Recipe of `camera_dagger.py` unchanged except: 24 episodes per round instead of 40 (schedule: ~15–20 min per camera episode, 3 days to the paper deadline), balanced weather list with the evaluation probes removed, horizon 600, rounds 0–3, validation only on rounds 2 and 3 (clear, fog 1.0, rain 1.0 × seeds 3100–3103, horizon 1600). Selection rule: (mean success, −mean(crash+off-road)) as in the original. Teacher B behaviour is validated across clear, fog and rain with the identity mapping (`validate_teacher_identity_caution.py`, n=50 per condition, seeds 3100–3149) before its demonstrations are used; a failure (< 0.90 success in any validated condition) stops the pipeline for review.

## 5. Evaluation
* Density 0.05; primary horizon 1600 steps; horizon 800 derived from the same traces (secondary).
* Primary conditions: clear, fog 1.0, rain 1.0, seeds 5200–5219 (20 paired seeds); probe conditions fog 0.6 and rain 0.45, seeds 5200–5209 (10 paired seeds). Seeds are never used for training or tuning.
* Arms A, B, C on the same scenario seed with identical per-step weather noise.
* **Pilot-scale evidence.** With n=20 paired seeds a formal non-inferiority claim is not possible (a 10-pp margin needs ≳79 pairs at observed discordance 0.10). Results are reported with exact paired (McNemar) tests, Wilson / bootstrap CIs and are labelled pilot.
* Primary outcomes: success; collision and off-road rates; paired Wh/km on episodes where both compared policies succeed (all-episode energy reported separately). Secondary: route completion, mean speed, duration, minimum gap, braking (share of steps with deceleration below −2 m/s²), estimator accuracy, clear-weather false caution, speed vs severity.
* Energy = `bev_rl/energy.py` unchanged (Wh per episode, Wh/km, distance from the odometer). Mass is read from the simulator at run time.

## 6. Acceptance criteria (descriptive at pilot scale; no formal non-inferiority claim)
Inherited from the earlier pre-registrations: paired exact McNemar + Wilson CIs, cluster bootstrap (2,000 resamples, seed 20261115), the estimator-gate structure (kind accuracy, per-kind MAE against a constant baseline) and the Gate-3 style comparison of estimated vs oracle caution. New decisions in this amendment (rationale: the pilot baseline's own sampling uncertainty at n=20 is ±15–20 pp, so only large harm is detectable):
1. **Success:** B and C point estimates not more than 10 pp below A in each primary condition (pooled and per condition). A single-condition drop > 10 pp is reported as a failure of the adaptation, not hidden by pooling.
2. **Safety:** collisions + off-road for C (and B) not higher than A's in the pooled adverse-weather conditions (fog 1.0 + rain 1.0), and in clear weather not more than 1 additional event.
3. **Selective adaptation:** mean speed of C in clear weather within 5% of A; mean speed of C in fog 1.0 and rain 1.0 lower than in clear and monotone in true severity across the probe levels.
4. **Energy** is declared a positive result only if criteria 1–2 hold: paired both-success Wh/km of C below A with the 95% CI of the paired mean difference excluding 0 in fog 1.0 and rain 1.0, effect reported as a percentage. If safety does not hold, energy is reported but not claimed.
5. **Cost of estimation (C vs B):** C within 10 pp of B on success, and no more events than B + 1 in adverse weather. If violated: investigate in-loop training as the corrective step.
6. **Estimator:** held-out kind macro recall ≥ 0.95 and per-kind MAE below 0.5× the constant-prediction baseline (inherited structure; numeric values taken from the feasibility check ≈ 0.99 / 0.06 / 0.035, constant baselines 0.225 / 0.237).

## 7. Known limitations stated in advance
Identity mapping is an assumption; the always-cautious reference (slow in every weather) reaches similar energy in the earlier BEV-level proxy, so an energy claim is only a claim about *selective* adaptation if clear-weather speed/energy is preserved (criterion 3); the camera evaluation is pilot-scale; vehicle mass is read live and not otherwise recorded; the bumper-gap logging is privileged simulator truth (analysis only); the plug-in design means C is evaluated on a network trained with oracle c (inputs from the estimator may lie slightly off the training distribution).

## 8. Relation to the older architecture diagram and methodology document (recorded 2026-11-16, at the user's instruction)
The user confirmed that the **Nov 16 specification governs**: weather-kind classifier + separate fog and rain intensity regressors, identity caution mapping, teacher target speed 30*(1-c/3) km/h and following distance 10+15c m. Training is NOT restarted.

| Item | `report/Architecture_Live_Caution_Branch_v2.drawio` (Nov 14) | This work (Nov 16 spec, implemented) |
|---|---|---|
| Image estimator | front-left frame -> effective visibility range R_eff (regression on probe measurements) | front-left frame -> 7 existing features -> kind (clear/fog/rain) + fog or rain intensity |
| Caution | c = 1 - min(R_eff, R_ref)/R_ref | identity: clear 0, fog = fog level, rain = rain level |
| Teacher | IDM + SafeTeacher + curve guard 4.0 with a visibility cap v_vis = -b*t_r + sqrt(b^2 t_r^2 + 2b(R-m)), target = min(v_IDM, v_vis) | caution IDM (30*(1-c/3) km/h, 10+15c m) + SafeTeacher + curve guard 4.0 (`cs0_g4.0`) |
| Supervision of the estimator | simulator-truth range labels from probe measurements | known synthetic corruption parameters (kind, level) |
| Unchanged | cameras, weather intervention, stereo SGM + YOLOv8n, HD map, frozen BEV builder, scalar state, CNN/MLP/caution branch (FC 1->16), fusion 176D, action head 176->128->2 | same |

`Final_Methodology_and_Approach.tex` (retention-conditioned caution; 40 episodes per round; rounds 0-5) is likewise superseded where it conflicts; its retention-based caution is retired. The diagram and that document are not edited; they remain the record of the earlier plans. Any figure of the system in the paper must show the Nov 16 estimator and teacher, not the R_eff box.
