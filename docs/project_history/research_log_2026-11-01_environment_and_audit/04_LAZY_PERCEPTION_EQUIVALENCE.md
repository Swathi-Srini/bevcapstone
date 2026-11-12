# Lazy-stereo perception vs full perception

Claim: `LazyStereoPerception` (skips SGBM when the front camera has no detection) returns the same BEV and
scalar state as the repository's `FullPerception` (`BEVStateAssembler.assemble`).

Run 1 (2026-11-01, `python -m bev_rl.validate_lazy_perception`, seed 300, IDM ego, traffic density 0.3, 70 frames;
log: `lazy_equivalence_run1.log`; an earlier larger run was killed by the OS for low memory and produced no result):

| frames | bit-identical (BEV and scalars) | frames with occupied BEV cells | mean s/frame full | mean s/frame lazy |
|---|---|---|---|---|
| 70 | 70 | 1 | 2.11 | 1.85 |

Limits of this evidence: only ONE of the 70 frames had a detection that put occupied cells in the BEV, so the
stereo-using branch was exercised rarely. The equivalence additionally rests on reading the code
(`objects_from_detections` uses the depth map only for `front_left_camera` detections). Timing here is for a
traffic scene, where lazy skipping is rarely possible; the zero-traffic speed-up has not been re-measured.
