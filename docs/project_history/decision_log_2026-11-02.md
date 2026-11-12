# Decisions
- D1: Original repo untouched; copies recorded in research_log/02_TRANSFER_LOG.md.
- D2: BEV v1 discarded (constant wedge+detections; 1 unique grid over 34 frames). Replaced by camera-derived BEV v2.
- D3: Camera geometry re-derived (side cameras were mirrored; camera height 1.935 m, not 1.4). Legacy YOLO+weather module left unedited; corrections live in `bev_state/ground_geometry.py`.
- D4: Fog = Koschmieder with per-pixel flat-ground depth; rain = seeded alpha-composited streaks without global brightness change. Legacy filter was image-space and darkened rain.
- D5: Surface classifier = small MLP on 15 patch features, trained on calibration seeds 3000–3002 (validated on 3003), inverse-sqrt class weights. Lane-line class merges solid and broken corridor because dash phase is unresolved.
- D6: PPO paused until BEV validated and integrated; cached-BEV shortcut retired because the BEV now varies.
- D7: 2026-11-02 user granted autonomy over the technical workflow; pause only for the items the user listed (objective change, claim-changing choices, destructive ops, external resources, unsafe hardware).
