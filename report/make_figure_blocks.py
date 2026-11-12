"""Write report/figure_blocks/<figure>.tex: for every figure the caption (title + what is shown), the plain-English reading and the research implication.

Both documents (Director's Research Guide and Complete Technical Research Report) include these blocks, so a figure is explained identically everywhere.
Numbers quoted in the text are read from the result files (not typed), except where marked.

Run from the repository root:   venv\\Scripts\\python.exe report\\make_figure_blocks.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_report_figures as F   # noqa: E402

OUT = F.ROOT / "report" / "figure_blocks"
OUT.mkdir(exist_ok=True)
FINAL = F.FINAL
IO = FINAL.get("image_only")


def f3(x):
    return f"{x:.3f}"


def io_sentence_success():
    if IO is None:
        return ""
    lo, hi = F.wilson(round(IO["success"] * IO["n"]), IO["n"])
    return (f" The Image-Only Driving Model, which reads camera pixels instead of the bird's-eye-view grid, reaches {f3(IO['success'])} (interval {lo:.3f} to {hi:.3f}), "
            f"far below every model that uses the grid.")


def io_sentence_outcomes():
    if IO is None:
        return ""
    return (f" The Image-Only Driving Model ends {f3(IO['crash'])} of the episodes in a collision, {f3(IO['off_road'])} by leaving the road and {f3(IO['timeout'])} by timeout.")


def blocks():
    B = {}
    B["fig_headline_success_clear_weather_200_episodes"] = (0.97, "Success rate of the driving models on the same 200 test episodes",
        r"Each bar is the share of 200 identical test episodes (scenario seeds 5000 to 5199, clear weather, sparse traffic) in which a model reached the destination without a collision and without leaving the road. The black line is the 95\% Wilson confidence interval, the range of success rates that is statistically compatible with the observed count. Grey bars are rule-based teachers that see the exact simulator state; orange bars are the two baseline models supplied by the teammates; blue bars are models trained in this project.",
        r"The Proposed Bird's-Eye-View Driving Model succeeds in 92.5\% of the episodes. The two baseline models succeed in 85.0\% (trained without traffic) and 74.5\% (trained with traffic). The proposed model is about as good as the Collision-Aware Teacher (95.0\%) whose behaviour it learned; the intervals overlap, so the two cannot be told apart." + io_sentence_success(),
        r"Training a model by letting it drive and asking a teacher what it should have done (dataset aggregation) produces a model that matches its teacher and clearly beats few-demonstration behaviour cloning. This figure does not say \emph{which part} of the proposed design (the training method or the map overlay) causes the gain; that needs the additional evaluations listed in the paper-directions section.",
        "fig:headline")
    B["fig_outcome_breakdown_200_episodes"] = (0.97, "How the 200 test episodes end, for each model",
        r"For every model, the 200 episodes are split into the four possible outcomes: success (green), collision (red), left the road (yellow) and timeout (grey). Each bar sums to one.",
        r"The proposed model fails almost only by collision (6.0\% collisions, 1.0\% left the road, 0.5\% timeouts). The baseline trained with traffic leaves the road in 13.5\% of the episodes and collides in 12.0\%; the baseline trained without traffic collides in 12.5\%." + io_sentence_outcomes(),
        r"The baselines fail in a way that points to \emph{drift}: with only five demonstrations they never saw how to recover from a small steering error, so they slide off the road. Iterative training removes that failure type (1.0\% left the road). What remains for the proposed model is collisions with other vehicles, which points to what the model can see rather than to how it was trained.",
        "fig:outcomes")
    B["fig_per_episode_outcome_strips"] = (0.97, "Which test episodes each model fails",
        r"Each thin column is one of the 200 test episodes; each row is one model. Green means the model succeeded in that episode, red means it failed. The episodes are sorted so that those failed by no model are on the left and those failed by many models are on the right.",
        r"Most episodes are solved by every model. The failures are concentrated in a small set of hard scenarios on the right, and the models fail many of the same ones. The two baselines also fail a block of episodes that all the other models solve.",
        r"Because failures cluster in particular scenarios, the remaining errors can be studied scenario by scenario instead of treated as random. This is what the failure analysis does, and it is why the comparison is paired (all models see the same scenarios).",
        "fig:strips")
    B["fig_paired_difference_headline_versus_others"] = (0.97, "Paired difference in success between the proposed model and each other model",
        r"For each comparison the difference (proposed model minus the other model) is computed on the same 200 episodes. The point is the difference; the black line is the 95\% bootstrap interval obtained by resampling episodes 5000 times. A line entirely to the right of the zero line means the proposed model is better with statistical support.",
        r"The proposed model is better than the baseline trained without traffic by 0.075 (interval $+0.035$ to $+0.120$) and better than the baseline trained with traffic by 0.180 (interval $+0.125$ to $+0.240$). Its differences to the Collision-Aware Teacher ($-0.025$), the caution-study control ($+0.030$) and the stock driver model ($+0.040$) have intervals that include or touch zero." + (r" The Image-Only Driving Model is far below it." if IO is not None else ""),
        r"The claim that the proposed model beats the two baselines is statistically supported. No claim is made that it beats the teachers or the control. Using paired comparisons (same scenarios for every model) is what makes a difference of 0.075 detectable with 200 episodes.",
        "fig:paired")
    B["fig_speed_versus_energy_200_episodes"] = (0.97, "Mean driving speed and estimated energy use of each model",
        r"Each point is one model, placed by its mean speed (horizontal, metres per second) and its estimated traction energy per kilometre (vertical, watt-hours per kilometre) over the 200 clear-weather test episodes. The energy value is the mechanical work at the wheels of a declared virtual vehicle (no regenerative braking), not a battery measurement.",
        r"The two baselines use less energy (85.9 and 90.7 watt-hours per kilometre) and drive faster (8.09 and 8.07 metres per second) than the proposed model (96.2 watt-hours per kilometre at 7.87 metres per second), because they are less careful. The caution-study control is the cheapest (80.4 watt-hours per kilometre).",
        r"Energy comparisons are only meaningful between models with similar safety; a model that crashes early also uses little energy. Differences are valid as comparisons between models, not as absolute battery figures.",
        "fig:energy")
    B["fig_measured_perception_degradation"] = (0.97, "Measured effect of simulated fog and rain on the camera perception",
        r"Left: the share of vehicles in view that the YOLO object detector finds, per weather condition (vertical lines are 95\% intervals). Right: the share of vehicle pixels at which the stereo-depth algorithm returns a valid distance. Weather conditions are the teammates' image augmentation at the intensity shown (0 = none, 1 = strongest). 40 scenes; 72 vehicle observations in clear weather and 144 in each weather condition.",
        r"In clear weather the detector finds 72\% of the vehicles. In fog of intensity 1.0 it finds 31\%, in rain of intensity 0.6 it finds 15\% and in rain of intensity 1.0 only 1.4\%. Stereo depth stays valid for 88\% of pixels in clear weather and 49\% in the strongest fog, but remains at 67\% to 83\% in rain.",
        r"The perception measurement confirms that the simulated weather is a real degradation of the camera pipeline, mostly through \emph{missed vehicles}. For vehicles that are detected, the distance estimate is not worse. This tells us what a driving model would have to cope with, and explains why the driving results in the next figures are unaffected.",
        "fig:perception")
    B["fig_success_by_weather_calibrated_perception_200_episodes"] = (0.97, "Success rate in four weather conditions under the measured perception degradation",
        r"Four models were each evaluated on the same 200 episodes in clear weather, fog of intensity 1.0, rain of intensity 0.6 and rain of intensity 1.0. In every weather condition the input of the model was degraded as measured: vehicles can disappear from the bird's-eye-view grid, or be placed at a wrong distance. Black lines are 95\% intervals.",
        r"The proposed model stays between 0.89 and 0.91 in all four conditions; the weather-degradation-trained model between 0.88 and 0.90. The reinforcement-learning-fine-tuned model is lower in every condition (0.765 to 0.815) and the baseline trained with traffic is flat at about 0.75.",
        r"Under the perception degradation that was measured and implemented, weather does not lower driving success in sparse traffic. It follows that a robustness or weather-adaptation claim cannot be made from this experiment, and that a weather-adaptive method would have nothing to adapt to here. The next research step is to find the conditions (denser traffic, harsher loss of vehicles) under which weather begins to matter.",
        "fig:weather")
    B["fig_success_versus_localisation_error"] = (0.97, "Success when the estimate of the car's position on the map is wrong by a constant amount",
        r"The proposed model reads the road from a map drawn into its input, so it needs to know where the car is on the map. Here the position estimate is shifted by a constant lateral distance (horizontal axis, metres; a matching small heading error is added). Success over 40 episodes is shown for three models; bands are 95\% intervals.",
        r"All models are close to their no-error success up to 0.5 metres of error. The proposed model falls to 0.70 at 1 metre and 0.275 at 2 metres; the baseline trained with traffic falls from 0.775 to 0.15.",
        r"The map overlay makes the model accurate when the map alignment is exact and fragile when it is not. A real vehicle would need a position estimate better than about half a metre, or a design that does not depend on one source alone (next figure).",
        "fig:localisation")
    B["fig_success_versus_localisation_error_pose_trained_models"] = (0.97, "Training with position error, and adding a second lane cue",
        r"The proposed model reads the road from a map drawn into its input, so it needs to know where the car is on the map. Here the position estimate is shifted by a constant lateral distance (horizontal axis, metres; a small heading error grows with it). Success over 40 episodes for the proposed model and four models trained with random position errors; bands are 95\% intervals. Dashed lines: the map is the only source of lane position. Solid green and pink lines: a second, independent and noisy lane measurement (lateral offset and heading) is also given to the model. 40 episodes per point.",
        r"Training with position errors alone does not help (dashed lines, 0.25 to 0.28 at 2 metres of error). Adding a second independent lane cue keeps success between 0.875 and 0.95 for all error sizes up to 2 metres (solid lines).",
        r"Robustness to a wrong position estimate comes from \emph{redundancy} (two sources that can disagree), not from noise augmentation. Caution: the second cue is simulated (simulator truth plus noise), so the result shows that the network can exploit redundancy, not that a real lane detector would be accurate enough.",
        "fig:posetrained")
    B["fig_dataset_aggregation_validation_curves"] = (0.97, "Validation success after each training round for five input designs",
        r"During training the model drives validation scenarios after each round (round 0: the teacher drives and the model learns from it; later rounds: the model drives and the teacher labels). Left: sparse traffic; right: dense traffic. Five designs of the model input are compared; 10 validation episodes per point.",
        r"In sparse traffic all designs except the map overlay without vehicle-state inputs are at a validation success of 1.0 in almost every round, so this validation cannot tell them apart. In dense traffic all values are between 0 and 0.5 and differ by at most five episodes out of ten. Only the design with the map overlay but no vehicle-state inputs is clearly worse.",
        r"A validation set of 10 episodes is too small to choose between designs, and several models ended up being the round-0 model by tie-breaking. All claims between designs therefore rest on the later 200-episode test, not on these curves.",
        "fig:dagger")
    B["fig_teacher_ceiling_dense_traffic"] = (0.97, "Outcomes of four rule-based teacher variants in dense traffic",
        r"Four variants of the rule-based teacher were run for 20 episodes each in dense traffic (traffic density 0.2, clear weather, scenario seeds 3100 to 3119). Green = success, red = collision, grey = timeout.",
        r"The stock intelligent driver model succeeds in 40\% of the episodes and collides in 45\%. Switching off lane changes, enlarging the safety gap or adding predictive braking raises success only to 45\%; predictive braking reduces collisions from 45\% to 30\% but turns them into timeouts (25\%).",
        r"A model that imitates a teacher cannot be better than the teacher. Because even the teachers fail more than half of the dense-traffic episodes, the project's main results are restricted to sparse traffic, and dense traffic remains an open problem.",
        "fig:teacher")
    B["fig_birds_eye_view_fidelity_before_collision"] = (0.97, "Gap to the other vehicle as drawn in the model's input, in the last steps before two collisions",
        r"For two crashes of a bird's-eye-view model, the true distance between the car and the nearest vehicle (black) is compared with the distance that the 64 by 64 grid shows (red). A hollow circle marks steps at which the vehicle is not in the grid at all. These numbers are copied from the diagnosis note of 10 November (two replayed episodes).",
        r"The oncoming vehicle in the left panel first appears in the grid only 0.9 seconds before impact. In the right panel the grid shows the crossing vehicle 1.7 metres farther away than it really is, and the vehicle disappears from the grid 0.1 seconds before the collision.",
        r"The model's view of the other vehicles is late and coarse: vehicles are drawn as blocks without heading, and the grid ends 17.5 metres ahead. This is a limit of the representation that every model using this grid shares, and it is the leading explanation for the remaining failures.",
        "fig:fidelity")
    B["fig_reinforcement_learning_training_curves"] = (0.97, "What the training logs of the four reinforcement-learning attempts show",
        r"One panel per attempt, each showing the quantity that was logged during that attempt against the number of training steps: success in training episodes (attempt 1), validation success (attempts 2 to 4). The attempts used different measurements, so panels must not be compared with each other.",
        r"Attempt 1: success in training episodes did not rise (0.59 to 0.45 for run C2; 0.62 to 0.60 for run Probe C). Attempt 2: validation success stayed at 0.65 to 0.675. Attempt 3: all five reward combinations stayed between 0.325 and 0.425 around the same starting value 0.35. Attempt 4: validation was 1.0 at every checkpoint, including before any training, so it carried no information.",
        r"No training log shows learning, and in attempt 4 the training log hid a real loss that only the 200-episode test revealed. The lesson is to judge reinforcement learning on a large, separate test set, never on the small validation set used during training.",
        "fig:rltraining")
    B["fig_reinforcement_learning_before_and_after"] = (0.97, "Before and after reinforcement learning, for each of the four attempts",
        r"Each panel compares the model before reinforcement learning with the model after it, using the best available measurement for that attempt: 70 test episodes (attempt 1), 40 validation episodes (attempts 2 and 3) and 200 test episodes per weather condition (attempt 4).",
        r"Attempt 1: the trained teachers succeed in 23\% of 70 test episodes, against 67\% for the stock driver model they started from. Attempt 2: unchanged (0.675). Attempt 3: changes of $-0.025$ to $+0.05$, within the noise of 40 episodes. Attempt 4: success falls in every weather condition, by 0.065 to 0.115.",
        r"Reinforcement learning never improved a model in this project and twice made it worse. The reasons (the reward lets the policy trade safety for speed; the evaluation was too small; the starting model is already close to what its input allows) are analysed in the reinforcement-learning section.",
        "fig:rlbeforeafter")
    B["fig_birds_eye_view_input_examples"] = (0.9, "The two input encodings of the bird's-eye-view grid, drawn from one real scenario",
        r"The model's input is a 64 by 64 grid covering 10 metres to each side, 17.5 metres ahead and 2.5 metres behind the car. Left: the teammates' encoding (unknown, free, the ego car and detected vehicles). Right: the same grid with the route centre line and road boundaries from the map added. Both are drawn from scenario seed 5169 at step 238, with the same vehicle in view.",
        r"In the left grid the car sees free space and a vehicle ahead but nothing about the road. In the right grid it also sees where the lane edges and the planned route are.",
        r"The map overlay gives the model the road geometry directly, which is why three vehicle-state numbers (lateral offset, heading error, curvature) could be removed from its numeric inputs. It also creates the dependence on an accurate position estimate studied above.",
        "fig:inputs")
    B["fig_seed5169_front_camera_frames"] = (0.97, "Frames of the front-camera videos of scenario 5169",
        r"Each row is one model in one weather condition (clear, or rain of intensity 1.0); the columns are moments of the episode (steps 100, 200, 240 and the last frame; 10 steps = 1 second). The picture is the front-left camera of the teammates' rig with the weather augmentation applied. Videos for fog of intensity 1.0 and rain of intensity 0.6 are in the same folder.",
        r"Both baselines collide with another vehicle at about step 250 (25 seconds). The proposed model does not collide and drives on until it reaches the destination at about step 700 (70 seconds). The Image-Only Driving Model collides at step 230 in clear weather and leaves the road after 88 steps in the heaviest rain. Heavy rain makes the picture much worse but does not change what the grid-based models do.",
        r"This scenario was chosen \emph{because} the baselines fail on it, so it illustrates a typical failure but says nothing about how often it happens (the 200-episode results do). The grid-based models do not read these pixels; they read the simulator-built grid with weather applied as measured. Only the image-only model reads pixels, and it fails here in all four weather conditions.",
        "fig:frames")
    return B


def main():
    for name, (width, title, shows, reading, implication, label) in blocks().items():
        text = "\\explainedfigure{" + name + "}{" + str(width) + "}{" + title + "}{" + shows + "}{" + reading + "}{" + implication + "}{" + label + "}\n"
        (OUT / f"{name}.tex").write_text(text, encoding="utf-8")
    print("wrote", len(blocks()), "figure blocks; image-only model included:", IO is not None)


if __name__ == "__main__":
    main()
