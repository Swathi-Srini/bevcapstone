"""Insert an 'In plain words' box (and, where relevant, a 'Compared with GitHub' box) after the section heading of the technical-report sections.
Run once from the repository root after transform_sections_to_plain_language.py:
    venv\\Scripts\\python.exe report\\insert_plain_words_boxes.py
"""
from pathlib import Path

DIR = Path(__file__).resolve().parent / "technical"

PLAIN = {
"03_experimental_setup.tex": r"""\begin{plainwords}
Every model is tested in the same simulator, on the same driving scenarios, and counted the same way. This section defines the simulator, how an episode ends, which scenarios are used for training, for choosing a model and for the final test, the measures reported, and the statistics used to say whether one model is really better than another.
\end{plainwords}
""",
"04_perception_measurement.tex": r"""\begin{plainwords}
A real car would see through cameras. Fog and rain make cameras worse. Before testing whether weather hurts a driving model, we measured how much the weather filter of the project really damages the two perception steps that feed the model: the object detector that finds vehicles, and the stereo-depth algorithm that estimates how far they are. The result is that the damage is large, and that it consists mostly of \emph{missed vehicles}.
\end{plainwords}
\begin{cautionbox}[Compared with GitHub]
The functions measured here are the teammates' modules committed on GitHub (\pth{YOLO+weather module/}, \pth{bev_state/}); they were only called, never modified. The GitHub README describes the weather filter as ``preliminary tooling'' that ``is not evidence of physical fog/rain robustness''. GitHub contains \emph{no measurement} of the filter's effect on detection or depth; the measurement in this section is local only (Section~\ref{sec:github}).
\end{cautionbox}
""",
"05_model_architectures.tex": r"""\begin{plainwords}
All the learned drivers are built from the same small neural network, so that differences between drivers come from \emph{what they are shown} and \emph{how they were taught}, not from how big they are. This section draws the network, defines each of its inputs, and describes the two rule-based teachers.
\end{plainwords}
\begin{cautionbox}[Compared with GitHub]
The network is the teammates' \code{BEVScalarPolicy} from \pth{bev_policy/model.py} on GitHub (convolution layers of 32, 64 and 96 maps, then a layer of 128 numbers; a small network for the six vehicle-state numbers; a head of 128 units). The only difference is an optional extra-input branch added in a local, uncommitted edit; without it the network is identical to GitHub's and the two committed baseline checkpoints load unchanged. The map overlay, the collision-aware teacher and the image-only network do not exist on GitHub.
\end{cautionbox}
""",
"06_training_strategies.tex": r"""\begin{plainwords}
There are three ways to teach a driver in this project. (1) \emph{Copy a few recordings} of a good driver (behaviour cloning). (2) \emph{Let the driver drive, and ask a teacher what it should have done} in the situations it actually reached (dataset aggregation). (3) \emph{Trial and error with a score} (reinforcement learning). This section describes the first two in detail; the third has its own section.
\end{plainwords}
\begin{cautionbox}[Compared with GitHub]
GitHub contains only the first way: the collection script and the training script for behaviour cloning (\pth{bev_policy/collect_demos.py}, \pth{train_bc.py}), and the two checkpoints trained from five demonstrations each. Dataset aggregation, both teachers' use as labellers, the caution, position-error and degraded-perception variants, and all reinforcement learning exist only locally (Section~\ref{sec:github}).
\end{cautionbox}
""",
"08_results_headline.tex": r"""\begin{plainwords}
Seven drivers (and two rule-based teachers) drove the same 200 scenarios. The driver we propose succeeds in 92.5\% of them. The two baseline drivers that are committed on GitHub succeed in 85.0\% and 74.5\%. The proposed driver is statistically as good as the teacher it learned from.
\end{plainwords}
\begin{cautionbox}[Compared with GitHub]
The GitHub README states 90\% success (9 of 10 episodes) for the traffic-trained baseline. On 200 episodes the same checkpoint scores 74.5\% (interval 0.680 to 0.800); the README figure comes from ten episodes with the full camera pipeline, whose interval (0.60 to 0.98) is compatible with it. Table~\ref{tab:githubbaseline} gives the comparison.
\end{cautionbox}
""",
"09_results_weather_and_robustness.tex": r"""\begin{plainwords}
This section answers the original question: does the driving get worse in fog and rain? The answer is no, in sparse traffic, for the perception damage we measured. It also reports three side studies: a driver that becomes cautious when its perception is unreliable, which input design works best, and what happens when the car's position estimate on the map is wrong.
\end{plainwords}
\begin{cautionbox}[Compared with GitHub]
The GitHub README lists the weather evaluation as ``preliminary tooling'' and states that it ``does not change vehicle dynamics or road friction'' and is ``not evidence of physical fog/rain robustness or adaptive driving''. The results below complete that picture within the same limits: weather acts only through the camera pipeline and, in this report, only through a measured degradation of what the model sees.
\end{cautionbox}
""",
"11_failure_analysis.tex": r"""\begin{plainwords}
A good evaluation does not stop at ``92.5\% succeed''; it asks where the other 7.5\% go wrong and why. This section examines the proposed model's 15 failed episodes, the limits of the bird's-eye-view input, the difficulty of dense traffic, and what the baselines' failures look like.
\end{plainwords}
\begin{cautionbox}[Compared with GitHub]
The limitation of the grid (vehicles drawn without heading, a window of 17.5 metres ahead, a centre-based inclusion rule) is in the teammates' own code on GitHub, \pth{bev_state/assembler.py}. It is therefore a property of the repository, shared by every model that uses that grid, including the two committed baselines. A fix would belong in that module.
\end{cautionbox}
""",
"12_videos.tex": r"""\begin{plainwords}
Videos make the results tangible. We recorded one scenario, number 5169, in which both baseline drivers crash and the proposed driver does not, in four weather conditions, with the front camera only and with smooth, real-time playback. The image-only driver, which really does read these pictures, was recorded as well.
\end{plainwords}
\begin{cautionbox}[Compared with GitHub]
The video format follows the teammates' reference recording (the front-left camera frame with the weather filter applied, ten frames per second, one frame per simulation step). The recordings themselves are local only and excluded from git by the repository's \code{*.mp4} rule.
\end{cautionbox}
""",
"13_what_worked_what_did_not.tex": r"""\begin{plainwords}
One table of everything that was tried, with the verdict and how strong the evidence is, followed by the lessons that apply beyond this project.
\end{plainwords}
""",
"14_paper_directions.tex": r"""\begin{plainwords}
Seven possible papers, ranked by how much evidence already exists for each, what is still missing, and what it would cost to finish. The best single paper combines the first three directions.
\end{plainwords}
""",
"15_related_work.tex": r"""\begin{plainwords}
Twenty papers on arXiv, from 2025 and 2026 (plus the 2021 paper that introduced the simulator), on MetaDrive, weather and robustness, bird's-eye-view perception, and teacher--student imitation. For each: what problem it addresses, what method it uses, what it achieved, and how it relates to this project. The finding that matters most: none of them studies weather-degraded perception in MetaDrive.
\end{plainwords}
""",
}


def main():
    for name, text in PLAIN.items():
        f = DIR / name
        s = f.read_text(encoding="utf-8")
        lines = s.split("\n")
        # insert after the first line (the \section line)
        lines.insert(1, "\n" + text)
        f.write_text("\n".join(lines), encoding="utf-8")
        print("inserted into", name)


if __name__ == "__main__":
    main()
