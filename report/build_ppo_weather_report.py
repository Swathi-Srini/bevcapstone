from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path

ROOT=Path(r'E:\Capstone\bevcapstone')
OUT=ROOT/'report'/'PPO_Weather_Study_Complete_Explanation.docx'
FIG=ROOT/'report'/'figures'/'ppo_weather_pipeline.png'
D=Document(); sec=D.sections[0]
sec.top_margin=Inches(.52);sec.bottom_margin=Inches(.48);sec.left_margin=Inches(.7);sec.right_margin=Inches(.7)
styles=D.styles
styles['Normal'].font.name='Aptos';styles['Normal'].font.size=Pt(9.0);styles['Normal'].font.color.rgb=RGBColor(38,52,66)
styles['Normal'].paragraph_format.space_after=Pt(2)
for n,size,color in [('Title',27,'15324B'),('Heading 1',18,'15324B'),('Heading 2',13,'1E6078'),('Heading 3',10.5,'1E6078')]:
 s=styles[n];s.font.name='Aptos Display';s.font.size=Pt(size);s.font.bold=True;s.font.color.rgb=RGBColor.from_string(color)
 s.paragraph_format.space_before=Pt(9);s.paragraph_format.space_after=Pt(4)
# Remove the blue bottom rule inherited from Word's built-in Title paragraph style.
title_ppr=styles['Title']._element.get_or_add_pPr()
title_border=title_ppr.find(qn('w:pBdr'))
if title_border is not None: title_ppr.remove(title_border)
# header/footer
h=sec.header.paragraphs[0];h.text='BEV DRIVING RESEARCH  |  PPO WEATHER STUDY';h.style='Caption';h.runs[0].font.color.rgb=RGBColor(80,105,125)
f=sec.footer.paragraphs[0];f.alignment=WD_ALIGN_PARAGRAPH.RIGHT
run=f.add_run('MetaDrive simulation study  •  ');run.font.size=Pt(8);run.font.color.rgb=RGBColor(100,115,125)
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');f._p.append(fld)

def shade(cell,fill):
 tcPr=cell._tc.get_or_add_tcPr();shd=OxmlElement('w:shd');shd.set(qn('w:fill'),fill);tcPr.append(shd)
def set_cell(cell,text,bold=False,color='263442',size=8.2):
 cell.text='';p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(1);p.paragraph_format.space_before=Pt(1)
 r=p.add_run(str(text));r.bold=bold;r.font.name='Aptos';r.font.size=Pt(size);r.font.color.rgb=RGBColor.from_string(color)
 cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER

def table(headers,rows,widths=None,size=8.2):
 t=D.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.style='Light Shading Accent 1';t.autofit=False
 for i,x in enumerate(headers):set_cell(t.rows[0].cells[i],x,True,'FFFFFF',size);shade(t.rows[0].cells[i],'15324B')
 trPr=t.rows[0]._tr.get_or_add_trPr();rep=OxmlElement('w:tblHeader');rep.set(qn('w:val'),'true');trPr.append(rep)
 for ri,row in enumerate(rows):
  cells=t.add_row().cells
  for i,x in enumerate(row):
   set_cell(cells[i],x,False,'263442',size)
   if ri%2==1:shade(cells[i],'EFF4F8')
 if widths:
  for row in t.rows:
   for i,w in enumerate(widths):row.cells[i].width=Inches(w)
 return t

def para(text='',boldlead=None,style=None):
 p=D.add_paragraph(style=style)
 if boldlead and text.startswith(boldlead):p.add_run(boldlead).bold=True;p.add_run(text[len(boldlead):])
 else:p.add_run(text)
 return p
def bullet(text):
 p=D.add_paragraph(style='List Bullet');p.paragraph_format.space_after=Pt(0);p.add_run(text);return p
def note(title,text,fill='EAF2F7'):
 t=D.add_table(rows=1,cols=1);t.style='Table Grid';c=t.cell(0,0);shade(c,fill);c.text=''
 p=c.paragraphs[0];p.paragraph_format.space_after=Pt(2);r=p.add_run(title);r.bold=True;r.font.color.rgb=RGBColor(21,50,75);r.font.size=Pt(10)
 p=c.add_paragraph(text);p.paragraph_format.space_after=Pt(2);p.runs[0].font.size=Pt(9)

# Page 1
p=D.add_paragraph(style='Title');p.add_run('BEV PPO Weather Study')
p=D.add_paragraph();r=p.add_run('Complete project explanation from inputs to 100-episode evaluation');r.font.size=Pt(14);r.font.color.rgb=RGBColor(30,96,120)
para('This report explains the repository’s saved BC-initialized Proximal Policy Optimization (PPO) driving experiment, its architecture, the exact 100 held-out simulator episodes, and what the results do and do not establish. The main finding is clear: the integrated pipeline runs, but the current checkpoint is an early pilot rather than a reliably trained weather-adaptive driver.')
note('The 100 episodes were evaluation, not training','The evaluated checkpoint started from a behavior-cloning (BC) policy and received 2,048 PPO environment steps. Then the same checkpoint was evaluated for 20 episodes in each of five fixed weather conditions. Those evaluation episodes do not update PPO. The observed success rate was 50%–65%, with collisions in 35%–40% of episodes.', 'FFF2D9')
D.add_heading('At a glance',level=1)
table(['Item','What the saved experiment did'],[
('Learning method','PPO fine-tuning initialized from a BC checkpoint; BC action matching remains as a frozen reference loss.'),
('Driving representation','One 64 × 64 BEV occupancy/route grid plus six route and ego-state values.'),
('Control','Two continuous outputs: steering and combined throttle/brake, each bounded to [−1, 1].'),
('Weather','Synthetic visual fog/rain intervention plus condition-derived visibility masking in the BEV; friction and dynamics unchanged.'),
('Evaluation','Five conditions × 20 seeds = 100 episodes; traffic density 0.10; held-out seed range 30000–30019.'),
('Readiness','Closed-loop integration and a finite pilot are demonstrated. PPO convergence, safe driving, physical weather robustness, and energy savings are not demonstrated.')],size=9)
D.add_heading('How to read this report',level=2)
para('Sections 1–4 explain the pipeline, policy and reward. Sections 5–7 report the experiment and its limits. Sections 8–9 give reproducibility details, an honest project-stage assessment and the work needed before making stronger research claims.')

# Page 2
h_arch=D.add_heading('1. System architecture',level=1);h_arch.paragraph_format.page_break_before=True
para('The figure follows a single simulation decision from the world state, through cameras and perception, into the policy, and back to MetaDrive. It also shows where PPO learns and how the held-out study was measured.')
p=D.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run().add_picture(str(FIG),width=Inches(5.8))
p=D.add_paragraph('Figure 1. Implemented BC-initialized BEV PPO pipeline. The weather-derived visibility range is supplied from the selected test condition; it is not estimated from camera images.',style='Caption');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
D.add_heading('Driving task: route following with obstacle awareness',level=2)
para('The route is supplied by MetaDrive’s generated scenario and lane/map state. The BEV contains route center and lane boundaries, while nearby detected objects become obstacle footprints. Therefore, this is route following with a perception channel for obstacles. It is not a car-following policy trained to imitate the vehicle in front. A vehicle may appear ahead because the simulated route contains traffic; it is not the route teacher.')
para('This setup also contains privileged simulator/map inputs: lane geometry and parts of the six-value state are read from MetaDrive. The experiment is consequently a controlled BEV policy study, not an end-to-end camera-only autonomous-driving system.')

# Page 3
D.add_heading('2. From raw cameras to the policy observation',level=1)
D.add_heading('Sensors and weather inputs',level=2)
para('The environment mounts five RGB views: front-left, front-right, left, right and rear. The front pair is processed as a stereo pair. At each observation, the selected clear/fog/rain image transformation is applied before detection and depth estimation. The study uses a 320-pixel YOLO input, confidence threshold 0.25, and stereo matching at half scale.')
D.add_heading('Detection and depth',level=2)
para('YOLOv8n runs on front-left, left, right and rear images. It emits class-labeled 2D boxes. StereoSGBM estimates depth from the weather-conditioned front-left/front-right images. The BEV assembler uses those detections and depth to form object footprints in ego-relative coordinates. The side and rear views use a ground-plane projection approximation because they do not have a stereo pair in this experiment.')
D.add_heading('BEV grid',level=2)
para('The policy receives one 64 × 64 grid spanning 20 m laterally and 20 m from the rear extent through 17.5 m forward. The grid combines visible-free cells, unknown cells, route center, boundaries, detected-object footprints and the ego vehicle. It is a compact, single-channel encoded map, not a learned multi-camera feature volume.')
para('For fog and rain evaluation, a horizon computed from the known weather name and intensity masks BEV cells beyond a conservative distance as unknown. This is a controlled intervention to test a weather-conditioned perception state. It is a limitation for claims about autonomous weather recognition because the policy pipeline receives a condition-derived horizon.')
D.add_heading('Six scalar inputs',level=2)
table(['Index','State value','Role'],[
('1','Speed (m/s)','Current ego speed'),('2','Route progress','Fraction of route completed'),('3','Lateral deviation (m)','Offset from lane center'),('4','Heading error (rad)','Difference from lane direction'),('5','Curvature ahead','Approximate lane heading change ahead'),('6','Distance to goal (m)','Remaining route distance')],size=8.8)
para('These features are normalized using mean and standard deviation saved alongside the PPO checkpoint. Several are read directly from simulator navigation/lane state. They are not reconstructed from YOLO detections.')

# Page 4
D.add_heading('3. Policy network and PPO learning',level=1)
D.add_heading('Network',level=2)
table(['Component','Implementation in the evaluated checkpoint'],[
('BEV encoder','CNN: Conv(1→32, 5×5), Conv(32→64, 3×3), Conv(64→96, 3×3), pooling and 128-unit projection.'),
('Scalar encoder','MLP: 6→64→32.'),
('Fusion','Concatenate 128 BEV features and 32 scalar features = 160 features.'),
('Actor','160→128→2, tanh bounded. PPO generalized state-dependent exploration (gSDE) is enabled during learning.'),
('Critic','160→128→1 estimates the value of a state for PPO advantage calculation.')],size=8.8)
D.add_heading('BC warm start and PPO update',level=2)
para('The BC checkpoint provides an initial driving policy. PPO warm-starts compatible BEV/scalar encoders and action-mean layers from that checkpoint. During each PPO update, a frozen copy of the BC policy is also queried on rollout observations. An action-matching mean-squared-error term discourages large departures from the BC actions while the policy improves against simulator reward. This is a regularizer; it does not guarantee the BC policy itself is safe or accurate.')
para('The PPO policy objective uses the clipped probability ratio, a critic value loss and an entropy term, plus the weighted BC anchor loss. The saved checkpoint metadata records seed 2027, learning rate 3×10⁻⁵, four update epochs, target KL 0.01, BC loss coefficient 0.5, adaptive-speed penalty 0.05, collision penalty 30, randomized weather up to intensity 0.8, traffic density 0.10, and a 400-step training episode cap. The trainer source defaults to 128 rollout steps and 64-sample minibatches; these two values are not recorded in the checkpoint metadata, so they cannot be verified as the exact run settings. Total recorded PPO training was 2,048 environment steps.')
note('Why the training result cannot be called converged','2,048 interaction steps are short for a visual-input driving policy. The saved evaluation table does not include a learning curve, repeated training seeds, checkpoint-selection protocol, or a BC-versus-PPO matched comparison. The 100 evaluation episodes estimate this checkpoint’s outcomes under the listed conditions; they cannot establish that PPO is optimized or that it generalizes broadly.', 'FFF2D9')
D.add_heading('What does “PPO working” mean here?',level=2)
para('The implementation performs PPO rollout collection and clipped policy/value updates; the action comes from the learned actor, and the environment supplies reward and next observation. That establishes the algorithmic loop is wired. Whether the policy has learned a good driving strategy is a separate question. With these metrics, the answer is: PPO ran, but driving reliability is not yet good enough to claim a successful trained agent.')

# Page 5
D.add_heading('4. Actions, reward and energy accounting',level=1)
D.add_heading('Control actions',level=2)
para('At each decision, the actor produces steering and longitudinal control in [−1, 1]. The second value is a combined throttle/brake command: positive values request throttle and negative values request braking. The action is clipped to the valid range before MetaDrive receives it.')
D.add_heading('Reward composition',level=2)
para('The environment retains MetaDrive’s native reward and adds experiment-specific shaping. The saved safety checkpoint uses an adaptive overspeed penalty of 0.05, an additional collision/off-road penalty of 30, and a +10 destination-arrival bonus. A speed target is calculated from the visibility horizon. If the front-center BEV contains an obstacle, an obstacle-distance speed envelope can lower the target further. The penalty is 0.05 × max(speed − target speed, 0)². No NPC ground-truth position is used for this obstacle calculation.')
para('This shaping makes the reward action- and outcome-dependent: PPO receives rewards and penalties after stepping the simulator. However, the reported study does not include a reward decomposition plot or training logs that would show whether the policy reliably learned from each term.')
D.add_heading('Energy: what is and is not included',level=2)
para('The energy values in the table are post-run estimates from a flat-road longitudinal tractive-work model using speed and acceleration, with assumed vehicle mass 1,800 kg, drag coefficient 0.23, frontal area 2.22 m², air density 1.225 kg/m³, rolling coefficient 0.01 and 0.1 s time step. The model clips negative work to zero. It omits regenerative braking, grade, drivetrain and battery efficiency, tire slip and calibrated MetaDrive energy.')
para('The evaluated safety checkpoint has energy_reward_weight = 0.0. Therefore, the PPO agent was not trained to optimize energy in this experiment. Wh/km is only a post-hoc proxy column. A separate energy-weighted candidate exists in the repository, but it has no matched 100-episode comparison in this study and cannot be claimed to save energy.')

# Page 6
D.add_heading('5. Held-out evaluation design',level=1)
para('One saved PPO checkpoint was evaluated with five fixed conditions. Each condition used the same seed set (30000–30019), creating paired scenarios across weather conditions. The runner used a MetaDrive scenario pool size of 30050, traffic density 0.10 and a 500-step episode cap. Seeds 30000–30019 are outside the training scenario-index range stated by the experiment source (10–10009).')
table(['Condition','Weather intensity','Episodes','Condition meaning'],[
('Clear','0.00','20','No synthetic fog/rain corruption'),('Light fog','0.25','20','Lower fog-corruption level'),('Heavy fog','0.75','20','Higher fog-corruption level'),('Light rain','0.25','20','Lower rain-corruption level'),('Heavy rain','0.75','20','Higher rain-corruption level')],size=8.8)
para('For each episode, the evaluator records outcome, number of steps, average speed, lateral and heading error, estimated energy, steering/throttle/brake summaries, YOLO box totals by class and camera, frames with boxes, projected-object frames, and front stereo-valid fraction. The saved study aggregates continuous episode metrics as mean ± sample standard deviation. Success has 95% Wilson intervals.')
D.add_heading('Meaning of held out',level=2)
para('The evaluation seed numbers were kept outside the stated training seed range, which is a useful scenario-seed holdout. It is not a fully independent test of geography, maps, traffic behavior, weather physics or simulator version: the scenario generator and map family remain MetaDrive. Because the study has 20 episodes per condition and only one trained checkpoint/seed, the results are a pilot generalization test, not broad statistical validation.')
D.add_heading('Traffic and “little traffic”',level=2)
para('Traffic density was fixed at 0.10, a relatively low-traffic setting. This is consistent with a light-traffic pilot; it does not establish performance in dense traffic. “20 episodes” also means 20 scenario episodes per weather condition, not 20 vehicles and not 20 independent training runs.')

# Page 7 results
D.add_heading('6. Results from the 100 episodes',level=1)
para('The candidate completed a portion of scenarios in every tested condition, but collision frequency remains high. Success rates vary from 50% to 65%, and the confidence intervals are wide because each condition has only 20 episodes.')
rows=[
('Clear 0.00','12/20 (60%; 38.7–78.1%)','8/20 (40%)','0','7.95 ± 0.37','0.23 ± 0.07','163.94 ± 69.02'),
('Fog 0.25','11/20 (55%; 34.2–74.2%)','8/20 (40%)','1','7.98 ± 0.37','0.27 ± 0.08','168.89 ± 70.64'),
('Fog 0.75','12/20 (60%; 38.7–78.1%)','8/20 (40%)','0','8.09 ± 0.38','0.31 ± 0.06','170.30 ± 72.85'),
('Rain 0.25','13/20 (65%; 43.3–81.9%)','7/20 (35%)','0','7.98 ± 0.37','0.25 ± 0.08','162.92 ± 70.31'),
('Rain 0.75','10/20 (50%; 29.9–70.1%)','8/20 (40%)','2','8.04 ± 0.37','0.29 ± 0.06','169.90 ± 71.46')]
table(['Weather','Success (95% Wilson CI)','Collision','Off-road','Speed m/s','|Lateral error| m','Energy proxy Wh/km'],rows,size=7.1)
p=para('Heading error (mean ± SD) by condition was: clear 0.017 ± 0.015 rad; light fog 0.020 ± 0.018; heavy fog 0.021 ± 0.017; light rain 0.018 ± 0.015; heavy rain 0.021 ± 0.017. No episode ended only because it hit the 500-step cap.')
D.add_heading('YOLO prediction-count ablation',level=2)
para('Counts are detector outputs accumulated across camera frames. They are not unique cars, tracked objects, ground-truth counts, precision or recall. The same vehicle may be counted repeatedly across frames and/or views; false positives are possible.')
rows=[('Clear 0.00','1.619','10,644','32.9%','0.0101'),('Fog 0.25','1.558','9,926','34.5%','0.0101'),('Fog 0.75','1.215','7,833','28.3%','0.0100'),('Rain 0.25','1.904','12,831','47.7%','0.0079'),('Rain 0.75','0.907','5,701','32.3%','0.0045')]
table(['Weather','Vehicle boxes / frame','Vehicle box instances','Frames with any box','Mean stereo-valid fraction'],rows,size=7.6)
para('Heavy rain had the fewest predicted vehicle boxes per frame (0.907), while light rain had the most (1.904). The front stereo-valid fraction is low in every condition, suggesting depth coverage is a bottleneck in this setup. Without labeled frames and tracking, detector quality cannot be inferred from box volume.')

# Page 8
D.add_heading('7. Interpretation and limits',level=1)
D.add_heading('What the data support',level=2)
for s in [
'The complete simulator loop is integrated: camera sensing, synthetic weather, perception, BEV/scalar observation, PPO policy action, simulator reward and episode reporting.',
'The tested BC-initialized PPO checkpoint drove some held-out low-traffic MetaDrive scenarios in each fixed weather condition.',
'Visual degradation is associated with changes in box-count and stereo-coverage summaries; heavy rain has the smallest vehicle box rate in this run.',
'The study has explicit per-condition outcomes and uncertainty intervals, which is a useful baseline for a stronger experiment.'
]:bullet(s)
D.add_heading('What the data do not support',level=2)
for s in [
'That PPO converged, is free of overfitting, or is safe: training was 2,048 steps, one checkpoint, with a 35–40% collision rate.',
'That the car is robust to real fog/rain: weather is a synthetic camera effect and tire friction/vehicle dynamics are unchanged.',
'That it understands weather from camera images: visibility masking is driven by a known condition-derived horizon.',
'That detector accuracy is high or low: YOLO counts have no annotated ground truth and are not unique-object tracks.',
'That energy use improved: energy reward weight was zero for this checkpoint and Wh/km is only an uncalibrated post-hoc proxy.',
'That generalization across roads or traffic is established: scenarios use the same MetaDrive generator and traffic density is only 0.10.'
]:bullet(s)
note('Research-ready wording for this result','“We evaluated a BC-initialized, safety-shaped BEV PPO pilot on 100 held-out MetaDrive episodes across five synthetic visual weather conditions. Success ranged from 50% to 65%; collision rates ranged from 35% to 40%. These findings demonstrate end-to-end experimental integration, not PPO convergence or real-world weather robustness.”','E7F2FF')

# Page 9
D.add_heading('8. Reproducibility and repository map',level=1)
para('The following commands assume Windows PowerShell, the repository root, an installed project environment with MetaDrive/PyTorch/Stable-Baselines3/Ultralytics/OpenCV, and the checkpoint/data files present. Training and evaluation are computationally expensive; the command below reproduces the recorded 100-episode study, not PPO training.')
D.add_heading('Re-run the saved study',level=2)
p=D.add_paragraph();p.paragraph_format.left_indent=Inches(.18);r=p.add_run('python -m bev_policy.run_weather_generalization_study');r.font.name='Consolas';r.font.size=Pt(9);r.font.color.rgb=RGBColor(30,96,120)
para('The runner loads bev_policy/checkpoints/ppo_bev_adaptive_safety_v1.zip and writes condition JSON/CSV files plus paper_summary.json and paper_table.md under bev_policy/outputs/ppo_weather_generalization_20_stereo05/. Running it does not train or update the checkpoint.')
D.add_heading('Relevant implementation files',level=2)
table(['Repository path','Purpose'],[
('bev_policy/train_ppo.py','CLI and metadata writing for BC-initialized PPO training.'),
('bev_policy/ppo.py','BEV fusion network, BC anchor, simulator wrapper, reward shaping and observation loop.'),
('bev_policy/perception.py','Weather conditioning, stereo depth and YOLO execution.'),
('bev_state/assembler.py','Camera detections/depth and simulator route features assembled into BEV + six scalars.'),
('bev_policy/evaluate.py','Loads checkpoint, runs episodes, records metrics, videos/dashboard support.'),
('bev_policy/run_weather_generalization_study.py','Fixed five-condition, 20-seed-per-condition study and aggregation.'),
('bev_policy/outputs/ppo_weather_generalization_20_stereo05/paper_table.md','Readable study summary and interpretation limits.'),
('bev_policy/checkpoints/ppo_bev_adaptive_safety_v1.json','Training configuration, action shape, architecture and checkpoint metadata.')],size=8)
para('The JSON/CSV episode reports in the same output directory are the auditable per-condition records. The checkpoint’s matching .json stores scalar normalization values needed for PPO evaluation.')
D.add_heading('How the training command is configured',level=2)
para('The training script defaults to 100,000 timesteps, but the evaluated checkpoint metadata says this particular run used 2,048. To reproduce that exact length, pass --timesteps 2048 and the recorded BC checkpoint and settings. Merely executing the defaults would produce a different experiment and a different checkpoint.')

# Page 10
D.add_heading('9. Project stage and next research steps',level=1)
D.add_heading('Current stage',level=2)
para('The project is at an integrated simulation prototype plus short PPO pilot. It has an implemented perception-to-BEV-to-control loop and a held-out weather evaluation harness. It is not yet at a validated adaptive-driving result. The table should be presented as a baseline/diagnostic study rather than a headline performance claim.')
D.add_heading('Recommended next sequence',level=2)
for s in [
'Fix and freeze the experiment contract: record exact commit, package versions, scenario seeds, weather seeds, checkpoint hash, normalization, and all hyperparameters in each report.',
'Collect a larger and more diverse BC dataset with braking, obstacle encounters, lane changes/curves, and seed diversity; compare BC alone against PPO on identical held-out seeds.',
'Train PPO substantially longer with multiple training seeds and periodic held-out validation; keep test seeds untouched for final reporting. Plot success, collision, off-road, reward, KL, value loss and BC-anchor loss against environment steps.',
'Use factorial ablations: BC vs PPO; BC anchor on/off; adaptive speed shaping on/off; camera weather vs weather horizon masking; detector/depth input swaps; energy reward off/on. Match seeds and report confidence intervals.',
'Improve perception evaluation with labeled sampled frames, precision/recall by weather and distance, unique-object tracking, calibration and stereo-depth error against simulator ground truth.',
'For a stronger generalization claim, test unseen maps/scenario families, multiple traffic densities and held-out weather intensities. Add friction/rain dynamics only as a separate physical-effects experiment.',
'For energy results, compare matched successful trips and report distance-normalized proxy plus speed/arrival/safety metrics. Calibrate the model or use simulator energy signals before claiming energy efficiency.',
'Repeat the final locked protocol across multiple PPO seeds and report both per-seed and pooled uncertainty; archive videos and failures for qualitative review.'
]:bullet(s)
D.add_heading('Glossary',level=2)
table(['Term','Meaning in this project'],[
('BC','Behavior cloning: supervised actions learned from demonstrations, used as PPO initialization and action anchor.'),
('BEV','Bird’s-eye-view, ego-relative raster grid that summarizes route, free/unknown space and detected objects.'),
('PPO','Proximal Policy Optimization, a policy-gradient RL algorithm with clipped policy updates and a value critic.'),
('Held-out seed','A scenario seed not used in the stated training range; it is not necessarily a different map family.'),
('Stereo-valid fraction','Fraction of pixels with nonzero valid depth output from front StereoSGBM.'),
('Energy proxy','Model-estimated positive tractive work divided by distance; not battery-meter energy.')],size=7.3)
p_source=para('Source of record: checkpoint metadata, study runner, evaluator, policy/reward implementation, state assembler, and saved study report in the repository.')
p_source.paragraph_format.space_before=Pt(0);p_source.paragraph_format.space_after=Pt(0);p_source.paragraph_format.line_spacing=1
for rr in p_source.runs: rr.font.size=Pt(7.3)

# Document properties
D.core_properties.title='BEV PPO Weather Study Complete Explanation'
D.core_properties.subject='Architecture, training, reward, evaluation and interpretation of the 100-episode MetaDrive weather study'
D.core_properties.author='BEV Driving Project'
D.save(OUT)
print(OUT)
