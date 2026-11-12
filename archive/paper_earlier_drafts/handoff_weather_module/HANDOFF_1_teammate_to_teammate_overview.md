# Handoff 1 of 2: Teammate to teammate (overview)

**For:** the teammate who owns the weather module (`YOLO+weather module/`).
**Companion file:** `HANDOFF_2_agent_to_agent_work_order.md` is the step-by-step work order for your AI agent. Give your agent both files.
**From:** the policy side. Date: 2026-11-05.

---

## 1. The goal in one sentence

Turn the weather module from "a 0-to-1 effects slider" into **physically based fog and rain, checked against real photos, that tells our driving policy how far it can actually see**. If that works, the weather module becomes a central contribution of our paper, and you own it.

---

## 2. Where the project stands

**What we are building.** A driving policy that stays safe and energy-efficient when weather degrades perception. The pipeline is: MetaDrive simulator, then camera and YOLOv8, then a 64x64 BEV (bird's-eye-view) map, then the policy, which outputs steering and throttle.

**What we measured.**

| Result | Meaning |
|---|---|
| Your fog and rain sliders cut YOLO detections (rain 0.6 keeps 16%, fog 0.6 keeps 102%). | Weather matters to perception, and rain and fog behave very differently. |
| In sparse traffic, losing detections did not change whether the car reached its goal (success 0.78 to 0.95 in all weather). | There are few cars on the road, so detection loss alone is not our story. |
| **A position error of the car against the HD map breaks the policy**: 0.90 success up to 0.5 m error, 0.70 at 1 m, 0.28 at 2 m. | Weather likely hurts us by degrading map alignment. That needs a physical model. |
| Our "caution" rule (slow from 30 to 20 km/h in bad weather) is hand-set. | Physics can replace it with a formula. |

---

## 3. The two discoveries that make this worth doing (I checked the code)

**Discovery A: today's fog is not fog physics.** In `weather/weather_utils.py`, `add_fog` whitens each image row by an amount that grows linearly down the image: the top row is unchanged and the bottom row is whitest. I ran it on a flat grey image to confirm: the top row stays at 127.5 and the bottom row goes from 161 (level 0.3) to 240 (level 1.0). Real fog does the opposite: it whitens the **farthest** things most, and it depends on distance, not on image row. It also ignores sky.

**Discovery B: today's rain contains a darkening artifact.** In `add_rain`, the last filter (`filter2D` with a kernel whose entries add to 0.75) dims the whole image by about a quarter and smears it vertically, at every intensity. On a flat grey image of value 128, the output averages 93.6 at level 0.3, 93.2 at 0.6 and 96.0 at 1.0. So the drop from 128 to about 94 does not depend on rain level at all. Part of why rain destroys YOLO in our measurements may be this artifact, not rain.

**Why it matters.** Our measured "retention" numbers are the foundation of the caution idea. If they come from a vertical gradient and a darkening filter, a reviewer will find it. If we replace them with physics and the numbers survive, the story is solid. If they change a lot, that is an important finding too.

---

## 4. What we need from you (four deliverables)

1. **Physical fog and rain renderers** that take real units: visibility in metres for fog, rain rate in mm/h for rain. They must use per-pixel depth for fog and sampled 3D raindrops for rain.
2. **A table that converts our old 0-1 levels to physical units**, so old results can be read in the new language.
3. **Detection retention re-measured** with the new renderers, using true recall and precision, on at least 100 paired scenes with traffic.
4. **A validation against real photos in fog and rain**, so we can say the renderers match reality.

The second half of the work is the link to driving, in section 6.

---

## 5. The physics, in plain language, equation by equation

Each equation lists its terms, its meaning in words, and the paper or book it comes from. Your agent has the code-ready version in Handoff 2. Items marked **[confirm]** are from memory: open the source and check them before you cite.

### 5.1 Fog dims and whitens things in proportion to distance

```
I(x) = J(x) * t(x) + A * (1 - t(x))
t(x) = exp(-beta * d(x))
```

| Symbol | Meaning | Unit |
|---|---|---|
| I(x) | the pixel you see in fog | image intensity |
| J(x) | what that pixel would look like without fog | image intensity |
| t(x) | how much of the object's light reaches the camera (1 = all, 0 = none) | 0 to 1 |
| A | the brightness of the fog itself ("airlight") | image intensity |
| d(x) | distance from the camera to that pixel | m |
| beta | extinction coefficient: how strongly the fog blocks light | 1/m |

**In words:** the farther an object is, the more it fades into the fog colour, exponentially with distance.
**Source:** Koschmieder (1924), the original visibility law. Used for image fog in Sakaridis, Dai and Van Gool, "Semantic Foggy Scene Understanding with Synthetic Data", IJCV 2018 (arXiv 1708.07819), which we read.

### 5.2 Turning "visibility in metres" into the fog strength

```
beta = 3.912 / V
```

| Symbol | Meaning | Unit |
|---|---|---|
| V | meteorological visibility: the distance at which a dark object's contrast has dropped to 2% | m |
| 3.912 | equals -ln(0.02), which comes from that 2% threshold | none |

**In words:** if you can see 100 m, then beta = 0.039 per metre. Some texts use a 5% threshold, which gives 3.0 instead of 3.912, so state which one you use. **[confirm]** the 2% convention in a meteorology reference.
**Why this line matters most:** it is what makes "fog 0.6" into "fog with 100 m visibility".

### 5.3 How far can YOLO see?

```
d_det = ln(C0 / C_min) / beta
```

| Symbol | Meaning | Unit |
|---|---|---|
| d_det | the farthest distance at which YOLO still detects a car | m |
| C0 | contrast of the car against its background with no fog | 0 to 1 |
| C_min | the smallest contrast YOLO needs to detect it. **Measure it, don't guess it.** | 0 to 1 |

**Source:** this is my own derivation from 5.1 (contrast at distance d is C0 * exp(-beta * d)). It is **not** from a paper. Treat it as a hypothesis your experiments test. If YOLO's behaviour does not follow it, report that: it is a finding.

### 5.4 Raindrops: how many, how big

```
N(D) = N0 * exp(-Lambda * D)        N0 = 8000 /(m^3 mm)      Lambda = 4.1 * R^(-0.21)  per mm
```

| Symbol | Meaning | Unit |
|---|---|---|
| N(D) | number of drops of diameter D in one cubic metre | per m^3 per mm |
| D | drop diameter | mm |
| R | rain rate. Light about 2.5, moderate about 10, heavy above 7.6, very heavy 50 | mm/h |
| N0, Lambda | fitted constants of the distribution | as shown |

**Source:** Marshall and Palmer (1948), "The distribution of raindrops with size", Journal of Meteorology. The standard rain-drop-size law.

### 5.5 How fast raindrops fall

```
v(D) = 9.65 - 10.3 * exp(-0.6 * D)      [m/s, D in mm]
```

**Source:** a fit to Gunn and Kinzer (1949) measurements, as given in Atlas et al. (1973). **[confirm]** the coefficients. Valid for drops of about 0.5 mm and larger.

### 5.6 The sanity check to run first

```
R = 6 * pi * 1e-4 * integral of ( v(D) * D^3 * N(D) ) dD       [mm/h]
```

If you take 5.4 and 5.5 and compute R this way, you must get back the rain rate you started with. If you don't, a unit is wrong. Do this before using any other rain equation. **Source:** the standard rain-rate integral from radar meteorology texts **[confirm]**.

### 5.7 Does rain block the view? Barely.

```
alpha_rain = pi * N0 / Lambda^3  ~=  0.365 * R^0.63    [per km]
V_rain = 3.912 / alpha_rain
```

| Symbol | Meaning |
|---|---|
| alpha_rain | how much rain weakens light over distance |
| V_rain | the visibility rain alone would give |

At 50 mm/h this gives about 0.9 km. Over 20 to 50 m, rain barely dims anything.
**Source:** I derived the first form from the Marshall-Palmer drops, where each drop blocks about twice its own cross-section. It reproduces the commonly quoted 0.365 * R^0.63 law. **[confirm]** that law in a rain-attenuation reference.
**Why it matters:** rain's damage to a camera comes from **streaks, water on the lens, and a haze of distant drops** (5.8), not from dimming. Fog is the reverse: it dims everything.

### 5.8 What rain looks like in a camera

```
streak length   ~=  (f / z) * v(D) * T
streak width    ~=  f * D / z
brightness gain ~=  (tau / T) * (E_drop - E_background)
```

| Symbol | Meaning | Unit |
|---|---|---|
| f | camera focal length (ours: about 1039 pixels) | pixels |
| z | distance from the camera to the drop | m |
| v(D) | the drop's fall speed from 5.5 | m/s |
| T | camera exposure time | s |
| D | drop diameter | m |
| tau | time the drop spends inside one pixel's field of view | s |
| E_drop, E_background | brightness of the drop and of the scene behind it | image intensity |

**Source:** Garg and Nayar, "Vision and Rain", International Journal of Computer Vision, 2007. **[confirm]** the exact forms.
**How to use it:** sample real drops from 5.4 inside the camera view and draw each one by these rules. Do not paste a texture. Distant drops merge into a haze (Li et al., CVPR 2019, heavy-rain restoration **[confirm]**).

### 5.9 Weather and the car's position on the map (second job)

We need `sigma_y(V, R)` (lateral position error in m) and `sigma_psi(V, R)` (heading error in rad) as functions of weather. A starting form, **my own suggestion**:

```
sigma_y = sqrt( sigma0^2 + (k * g(beta * d_ref))^2 ),    g(x) = 1 - exp(-x)
```

| Symbol | Meaning |
|---|---|
| sigma0 | position error in clear weather, about 0.1 to 0.3 m |
| d_ref | distance to the farthest landmark (for example a lane marking) still usable |
| g(...) | how much of that landmark's visibility the weather has removed |
| k | a scale you **fit to published data**, never invent |

**Where to look for numbers:** the rain and street-map localisation study arXiv 2409.01038 (a search snippet reported 6.05 m error in rain against 2.46 m clear; nobody here has read it, so confirm it), and the 17 cm lane-level requirement from the US DOT. Every number needs a source.

---

## 6. The link to driving (where it becomes a paper)

A car must be able to stop within the distance it can trust:

```
d_sight = v * t_r + v^2 / (2 * a_b)
v_safe  = a_b * ( -t_r + sqrt( t_r^2 + 2 * d_sight / a_b ) )
```

| Symbol | Meaning | Typical value |
|---|---|---|
| d_sight | the trusted distance: the smallest of visibility V, YOLO's range d_det, and the useful map range | m |
| v | speed | m/s |
| t_r | reaction time | 0.5 to 1.0 s |
| a_b | braking deceleration (use mu * g on a wet road) | about 3 m/s^2 |
| v_safe | the highest speed that still lets you stop within d_sight | m/s |

**Source:** standard stopping-distance kinematics (a textbook formula), solved for speed.

**The plan:** our hand-set caution rule becomes `c = 1 - v_safe(d_sight) / v_free`. You supply `d_det(V, R)` and `sigma(V, R)`. We supply the policy and the evaluation. If the physics-based caution matches or beats our hand-set one, and the policy trained on it is safer or more efficient in weather, that is a clean, defensible contribution.

---

## 7. How we will judge it

You pass when all of these hold:

- `beta = 3.912 / V` holds exactly in code.
- The rain-rate check in 5.6 comes back within 2%.
- `alpha_rain` is within 5% of `0.365 * R^0.63`.
- The fog strength recovered from a rendered image matches what you set, within 10%.
- Retention curves are monotone, with confidence intervals, on at least 100 paired scenes, using true recall and precision.
- At least one real fog dataset and one real rain dataset are compared, and nothing is tuned on the data you validate against.

## 8. Ground rules

- Report what the data says. If the physics changes our retention numbers, we want to know. If it reproduces them, we want to know that too.
- Read primary papers before you cite them. Keep a list of what you read in full and what only in abstract.
- Do not edit `bev_state/` or our policy files. Tell us if you change an interface.
- Rain affecting road friction is out of scope unless the whole team agrees to bring it back.

## 9. Search topics

"adverse weather perception-aware planning", "visibility-limited safe speed autonomous vehicle", "weather-dependent detector recall vs visibility", "physically based rain and fog simulation camera LiDAR", "localisation degradation rain map matching", "weather-conditioned reinforcement learning driving", "operational design domain weather speed adaptation".

**Already read in full by us:** Roach (arXiv 2108.08265: the policy design we build on, so that part is not novel); Manchingal et al. (2510.22680: uncertainty slowing a racing car); COLA (2210.03209: weather as a hidden mode); "Weather Adaptive Autonomous Driving" (2611.04305: per-weather detector thresholds); Sakaridis et al. 2018 (the fog model).

**Where to find our data:** `bev_rl/adaptive/results/retention_table.json` (current retention), `bev_rl/adaptive/caution.py` (how we use it), `paper/literature/LITERATURE_REVIEW.md` (what we read), `bev_rl/adaptive/results/localisation_error_sensitivity_*.json` (policy vs position error), `.claude/PROJECT_STATE.md` (decision log).
