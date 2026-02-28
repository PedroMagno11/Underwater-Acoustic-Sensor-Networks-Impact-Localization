# 🌊 Deployment of Underwater Acoustic Sensor Networks for Projectile Impact Localization

## 📌 Research Framework – FUSION 2026

This repository contains the complete simulation framework used in the research 
submitted to **FUSION 2026**, investigating optimal deployment strategies for 
underwater acoustic sensor networks under realistic propagation conditions.

The study evaluates the impact of:

* 3D volumetric geometry
* Depth-dependent sound speed profiles (SSP)
* Multi-impact operational scenarios
* Time-of-arrival (TOA) localization
* Genetic Algorithm (GA) optimization
* Particle Swarm Optimization (PSO) validation
* Comparison against regular polygon deployments

---

# 🎯 Research Objective

The main objective of this work is to evaluate whether symmetric geometric layouts (e.g., regular polygons) remain optimal when realistic underwater acoustic propagation and 3D spatial modeling are considered.

The framework investigates:

* Localization accuracy
* Coverage violations
* Sensitivity to depth-dependent sound speed variation
* Stability of optimized solutions across different metaheuristics

---

# 🧠 Methodological Overview

For each optimization generation:

1. A shared set of impact points is generated.
2. Arrival times are computed using depth-dependent SSP.
3. Noise is injected into TOA measurements.
4. Maximum Likelihood Estimation (MLE) is used for localization.
5. A fitness function evaluates:

   * Localization error
   * Coverage penalties
   * Detection constraints
6. Optimization is performed using:

   * Genetic Algorithm
   * Particle Swarm Optimization
7. Results are statistically compared to regular polygon baselines.

---

# 📂 Project Structure (Based on Repository)

```
acoustic/
    arrival_time.py
    sound_speed_profile.py
    sound_speed_profile_builder.py

geometry/
    grid_geometry.py
    distance.py

localization/
    mle_estimator.py

evaluation/
    cost_function.py

genetic_algorithm/
    chromosome.py
    crossover.py
    genetic_algorithm.py
    mutation.py
    penalty.py
    population.py
    selection.py

particle_swarm/
    particle_swarm.py

regular_polygon/
    regular_polygon.py

performance/
    parallel_evaluation.py

settings/
    environment_settings.py
    genetic_algorithm_settings.py
    logging_settings.py
    particle_swarm_settings.py
    performance_settings.py
    simulation_settings.py

executable/
    ga_runner.py
    pso_runner.py
    regular_polygon_runner.py

visualization/
    plot_scenarios.py

scripts/
    plot_ga_scenarios.py
    plot_pso_scenarios.py
    plot_regular_polygon_scenarios.py

utils/
    seeding.py
    results.py
    loaders.py
```

---

# ⚙️ Configuration

All experiment parameters are defined in:

```
experiment_config.json
```

### Main configuration blocks:

### Environment

* Grid resolution
* Detection radius
* Minimum and maximum sensor depth
* Target region geometry

### Genetic Algorithm

* Population size
* Mutation probabilities
* Crossover probability
* Tournament size
* Elitism
* Random seed

### Particle Swarm

* Swarm size
* Inertia weight
* Cognitive/social coefficients
* Random seed

### Simulation

* Number of impact points per evaluation
* Time noise standard deviation

### Performance

* Parallel evaluation settings
* Worker configuration

---

# ▶️ Running Experiments

### Run Genetic Algorithm

```bash
python -m executable.ga_runner
```

### Run Particle Swarm Optimization

```bash
python -m executable.pso_runner
```

### Run Regular Polygon Baseline

```bash
python -m executable.regular_polygon_runner
```

---

# 📊 Output

The framework generates:

* JSON summary files
* CSV performance metrics
* Deployment figures
* Coverage visualizations
* Localization heatmaps

Visualization scripts are available in:

```
scripts/
```

Example:

```bash
python -m scripts.plot_ga_scenarios
```

---

# 🔁 Reproducibility

This framework was designed for deterministic scientific reproducibility.

Reproducibility is ensured by:

* Global seeding system (`utils/seeding.py`)
* Generation-level impact seeding
* Fixed GA and PSO seeds
* Shared impact set per generation
* Config-driven experiment execution

To reproduce the results reported in the paper:

1. Use the provided `experiment_config.json`
2. Do not modify random seeds
3. Run GA and PSO experiments
4. Compare metrics with regular polygon baseline

---

# 📐 Acoustic Model Assumptions

* First-arrival direct-path propagation only
* No multipath modeling
* No boundary reflections
* Static medium during evaluation
* Depth-dependent SSP applied

These assumptions isolate geometric and SSP effects on TOA consistency.

---

# 🧪 Validation Strategy

The robustness of optimized deployments is evaluated through:

* GA vs PSO cross-validation
* Multi-scenario averaging
* Statistical comparison against symmetric baseline layouts
* Coverage violation analysis

Agreement between GA and PSO indicates stable fitness landscape regions.

[//]: # (---)

[//]: # ()
[//]: # (# 📚 Citation)

[//]: # ()
[//]: # (If this framework is used in academic research, please cite:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@inproceedings{fusion2026_underwater_sensor_deployment,)

[//]: # (  title={Three-Dimensional Underwater Acoustic Sensor Deployment Optimization with Depth-Dependent Sound Speed Profiles},)

[//]: # (  author={Magno, Pedro},)

[//]: # (  booktitle={Proceedings of IEEE FUSION 2026},)

[//]: # (  year={2026})

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (---)

# 🚀 Future Work

Planned extensions include:

* Multipath propagation modeling
* Ray-tracing acoustic modeling
* Adaptive mesh refinement
* Real buoy deployment validation
* Embedded real-time implementation

---

# 👨‍💻 Authors

Pedro Magno Almeida Nogueira

José Gomes de Carvalho Junior

Pablo Rangel

Júlio César
