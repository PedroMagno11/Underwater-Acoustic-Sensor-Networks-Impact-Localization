# 🌊 Deployment of Underwater Acoustic Sensor Networks for Projectile Impact Localization

## 📌 Research Framework

This repository contains the complete simulation framework used in the research 
submitted to **FUSION 2026**, investigating optimal deployment strategies for underwater acoustic sensor networks under realistic acoustic propagation conditions.

The study evaluates the impact of:

* 3D volumetric geometry
* Depth-dependent sound speed profiles (SSP)
* Multi-impact operational scenarios
* Time-of-Arrival (TOA) localization
* Genetic Algorithm (GA) optimization
* Particle Swarm Optimization (PSO) validation
* Comparison against regular polygon baseline deployments

---

# 🎯 Research Objective

The primary objective of this work is to assess whether symmetric geometric layouts (e.g., regular polygon configurations) remain optimal when realistic underwater acoustic propagation and full three-dimensional spatial modeling are considered.

The framework investigates:

* Localization accuracy
* Coverage violations
* Sensitivity to depth-dependent sound speed variations
* Stability of optimized solutions across different metaheuristic approaches

---

# 🧠 Methodological Overview

For each optimization generation:

1. A shared set of impact points is generated.
2. Arrival times are computed using a depth-dependent sound speed profile (SSP).
3. Measurement noise is injected into TOA observations.
4. Maximum Likelihood Estimation (MLE) is applied for impact localization.
5. A fitness function evaluates:

   * Localization error
   * Coverage penalties
   * Detection constraints
6. Optimization is performed using:

   * Genetic Algorithm (GA)
   * Particle Swarm Optimization (PSO)
7. Results are statistically compared against regular polygon baseline configurations.

---

# 📂 Project Structure

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

### Main Configuration Blocks

### Environment

* Grid resolution
* Detection radius
* Minimum and maximum sensor depth
* Target region geometry

### Genetic Algorithm

* Population size
* Mutation probability
* Crossover probability
* Tournament size
* Elitism
* Random seed

### Particle Swarm

* Swarm size
* Inertia weight
* Cognitive and social coefficients
* Random seed

### Simulation

* Number of impact points per evaluation
* TOA noise standard deviation

### Performance

* Parallel evaluation settings
* Worker configuration

---

# ▶️ Running Experiments

First, ensure that Python (version 3.10 or higher is recommended) is installed on your system.

### Create a Virtual Environment
```bash
python -m venv .venv
```
### Activate the Virtual Environment

#### On Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate
```

#### On Windows (Command Prompt):

```cmd
.\.venv\Scripts\activate.bat
```

#### On Linux / macOS:

```bash
source .venv/bin/activate
```

After activation, your terminal prompt should display `(.venv)`.


### Install Dependencies
```bash
pip install -r requirements.txt
```

---

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


All raw optimization outputs are stored under:
```
outputs/
```

Comparative results between GA, PSO, and regular polygon baseline deployments are available in:
```
outputs_comparison/
```
This directory contains aggregated metrics, scenario-level comparisons, and statistical summaries used in the 
analysis presented in the submitted paper.

Visualization scripts are available in:

```
scripts/
```

To generate scenarios for the Genetic Algorithm:

```bash
python -m scripts.plot_ga_scenarios
```

To generate scenarios for Particle Swarm Optimization:

```bash
python -m scripts.plot_pso_scenarios
```

To generate scenarios for regular polygon deployments:

```bash
python -m scripts.plot_regular_polygon_scenarios
```

---

# 🔁 Reproducibility

This framework was designed to ensure deterministic scientific reproducibility.

Reproducibility is achieved through:

* A centralized global seeding system (`utils/seeding.py`)
* Generation-level impact seeding
* Fixed GA and PSO random seeds
* Shared impact sets per generation
* Configuration-driven experiment execution

To reproduce the results reported in the paper:

1. Use the provided `experiment_config.json` without modification.
2. Do not alter random seed values.
3. Execute both GA and PSO experiments.
4. Compare the resulting metrics against the regular polygon baseline.

---

# 📐 Acoustic Model Assumptions

* First-arrival direct-path propagation only
* No multipath modeling
* No boundary reflections
* Static medium during evaluation
* Depth-dependent SSP applied

These assumptions isolate the geometric and sound-speed-profile effects on TOA consistency and localization performance.

---

# 🧪 Validation Strategy

The robustness of optimized deployments is evaluated through:

* GA vs. PSO cross-validation
* Multi-scenario averaging
* Statistical comparison against symmetric baseline layouts
* Coverage violation analysis

Agreement between GA and PSO results indicates convergence toward stable regions of the fitness landscape rather than algorithm-specific artifacts.

---

# 🚀 Future Work

Planned extensions include:

* Multipath propagation modeling
* Ray-tracing-based acoustic modeling
* Adaptive mesh refinement strategies
* Real-world buoy deployment validation
* Embedded real-time implementation

---

# 👨‍💻 Authors

- Pedro Magno Almeida Nogueira

- José Gomes de Carvalho Junior

- Pablo Rangel

- Julio Cesar
