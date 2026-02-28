from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ParticleSwarmSettings:
    # Seed do PSO (independente do GA)
    random_seed: int = 321

    # Budget do PSO: swarm_size * number_of_iterations
    # Para bater seu GA: 250*500 = 125k avaliações
    # Ex: swarm_size=100 e iterations=1250 => 125k
    number_of_iterations: int = 300
    swarm_size: int = 60

    # Coeficientes clássicos (estáveis)
    inertia_w: float = 0.72
    cognitive_c1: float = 1.49
    social_c2: float = 1.49

    # vmax como fração do range por gene:
    # com target_region_radius=1500 => range_xy ~ 3000 => 0.05 => 150m/iter (bem alinhado ao seu GA)
    vmax_fraction_of_range: float = 0.05