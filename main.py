# Import necessary libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

"""
This model simulates the interaction between:
- Plant Biomass (B): 0 = seedling, 1 = maximum size
- Plant Vigor (V): 0 = no energy reserves, 1 = maximum metabolic energy
- Pathogen Load (P): 0 = no pathogens, 1 = maximum pathogen capacity

Key relationships:
- Biomass growth depends on both vigor and current size
- Vigor represents energy reserves that support growth and defense
- Pathogens primarily affect vigor, which then impacts growth
"""

# Define parameters (adjusted for daily timesteps)
r0 = 0.15     # Pathogen reproduction rate
beta = 0.8    # Plant defense impact (now depends on vigor)
delta = 0.1   # Pathogen death rate
g = 0.06      # Base growth rate
alpha = 0.1   # Pathogen impact on vigor
eta = 0.08    # Vigor recovery rate
gamma = 0.05  # Vigor consumption rate for growth

# Initial conditions
P0 = 0.01     # Initial pathogen load (1% of max)
V0 = 0.8      # Initial vigor (80% of max - healthy seedling)
B0 = 0.05     # Initial biomass (5% of max - seedling size)

def system(t, y):
    """
    Modified system with three variables:
    dP/dt = (r0 * e^(-β*V) - δ) * P * (1-P)     # Pathogen growth
    dV/dt = η*(1-V) - γ*g*B - α*P*V              # Vigor dynamics
    dB/dt = g * V * B * (1-B)                    # Biomass growth
    
    Where:
    - η (eta) is vigor recovery rate
    - γ (gamma) is vigor consumption for growth
    """
    P, V, B = y
    
    # Ensure values stay within bounds
    P = np.clip(P, 0, 1)
    V = np.clip(V, 0, 1)
    B = np.clip(B, 0, 1)
    
    # Calculate rates
    r = r0 * np.exp(-beta * V)  # Pathogen reproduction affected by vigor
    
    # Differential equations
    dP_dt = (r - delta) * P * (1 - P)
    dV_dt = eta * (1 - V) - gamma * g * B - alpha * P * V  # Vigor recovers naturally but is consumed by growth
    dB_dt = g * V * B * (1 - B)  # Growth depends on vigor
    
    return [dP_dt, dV_dt, dB_dt]

# Simulation setup
t_start = 0
t_end = 120
num_points = 120
t_eval = np.linspace(t_start, t_end, num_points)

# Solve system
solution = solve_ivp(system, [t_start, t_end], [P0, V0, B0], 
                    t_eval=t_eval, method='RK23')

# Extract solutions
t = solution.t
P = solution.y[0]
V = solution.y[1]
B = solution.y[2]

# Create visualization
plt.figure(figsize=(12, 8))

# Plot pathogen load
plt.subplot(3, 1, 1)
plt.plot(t, P, label='Pathogen Load (P)', color='red')
plt.ylabel('Pathogen Load')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

# Plot vigor
plt.subplot(3, 1, 2)
plt.plot(t, V, label='Plant Vigor (V)', color='blue')
plt.ylabel('Plant Vigor')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

# Plot biomass
plt.subplot(3, 1, 3)
plt.plot(t, B, label='Plant Biomass (B)', color='green')
plt.xlabel('Days')
plt.ylabel('Plant Biomass')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
