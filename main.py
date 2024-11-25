# Import necessary libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

"""
This model simulates the interaction between plant pathogens and plant vigor in the root zone.
Values are normalized between 0 and 1 where:
- Plant Vigor (V): 0 = dead plant, 1 = maximum health
- Pathogen Load (P): 0 = no pathogens, 1 = maximum pathogen capacity
"""

# Define parameters (adjusted for daily timesteps)
r0 = 0.15     # Pathogens can increase by up to 15% per day under ideal conditions
beta = 0.8    # Strong plant defense impact on pathogen reproduction
delta = 0.1   # About 10% of pathogens die naturally each day
g = 0.06      # ~6% daily growth rate (leads to ~80% biomass at 60 days)
alpha = 0.1   # Pathogens reduce plant vigor by up to 10% per day at maximum load

# Initial conditions
P0 = 0.01     # Starting with very small pathogen presence (1% of max)
V0 = 0.05     # Starting with small seedling (5% of max potential biomass)

print("Starting simulation...")

# Simulation timeframe
t_start = 0
t_end = 105   # Simulate for 120 days to see full growth cycle
num_points = 105  # One point per day
t_eval = np.linspace(t_start, t_end, num_points)

def system(t, y):
    """
    Defines the system of differential equations with bounds checking:
    dP/dt = (r0 * e^(-β*V) - δ) * P    # Change in pathogen load
    dV/dt = g * V * (1 - V) - α * P * V # Change in plant vigor
    
    All values are constrained between 0 and 1
    """
    P, V = y  # Current state
    
    # Ensure values stay within bounds
    P = np.clip(P, 0, 1)
    V = np.clip(V, 0, 1)
    
    # Calculate pathogen reproduction rate
    r = r0 * np.exp(-beta * V)
    
    # Differential equations with logistic growth for plant vigor
    dP_dt = (r - delta) * P * (1 - P)  # Logistic growth for pathogens
    dV_dt = g * V * (1 - V) - alpha * P * V  # Logistic growth for plant vigor
    
    # Debug output
    if int(t) % 2 == 0:
        print(f"Time: {t:.1f}, Pathogen Load: {P:.2f}, Plant Vigor: {V:.2f}")
    
    return [dP_dt, dV_dt]

# Initial state vector
y0 = [P0, V0]

print("Solving differential equations...")
# Solve the system using scipy's solve_ivp (Initial Value Problem solver)
try:
    solution = solve_ivp(system, [t_start, t_end], y0, t_eval=t_eval, method='RK23')
    print("Solver completed successfully!")
except Exception as e:
    print(f"Error in solver: {e}")
    exit(1)

# Extract solutions
P = solution.y[0]  # Pathogen load over time
V = solution.y[1]  # Plant vigor over time
t = solution.t     # Time points

# Create visualization
plt.figure(figsize=(12, 6))

# Plot pathogen population dynamics
plt.subplot(2, 1, 1)
plt.plot(t, P, label='Pathogen Load (P)', color='red')
plt.title('Pathogen Load and Plant Vigor Over Time')
plt.ylabel('Pathogen Load\n(0 = none, 1 = maximum)')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

# Plot plant vigor dynamics
plt.subplot(2, 1, 2)
plt.plot(t, V, label='Plant Vigor (V)', color='green')
plt.xlabel('Time')
plt.ylabel('Plant Vigor\n(0 = dead, 1 = maximum health)')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

plt.tight_layout()

print("Creating plots...")
plt.show()

print("Done! Check for the plot window.")
