import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def run_simulation(params):
    """
    Runs the pathogen-plant simulation with given parameters
    """
    # Unpack parameters
    r0, beta, delta, g, alpha, eta, gamma, P0, V0, B0, t_end = params
    
    # Setup time points (one per day)
    t_start = 0
    num_points = t_end
    t_eval = np.linspace(t_start, t_end, num_points)

    def system(t, y):
        P, V, B = y
        P = max(0, P)
        V = max(0, V)
        B = np.clip(B, 0, 1)
        
        r = r0 * np.exp(-beta * V)
        dP_dt = (r - delta) * P * (1 - P)
        dV_dt = eta * (1 - V) - gamma * g * B - alpha * P * V
        dB_dt = g * V * B * (1 - B)
        
        return [dP_dt, dV_dt, dB_dt]

    # Solve system
    solution = solve_ivp(system, [t_start, t_end], [P0, V0, B0], t_eval=t_eval, method='RK23')
    
    return solution.t, solution.y[0], solution.y[1], solution.y[2]

def create_plot(t, P, V, B):
    """
    Creates matplotlib figure with three subplots showing daily progression
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot pathogen load
    ax1.plot(t, P, color='red', label='Pathogen Load (P)')
    ax1.set_ylabel('Pathogen Load\n(0 = none, 1 = maximum)')
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    ax1.legend()
    
    # Plot vigor
    ax2.plot(t, V, color='blue', label='Plant Vigor (V)')
    ax2.set_ylabel('Plant Vigor\n(0 = no energy, 1 = maximum)')
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    ax2.legend()
    
    # Plot biomass
    ax3.plot(t, B, color='green', label='Plant Biomass (B)')
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Plant Biomass\n(0 = seedling, 1 = maximum)')
    ax3.set_ylim(0, 1)
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    return fig

def main():
    st.title("Plant-Pathogen Interaction Model")
    
    st.markdown("""
    This model simulates the daily interaction between plant pathogens, plant vigor, and biomass.
    
    ### Model Variables:
    - **Plant Biomass (B)**: Physical size/mass of the plant
        - B = 0: Initial seedling size
        - B = 1: Maximum mature size
        - Growth depends on available vigor and current size
    
    - **Plant Vigor (V)**: Root metabolic energy reserves
        - V = 0: Completely depleted energy reserves
        - V = 1: Maximum energy storage capacity
        - Supports both growth and pathogen defense
    
    - **Pathogen Load (P)**: Root pathogen population
        - P = 0: No pathogens present
        - P = 1: Maximum pathogen carrying capacity
        - Growth inhibited by plant defenses
    
    ### System Equations:
    1. **Pathogen Dynamics** (dP/dt):
        ```
        dP/dt = (r0 * e^(-β*V) - δ) * P * (1-P)
        ```
        - Pathogen growth reduced by plant vigor through β
        - Natural death rate δ
        - Logistic growth limitation (1-P)
    
    2. **Vigor Dynamics** (dV/dt):
        ```
        dV/dt = η*(1-V) - γ*g*B - α*P*V
        ```
        - Natural recovery rate η
        - Energy cost of growth γ*g*B
        - Loss from pathogen damage α*P*V
    
    3. **Biomass Growth** (dB/dt):
        ```
        dB/dt = g * V * B * (1-B)
        ```
        - Base growth rate g
        - Growth requires vigor V
        - Logistic growth limitation (1-B)
    """)

    # Create three columns for parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Pathogen Parameters")
        r0 = st.slider("r0 (reproduction rate)", 0.0, 0.3, 0.15, 0.01,
                     help="Maximum pathogen reproduction rate")
        beta = st.slider("β (defense impact)", 0.0, 2.0, 0.8, 0.1,
                      help="How effectively vigor reduces pathogen reproduction")
        delta = st.slider("δ (death rate)", 0.0, 0.3, 0.1, 0.01,
                       help="Natural pathogen death rate")
        P0 = st.slider("Initial Pathogen Load", 0.0, 1.0, 0.2, 0.01,
                     help="Starting pathogen population")

    with col2:
        st.subheader("Vigor Parameters")
        eta = st.slider("η (vigor recovery)", 0.0, 0.2, 0.08, 0.01,
                     help="Natural vigor recovery rate")
        gamma = st.slider("γ (growth cost)", 0.0, 0.2, 0.05, 0.01,
                       help="How much growth depletes vigor")
        alpha = st.slider("α (pathogen impact)", 0.0, 0.5, 0.1, 0.01,
                       help="How much pathogens reduce vigor")
        V0 = st.slider("Initial Vigor", 0.0, 1.0, 0.75, 0.05,
                     help="Starting vigor level")

    with col3:
        st.subheader("Growth Parameters")
        g = st.slider("g (growth rate)", 0.0, 0.2, 0.06, 0.01,
                    help="Base growth rate")
        B0 = st.slider("Initial Biomass", 0.0, 0.2, 0.05, 0.01,
                     help="Starting plant size")
        t_end = st.slider("Simulation Days", 30, 180, 105, 15,
                       help="Length of simulation")

    # Run simulation with current parameters
    params = (r0, beta, delta, g, alpha, eta, gamma, P0, V0, B0, t_end)
    t, P, V, B = run_simulation(params)

    # Create and display plot
    fig = create_plot(t, P, V, B)
    st.pyplot(fig)

    # Display final values
    st.subheader("Final Values")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Pathogen Load", f"{P[-1]:.3f}")
    with col2:
        st.metric("Final Plant Vigor", f"{V[-1]:.3f}")
    with col3:
        st.metric("Final Biomass", f"{B[-1]:.3f}")

    # Update the interpretation section
    st.markdown("""
    ### Interpreting the Results:
    - **Plant Biomass** (green line):
        - Physical size of the plant
        - Growth rate proportional to vigor level
        - Shows logistic growth pattern
        - Typically reaches ~80% of maximum by day 60
    
    - **Plant Vigor** (blue line):
        - Root metabolic energy reserves
        - Naturally recovers but depleted by:
            1. Supporting biomass growth
            2. Defending against pathogens
        - Key mediator between growth and defense
    
    - **Pathogen Load** (red line):
        - Population of root pathogens
        - Growth suppressed by plant vigor
        - Competes for plant resources
        - Can cause plant stress by depleting vigor
    
    The model demonstrates the trade-off between growth and defense, mediated
    through the plant's energy reserves (vigor). High pathogen loads can deplete
    vigor, which then limits both defense and growth capabilities.
    """)

if __name__ == "__main__":
    main() 