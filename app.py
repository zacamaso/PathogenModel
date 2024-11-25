import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def run_simulation(params):
    """
    Runs the pathogen-plant simulation with given parameters
    """
    # Unpack parameters
    r0, beta, delta, g, alpha, P0, V0, t_end = params
    
    # Setup time points (one per day)
    t_start = 0
    num_points = t_end  # One point per day
    t_eval = np.linspace(t_start, t_end, num_points)

    def system(t, y):
        P, V = y
        P = np.clip(P, 0, 1)
        V = np.clip(V, 0, 1)
        r = r0 * np.exp(-beta * V)
        dP_dt = (r - delta) * P * (1 - P)
        dV_dt = g * V * (1 - V) - alpha * P * V
        return [dP_dt, dV_dt]

    # Solve system
    solution = solve_ivp(system, [t_start, t_end], [P0, V0], t_eval=t_eval, method='RK23')
    
    return solution.t, solution.y[0], solution.y[1]

def create_plot(t, P, V):
    """
    Creates matplotlib figure with two subplots showing daily progression
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot pathogen load
    ax1.plot(t, P, color='red', label='Pathogen Load (P)')
    ax1.set_ylabel('Pathogen Load\n(0 = none, 1 = maximum)')
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    ax1.legend()
    
    # Plot plant vigor
    ax2.plot(t, V, color='green', label='Plant Vigor (V)')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Plant Vigor\n(0 = dead, 1 = maximum health)')
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def main():
    st.title("Plant-Pathogen Interaction Model")
    
    st.markdown("""
    This model simulates the daily interaction between plant pathogens and plant vigor in the root zone.
    The simulation uses realistic parameters based on typical plant growth patterns and pathogen behavior.
    
    ### Key Assumptions:
    - Plants reach ~80% of maximum biomass after 60 days
    - Initial state represents a small seedling
    - Values are normalized between 0 and 1
    - Time steps represent days
    
    ### Parameter Descriptions:
    - **r0**: Daily maximum pathogen reproduction rate (typically pathogens double every 4-7 days)
    - **β (beta)**: Plant defense coefficient - higher values mean stronger plant defense
    - **δ (delta)**: Daily natural death rate of pathogens
    - **g**: Daily plant growth rate (adjusted for ~80% biomass at 60 days)
    - **α (alpha)**: Daily impact of maximum pathogen load on plant vigor
    """)

    # Create two columns for parameters
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pathogen Parameters")
        r0 = st.slider("r0 (max daily reproduction rate)", 0.0, 0.3, 0.15, 0.01,
                     help="Pathogens can increase by up to this percentage per day under ideal conditions")
        beta = st.slider("β (plant defense coefficient)", 0.0, 2.0, 0.8, 0.1,
                      help="At β=0.8, maximum plant vigor reduces pathogen reproduction by ~55%")
        delta = st.slider("δ (natural death rate)", 0.0, 0.3, 0.1, 0.01,
                       help="Percentage of pathogens that die each day from natural causes")
        P0 = st.slider("Initial Pathogen Load", 0.0, 0.2, 0.01, 0.01,
                     help="Starting pathogen population (typically very low)")

    with col2:
        st.subheader("Plant Parameters")
        g = st.slider("g (daily growth rate)", 0.0, 0.2, 0.06, 0.01,
                    help="Daily growth rate (~6% leads to 80% biomass at 60 days)")
        alpha = st.slider("α (pathogen virulence)", 0.0, 0.3, 0.1, 0.01,
                       help="How much maximum pathogen load reduces plant vigor per day")
        V0 = st.slider("Initial Plant Vigor", 0.0, 0.2, 0.05, 0.01,
                     help="Starting plant size (typically 5% of maximum for seedling)")
        t_end = st.slider("Simulation Length (days)", 30, 180, 120, 10,
                       help="Typical growth cycle is 120 days")

    # Run simulation with current parameters
    params = (r0, beta, delta, g, alpha, P0, V0, t_end)
    t, P, V = run_simulation(params)

    # Create and display plot
    fig = create_plot(t, P, V)
    st.pyplot(fig)

    # Display final values
    st.subheader("Final Values")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Final Pathogen Load", f"{P[-1]:.3f}")
    with col2:
        st.metric("Final Plant Vigor", f"{V[-1]:.3f}")

    # Add explanation of results
    st.markdown("""
    ### Interpreting the Results:
    - **Plant Vigor** (green line):
        - Starts at seedling size (5% of maximum)
        - Should reach ~80% of maximum around day 60
        - Growth slows as plant approaches maturity
        - Can be reduced by pathogen pressure
    
    - **Pathogen Load** (red line):
        - Starts with small initial presence
        - Growth is limited by plant defenses
        - Natural death rate provides some control
        - Can increase rapidly if plant defenses fail
    
    The interaction between these values shows whether the plant can successfully defend against the pathogen
    or if the infection becomes severe enough to significantly impact plant health.
    """)

if __name__ == "__main__":
    main() 