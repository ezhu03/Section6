import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------
# 1) ANALYTIC LORENTZIAN FORM
# -----------------------------
def E_lorentzian(omega, F, omega0, gamma):
    """
    Returns the analytic energy per cycle:
        E(omega) = pi * F * [gamma*omega] / [(omega0^2 - omega^2)^2 + (gamma*omega)^2].
    """
    return (
        np.pi * F * (gamma * omega) 
        / ((omega0**2 - omega**2)**2 + (gamma*omega)**2)
    )

# -----------------------------
# 2) TIME-DOMAIN SIMULATION
# -----------------------------
def damped_forced_osc(t, y, omega_f, F, omega0, gamma):
    """
    Right-hand side of:
        x'' + gamma*x' + omega0^2*x = F*cos(omega_f * t).
    y = [x, x'], so y' = [x', x''].
    """
    x, v = y
    dxdt = v
    dvdt = -omega0**2 * x - gamma*v + F*np.cos(omega_f*t)
    return [dxdt, dvdt]

def energy_absorbed_per_cycle(omega_f, F, omega0, gamma,
                              x0=0.0, v0=0.0, n_cycles=20, sample_rate=2000):
    """
    Numerically integrate the damped forced oscillator for n_cycles+1 cycles
    and measure the energy delivered by F*cos(omega_f*t) over the *last* cycle.
    """
    T = 2*np.pi / omega_f  # one driving period
    t_span = (0, (n_cycles+1)*T)
    t_eval = np.linspace(t_span[0], t_span[1], (n_cycles+1)*sample_rate+1)
    
    # Solve the ODE from t=0 to t=(n_cycles+1)*T
    sol = solve_ivp(
        damped_forced_osc, 
        t_span, 
        [x0, v0], 
        t_eval=t_eval, 
        args=(omega_f, F, omega0, gamma)
    )
    
    x_vals = sol.y[0]
    v_vals = sol.y[1]
    t_vals = sol.t
    
    # Instantaneous power: P(t) = F*cos(omega_f*t) * v(t).
    # We'll integrate over the last cycle [n_cycles*T, (n_cycles+1)*T].
    start_index = np.searchsorted(t_vals, n_cycles*T)
    t_seg = t_vals[start_index:]
    v_seg = v_vals[start_index:]
    P_seg = F * np.cos(omega_f * t_seg) * v_seg
    
    # Trapezoidal integration over that last cycle
    E = np.trapz(P_seg, x=t_seg)
    return E

# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":
    # Parameters
    F = 1.0        # forcing amplitude
    omega0 = 2.0   # undamped natural frequency
    gamma = 0.2    # damping coefficient
    
    # Range of driving frequencies
    omega_values = np.linspace(0.1, 4.0, 40)
    
    # Compute the analytic (Lorentzian) curve
    E_theory = [E_lorentzian(w, F, omega0, gamma) for w in omega_values]
    
    # Compute numeric energy from time-domain simulations
    E_numerical = [energy_absorbed_per_cycle(w, F, omega0, gamma) for w in omega_values]
    
    # Plot results
    plt.figure(figsize=(8,5))
    plt.plot(omega_values, E_theory, 'r-', label='Lorentzian (analytic)')
    plt.plot(omega_values, E_numerical, 'bo', label='Time-domain simulation')
    plt.xlabel('Driving frequency  ω_f')
    plt.ylabel('Energy absorbed per cycle  E(ω_f)')
    plt.title('Driven-Damped Oscillator: Energy per Cycle vs. Frequency')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig('lorentzian.png')
