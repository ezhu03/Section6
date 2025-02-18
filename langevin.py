import numpy as np
import matplotlib.pyplot as plt

def simulate_langevin(v0, theta, D, T, dt, npaths):
    """
    Simulate the Langevin equation using the Euler–Maruyama method.
    
    Parameters:
        v0 (float): Initial velocity.
        theta (float): Drift coefficient (ϑ).
        D (float): Diffusion coefficient (D).
        T (float): Total simulation time.
        dt (float): Time step.
        npaths (int): Number of sample paths.
    
    Returns:
        t (numpy.ndarray): Array of time points.
        v_paths (numpy.ndarray): Array of shape (npaths, len(t)) containing sample paths.
    """
    nsteps = int(T / dt)
    t = np.linspace(0, T, nsteps + 1)
    # Initialize the array for storing paths
    v_paths = np.zeros((npaths, nsteps + 1))
    v_paths[:, 0] = v0

    for i in range(nsteps):
        # dW: increments of the Wiener process (Gaussian with mean 0 and variance dt)
        dW = np.sqrt(dt) * np.random.randn(npaths)
        # Euler–Maruyama update
        v_paths[:, i+1] = v_paths[:, i] + theta * v_paths[:, i] * dt + np.sqrt(2 * D) * dW

    return t, v_paths

# Parameters
v0 = 1.0       # initial condition
theta = 0.1    # drift coefficient
D = 0.5        # diffusion coefficient
T = 10.0       # total time
dt = 0.01      # time step
npaths = 10000 # number of trajectories

# Simulate the Langevin dynamics
t, v_paths = simulate_langevin(v0, theta, D, T, dt, npaths)

# Compute ensemble mean and variance at each time step
mean_v = np.mean(v_paths, axis=0)
var_v = np.var(v_paths, axis=0)

# Analytical expressions:
analytical_mean = v0 * np.exp(theta * t)
if theta != 0:
    analytical_var = (D/theta) * (np.exp(2 * theta * t) - 1)
else:
    analytical_var = 2 * D * t

# Plot the ensemble mean versus the analytical mean
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t, mean_v, label="Simulated Mean")
plt.plot(t, analytical_mean, '--', label="Analytical Mean")
plt.xlabel("Time")
plt.ylabel("Mean of v(t)")
plt.title("Mean of Velocity vs Time")
plt.legend()

# Plot the ensemble variance versus the analytical variance
plt.subplot(1, 2, 2)
plt.plot(t, var_v, label="Simulated Variance")
plt.plot(t, analytical_var, '--', label="Analytical Variance")
plt.xlabel("Time")
plt.ylabel("Variance of v(t)")
plt.title("Variance of Velocity vs Time")
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('langevin.png')