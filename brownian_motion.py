import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 0.1          # drift coefficient
sigma = 0.2       # volatility coefficient
T = 10         # total time

N = 100    # number of time steps
dt = T / N  # time step
t = np.linspace(0, T, N+1)

# Set random seed for reproducibility
np.random.seed(80)

# Generate increments for the Wiener process (Brownian motion)
dW = np.sqrt(dt) * np.random.randn(N)  # dW ~ N(0, dt)
W = np.concatenate(([0], np.cumsum(dW)))  # W_0 = 0 and then cumulative sum

# Compute the process X_t using the closed form solution:
# X_t = exp(mu*t + sigma*W_t)
X = np.exp(mu * t + sigma * W)

# Plot the sample path
plt.figure(figsize=(8, 4))
plt.plot(t, X, lw=2)
plt.xlabel('Time')
plt.ylabel('$X(t)$')
plt.title(r'Sample Path of $X(t) = \exp(\mu t + \sigma W_t)$')
plt.grid(True)
plt.show()
plt.savefig('brownian_motion_X(t).png')

for i in range(N):
    X[i+1] = X[i] + X[i] * ( (mu + 0.5 * sigma**2)*dt + sigma * dW[i] )

# Compute x_I(t) = (X_I(t) - X_I(0)) / t, avoid division by zero at t=0.
xI = np.zeros(N+1)
xI[0] = 0  # define x_I(0)=0 by convention
xI[1:] = (X[1:] - X[0]) / t[1:]

# Plot the trajectory of x_I(t)
plt.figure(figsize=(10, 6))
plt.plot(t, xI, marker='o', label=r'$x_I(t)=\frac{X_I(t)-X_I(0)}{t}$')
plt.xlabel('Time $t$')
plt.ylabel(r'$x_I(t)$')
plt.title('Trajectory of $x_I(t)$')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('brownian_motion_xI(t).png')

# Heun scheme for the Stratonovich SDE:
# dX = X * [mu * dt + sigma * ◦ dW]
for i in range(N):
    # Evaluate drift and diffusion at the current point
    drift = mu * X[i]
    diff  = sigma * X[i]
    
    # Predictor step (Euler estimate)
    X_pred = X[i] + drift * dt + diff * dW[i]
    
    # Evaluate drift and diffusion at the predicted point
    drift_pred = mu * X_pred
    diff_pred  = sigma * X_pred
    
    # Corrector step (Heun update)
    X[i+1] = X[i] + 0.5 * (drift + drift_pred) * dt + 0.5 * (diff + diff_pred) * dW[i]

# Compute x_S(t) = (X(t) - X(0)) / t, avoiding division by zero at t = 0.
xS = np.zeros(N+1)
xS[0] = 0  # define x_S(0) = 0 by convention
xS[1:] = (X[1:] - X[0]) / t[1:]

# Plot the trajectory of x_S(t)
plt.figure(figsize=(10, 6))
plt.plot(t, xS, marker='o', linestyle='-', label=r'$x_S(t)=\frac{X_I(t)-X_I(0)}{t}$')
plt.xlabel('Time $t$')
plt.ylabel(r'$x_S(t)$')
plt.title('Trajectory of $x_S(t)$ via Stratonovich Integration')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('brownian_motion_xS(t).png')

# Choose N values logarithmically spaced between 10 and 1e4.
# We use 20 points in the logspace.
Ns = np.unique(np.logspace(np.log10(10), np.log10(1e4), num=20, dtype=int))

# Arrays to store final statistics for each integrator.
ito_means = []
ito_vars  = []
strato_means = []
strato_vars  = []

# Set the random seed once for reproducibility.
np.random.seed(42)
X0 = 1           # initial value: X(0)=1
num_MC = 100
# -----------------------------
# Loop over different N (time resolution)
# -----------------------------
for N in Ns:
    dt = T / N
    # Arrays to store the final x value for each trajectory.
    x_ito_vals = np.zeros(num_MC)
    x_strato_vals = np.zeros(num_MC)
    
    for mc in range(num_MC):
        # Generate Brownian increments for this trajectory
        dW = np.sqrt(dt) * np.random.randn(N)
        
        # -----------------------------
        # Itô integrator (Euler–Maruyama for the Itô SDE)
        # SDE: dX = X[(mu + 0.5*sigma**2)*dt + sigma*dW]
        X_ito = X0
        for i in range(N):
            X_ito = X_ito + X_ito * ((mu + 0.5 * sigma**2) * dt + sigma * dW[i])
        # Compute observable x = (X(T)-X(0))/T at final time
        x_ito_vals[mc] = (X_ito - X0) / T
        
        # -----------------------------
        # Stratonovich integrator (Heun scheme)
        # SDE: dX = X[mu*dt + sigma o dW]
        X_strato = X0
        for i in range(N):
            # Evaluate drift and diffusion at the current state:
            drift = mu * X_strato
            diff  = sigma * X_strato
            # Predictor step (Euler step):
            X_pred = X_strato + drift * dt + diff * dW[i]
            # Evaluate drift and diffusion at the predicted state:
            drift_pred = mu * X_pred
            diff_pred  = sigma * X_pred
            # Heun (corrector) update:
            X_strato = X_strato + 0.5 * (drift + drift_pred) * dt + 0.5 * (diff + diff_pred) * dW[i]
        x_strato_vals[mc] = (X_strato - X0) / T
    
    # Compute sample mean and variance for this N
    ito_means.append(np.mean(x_ito_vals))
    ito_vars.append(np.var(x_ito_vals))
    strato_means.append(np.mean(x_strato_vals))
    strato_vars.append(np.var(x_strato_vals))

# -----------------------------
# Plotting: 4 subplots (2 rows x 2 columns)
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Itô mean vs. N
axs[0, 0].plot(Ns, ito_means, marker='o', linestyle='-')
axs[0, 0].set_xscale('log')
axs[0, 0].set_title('Itô Integrator: Mean of x(T)')
axs[0, 0].set_xlabel('N (log scale)')
axs[0, 0].set_ylabel('Mean')

# Itô variance vs. N
axs[0, 1].plot(Ns, ito_vars, marker='o', linestyle='-')
axs[0, 1].set_xscale('log')
axs[0, 1].set_title('Itô Integrator: Variance of x(T)')
axs[0, 1].set_xlabel('N (log scale)')
axs[0, 1].set_ylabel('Variance')

# Stratonovich mean vs. N
axs[1, 0].plot(Ns, strato_means, marker='o', linestyle='-')
axs[1, 0].set_xscale('log')
axs[1, 0].set_title('Stratonovich Integrator: Mean of x(T)')
axs[1, 0].set_xlabel('N (log scale)')
axs[1, 0].set_ylabel('Mean')

# Stratonovich variance vs. N
axs[1, 1].plot(Ns, strato_vars, marker='o', linestyle='-')
axs[1, 1].set_xscale('log')
axs[1, 1].set_title('Stratonovich Integrator: Variance of x(T)')
axs[1, 1].set_xlabel('N (log scale)')
axs[1, 1].set_ylabel('Variance')

plt.tight_layout()
plt.show()
plt.savefig('brownian_motion_statistics.png')

# Define a set of time-step numbers, logarithmically spaced from 10 to 1e4
Ns = np.unique(np.logspace(np.log10(10), np.log10(1e4), num=20, dtype=int))

# Containers for statistics for the functionals
ito_F_means = []
ito_F_vars  = []
strato_F_means = []
strato_F_vars  = []

# Set random seed for reproducibility.
np.random.seed(42)

# -----------------------------
# Loop over different N (time resolution)
# -----------------------------
for N in Ns:
    dt = T / N

    # Arrays to store the final F values for each Monte Carlo run
    F_I_vals = np.zeros(num_MC)
    F_S_vals = np.zeros(num_MC)
    
    for mc in range(num_MC):
        # --- Itô simulation ---
        X_ito = X0
        F_I = 0.0
        # Generate dW increments for this trajectory
        dW = np.sqrt(dt)*np.random.randn(N)
        for i in range(N):
            # Euler-Maruyama update for X (Itô SDE):
            dX_ito = X_ito * ((mu + 0.5 * sigma**2)*dt + sigma*dW[i])
            # Update the functional F_I using the left-point rule:
            F_I += (X_ito**2) * dX_ito
            # Update X
            X_ito += dX_ito

        # --- Stratonovich simulation ---
        X_strato = X0
        F_S = 0.0
        # Use the same dW for fair comparison:
        # (Note: In a proper simulation you would generate new dW's but here we want to match the noise)
        # For Stratonovich, we use the Heun scheme for X and the midpoint rule for F_S.
        for i in range(N):
            # Current drift and diffusion:
            drift = mu * X_strato
            diff  = sigma * X_strato
            # Predictor (Euler step):
            X_pred = X_strato + drift*dt + diff*dW[i]
            # Heun (corrector) update for X:
            drift_pred = mu * X_pred
            diff_pred  = sigma * X_pred
            dX_strato = 0.5*(drift+drift_pred)*dt + 0.5*(diff+diff_pred)*dW[i]
            X_new = X_strato + dX_strato
            # For the Stratonovich integral, use the midpoint of X^2:
            F_S += ((X_strato**2 + X_new**2)/2) * dX_strato
            # Update X
            X_strato = X_new

        # Save the final accumulated functionals at time T
        F_I_vals[mc] = F_I
        F_S_vals[mc] = F_S

    # Compute sample mean and variance for this N (Itô)
    ito_F_means.append(np.mean(F_I_vals))
    ito_F_vars.append(np.var(F_I_vals))
    # And for Stratonovich
    strato_F_means.append(np.mean(F_S_vals))
    strato_F_vars.append(np.var(F_S_vals))

# -----------------------------
# Plotting the statistics: 4 subplots
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Itô Functional Mean
axs[0, 0].plot(Ns, ito_F_means, marker='o', linestyle='-')
axs[0, 0].set_xscale('log')
axs[0, 0].set_title('Itô: Mean of $F_I(T)$')
axs[0, 0].set_xlabel('N (log scale)')
axs[0, 0].set_ylabel('Mean')

# Itô Functional Variance
axs[0, 1].plot(Ns, ito_F_vars, marker='o', linestyle='-')
axs[0, 1].set_xscale('log')
axs[0, 1].set_title('Itô: Variance of $F_I(T)$')
axs[0, 1].set_xlabel('N (log scale)')
axs[0, 1].set_ylabel('Variance')

# Stratonovich Functional Mean
axs[1, 0].plot(Ns, strato_F_means, marker='o', linestyle='-')
axs[1, 0].set_xscale('log')
axs[1, 0].set_title('Stratonovich: Mean of $F_S(T)$')
axs[1, 0].set_xlabel('N (log scale)')
axs[1, 0].set_ylabel('Mean')

# Stratonovich Functional Variance
axs[1, 1].plot(Ns, strato_F_vars, marker='o', linestyle='-')
axs[1, 1].set_xscale('log')
axs[1, 1].set_title('Stratonovich: Variance of $F_S(T)$')
axs[1, 1].set_xlabel('N (log scale)')
axs[1, 1].set_ylabel('Variance')

plt.tight_layout()
plt.show()
plt.savefig('brownian_motion_functional_statistics.png')
# Parameters for the simulation
T_max = 50.0        # maximum simulation time (must be > max{t0}+max τ)
N = 500  # number of time points
dt = T_max / N  # time step
time_grid = np.linspace(0, T_max, N)

# Model parameters for the geometric Brownian motion
mu = 0.1
sigma = 0.2
X0 = 1.0

num_MC = 100  # number of Monte Carlo trajectories

# Preallocate an array to store the accumulated functional F(t)
# For each trajectory we will compute F(t)= sum_{n} X_n^2*(X_{n+1}-X_n)
F_all = np.zeros((num_MC, N))

# Simulate trajectories for X(t) and compute F(t) along each path
np.random.seed(42)  # for reproducibility

for i in range(num_MC):
    X = np.zeros(N)
    F = np.zeros(N)   # F(0) = 0
    X[0] = X0
    for n in range(N-1):
        dW = np.sqrt(dt)*np.random.randn()
        # Euler-Maruyama update for X(t)
        dX = X[n] * ((mu + 0.5*sigma**2)*dt + sigma*dW)
        X[n+1] = X[n] + dX
        # Update the functional F using the left-point rule:
        F[n+1] = F[n] + (X[n]**2)*dX
    F_all[i, :] = F

# Now, for fixed stopping times t0 = 5, 10, 20, 30, compute the autocorrelation function:
# C(τ) = <F(t0)*F(t0+τ)> as a function of the lag τ.
t0_list = [5, 10, 20, 30]

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for idx, t0 in enumerate(t0_list):
    t0_index = int(t0/dt)
    # We require that t0+τ remains within our simulation time.
    # Maximum lag index available is:
    max_lag = N - t0_index  
    tau_values = np.arange(max_lag) * dt  # convert lag indices to time lag τ

    # For each τ, compute the product F(t0)*F(t0+τ) over all trajectories
    # and then average over the ensemble.
    C_tau = np.zeros(max_lag)
    # Get the array of F(t0) values for all trajectories.
    F_t0 = F_all[:, t0_index]
    for lag in range(max_lag):
        F_t0_lag = F_all[:, t0_index + lag]
        C_tau[lag] = np.mean(F_t0 * F_t0_lag)
    
    axs[idx].plot(tau_values, C_tau, marker='o', linestyle='-')
    axs[idx].set_title(f"Autocorrelation C(τ) at t = {t0}")
    axs[idx].set_xlabel("τ")
    axs[idx].set_ylabel("C(τ)")
    axs[idx].grid(True)

plt.tight_layout()
plt.show()
plt.savefig('brownian_motion_autocorrelation.png')