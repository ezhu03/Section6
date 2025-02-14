import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Parameters
# --------------------------
n = 0.01       # star density (stars per unit volume)
N_sim = 100000  # number of simulations (experiments)
R_max = 10.0    # maximum radius for simulation (chosen large enough)

# --------------------------
# Theoretical PDF function
# --------------------------
def theoretical_pdf(R, n):
    """
    Returns the probability density f(R) for the nearest star being at distance R.
    """
    return 4 * np.pi * n * R**2 * np.exp(- (4/3) * np.pi * n * R**3)

# --------------------------
# Monte Carlo Simulation
# --------------------------
def simulate_nearest_distance(n, R_max):
    """
    Simulate the distance to the nearest star from the origin in a sphere of radius R_max.
    Stars are distributed with density n (Poisson process).
    
    Returns:
        The distance to the nearest star (or R_max if no star is found).
    """
    # Calculate the volume of the sphere of radius R_max.
    volume = (4/3) * np.pi * R_max**3
    
    # Draw the number of stars from a Poisson distribution.
    num_stars = np.random.poisson(n * volume)
    
    # If there are no stars, we return R_max (could also choose to discard such trials).
    if num_stars == 0:
        return R_max
    
    # For stars uniformly distributed in volume, the cumulative distribution for the distance r is:
    #   F(r) = (r / R_max)^3  for 0 <= r <= R_max.
    # Thus, to generate a random distance, we use the inverse transform:
    #   r = R_max * u^(1/3) where u is uniformly distributed in [0, 1].
    u = np.random.uniform(size=num_stars)
    distances = R_max * (u)**(1/3)
    
    # Return the smallest distance among the stars (the nearest one).
    return np.min(distances)

# Run the simulation many times
simulated_distances = np.array([simulate_nearest_distance(n, R_max) for _ in range(N_sim)])

# --------------------------
# Plotting the results
# --------------------------
# Define a range of R values for the theoretical pdf
R_values = np.linspace(0, R_max, 1000)
pdf_values = theoretical_pdf(R_values, n)

plt.figure(figsize=(10, 6))
# Plot histogram of simulated nearest distances; density=True normalizes the histogram.
plt.hist(simulated_distances, bins=50, density=True, alpha=0.5, label='Simulation')
# Plot the theoretical pdf
plt.plot(R_values, pdf_values, 'r-', lw=2, label='Theoretical PDF')
plt.xlabel('Distance R')
plt.ylabel('Probability Density f(R)')
plt.title('Distribution of the Nearest Star Distance')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('poisson.png')