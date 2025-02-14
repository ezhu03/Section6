import numpy as np
from numpy.linalg import eig
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
from scipy.linalg import logm
from scipy.integrate import solve_ivp

def build_heisenberg_xxx(N, J=1.0):
    """
    Build the Heisenberg XXX Hamiltonian matrix for N sites with periodic
    boundary conditions, returning a sparse (2^N x 2^N) matrix in CSR format.

    H = J * sum_{i=0}^{N-1} [ S^z_i S^z_{i+1} + 0.5 * ( S^+_i S^-_{i+1} + S^-_i S^+_{i+1} ) ],
    with site indices taken modulo N.
    """
    dim = 2**N

    # Precompute S^z for each site in each basis state
    # s_z_state[s, i] = +1/2 if spin i is up in basis state s, else -1/2
    s_z_state = np.zeros((dim, N), dtype=np.float64)
    for s in range(dim):
        for i in range(N):
            if ((s >> i) & 1) == 1:  # bit i set => spin up
                s_z_state[s, i] = +0.5
            else:
                s_z_state[s, i] = -0.5

    # Helper function: flip bit i in state s
    def flip_spin(state, i):
        return state ^ (1 << i)

    # Lists to accumulate the nonzero entries in COO format
    row_indices = []
    col_indices = []
    values = []

    # Build the Hamiltonian
    for s in range(dim):
        # ----- Diagonal part: sum over neighbors i, (i+1) -----
        diag_val = 0.0
        for i in range(N):
            j = (i + 1) % N
            diag_val += s_z_state[s, i] * s_z_state[s, j]

        if diag_val != 0.0:
            row_indices.append(s)
            col_indices.append(s)
            values.append(diag_val)

        # ----- Off-diagonal (flip-flop) -----
        for i in range(N):
            j = (i + 1) % N

            bit_i = (s >> i) & 1
            bit_j = (s >> j) & 1

            # S^+_i S^-_j: site i down (0), site j up (1)
            if bit_i == 0 and bit_j == 1:
                s_prime = flip_spin(s, i)
                s_prime = flip_spin(s_prime, j)
                row_indices.append(s_prime)
                col_indices.append(s)
                values.append(0.5)  # coefficient

            # S^-_i S^+_j: site i up (1), site j down (0)
            if bit_i == 1 and bit_j == 0:
                s_prime = flip_spin(s, i)
                s_prime = flip_spin(s_prime, j)
                row_indices.append(s_prime)
                col_indices.append(s)
                values.append(0.5)  # coefficient

    # Build a COO matrix then convert to CSR
    H_coo = coo_matrix(
        (values, (row_indices, col_indices)),
         shape=(dim, dim),
         dtype=np.float64
    )
    
    # Multiply by J if desired
    H_coo.data *= J
    
    # Finally convert to CSR for more efficient linear algebra
    H_csr = H_coo.tocsr()
    return H_csr

def build_markov_chain_from_heisenberg(H):
    """
    Construct a discrete-time Markov chain transition matrix P
    from the off-diagonal structure of the Heisenberg Hamiltonian
    for N=3 spins (or any N, but let's focus on N=3 for clarity).

    The idea:
      - For each basis state 's' (row in H), find all states 's_prime' != s
        where H[s_prime, s] != 0 (i.e., flip-flop transitions).
      - Let k = number of such neighbors.
      - If k > 0, assign P[s->s_prime] = 1/k for each s_prime in that set;
        else if k=0, P[s->s] = 1 (absorbing state).
    """
    dim = H.shape[0]
    # We'll build a dense matrix for clarity, though one could keep it sparse.
    P = np.zeros((dim, dim), dtype=float)

    # Convert H to a (row->list_of_columns) adjacency or check H off-diagonal directly
    # We'll do a simple pass: for each state s, identify the nonzero off-diagonal in the s-th column of H.
    # NOTE: H is in CSR, so H[:,s] is not as direct, but we can transpose or just parse rows.
    # Alternatively, parse row s for neighbors s' (the flip direction doesn't matter for adjacency).
    #
    # However, from a "Markov chain from flip-flop" perspective, we can read row s:
    #   Off-diagonal elements of row s (H[s, x]) with value = 0.5 => there's a flip from x->s or s->x.
    # Actually, let's do it by searching the columns in row s that have 0.5 or -some_value? 
    # In the Heisenberg code, the flip-flop terms have 0.5. The diagonal has the S^z S^z terms.
    #
    # But to match the usual "from s to s'", we want P[s->s'].
    # That means we look at the s-th row of H, find columns s' != s that have 0.5.
    # If H[s, s'] = 0.5 or so, that means H has off-diagonal for the pair (s, s').
    # For the Heisenberg model we built, it's symmetrical for the flip-flop part, but let's be explicit.

    for s in range(dim):
        # Extract the row slice
        row_start = H.indptr[s]
        row_end   = H.indptr[s+1]
        col_indices = H.indices[row_start:row_end]
        vals       = H.data[row_start:row_end]

        # Identify the off-diagonal positions that correspond to a single flip (coefficient 0.5).
        # We only want s' != s
        neighbors = []
        for c, val in zip(col_indices, vals):
            if c != s and abs(val - 0.5) < 1e-12:  # check if ~0.5
                neighbors.append(c)

        k = len(neighbors)
        if k == 0:
            # absorbing state
            P[s, s] = 1.0
        else:
            for s_prime in neighbors:
                P[s, s_prime] = 1.0 / k

    return P
def site_basis_energies(N, J=1.0):
    """
    Compute the energy E[s] of each of the 2^N spin configurations
    under a simplified Heisenberg-XXZ or Ising-like model.
    
    Here, for demonstration, we'll use a simple 'Ising-like' diagonal energy:
       H = -J sum_{i} (S^z_i * S^z_{i+1})
    with S^z = ±1, and periodic boundary conditions.
    
    If you wanted the true Heisenberg XXX energies, you would need
    diagonal + off-diagonal terms. But for a classical Metropolis,
    we only need the diagonal energies to define acceptance.
    """
    dim = 2**N
    energies = np.zeros(dim, dtype=float)

    for s in range(dim):
        # Convert bit pattern to spin +1 or -1
        spins = np.array([+1 if ((s >> i) & 1)==1 else -1 for i in range(N)])
        # Sum neighbor products
        E = 0
        for i in range(N):
            j = (i+1) % N
            E += -J * spins[i]*spins[j]
        energies[s] = E
    return energies
def find_stationary_distribution_eig(P):
    """
    Finds a stationary distribution pi for the Markov chain with transition matrix P,
    by computing the eigenvector of P^T corresponding to eigenvalue 1 and
    normalizing so that sum(pi) = 1.

    Returns:
        pi (1D numpy array): A stationary distribution (row vector).
    """
    # P is size (n x n); we want to solve pi * P = pi.
    # This is equivalent to (P^T) * pi^T = pi^T, i.e. pi^T is an eigenvector of P^T with eigenvalue 1.

    w, v = eig(P.T)   # v[:, j] is an eigenvector of P^T corresponding to eigenvalue w[j]
    # Find the eigenvalue closest to 1
    idx = np.argmin(np.abs(w - 1.0))

    # Extract that eigenvector, make real (small imaginary parts can arise from numerical noise)
    pi_vec = v[:, idx].real

    # Normalize so that components sum to 1
    pi_sum = np.sum(pi_vec)
    if abs(pi_sum) < 1e-15:
        raise ValueError("Eigenvector for eigenvalue ~1 has near-zero sum. Chain might be degenerate.")
    pi_vec /= pi_sum

    return pi_vec

def find_stationary_distribution_iter(P, max_iter=1000, tol=1e-12):
    """
    Simple power-method / iterative approach to find a stationary distribution:
      1. Start from a uniform distribution.
      2. Repeatedly do v <- v * P
      3. Stop when changes become small or we hit max_iter.

    Returns:
        v (1D numpy array): The converged distribution vector.
    """
    n = P.shape[0]
    v = np.ones(n) / n  # uniform initial distribution
    for _ in range(max_iter):
        v_next = v.dot(P)
        if np.linalg.norm(v_next - v, 1) < tol:
            return v_next
        v = v_next
    return v  # might not be fully converged if max_iter too small
def power_iteration(P, pi0, num_steps=10):
    """
    Repeatedly apply pi_{k+1} = pi_k * P for 'num_steps'.
    Returns a list of distributions [pi^0, pi^1, pi^2, ...].
    """
    distributions = [pi0]
    pi = pi0.copy()
    for _ in range(num_steps):
        pi = pi @ P
        distributions.append(pi)
    return distributions

import numpy as np

def build_magnon_energies(N, J):
    """
    For an N-spin Heisenberg ring in the single-magnon subspace,
    E_k = 2 J sin^2(pi*k/N),  k=0..N-1.
    """
    ks = np.arange(N)
    E = 2*J * np.sin(np.pi * ks / N)**2
    return E

def build_magnon_boltzmann_chain(E, T, kB=1.0):
    """
    Construct the 3x3 Markov chain P_{k->k'} for N=3 (or NxN in general)
    via a Boltzmann-like rule:
        P_{k->k'} ~ exp( -(E_{k'} - E_k)/(kB*T) ).
    Each row is normalized to sum=1.
    """
    n = len(E)
    P = np.zeros((n, n), dtype=float)
    for k in range(n):
        # Numerator for each possible k'
        row_unnormalized = []
        for kprime in range(n):
            exponent = -(E[kprime] - E[k])/(kB*T)
            row_unnormalized.append(np.exp(exponent))
        row_sum = sum(row_unnormalized)
        for kprime in range(n):
            P[k, kprime] = row_unnormalized[kprime]/row_sum
    return P

def build_site_basis_metropolis_chain(energies, T, kB=1.0):
    """
    Build the Metropolis single-spin-flip transition matrix P (dim x dim),
    where dim=2^N.  For each state s, we consider flipping one of N spins,
    compute energy difference dE, accept with probability min(1, e^{-dE/kBT}).
    
    We normalize so that each row sums to 1.
    """
    dim = len(energies)
    N = int(np.log2(dim))

    P = np.zeros((dim, dim), dtype=float)

    for s in range(dim):
        # We'll consider flipping each of the N spins with equal probability 1/N
        rates = []
        flips = []
        for i in range(N):
            s_prime = s ^ (1 << i)  # flip spin i
            dE = energies[s_prime] - energies[s]
            # Metropolis acceptance
            a = 1.0 if dE <= 0 else np.exp(-dE/(kB*T))
            rates.append(a)
            flips.append(s_prime)

        rate_sum = sum(rates)
        # Probability out of s to each accepted flip:
        for (sp, r) in zip(flips, rates):
            P[s, sp] = r / rate_sum

    return P
def find_stationary_distribution(P, tol=1e-14, max_iter=1000):
    """
    Solve pi P = pi by power iteration
    """
    n = P.shape[0]
    pi = np.ones(n)/n  # uniform start
    for _ in range(max_iter):
        pi_next = pi @ P
        if np.linalg.norm(pi_next - pi, 1) < tol:
            return pi_next
        pi = pi_next
    return pi

def find_stationary_distribution_eig(P):
    """
    Solve pi P = pi by looking for eigenvector of P^T with eigenvalue 1.
    """
    w, v = eig(P.T)
    # Pick eigenvector whose eigenvalue is closest to 1
    idx = np.argmin(np.abs(w - 1.0))
    pi_vec = v[:, idx].real
    # Normalize so it sums to 1
    pi_vec /= pi_vec.sum()
    return pi_vec

def master_equation_ode(t, pi, Q):
    """
    ODE for pi(t) in row-vector form:
      d pi(t) / dt = pi(t) * Q.
    pi is shape (n,), Q is (n,n).
    We return d pi / dt as shape (n,).
    """
    # Convert pi (1D) to row vector if we want: but we can just do direct multiplication
    dpi_dt = pi @ Q
    return dpi_dt
# ---------------------------------------------------------------------
# Example usage for N=3
# ---------------------------------------------------------------------
if __name__ == "__main__":
    N = 3
    J = 1.0
    kB = 1.0
    H = build_heisenberg_xxx(N, J)
    print("Heisenberg Hamiltonian (dense) for N=3:")
    print(H.toarray())

    P = build_markov_chain_from_heisenberg(H)
    print("\nMarkov transition matrix P (from flip-flop structure):")
    # We can print it with nicer formatting
    np.set_printoptions(precision=3, suppress=True)
    print(P)

    # Check that each row sums to 1
    row_sums = P.sum(axis=1)
    print("\nRow sums (should be all 1):", row_sums)
    # 1) Solve via eigenvector
    pi_eig = find_stationary_distribution_eig(P)
    print("\nStationary distribution (Eigenvalue method):\n", pi_eig)
    print("Check sum(pi) =", np.sum(pi_eig))
    print("Check pi*P - pi =", pi_eig @ P - pi_eig, "\n")

    # 2) Solve via iterative approach
    pi_iter = find_stationary_distribution_iter(P)
    print("Stationary distribution (Iterative method):\n", pi_iter)
    print("Check sum(pi) =", np.sum(pi_iter))
    print("Check pi*P - pi =", pi_iter @ P - pi_iter)
    # We'll label the 8 basis states as s=0..7 in binary:
    #   s=0 -> |↓↓↓>, s=7 -> |↑↑↑>, etc.
    # For clarity, let's define a helper to show distribution as "8 components".
    def show_dist(pi, label=""):
        print(f"{label} [{', '.join(f'{p:5.3f}' for p in pi)}]  (sum={sum(pi):.3f})")

    #
    # 1) Start from Pr(|↑↑↑>)=1
    #
    pi0_case1 = np.zeros(8); pi0_case1[7] = 1.0  # all in state s=7
    dists_case1 = power_iteration(P, pi0_case1, num_steps=5)
    print("\nCase 1) Initial distribution = 100% in |↑↑↑> (s=7)")
    for k, dist in enumerate(dists_case1):
        show_dist(dist, label=f"k={k}")

    #
    # 2) Start from half in |↑↑↑> (s=7) and half in |↓↑↓> (s=2)
    #
    pi0_case2 = np.zeros(8); pi0_case2[7] = 0.5; pi0_case2[2] = 0.5
    dists_case2 = power_iteration(P, pi0_case2, num_steps=5)
    print("\nCase 2) Initial distribution = 50% in s=7, 50% in s=2 (|↓↑↓>)")
    for k, dist in enumerate(dists_case2):
        show_dist(dist, label=f"k={k}")

    #
    # 3) Uniform initial distribution
    #
    pi0_case3 = np.ones(8) / 8.0
    dists_case3 = power_iteration(P, pi0_case3, num_steps=5)
    print("\nCase 3) Uniform initial distribution = 1/8 for each of the 8 states")
    for k, dist in enumerate(dists_case3):
        show_dist(dist, label=f"k={k}")
    
    T = 1.0
    E = build_magnon_energies(N, J)
    P_magnon = build_magnon_boltzmann_chain(E, T)
    print("Magnon energies:", E)
    print("Magnon-basis transition matrix P (3x3):\n", P_magnon)
    print("Row sums (should be 1):", P_magnon.sum(axis=1))

    energies_site = site_basis_energies(N, J=J)
    P_site = build_site_basis_metropolis_chain(energies_site, T, kB=kB)
    print("\nSite-basis Markov chain (8x8) with Metropolis flips:\n", P_site)
    print("Row sums (should be 1):", P_site.sum(axis=1))
    print("Example energies of the 8 states:", energies_site)
    # Let's sweep over a range of temperatures
    # We'll sweep a small set of T values
    T_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    # Precompute site-basis energies (8 states for N=3)
    

    # Precompute magnon energies (3 states)
    energies_magnon = build_magnon_energies(N, J=J)

    # We'll store the resulting stationary distributions for plotting
    # in the site basis (8D) and in the magnon basis (3D).
    site_dists = []
    magnon_dists = []

    for T in T_values:
        # Build/solve site-basis chain:
        P_site = build_site_basis_metropolis_chain(energies_site, T, kB=kB)
        pi_site = find_stationary_distribution_eig(P_site)

        # Build/solve magnon-basis chain:
        P_magnon = build_magnon_boltzmann_chain(energies_magnon, T, kB=kB)
        pi_magnon = find_stationary_distribution_eig(P_magnon)

        site_dists.append(pi_site)
        magnon_dists.append(pi_magnon)

    site_dists = np.array(site_dists)       # shape = (len(T), 8)
    magnon_dists = np.array(magnon_dists)   # shape = (len(T), 3)

    # ----------------------------------------------------------------------
    # Print some results
    # ----------------------------------------------------------------------
    for i, T in enumerate(T_values):
        print(f"\n=== T = {T} ===")
        print("Site-basis pi (8 states):", site_dists[i])
        print("Magnon-basis pi (3 states):", magnon_dists[i])

    # ----------------------------------------------------------------------
    # Plot:  We'll make 2 subplots
    #   (A) site-basis: group states by # of spins up, or show each state separately
    #   (B) magnon-basis: 3 states
    # ----------------------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))

    # (A) For the site basis, let's group states by "number of spins up"
    # s in [0..7], each has 0..3 spins up.  We'll group probability by that count.
    # Then we plot total probability for 0 up, 1 up, 2 up, 3 up vs T.
    pop_0up = []; pop_1up = []; pop_2up = []; pop_3up = []
    for i, dist in enumerate(site_dists):
        # dist is shape (8,)
        # spin count for each s:
        counts = [bin(s).count('1') for s in range(8)]
        p0 = sum(dist[s] for s in range(8) if counts[s]==0)  # all-down
        p1 = sum(dist[s] for s in range(8) if counts[s]==1)
        p2 = sum(dist[s] for s in range(8) if counts[s]==2)
        p3 = sum(dist[s] for s in range(8) if counts[s]==3)  # all-up
        pop_0up.append(p0)
        pop_1up.append(p1)
        pop_2up.append(p2)
        pop_3up.append(p3)

    ax1.plot(T_values, pop_0up, 'o--', label='0 spins up')
    ax1.plot(T_values, pop_1up, 'o--', label='1 spin up')
    ax1.plot(T_values, pop_2up, 'o--', label='2 spins up')
    ax1.plot(T_values, pop_3up, 'o--', label='3 spins up')
    ax1.set_xlabel("Temperature T")
    ax1.set_ylabel("Probability sum in site basis")
    ax1.set_title("Site Basis (grouped by # spins up)")
    ax1.legend()

    # (B) For the magnon basis, we have just 3 states: k=0, k=1, k=2, energies [0, 1.5, 1.5]
    pi_k0 = magnon_dists[:,0]
    pi_k1 = magnon_dists[:,1]
    pi_k2 = magnon_dists[:,2]
    ax2.plot(T_values, pi_k0, 'o--', label='k=0 (E=0)')
    ax2.plot(T_values, pi_k1, 'o--', label='k=1 (E=1.5J)')
    ax2.plot(T_values, pi_k2, 'o--', label='k=2 (E=1.5J)')
    ax2.set_xlabel("Temperature T")
    ax2.set_ylabel("Probability in magnon basis")
    ax2.set_title("Magnon Basis")
    ax2.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('site_magnon_distributions.png')
    N=5
    # 1) Compute single-magnon energies
    E = build_magnon_energies(N, J)
    print(f"Magnon energies (N={N}): {E}")

    # 2) Build the Boltzmann-type Markov chain in magnon basis
    P = build_magnon_boltzmann_chain(E, T, kB)
    print("\nMagnon-basis transition matrix P:\n", P)
    print("Check row-sums:", P.sum(axis=1))

    # 3) For reference, find the stationary distribution by eigenvalue
    pi_stationary = find_stationary_distribution_eig(P)
    print("\nStationary distribution (via eigen-decomposition):\n", pi_stationary)
    print("Check sum:", pi_stationary.sum())

    # ------------------------------------------------------------------
    # Question 6: Three initial guesses and use power iteration
    #    1) Pr(|k=1>) = 1
    #    2) Pr(|k=1>) = 1/2, Pr(|k=4>)=1/2
    #    3) Uniform over k=0..4
    # We'll do about 5 or 6 iterations to see the approach to stationarity.
    # ------------------------------------------------------------------

    nstates = N  # i.e. 5 in this example

    # (1) All in k=1
    pi0_case1 = np.zeros(nstates); pi0_case1[1] = 1.0
    dists_case1 = power_iteration(P, pi0_case1, num_steps=6)
    print("\nCase 1) Initial distribution = 100% in |k=1>")
    for step, dist in enumerate(dists_case1):
        print(f"Iteration {step}, pi = {dist}")

    # (2) Half in k=1, half in k=4
    pi0_case2 = np.zeros(nstates)
    pi0_case2[1] = 0.5
    pi0_case2[4] = 0.5
    dists_case2 = power_iteration(P, pi0_case2, num_steps=6)
    print("\nCase 2) Initial distribution = 50% in |k=1>, 50% in |k=4>")
    for step, dist in enumerate(dists_case2):
        print(f"Iteration {step}, pi = {dist}")

    # (3) Uniform
    pi0_case3 = np.ones(nstates)/nstates
    dists_case3 = power_iteration(P, pi0_case3, num_steps=6)
    print("\nCase 3) Uniform initial distribution over k=0..4")
    for step, dist in enumerate(dists_case3):
        print(f"Iteration {step}, pi = {dist}")
    # 1) Build discrete-time chain P
    E = build_magnon_energies(N, J)
    P = build_magnon_boltzmann_chain(E, T, kB=kB)

    # 2) Convert P -> Q = (1 / delta_t) logm(P). We'll take delta_t=1 for simplicity
    #    We'll use scipy.linalg.logm for matrix log
    from scipy.linalg import logm
    Q_matrix = logm(P)  # complex matrix in general, but for a valid chain we want real
    # We'll keep only the real part in case of small floating imaginary errors
    Q = Q_matrix.real

    # Check if row sums of Q are ~0 (they should be, up to numerical precision)
    row_sums = Q.sum(axis=1)
    print("Q row sums (should be ~0):", row_sums)

    # 3) Solve d pi/dt = pi Q, with initial condition pi(0)= e_k=1
    #    i.e. all population in state k=1 at t=0
    n = P.shape[0]
    pi0 = np.zeros(n)
    pi0[1] = 1.0  # Pr(|k=1>)=1 initially

    # We'll integrate from t=0 to t=5.  The chain is small so that's plenty
    t_span = (0, 5)

    # Our ODE is: d pi / dt = pi * Q
    def ode_func(t, y):
        return master_equation_ode(t, y, Q)

    sol = solve_ivp(ode_func, t_span, pi0, dense_output=True, max_step=0.01)
    # "sol.y" has shape (n, number_of_time_points)
    # "sol.t" is the time array

    # 4) Plot pi_k(t) vs t
    ts = np.linspace(0, 5, 200)  # times for sampling the solution
    sol_y = sol.sol(ts)         # shape = (n, len(ts))
    # Each row is pi_k(t). We'll transpose for convenience
    pi_of_t = sol_y.T  # shape = (len(ts), n)

    plt.figure(figsize=(6,4))
    for k in range(n):
        plt.plot(ts, pi_of_t[:,k], label=f"pi_{k}(t)")

    plt.title("Master Eq in Magnon Basis (continuous time)")
    plt.xlabel("Time t")
    plt.ylabel("Probability pi_k(t)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('master_eq_magnon_basis.png')

    # 5) Compare final pi(t->infinity) with the discrete-time stationary distribution
    #    from an eigenvalue solve on P.
    #    In principle, they should match if the logm was consistent.
    #    Let's see the final pi(t=5) from the ODE:
    pi_final_ODE = pi_of_t[-1,:]
    print("\nFinal pi from ODE (t=5):", pi_final_ODE, " sum=", pi_final_ODE.sum())

    # Stationary distribution from discrete chain:
    w, v = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(w - 1.0))
    pi_stationary = v[:,idx].real
    pi_stationary /= pi_stationary.sum()
    print("Stationary dist from discrete chain:", pi_stationary)
