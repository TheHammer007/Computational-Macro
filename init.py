import numpy as np
from scipy import special

# Step 1: Generate a wage distribution
def generate_wage_distribution(n=60, a=600, b=400, wage_min=10, wage_max=20):
    """
    Generate a wage distribution following a Beta-binomial distribution.
    
    Parameters:
    -----------
    n : int
        Number of wage points
    a, b : float
        Shape parameters for the Beta-binomial distribution
    wage_min, wage_max : float
        Minimum and maximum wages
    
    Returns:
    --------
    wages : numpy.ndarray
        Array of wage values
    probs : numpy.ndarray
        Corresponding probabilities for each wage
    """
    # Create equidistant wages in the interval [wage_min, wage_max]
    wages = np.linspace(wage_min, wage_max, n)
    
    # Calculate probabilities using the Beta-binomial distribution
    k = np.arange(n)
    probs = np.zeros(n)
    
    for i in range(n):
        probs[i] = special.binom(n-1, i) * special.beta(i+a, n-1-i+b) / special.beta(a, b)
    
    # Normalize probabilities to sum to 1
    probs = probs / np.sum(probs)
    
    return wages, probs

# Step 3: Solve the fixed point problem
def solve_mccall_model(wages, probs, beta=0.95, c=6, tol=1e-6, max_iter=10000):
    """
    Solve the McCall job search model using value function iteration.
    
    Parameters:
    -----------
    wages : numpy.ndarray
        Array of possible wage values
    probs : numpy.ndarray
        Corresponding probabilities for each wage
    beta : float
        Discount factor
    c : float
        Unemployment compensation
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    
    Returns:
    --------
    V : numpy.ndarray
        Value function at each wage level
    reservation_wage : float
        The minimum wage the worker will accept
    """
    n = len(wages)
    
    # Step 2: Set an arbitrary first guess for V
    V_old = np.zeros(n)  # Initialize with zeros
    
    # Alternative initialization: worker always accepts wage offer
    # V_old = wages / (1 - beta)
    
    # Value function iteration
    for i in range(max_iter):
        # For each possible wage offer, calculate the value of accepting
        V_accept = wages / (1 - beta)
        
        # Calculate the value of rejecting: c + β E[V(w)]
        V_reject = c + beta * np.sum(V_old * probs)
        
        # The value function is the maximum of accepting or rejecting
        V_new = np.maximum(V_accept, V_reject * np.ones(n))
        
        # Check for convergence
        if np.max(np.abs(V_new - V_old)) < tol:
            break
        
        V_old = V_new.copy()
        
        if i == max_iter - 1:
            print("Warning: Maximum iterations reached without convergence")
    
    # Find the reservation wage (smallest wage where accepting is better than rejecting)
    V_reject_final = c + beta * np.sum(V_new * probs)
    accept_indices = np.where(wages / (1 - beta) >= V_reject_final)[0]
    
    if len(accept_indices) > 0:
        reservation_wage = wages[accept_indices[0]]
    else:
        reservation_wage = None
        
    return V_new, reservation_wage

# Run the model
if __name__ == "__main__":
    # Generate the wage distribution
    wages, probs = generate_wage_distribution()
    
    # Solve the model
    V, reservation_wage = solve_mccall_model(wages, probs)
    
    print(f"Reservation wage: {reservation_wage:.4f}")
    
    # Summary statistics
    print(f"Number of wage points: {len(wages)}")
    print(f"Wage range: [{min(wages):.2f}, {max(wages):.2f}]")
    print(f"Value function - Min: {min(V):.4f}, Max: {max(V):.4f}")
