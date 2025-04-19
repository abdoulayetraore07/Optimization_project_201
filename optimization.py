import numpy as np
import time

def calculate_convergence_rate(distances):
    """
    Calculate the convergence rate based on a list or array of distances.

    Parameters:
        distances (list or np.ndarray): A sequence of Lp distances or errors.

    Returns:
        tuple: (mu, rate)
            - mu: Estimated convergence rate (last valid ratio).
            - rate: String indicating the type of convergence 
                    ('Superlinear or Quadratic', 'Linear', or 'Sublinear or Divergent').
    """
    # Calculate successive differences
    differences = np.diff(distances)

    # Ensure enough data for convergence analysis
    if len(differences) < 2:
        return None, "Insufficient data to determine rate"

    # Calculate successive ratios with protection against division by zero
    valid_differences = differences[1:]  # Denominators for ratios
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.divide(differences[:-1], valid_differences, where=valid_differences != 0)
        ratios[np.isnan(ratios)] = 0  # Replace NaNs with 0

    # Use the last valid ratio
    mu = ratios[-1] if len(ratios) > 0 else None

    # Determine convergence rate
    if mu is not None:
        if mu < 1:
            rate = "Superlinear or Quadratic"
        elif mu == 1:
            rate = "Linear"
        else:
            rate = "Sublinear or Divergent"
    else:
        rate = "Unable to determine"

    return mu, rate



def NewtonOptimization(initial_values, lagrange_values, max_iterations, tolerance, lp_norm, matrix, vector):
    """Run the Newton optimization method and calculate important metrics."""
    # Initialize solution and settings
    current_solution = np.zeros(8, dtype=float)
    current_solution[:5] = np.array(initial_values, dtype=float)
    current_solution[5:] = np.array(lagrange_values, dtype=float)

    all_solutions = [current_solution.copy()]
    distances = []

    # Timing the process
    start_time = time.time()

    for iteration in range(max_iterations):
        # Substitute current solution into symbolic matrix and vector
        substituted_matrix = matrix.subs({
            'x1': current_solution[0], 'x2': current_solution[1], 
            'x3': current_solution[2], 'x4': current_solution[3], 
            'x5': current_solution[4], 'lambda1': current_solution[5], 
            'lambda2': current_solution[6], 'lambda3': current_solution[7]
        })

        substituted_vector = vector.subs({
            'x1': current_solution[0], 'x2': current_solution[1], 
            'x3': current_solution[2], 'x4': current_solution[3], 
            'x5': current_solution[4], 'lambda1': current_solution[5], 
            'lambda2': current_solution[6], 'lambda3': current_solution[7]
        })

        # Convert symbolic data to numerical arrays
        matrix_np = np.array(substituted_matrix).astype(float)
        vector_np = np.array(substituted_vector).astype(float).ravel()

        # Add stability factor
        matrix_np += 1e-9 * np.eye(matrix_np.shape[0])

        # Solve the linear system
        step = np.linalg.solve(matrix_np, vector_np)

        # Update solution
        previous_solution = current_solution.copy()
        current_solution -= step
        all_solutions.append(current_solution.copy())

        # Calculate and store distance
        distance = np.linalg.norm(vector_np, ord=lp_norm)
        distances.append(distance)

        # Stopping condition 1: Check step size
        if np.linalg.norm(step, ord=lp_norm) < tolerance:
            print(f"Stopping due to small step size: ||step|| < {tolerance}")
            break

        # Stopping condition 2: Check difference between successive solutions
        if np.linalg.norm(current_solution - previous_solution, ord=lp_norm) < tolerance:
            print(f"Stopping due to small solution change: ||X_k+1 - X_k|| < {tolerance}")
            break

        # Stopping condition 3: Gradient convergence (optional, if gradient distances are tracked)
        if len(distances) > 2:
            gradient = np.gradient(distances, edge_order=2)
            if np.linalg.norm(gradient, ord=lp_norm) < tolerance:
                print(f"Stopping due to gradient convergence: ||∇L(x_k, λ_k)|| < {tolerance}")
                break

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Convergence rate
    mu, rate = calculate_convergence_rate(distances)

    return {
        "final_solution": current_solution,
        "all_solutions": np.array(all_solutions),
        "distances": np.array(distances),
        "iterations": len(distances),
        "cpu_time": total_time,
        "convergence_rate": rate,
        "rate_constant": mu,
    }
