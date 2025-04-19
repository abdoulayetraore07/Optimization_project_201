import numpy as np
from optimization import NewtonOptimization
from computation import initialize_matrices
from visualization import display_convergence_plot
from settings import *

def run_program():
    # Step 1: Prepare symbolic matrices for the problem
    symbolic_matrix, constraint_vector = initialize_matrices()

    # Step 2: Define starting conditions
    starting_point = [-1.71, 1.59, 1.82, -0.763, -0.763]
    #starting_point = [-1.9, 1.82, 2.02, -0.9, -0.9]
    #starting_point = [1, 0, 3, 0, 0]
    initial_lagrange = [1, 1, 1]

    # Step 3: Run the optimization method
    results = NewtonOptimization(
        starting_point, 
        initial_lagrange, 
        max_iterations, 
        tolerance, 
        lp_norm, 
        symbolic_matrix, 
        constraint_vector
    )

    # Step 4: Display results
    result_x= (results['final_solution'])[:5] 
    result_lambda =  (results['final_solution'])[5:8] 
    all_solution_1= results['all_solutions'][:3]
    all_solution_2= results['all_solutions'][9:12]
    all_solution_3= results['all_solutions'][15:18]
    # Transformer en numpy
    matrix_np = np.array(result_x).astype(float)  

    # Produit de tous les éléments
    product = np.prod(matrix_np)
    print("\n\n")
    expo_product = np.exp(product)
    print("=== Optimization Results 1 ===")
    print(f"Final Solution x: {result_x}")
    print(f"Final Solution Lambda: {result_lambda}")
    print("Value of the Objective function of PCE:", product)
    print("Value of the Objective function initial:", expo_product)
    print(f"Total Iterations: {results['iterations']}")
    print(f"CPU Time: {results['cpu_time']} seconds")
    print(f"Convergence Rate: {results['convergence_rate']} (Rate Constant: {results['rate_constant']})")
    print(f"Generated sequence begin: {all_solution_1}")
    print(f"Generated sequence middle: {all_solution_2}")
    print(f"Generated sequence end: {all_solution_3}")
    print("\n\n")

    # Step 5: Visualize convergence
    display_convergence_plot(
        results, 
        title="Convergence Analysis for Starting Point 1", 
        legend_label="Optimization Path"
    )

if __name__ == "__main__":
    run_program()
