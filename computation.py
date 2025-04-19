import sympy as sp

def initialize_matrices():
    # Define symbolic variables
    x1, x2, x3, x4, x5, lambda1, lambda2, lambda3 = sp.symbols('x1 x2 x3 x4 x5 lambda1 lambda2 lambda3')

    # Define the objective function
    objective_function = x1 * x2 * x3 * x4 * x5

    # Define constraints
    constraint_1 = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 - 10
    constraint_2 = 5 * x5 * x4 - x2 * x3
    constraint_3 = x1**3 + x2**3 + 1

    # Create the Hessian matrix
    hessian_f = sp.hessian(objective_function, [x1, x2, x3, x4, x5, lambda1, lambda2, lambda3])
    hessian_constraints = sp.hessian(
        lambda1 * constraint_1 + lambda2 * constraint_2 + lambda3 * constraint_3, 
        [x1, x2, x3, x4, x5, lambda1, lambda2, lambda3]
    )

    # Combine to form the symbolic matrix
    symbolic_matrix = hessian_f + hessian_constraints

    # Create the constraint vector
    variables = [x1, x2, x3, x4, x5, lambda1, lambda2, lambda3]
    constraint_vector = sp.Matrix([
        sp.diff(objective_function + lambda1 * constraint_1 + lambda2 * constraint_2 + lambda3 * constraint_3, var) 
        for var in variables
    ])

    return symbolic_matrix, constraint_vector
