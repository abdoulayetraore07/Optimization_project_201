import matplotlib.pyplot as plt
import numpy as np

def display_convergence_plot(results, title, legend_label):
    """Display the distance convergence graph."""
    distances = results['distances']
    iterations = np.arange(len(distances))
    
    plt.plot(iterations, distances, label=legend_label, color='blue', marker='o')
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Distance (L2 Norm)")
    plt.legend()
    plt.grid()
    plt.show()
