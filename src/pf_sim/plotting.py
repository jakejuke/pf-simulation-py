# src/pf_simulation/plotting.py
# Auto generated from Gemini as an example of a plotting function
"""
Functions for visualizing phase field simulation results.
This module provides functions to plot the results of phase field simulations,
including the final state of the simulation grid.
"""

import matplotlib.pyplot as plt
from .simulation import Simulation # Import the class definition

def plot_final_state(sim_object: Simulation, save_path: str):
    """
    Plots the final state of the simulation grid.

    Args:
        sim_object: The completed Simulation object.
        save_path: The file path to save the figure to.
    """
    # Access the data directly from the passed-in object's attributes
    grid_data = sim_object.mesh.data 
    timestep = sim_object.timestep

    plt.figure()
    plt.imshow(grid_data, cmap='viridis')
    plt.title(f"Final State at Timestep {timestep}")
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()
    