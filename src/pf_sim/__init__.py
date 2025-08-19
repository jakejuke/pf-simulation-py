# src/pf_simulation/__init__.py

"""
pf_simulation: A Python package for phase field simulations
"""

# Import the main classes to make them available at the top-level of the package
from .mesh import Mesh
from .simulation import Simulation

# Import key functions for convenience
from .io import load_simulation_from_file, save_simulation
from .plotting import plot_grid

# Define what gets imported when a user types 'from pf_simulation import *'
# This is a good practice for defining the public API.
__all__ = [
    'Mesh',
    'Simulation',
    'load_simulation_from_file',
    'save_simulation',
    'plot_grid'
]

# You can also define a package-level version number
__version__ = "0.1.0"