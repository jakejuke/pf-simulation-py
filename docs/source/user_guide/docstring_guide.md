# Best Practices for Writing Docstrings

The goal of a docstring is to explain **what** a function or class does and **why**, not _how_ it does it. The code itself shows the "how."

1. **Use the Imperative Mood:** Start the summary line with a command or verb.
    
    - **Good:** `"""Calculate the Laplacian of the grid."""`
        
    - **Bad:** `"""This function calculates the Laplacian of the grid."""`
        
2. **Be Consistent:** Stick to a single, consistent format like the **NumPy style** throughout your entire project.
    
3. **Describe Inputs and Outputs:** Clearly document each parameter (`Parameters`) and what the function returns (`Returns`), including their types.
    
4. **Keep it Concise:** A one-line docstring is often sufficient for simple functions.
    

## Example Docstring (NumPy Style)

``` python
def calculate_laplacian(grid: np.ndarray, dx: float) -> np.ndarray:
    """Calculate the Laplacian of a 2D grid using a 5-point stencil.

    This function applies a finite difference approximation to compute the
    Laplacian, assuming periodic boundary conditions.

    Parameters
    ----------
    grid : np.ndarray
        The 2D input array on which to compute the Laplacian.
    dx : float
        The grid spacing.

    Returns
    -------
    np.ndarray
        A new array of the same shape as `grid` containing the Laplacian.
    """
    # ... function code ...
```

## What Goes at the Top of a Python File

Follow the standard order recommended by Python's official style guide (PEP 8).

1. **Shebang (Optional):** For scripts you intend to make directly executable on Linux/macOS.
    
    ```
    #!/usr/bin/env python3
    ```
    
2. **Module Docstring:** A triple-quoted string that explains the purpose of the entire file/module. This is the most important part.
    
3. **Imports:** All import statements come after the module docstring and should be grouped in this order:
    
    1. Standard library imports (e.g., `os`, `math`).
        
    2. Third-party library imports (e.g., `numpy`, `matplotlib`).
        
    3. Your own local application/library imports (e.g., `from .mesh import Mesh`).
        

## Example File Structure

``` python
#!/usr/bin/env python3
"""
The main simulation engine for the phase-field package.

This module contains the `Simulation` class, which acts as the high-level
interface for setting up and running simulations. It handles state management,
timestepping, and dispatching computations to the appropriate backend
(CPU or GPU).
"""

# 1. Standard library imports (none in this case)

# 2. Third-party library imports (alphabetized)
import numpy as np

# 3. Local application imports (alphabetized)
from .cpu_kernels import run_step_cpu_2d, run_step_cpu_3d
from .mesh import Mesh


# --- Start of your code (e.g., the Simulation class definition) ---
class Simulation:
    # ...
```