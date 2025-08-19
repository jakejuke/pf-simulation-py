# Python Phase-Field Simulation

A Python-based library for running phase field simulations, designed for high performance and extensibility.

## Features

-   Modular and object-oriented design.
-   CPU backend accelerated with Numba.
-   GPU backend ready for implementation with Numba/CuPy.
-   Clear separation of simulation logic and analysis.

## Installation

1.  Clone the repository:
    ```bash
    git clone [your-repo-url]
    cd pf_simulation_py
    ```

2.  Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Quick Start

See the example in `notebooks/01-run-simulation.ipynb` to run your first simulation.
