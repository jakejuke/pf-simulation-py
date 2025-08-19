## The Goal: A Python Front End for a C++ Engine

This guide explains how to connect a high-performance C++ backend to a user-friendly Python front end. We will use **`pybind11`**, a modern and powerful C++ library, to create the "bindings" that allow Python to call your C++ code.

This approach gives you the best of both worlds: the raw speed of C++ for heavy computations and the simplicity of Python for everything else.

### Prerequisites

1. **A C++ Compiler:** You'll need a C++ compiler installed on your system (e.g., GCC on Linux, Clang on macOS, or MSVC on Windows).
    
2. **pybind11:** Install it into your Python virtual environment. It's a good idea to add `pybind11` to your `requirements.txt` file.
    
    ```
    pip install pybind11
    ```
    

### Step 1: Organize Your Project

First, structure your repository to cleanly separate the C++ and Python source code.

```
pf_simulation_py/
├── cpp_src/              # <-- Your C++ code lives here
│   ├── simulation.hpp
│   └── bindings.cpp
├── src/
│   └── pf_simulation/    # <-- Your Python package lives here
│       ├── __init__.py
│       └── simulation.py
└── setup.py              # <-- The build script
```

### Step 2: Write Your C++ Engine

This is your core C++ code. For this example, let's imagine a simple C++ class in `cpp_src/simulation.hpp`.

```
// cpp_src/simulation.hpp

#include <vector>
#include <string>

class CppSimulation {
public:
    CppSimulation(int size) : size(size) {
        // Constructor
    }

    void run_step() {
        this->timestep++;
    }

    int get_timestep() const {
        return this->timestep;
    }

private:
    int size;
    int timestep = 0;
};
```

### Step 3: Write the Binding Code

This is the bridge between C++ and Python. You create a file (e.g., `cpp_src/bindings.cpp`) that uses `pybind11` to expose your C++ classes and functions.

```
// cpp_src/bindings.cpp

#include <pybind11/pybind11.h>
#include "simulation.hpp" // Include your C++ engine's header

namespace py = pybind11;

// The PYBIND11_MODULE macro creates the entry point for the Python module.
// The first argument, "cpp_engine", is the name of the module in Python.
PYBIND11_MODULE(cpp_engine, m) {
    m.doc() = "Python bindings for the C++ phase-field engine"; // Optional module docstring

    // Expose the CppSimulation class to Python
    py::class_<CppSimulation>(m, "Simulation")
        .def(py::init<int>())  // Expose the constructor (takes an int)
        .def("run_step", &CppSimulation::run_step) // Expose the run_step method
        .def("get_timestep", &CppSimulation::get_timestep); // Expose the get_timestep method
}
```

### Step 4: Create the Build Script (`setup.py`)

This script tells Python how to find and compile your C++ code into a Python extension. This is the most critical part. Create a `setup.py` file in your project's root directory.

```
# setup.py

from setuptools import setup, Extension
import pybind11

# Define the C++ extension module
cpp_args = ['-std=c++11', '-O3'] # Compiler flags

ext_modules = [
    Extension(
        'pf_simulation.cpp_engine', # The name of the module in the Python package
        ['cpp_src/bindings.cpp'],   # List of C++ source files
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='pf_simulation',
    version='0.1.0',
    packages=['pf_simulation'],
    package_dir={'': 'src'},
    ext_modules=ext_modules,
)
```

### Step 5: Build and Install

Now, from your project's root directory, you can run an "editable" install. This command will compile the C++ code and link it into your `src/pf_simulation` directory so you can import it.

```
pip install -e .
```

If this command succeeds, you have successfully built your hybrid package!

### Step 6: Use It in Python

Finally, you can use your high-speed C++ class just like any other Python class.

```
# In a Python script or notebook...

# Import the C++ class from the compiled module
from pf_simulation.cpp_engine import Simulation

# Create an instance of the C++ object
sim = Simulation(size=256)

# Call its methods
sim.run_step()
sim.run_step()

# It works just like a native Python object!
print(f"Current timestep from C++ engine: {sim.get_timestep()}")
# Output: Current timestep from C++ engine: 2
```