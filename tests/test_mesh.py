# tests/test_mesh.py

from src.pf_simulation.mesh import Mesh

def test_mesh_initialization_2d():
    """Tests if a 2D Mesh object is created with the correct attributes."""
    # Setup
    shape = (128, 64)
    dx = 0.5
    
    # Action
    mesh = Mesh(shape=shape, dx=dx)
    
    # Assertions (check if the results are what you expect)
    assert mesh.ndim == 2
    assert mesh.shape == shape
    assert mesh.dx == dx
    assert mesh.data.shape == shape

def test_mesh_initialization_3d():
    """Tests if a 3D Mesh object is created with the correct attributes."""
    # Setup
    shape = (32, 32, 32)
    dx = 1.5
    
    # Action
    mesh = Mesh(shape=shape, dx=dx)
    
    # Assertions
    assert mesh.ndim == 3
    assert mesh.shape == shape
    assert mesh.dx == dx

