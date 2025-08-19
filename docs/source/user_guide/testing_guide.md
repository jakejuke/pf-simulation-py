## The Goal: Write Code That's Easy to Test

Thinking about testing _while you code_ is a best practice that leads to better, more modular software design. The goal is to write **testable code**.

## The Core Idea: Isolate and Verify 🔬

The fundamental principle of testing is to **isolate** a small piece of your code (a single function or method) and **verify** that it behaves exactly as you expect. You provide a known input and assert that it produces a known output.

## Best Practices to Follow While Coding

### 1. Keep Functions Small and Focused

A function should do **one thing and do it well**. This is the most important rule for creating testable code.

- **Bad (Hard to Test):** A single function that loads data, performs a calculation, and saves the result. You can't test the calculation without involving the file system.
    
- **Good (Easy to Test):** Break the logic into separate functions:
    
    - `load_data(filepath)`: Only reads a file.
        
    - `perform_calculation(data)`: Only does the math.
        
    - `save_data(result, filepath)`: Only writes a file.
        

This allows you to write a simple test for `perform_calculation` without any external dependencies like files.

### 2. Separate Calculations from I/O

Your core scientific logic (the math) should **never** be mixed with code that has side effects, such as:

- Reading or writing files
    
- Plotting figures
    
- Printing to the console
    

By separating these concerns, you can test your critical numerical code in a clean, predictable environment.

### 3. Use "Pure" Functions When Possible

A pure function is one whose output depends **only on its input arguments**. It does not modify its inputs or any global state.

- **Bad (Impure Function - Modifies Input):**
    
    ```
    def run_step_impure(grid):
        # This function modifies the grid it receives
        laplacian = calculate_laplacian(grid)
        grid += laplacian # Modifies the original grid in-place
    ```
    
    Testing this is harder because you have to check if the original input variable was changed correctly.
    
- **Good (Pure Function - Returns New Data):**
    
    ```
    def run_step_pure(grid):
        # This function returns a *new* grid, leaving the original unchanged
        laplacian = calculate_laplacian(grid)
        new_grid = grid + laplacian
        return new_grid
    ```
    
    Testing this is much simpler: you provide an input and assert that the returned value is what you expect.
