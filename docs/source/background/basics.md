# Basics

Image of 2D microstructure

How many fields (parameters) do we use? One per grain?

Cahn-Hillard Equation

Ginzburg-Landau Equation  
$\frac{\partial \eta_i}{\partial t} = -L_i \frac{\delta F}{\delta \eta_i}$


### The del or nabla operator

$\nabla = \sum_i \vec{e}_i \frac{\partial}{\partial x_i}$,

where x and e...  Familiar expression used for 3D Euclidean space:

$\hat{\textbf{\i}}\frac{\partial}{\partial x} + \hat{\textbf{\j}}\frac{\partial}{\partial y} + \hat{\textbf{k}}\frac{\partial}{\partial z}$


### The Laplace operator or Laplacian

$ \Delta = \nabla \cdot \nabla = \nabla^2 $

Divergence of the gradient...  
the sum off all unmixed second partial derivatives

$\Delta = \sum\limits_{i=1}^{p} \frac{\partial^2}{\partial^2 x_i}$


### Grain boundary thickness

The boundary energy and thickness vary with $\kappa_i$

## Forward Euler method 

This is the method currently used in the code.

## Links

https://en.wikipedia.org/wiki/Euler_method
https://en.wikipedia.org/wiki/Midpoint_method
https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
https://mooseframework.inl.gov/modules/phase_field/Grain_Growth_Model.html


Does it make sense to use another method to differentiate? Does it even make a 
difference in the result? How much more compute would we need?

## Variables



| variable | quantity |
| -------- | -------- |
| $\gamma$ | grain boundary energy |
| $M$      | grain boundary mobility |
| $\eta_i (\vec{r},t)$ | order parameter (field variable) |
| $\kappa_i$ | gradient energy coefficient |
| $\vec{r}$ | position vector in simulation cell |
| $\phi$   | field variable |
| $F$      | total free energy |
| $p$      | number of order parameters |



## Abbreviations

| abbr | full name |
| ---- | --------- |
| nop | number of order parameters |
| pnopA | pointer to nop of matrix A |
...


**Some conventions:**<br>
Abbreviations in code<br>
https://github.com/abbrcode/abbreviations-in-code


