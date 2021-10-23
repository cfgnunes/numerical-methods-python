# Numerical Methods [![Actions Status](https://github.com/cfgnunes/numerical-methods-python/workflows/build/badge.svg)](https://github.com/cfgnunes/numerical-methods-python/actions)

Numerical methods implementation in Python 3.

For the "MATLAB version", see [this repository](https://github.com/cfgnunes/numerical-methods-matlab).

## Getting Started

### Prerequisites

#### Using Conda (Recommended)

```sh
conda env create
conda activate numerical-methods-env
```

#### Using Pip

```sh
pip install -r requirements.txt
```

#### Using Ubuntu

This section assumes Ubuntu 16.04 (also tested on Ubuntu 20.04), but the procedure is similar for other Linux distributions.

```sh
sudo apt -y install python3-numpy
```

### Running the examples

To run the main example, use:

```sh
python3 main.py
```

## Implementations

### Solutions of equations

- Bisection method
- Newton method
- Secant method

### Interpolation

- Lagrange method
- Neville method

### Algorithms for polynomials

- Briot-Ruffini method
- Newton's Divided-Difference method

### Numerical differentiation

- Backward-difference method
- Three-Point method
- Five-Point method

### Numerical integration

- Composite Trapezoidal method
- Composite 1/3 Simpson's method

### Initial-value problems for ordinary differential equations

- Euler's method
- Taylor's (Order Two) method
- Taylor's (Order Four) method
- Runge-Kutta (Order Four) method

### Systems of differential equations

- Runge-Kutta (Order Four) method

### Methods for Linear Systems

- Gaussian Elimination
- Backward Substitution
- Forward Substitution

### Iterative Methods for Linear Systems

- Jacobi method
- Gauss-Seidel method

## Built With

- [Python](https://www.python.org/) - Programming language used
