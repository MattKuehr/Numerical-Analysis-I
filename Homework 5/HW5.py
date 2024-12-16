# Author: Matt Kuehr / mck0063@auburn.edu
# Date: 2024-11-08
# Assignment Name: hw05

import numpy as np

def p1(data, powers):
    """
    Implement the Richardson extrapolation. 

    Assume the expansion of 
    f(h) = f(0) + c_1 h^{alpha_1} + c_2 h^{alpha_2} + ... + c_n h^{alpha_n} - ...

    @param data: a list of values [f(2^(-1)), f(2^(-2)), ..., f(2^(-n))]
    @param powers: a list of powers [alpha_1, alpha_2, ..., alpha_{n-1}]

    @return: the extrapolated value of f(0) using Richardson extrapolation

    """
    n = len(data)
    
    # Create coefficient matrix to represent extrapolation
    # and populate first column with 1's
    coeff = [[1.0] for _ in range(n)]  
    
    # Fill in the coefficient matrix
    # Each row represents equation for f(2^(-i))
    for i in range(n):
        h = 2.0 ** (-i-1)  # h = 2^(-i-1)
        for j in range(len(powers)):
            coeff[i].append(h ** powers[j])
    
    # Solve system of equations using Gaussian elimination
    # Forward elimination
    for i in range(n):
        
        # Make diagonal element 1
        divisor = coeff[i][i]
        for j in range(i, n):
            coeff[i][j] /= divisor
        data[i] /= divisor
        
        # Eliminate column
        for j in range(i + 1, n):
            factor = coeff[j][i]
            for k in range(i, n):
                coeff[j][k] -= factor * coeff[i][k]
            data[j] -= factor * data[i]
    
    # Back substitution
    for i in range(n-1, -1, -1):
        for j in range(i-1, -1, -1):
            factor = coeff[j][i]
            coeff[j][i] = 0
            data[j] -= factor * data[i]
    
    return data[0]  # This is f(0) as approximated with Richardson extrapolation

# This is a helper function to call in p2
def compute_finite_sum(h, beta):
    """Helper function to compute finite sum up to 1/h terms"""
    n = int(1/h)
    total = 0
    for i in range(n + 1):
        total += (-1)**i / ((2*i + 1)**beta)
    return total

def p2(beta):
    """
    Compute the value of the series 
        sum_{k=0}^(infty) ((-1)^k /(2k + 1)^{beta})

    @param beta: a real value for the parameter beta on (0, 1]

    @return: the value of the series.
    """
    # Choose number of points for extrapolation (we want enough for good accuracy
    # but not too many to induce stability issues)
    m = min(8, int(15 - beta))
    
    # Compute data points f(h) for h = 2^(-k)
    data = []
    for k in range(1, m+1):
        h = 2.0**(-k)
        data.append(compute_finite_sum(h, beta))
    
    # Create powers array
    powers = [beta + i for i in range(m-1)]
    
    # Set up coefficient matrix for Richardson extrapolation
    A = np.zeros((m, m))
    
    # Fill coefficient matrix
    for i in range(m):
        A[i,0] = 1.0  # First column is all 1's
        h = 2.0**(-i-1)  # h = 2^(-i-1)
        for j in range(1, m):
            A[i,j] = h**powers[j-1]
    
    # Solve system of equations
    result = np.linalg.solve(A, data)
    
    # Return first coefficient (the extrapolated value)
    return result[0]


def p3(shifts):
    """
    Compute the coefficients of the finite difference scheme for f'(x)
    using the formula
    
    f'(x) approx (1/h) (c_0 f(x_0) + c_1 f(x_1) + ... + c_n f(x_n)) + O(h^n)

    @param: shifts: a list of real values (a_0, a_1, ..., a_n), the nodes are x_i = x + a_i h
    
    @return: coefs: a numpy array of coefficients (c_0, c_1, ..., c_n)

    """
    n = len(shifts)
    
    # Create matrix A for the system of equations
    # Each row represents a power in the Taylor expansion
    A = np.zeros((n, n))
    
    # Fill matrix A
    for i in range(n):  # row (equation number)
        for j in range(n):  # column (coefficient number)
            
            # For i=0, we want sum of coefficients = 0
            # For i=1, we want sum of coefficients*shifts = 1 (to match f')
            # For i>1, we want sum of coefficients*shifts^i = 0
            A[i,j] = shifts[j]**i if i > 0 else 1
            
    # Create right-hand side vector
    # Only the equation for the first derivative (i=1) has a non-zero RHS
    b = np.zeros(n)
    b[1] = 1
    
    # Solve the system of equations
    coefs = np.linalg.solve(A, b)
    
    return coefs


def p4(shifts, l):
    """
    For 6630 only. 
    
    Compute the coefficients of the finite difference scheme for f^{(l)}(x)
    using the formula

    f^{(l)}(x) approx (1/h^l) (c_0 f(x_0) + c_1 f(x_1) + ... + c_n f(x_n)) + O(h^{n + 1 - l})

    @param: shifts: a list of real values (a_0, a_1, ..., a_n), the nodes are x_i = x + a_i h
    @param: l: an integer n >= l >= 1, the order of the derivative

    @return: coefs: a numpy array of coefficients (c_0, c_1, ..., c_n)

    """
    coefs = np.zeros(len(shifts))
    # write your code here.
    return coefs

