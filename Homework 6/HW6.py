# Author: Matt Kuehr / mck0063@auburn.edu
# Date: 2024-12-02
# Assignment Name: hw06

import numpy as np
import matplotlib.pyplot as plt

# Problem 1
def p1(func, a, b, n, option):
    """
    Implement composite quadrature rules for numerical integration
    of a function over the interval [a, b] with n subintervals.
    The option parameter determines the type of quadrature rule to use: 
    
    1 for the midpoint rule, 2 for the trapezoidal rule, and 3 for Simpson's rule.

    @param func: The function to integrate, provided as a function handle.
    @param a: The lower bound of the integration interval.
    @param b: The upper bound of the integration interval.
    @param n: The number of subintervals to use for the integration.
    @param option: The type of quadrature rule to use (1, 2, or 3).
    
    @return: The approximate integral of the function over the interval [a, b].
    """

    if option == 1:
        # Composite midpoint rule
        h = (b - a) / n
        x_midpoints = a + (np.arange(n) + 0.5) * h
        ret = h * np.sum(func(x_midpoints))
    elif option == 2:
        # Composite trapezoidal rule
        h = (b - a) / n
        x = a + np.arange(n + 1) * h
        y = func(x)
        ret = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    elif option == 3:
        if n % 2 != 0:
            raise ValueError("The number of subintervals must be even for Simpson's rule.")
        # Composite Simpson's rule
        h = (b - a) / n
        x = a + np.arange(n + 1) * h
        y = func(x)
        ret = h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    else:
        raise ValueError("Invalid option value. Must be 1, 2, or 3.")
    
    return ret

# Problem 2
def p2():
    """
    run with the following command: hw06.p2(). Do not edit this function.
    
    It checks the convergence of the composite quadrature rules implemented in p1.
    
    Here we use some examples, 
    f_1(x) = exp(x), 
    f_2(x) = (1 - x^2)^3, this function's first 2 derivatives at the endpoints are zero.
    f_3(x) = (1 - x^2)^5, this function's first 4 derivatives at the endpoints are zero.
    f_4(x) = (1 - x^2)^7, this function's first 6 derivatives at the endpoints are zero.

    Run this function will plot the figures for the convergence of the composite quadrature rules.
    Make comments about the results obtained from the plots. 
    
    > For instance, you can comment on the convergence rates of the quadrature rules, and how they compare to the theoretical rates.
    > Here are a few example questions you can answer in your comments:
    > Does the Runge phenomenon of f1 (Runge's function) lower the convergence rate?
    > Does Simpson's rule have a faster convergence rate than the other two for these examples?
    > Based on your observations, what kind of functions can have a faster convergence rate with the given composite quadrature rules?

    Write your comments here.
    > For f1 the midpoint and trapezoidal rule exhibit convergence rates of O(h^2), while Simpson's rule exhibits O(h^4). This agrees
    > with the theoretical rates expected of these quadrature rules. Simpson's rule has faster convergence rate, as was expected 
    > because it utilizes the information from 3 points to approximate the underlying function, and it is reasonable to expect that e^x
    > on [-1,1] would be more accurately captured with this method due to its rate of change being more accurately captured with the 
    > quadratic approximation of Simpson's rule compared to the linear approximation of the others.
    >
    > For f2 all three rules exhibit convergence rates between O(h^2) and O(h^4), close to the boundary of the O(h^4) line. For this 
    > function, trapezoidal rule and midpoint rule exhibit very similar results, but Simpson's rule actually lags slightly behind them.
    > f2 is symmetric about 0, so midpoint and trapezoidal rules will do fairly well to approximate it. It is noteworthy that f2 is 
    > somewhat smooth about its endpoints, so the extra information Simpson's rule uses could actually lead to an accumulation of errors
    > due to Runge's phenomenon, leading it to converge slightly slower than the other rules and underperform its theoretical convergence.
    > The first two derivatives vanishing allows the midpoint and trapezoidal rules to cancel leading error terms. This results in these
    > rules overperforming slightly, while Simpson's rule underperforms. So the properties of the objective function allow these rules a 
    > slight edge over Simspon's rule.
    >
    > For f3 all three rules exhibit convergence between O(h^4) and O(h^6), with trapezoidal and midpoint rule being identical in 
    > performance, and Simpson's rule lagging slightly behind. So Simpson's rule is matching its theoretical convergence rate, while 
    > the other two rules are exceeding them. The midpoint and trapezoidal rules are likely achieving superconvergence because the 
    > symmetry of the interval leads to cancellations which reduce the error. Also, the first 4 derivatives vanish at the endpoints and
    > so the contributions for these terms will cancel, enabling midpoint and trapezoidal rules to exceed expectations and Simpson's rule
    > to overcome the error induced by the Runge Phenomenon and return to its expected theoretical convergence. 
    >
    > For f4 Simpson's rule exhibits convergence of nearly O(h^6) while midpoint and trapezoidal rules exceed this convergence and sit
    > between the lines delineating O(h^6) and O(h^8). The symmetry of the objective function as well as its vanishing endpoint derivatives 
    > allow several leading error terms to be cancelled, resulting in superconvergence for all methods, especially the midpoint and  
    > trapezoidal rules which are more receptive to these properties of the function.       
    >
    > The Runge phenomenon does not lower the convergence rate of f1, because e^x is not smooth on the endpoints of [-1,1] due to the  
    > exponential rate of change of the function. It also does not exhibit sharp oscillatory behavior. Thus, the Runge Phenomenon does 
    > not have the requisite conditions to occur.
    > 
    > Simpson's rule exhibits better convergence than the trapezoidal and midpoint rules for f1, but not for f2, f3 or f4. This is because
    > in general, Simpson's rule uses more information and can achieve a better approximation for f1, but the symmetry of f2, f3 and f4 
    > alongside the vanishing derivatives at the endpoints allow the midpoint and trapezoidal rules to achieve superconvergence by cancelling
    > several leading error terms. As a result, the structure of the objective functions allows the midpoint and trapezoidal rules to exceed 
    > expectations, and in some cases marginally outperform Simpson's rule. 
    >
    > Functions which are symmetric about the midpoint of the interval and have derivatives that vanish at the endpoints are prone to 
    > experiencing faster convergence rates. This is because the odd contributions in the quadrature rule's approximation error will
    > mostly cancel out due to symmetry. Also, since the error of quadrature rules depends on the derivatives of the objective function,  
    > the more vanishing derivatives a function has at the endpoints, the more leading error terms will effectively be cancelled.
    >        
    """
    funcs = [np.exp,
             lambda x: (1 - x**2)**3,
             lambda x: (1 - x**2)**5,
             lambda x: (1 - x**2)**7]

    funcs_names = ['exp(x)', '(1 - x^2)^3', '(1 - x^2)^5', '(1 - x^2)^7']

    exact = [np.exp(1) - np.exp(-1), 32/35, 512/693, 4096/6435]

    n = 2**np.arange(1, 9, dtype=float) # Changing to float array to avoid errors

    for k, func in enumerate(funcs):
        errors = np.zeros((3, len(n)))
        for i, n_i in enumerate(n):
            for j in range(3):
                errors[j, i] = np.abs(p1(func, -1, 1, n_i, j+1) - exact[k])

        plt.figure(k)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.plot(n, errors[0, :], 'r-', label='Midpoint Rule')
        plt.plot(n, errors[1, :], 'g-', label='Trapezoidal Rule')
        plt.plot(n, errors[2, :], 'b-', label="Simpson's Rule")
        plt.plot(n, 1/n**2, 'm--', label='2nd order convergence')
        plt.plot(n, 1/n**4, 'k-.', label='4nd order convergence')
        plt.plot(n, 1/n**6, 'm--d', label='6nd order convergence')
        plt.plot(n, 1/n**8, 'k--o', label='8nd order convergence')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('number of subintervals')
        plt.ylabel('Error')
        plt.title(f'Convergence of Quadrature Rules for {funcs_names[k]}')
        plt.legend()
    plt.show()

# p1 from hw05 to be called in p3
def p1_hw05(data, powers):
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

# Problem 3
def p3(func, a, b, N, option):
    """
    Use your implemented Richardson extrapolation function in HW05 to implement the Romberg integration method.
    
    @param func: The function to integrate, provided as a function handle.
    @param a: The lower bound of the integration interval.
    @param b: The upper bound of the integration interval.
    @param N: it means 2^N is the maximum number of subintervals to use for the integration. 
              The Romberg method will start with 2^1=2 subintervals and double the number of subintervals until 2^N
    @param option: The type of quadrature rule to use (1, 2, or 3). See p1.
    
    @return: The approximate integral of the function over the interval [a, b].

    Note, the "powers" used in Richardson extrapolation (see hw05.m) should be [2, 4, 6, ...] for option 1 and 2. 
    For option 3, the "powers" should be [4, 6, 8, ...].
    """
    # Collect data: approximations with different numbers of subintervals
    data = []
    for k in range(N):
        n_k = 2 ** (k + 1)  # Start with n = 2^1 = 2 subintervals
        T_k = p1(func, a, b, n_k, option)
        data.append(T_k)
    
    # Determine the appropriate powers for Richardson extrapolation
    if option == 1 or option == 2:
        # For midpoint and trapezoidal rules, error is O(h^2), so powers are [2, 4, 6, ...]
        powers = [2 * (i + 1) for i in range(N - 1)]
    elif option == 3:
        # For Simpson's rule, error is O(h^4), so powers are [4, 6, 8, ...]
        powers = [2 * (i + 2) for i in range(N - 1)]
    else:
        raise ValueError("Invalid option value. Must be 1, 2, or 3.")
    
    # Use a copy of data to prevent modification of the original list
    data_copy = data.copy()

    # Apply Richardson extrapolation using your p1() function from HW05
    result = p1_hw05(data_copy, powers)

    return result

# p1 from hw02 to be called in p4
def p1_hw02(f, a, b, epsilon, name, f_prime=None):
    """
    @param f: a function name
    @param a: real number, the left end of the interval
    @param b: real number, the right end of the interval
    @param epsilon: function tolerance
    @param name: the name of the method to use
    @param f_prime: the derivative of the function f (only needed for Newton's method)

    @return: tuple (c, n), 
             c is the root of the function f in the interval [a, b]
             n is the number of iterations
    """
    # Write your code here (bisection code is provided)
    if name=="bisection": # Good
        # bisection method
        n = 0
        c = (a+b)/2
        while abs(f(c))>epsilon:
            n += 1
            if f(a)*f(c)<0:
                b = c
            else:
                a = c
            c = (a+b)/2
        return c,n
    elif name=="newton": # Good
        # Newton's method
        n = 0
        #c = (a+b)/2
        # Trying 1st iteration of false position method for initial guess
        c = (f(b)*a -f(a)*b) / (f(b) - f(a))
        while abs(f(c)) > epsilon:
            c -= f(c) / f_prime(c)
            n += 1
        return c,n
    elif name == "secant": # Good
        # Secant method
        n = 0
        x0, x1 = a, b
        while True:
            n += 1
            x2 = x1  - f(x1)*((x1 - x0) / (f(x1) - f(x0)))
            if abs(f(x2)) < epsilon:
                return x2,n
            x0, x1  = x1, x2

    elif name=="regula_falsi": # Good
        # False position method
        n = 0
        while True:
            n += 1
            c = (f(b)*a -f(a)*b) / (f(b) - f(a))
            if abs(f(c)) < epsilon:
                return c,n  
            if f(c)*f(a) < 0:
                b = c
            else:
                a = c   
    elif name=="steffensen": # Good
        # Steffensen's method
        n = 0
        #c = (a+b) / 2
        # Trying 1st iteration of false position method for initial guess
        c = (f(b)*a -f(a)*b) / (f(b) - f(a))
        while True:
            n += 1
            c -= f(c) / ((f(c + f(c)) - f(c)) / (f(c)))
            if abs(f(c)) < epsilon:
                return c,n
    else:
        print("Invalid name")

# Problem 4
def p4():
    """
    Construct the Gauss quadrature rule using the roots of the Legendre polynomial of degree 6.
     
    To evaluate Legendre polynomial of degree 6, use the helper function legendre_poly_6 defined below.

    @return: A 6x2 numpy matrix containing the roots and weights of the Gauss quadrature rule. 
             The first column contains the roots and the second column contains the corresponding weights.
    """

    roots = np.zeros(6)
    weights = np.zeros(6)

    # Define intervals for positive roots
    intervals = [ (0.0, 0.5), (0.5, 0.8), (0.8, 1.0) ]

    epsilon = 1e-14

    for i in range(3):
        # Positive root
        a, b = intervals[i]
        c, n = p1_hw02(legendre_poly_6, a, b, epsilon, 'bisection')
        # Store positive root
        roots[3 + i] = c
        # Store negative root
        roots[2 - i] = -c  # symmetric about zero

    # Now compute weights
    for i in range(6):
        x_i = roots[i]
        P_prime = deriv_legendre_poly(6, x_i)
        weights[i] = 2 / ((1 - x_i**2) * (P_prime**2))

    ret = np.column_stack((roots, weights))

    return ret

# Problem 5
def p5(n):
    """
    For 6630 ONLY. 

    Construct the Gauss quadrature rule using the roots of the Legendre polynomial of degree n
    
    @param n: The degree of the Legendre polynomial for the nodes of the Gauss quadrature rule.
    @return: An nx2 numpy matrix containing the roots and weights of the Gauss quadrature rule.
    
    To evaluate Legendre polynomial or its derivative of a specific degree n, use the following two functions.
                  
    legendre_poly_n = lambda x: legendre_poly(n, x)
    deriv_legendre_poly_n = lambda x: deriv_legendre_poly(n, x)
    
    """
    roots = np.zeros(n)
    weights = np.zeros(n)

    # your code here.


    ret = np.column_stack((roots, weights))
    return ret

###############################################################################
#                                                                             #
# Helper functions for Problem 4 and 5, do not modify these functions         #
#                                                                             #
###############################################################################

# Helper function for p4
def legendre_poly_6(x):
    """
    Evaluate the Legendre polynomial of degree 6 at x.
    
    @param x: The value at which to evaluate the Legendre polynomial.
    @return: The value of the Legendre polynomial of degree 6 at x.
    """
    return (231*x**6 - 315*x**4 + 105*x**2 - 5)/16

# Helper functions for p5
def legendre_poly(n, x):
    """
    Evaluate the Legendre polynomial of degree n at x.
    
    @param n: The degree of the Legendre polynomial to evaluate.
    @param x: The value at which to evaluate the Legendre polynomial.
    @return: The value of the Legendre polynomial of degree n at x.
    """
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return (2*n - 1)/n*x*legendre_poly(n-1, x) - (n - 1)/n*legendre_poly(n-2, x)

# Helper functions for p5
def deriv_legendre_poly(n, x):
    """
    Evaluate the derivative of the Legendre polynomial of degree n at x.
    
    @param n: The degree of the Legendre polynomial whose derivative to evaluate.
    @param x: The value at which to evaluate the derivative of the Legendre polynomial.
    @return: The value of the derivative of the Legendre polynomial of degree n at x.
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return n/(x**2 - 1)*(x*legendre_poly(n, x) - legendre_poly(n-1, x))
    

