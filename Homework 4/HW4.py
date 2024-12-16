# Author: Matt Kuehr / mck0063@auburn.edu
# Date: 2024-09-01
# Assignment Name: hw04


import numpy as np

def p1(data, eval_pts):
    """
    Implement the divided difference method to interpolate the data points, 
    then evaluate the polynomial at the given point.

    @param data: a list of tuples [(x0, y0), (x1, y1), ..., (xn, yn)]
    @param eval_pts: a list of x values to evaluate the interpolating polynomial

    @return: a list of y values evaluated at the eval_pts

    """
    y = np.zeros(len(eval_pts))

    # Write your code here.

    # Unpack data points
    x_vals, y_vals = zip(*data)

    n = len(x_vals)
    
    # Make nxn array to store divided difference values
    dd_table = np.zeros((n,n))
    dd_table[:,0] = y_vals

    # Fill in divided difference table
    for j in range(1, n): 
        for i in range(n-j):

            # Form numerator and denominator for coefficient
            numerator = dd_table[i+1][j-1] - dd_table[i][j-1]
            denominator = x_vals[i+j] - x_vals[i]
            
            # Enter divided difference value into table
            dd_table[i][j] = numerator / denominator


    # Collect top row coefficients for building polynomial
    coeffs = dd_table[0]

    # Helper function to build the Newton polynomial
    def build_polynomial(coeffs, x_vals, x):

        # Initialize polynomial
        polynomial = coeffs[0]

        # Initialize product
        product = 1.0

        # Generate polynomial terms of deg > 0
        for i in range (n-1):
            product *= (x - x_vals[i])
            polynomial += coeffs[i+1]*product 

        # Return fully constructed polynomial
        return polynomial
    
    # Evaluate polynomial at the evaluation points
    for k in range(len(eval_pts)):
        y[k] = build_polynomial(coeffs, x_vals, eval_pts[k])

    return y


def p2(data, eval_pts):
    """
    For 6630 ONLY

    Implement the divided difference method to interpolate the data points, 
    then evaluate the polynomial at the given point.

    @param data: a list of tuples [(x0, y0, y0_1, y0_2, ..., y0_m1), 
                                   (x1, y1, y1_1, y1_2, ..., y1_m2),
                                    ..., 
                                   (xn, yn, yn_1, yn_2, ..., yn_mn)] 

                where x0, x1, ..., xn are the x values and the subsequent 
                values in the tuple are the derivatives of the function at the x values. 

                For example, 

                y0 = f(x0),
                y0_1 = f'(x0),
                y0_2 = f''(x0),
                ... ,
                y0_m1 = f^(m1)(x0)

    @param eval_pts: a list of x values to evaluate the interpolating polynomial

    @return: a list of y values evaluated at the eval_pts
    """
    y = np.zeros(len(eval_pts))

    # Write your code here.

    return y
