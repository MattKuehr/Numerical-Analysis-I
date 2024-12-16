# Author: Matt Kuehr / mck0063@auburn.edu
# Date: 2024-10-15
# Assignment Name: hw03

import numpy as np

def p1(data, eval_pts):
    """
    Implement the Lagrange interpolation method, and evaluate the interpolating polynomial 
    at the given points.

    @param data: a list of tuples [(x0, y0), (x1, y1), ..., (xn, yn)]
    @param eval_pts: a list of x values to evaluate the interpolating polynomial

    @return: a list of y values evaluated at the eval_pts
    """
    y = np.zeros(len(eval_pts))

    # Separate data points into x and y coordinates
    x_coords, y_coords = zip(*data)

    # Evaluate lagrange polynomial at each element of eval_pts
    for i, x in enumerate(eval_pts):
        
        # Form the basis polynomials for the particular eval_pt
        for j, xj in enumerate(x_coords):
            
            # Use list comprehension to generate terms of numerator and denominator
            # then evaluate the product of each collection of terms.
            num = np.prod([x - xk for k, xk in enumerate(x_coords) if k != j])
            denom = np.prod([xj - xk for k, xk in enumerate(x_coords) if k != j])

            # Add contribution to lagrange polynomial evaluation at index i
            y[i] += y_coords[j] * (num / denom)

    return y



def p2():
    """
    Use equally spaced nodes and Chebyshev nodes to compute the Lagrange polynomial interpolation of 
    f(x) = 1/(1 + 25x^2) and g(x) = sin(pi * x) on [-1, 1]. 
    This code uses your implementation of p1 to compute the
    interpolation, and record the maximum interpolation error at 1000 equally 
    spaced points in [-1, 1].
    ----------------------------------------------------------------------------------

    First, run this function and tabulate your result in the table below. 
    Then, make a comment/explanation on the trend of the error for **equally spaced nodes** 
    as n increases for each function.

    Write your comments here:
    > For f(x), the error increases montonically with n, using equally spaced nodes. When we increase the 
    > number of nodes, the degree of the lagrange polynomial becomes higher to interpolate properly. In trying 
    > to propely fit the polynomial to the ground truth function values at each node, the interpolating  
    > polynomial oscillates more erratically when its degree increases. As a result, around the (relatively)
    > flat endpoints of f(x), our interpolating polynomial actually oscillates very steeply, inducing a large
    > interpolation error between the ground truth function and interpolating polynomial. This issue only 
    > worsens as n grows larger, as more nodes means a higher degree polynomial, and thus more erratic behavior
    > at the endpoints to induce worse interpolation error. (This is called Runge's phenomenon).
    >
    > For g(x) the error steadily decreases as n goes from 5-20, before reversing and increasing as n goes from
    > 20-55. This is because for small n, the lagrange interpolation is fairly poor, because it doens't have enough
    > nodes to accurately approximate the behavior of g(x). So adding more points can allow for a better fit of g(x).
    > So the error steadily decreases, as we add more equally spaced nodes. However, eventually, Runge's phenomenon,
    > (the oscillatory behavior described in the previous comment block), induces a sizable error which grows with n,
    > eventually leading the accuracy of the interpolating polynomial to decrease and the error to increase.
    > Although the endpoints of g(x) are not as flat as f(x), the oscillations caused by increasing the degree of
    > the lagrange polynomial to fit the larger number of equally spaced nodes still leads to a deterioriation at the 
    > endpoints of the interval. After a certain point (around n=20), the decrease in error caused by adding more
    > nodes is overwhelmed by the increase in error resulting from Runge's phenomenon, thus the overall interpolation
    > error begins to climb instead of continuing to fall.
    
    |n  |                        Function       | Error with equally spaced nodes | Error with Chebyshev nodes  |
    |---|---------------------------------------|---------------------------------|-----------------------------|
    | 5 | 1 / (1 + 25 * x ** 2)                 |       4.3267e-01                |       5.5589e-01            |
    |10 | 1 / (1 + 25 * x ** 2)                 |       1.9156e+00                |       1.0915e-01            |
    |15 | 1 / (1 + 25 * x ** 2)                 |       2.1069e+00                |       8.3094e-02            |
    |20 | 1 / (1 + 25 * x ** 2)                 |       5.9768e+01                |       1.5333e-02            |
    |25 | 1 / (1 + 25 * x ** 2)                 |       7.5764e+01                |       1.1411e-02            |
    |30 | 1 / (1 + 25 * x ** 2)                 |       2.3847e+03                |       2.0613e-03            |
    |35 | 1 / (1 + 25 * x ** 2)                 |       3.1708e+03                |       1.5642e-03            |
    |40 | 1 / (1 + 25 * x ** 2)                 |       1.0438e+05                |       2.8935e-04            |
    |45 | 1 / (1 + 25 * x ** 2)                 |       1.4243e+05                |       2.1440e-04            |
    |50 | 1 / (1 + 25 * x ** 2)                 |       4.8178e+06                |       3.9629e-05            |
    |55 | 1 / (1 + 25 * x ** 2)                 |       6.6475e+06                |       2.9383e-05            |
    |---|---------------------------------------|---------------------------------|-----------------------------|
    | 5 | np.sin(np.pi * x)                     |       2.6754e-02                |       1.3193e-02            |
    |10 | np.sin(np.pi * x)                     |       5.1645e-05                |       6.0348e-06            |
    |15 | np.sin(np.pi * x)                     |       9.2162e-10                |       2.0995e-11            |
    |20 | np.sin(np.pi * x)                     |       8.4855e-13                |       1.4433e-15            |
    |25 | np.sin(np.pi * x)                     |       2.8680e-11                |       1.5543e-15            |
    |30 | np.sin(np.pi * x)                     |       5.4648e-10                |       1.1102e-15            |
    |35 | np.sin(np.pi * x)                     |       1.2912e-08                |       1.4433e-15            |
    |40 | np.sin(np.pi * x)                     |       2.6307e-07                |       1.8874e-15            |
    |45 | np.sin(np.pi * x)                     |       8.7102e-06                |       1.7764e-15            |
    |50 | np.sin(np.pi * x)                     |       2.0284e-04                |       1.4433e-15            |
    |55 | np.sin(np.pi * x)                     |       5.7611e-03                |       2.3315e-15            |
    |---|---------------------------------------|---------------------------------|-----------------------------|
    """

    eval_pts = np.linspace(-1, 1, 1000)
    f = lambda x: 1 / (1 + 25 * x ** 2)
    f.__name__ = '1 / (1 + 25 * x ** 2)'
    g = lambda x: np.sin(np.pi * x)
    g.__name__ = 'np.sin(np.pi * x)'
    funcs = [f, g]

    print('|n  |                        Function       | Error with equally spaced nodes | Error with Chebyshev nodes  |');
    print('|---|---------------------------------------|---------------------------------|-----------------------------|');

    for i in range(2):
        for n in range(5, 60, 5):
            eq_data = [(x, funcs[i](x)) for x in np.linspace(-1, 1, n+1)]
            y = p1(eq_data, eval_pts)
            eq_error = np.max(np.abs(funcs[i](eval_pts) - y))

            cheb_data = [(np.cos((2 * k + 1) * np.pi / (2 * n + 2)), funcs[i](np.cos((2 * k + 1) * np.pi / (2 * n + 2)))) for k in range(n+1)]
            y = p1(cheb_data, eval_pts)
            cheb_error = np.max(np.abs(funcs[i](eval_pts) - y))

            print(f'|{n:2d} | {funcs[i].__name__:30s}        | {eq_error:16.4e}                | {cheb_error: 16.4e}            |')
        print('|---|---------------------------------------|---------------------------------|-----------------------------|');


def p3():
    """
    For 6630 students only.

    Use the extreme Chebyshev nodes to compute the Lagrange polynomial interpolation of 
    f(x) = 1/(1 + 25x^2) and g(x) = sin(pi * x) on [-1, 1]. 
    This code uses your implementation of p1 to compute the
    interpolation, and record the maximum interpolation error at 1000 equally spaced points in [-1, 1].
    ----------------------------------------------------------------------------------

    Run this function and tabulate your result in the table below. 
    Then, make a comment on the performance of the extreme Chebyshev nodes compared to Chebyshev nodes.

    Write your comments here.
    >
    >
    >
    >
    >

    
    |n  |                        Function       | Error with extreme Chebyshev nodes  |
    |---|---------------------------------------|-------------------------------------|
    | 5 | 1 / (1 + 25 * x ** 2)                 |                                     |
    |10 | 1 / (1 + 25 * x ** 2)                 |                                     |
    |15 | 1 / (1 + 25 * x ** 2)                 |                                     |
    |20 | 1 / (1 + 25 * x ** 2)                 |                                     |
    |25 | 1 / (1 + 25 * x ** 2)                 |                                     |
    |30 | 1 / (1 + 25 * x ** 2)                 |                                     |
    |35 | 1 / (1 + 25 * x ** 2)                 |                                     |
    |40 | 1 / (1 + 25 * x ** 2)                 |                                     |
    |45 | 1 / (1 + 25 * x ** 2)                 |                                     |
    |50 | 1 / (1 + 25 * x ** 2)                 |                                     |
    |55 | 1 / (1 + 25 * x ** 2)                 |                                     |
    |---|---------------------------------------|-------------------------------------|
    | 5 | np.sin(np.pi * x)                     |                                     |
    |10 | np.sin(np.pi * x)                     |                                     |
    |15 | np.sin(np.pi * x)                     |                                     |
    |20 | np.sin(np.pi * x)                     |                                     |
    |25 | np.sin(np.pi * x)                     |                                     |
    |30 | np.sin(np.pi * x)                     |                                     |
    |35 | np.sin(np.pi * x)                     |                                     |
    |40 | np.sin(np.pi * x)                     |                                     |
    |45 | np.sin(np.pi * x)                     |                                     |
    |50 | np.sin(np.pi * x)                     |                                     |
    |55 | np.sin(np.pi * x)                     |                                     |
    |---|---------------------------------------|-------------------------------------|
    
    
    """
    eval_pts = np.linspace(-1, 1, 1000)
    f = lambda x: 1 / (1 + 25 * x ** 2)
    f.__name__ = '1 / (1 + 25 * x ** 2)'
    g = lambda x: np.sin(np.pi * x)
    g.__name__ = 'np.sin(np.pi * x)'
    funcs = [f, g]
    print('|n  |                        Function       | Error with extreme Chebyshev nodes  |');
    print('|---|---------------------------------------|-------------------------------------|');
    for i in range(2):
        for n in range(5, 60, 5):
            ex_cheb_data = [(np.cos((k) * np.pi / (n)), funcs[i](np.cos(k * np.pi / (n)))) for k in range(n+1)]
            y = p1(ex_cheb_data, eval_pts)
            cheb_error = np.max(np.abs(funcs[i](eval_pts) - y))
            print(f'|{n:2d} | {funcs[i].__name__:30s}        | {cheb_error: 16.4e}                    |')
        print('|---|---------------------------------------|-------------------------------------|');


