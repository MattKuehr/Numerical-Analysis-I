# Author: Matt Kuehr / mck0063@auburn.edu
# Date: 2024-09-01
# Assignment Name: hw07


import time
import numpy as np
import matplotlib.pyplot as plt

def p1(func, y0, tspan, n_steps, method):
    """
    # Implement the function to solve the ODE y' = f(t, y) 
    # using the specified method (euler, midpoint, rk4)
    # 
    # @param func: The function f(t, y).
    # @param y0: The initial condition y(0).
    # @param tspan: The time span [t0, tf].
    # @param n_steps: The number of time steps to take.
    # @param method: The method to use to solve the ODE.
    # @return: The solution array to the ODE at each time step.
    """

    # YOUR CODE HERE. Euler method is implemented for an example. Implement the other methods.

    h = (tspan[1] - tspan[0]) / n_steps
    t = np.linspace(tspan[0], tspan[1], n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    if method == 'euler':
        for i in range(n_steps):
            y[i + 1] = y[i] + h * func(t[i], y[i])
    elif method == 'midpoint':
        for i in range(n_steps):
            k1 = func(t[i], y[i])
            t_mid = t[i] + h / 2
            y_mid = y[i] + (h / 2) * k1
            y[i + 1] = y[i] + h * func(t_mid, y_mid)
    elif method == 'rk4':
        for i in range(n_steps):
            k1 = func(t[i], y[i])
            k2 = func(t[i] + h / 2, y[i] + (h / 2) * k1)
            k3 = func(t[i] + h / 2, y[i] + (h / 2) * k2)
            k4 = func(t[i] + h, y[i] + h * k3)
            y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    else:
        raise ValueError("Invalid method. Choose 'euler', 'midpoint', or 'rk4'.")

    return y

def p2(method): # Introduced 'method' argument to select which ODE solver to test
    """
    # Test the implemented methods on the ODE
    # y' = t(y - t sin(t)) with initial condition y(0) = 1 over the interval
    #  [0, 1], [0, 3], and [0, 5] with variable step lengths. 
    # 
    # Plot the solution y(t) over the interval for various step sizes h. And plot the 
    # exact solution y(t) = t sin(t) + cos(t) over the same interval.
    #
    # Use the commands below to test the implemented methods:
    #
    #> p2('euler');
    #> p2('midpoint');
    #> p2('rk4');
    #
    # Observe the solution and error plots for the numerical solutions with different step sizes.
    # Write your observations in the comments. 

    # Your comment here (e.g, how does the error change with step size and the time span, etc.): 
    # 
    # For regular Euler on [0,1] the method fits the general trajectory of the curve for all time steps but fails to accurately capture the 
    # vertical displacement. For the smallest step size, the deviation in displacement is the smallest and for the largest step size the 
    # deviation is the largest. This matches what we would expect, as decreasing step size should increase accuracy. The error sees an order
    # of convergence of roughly O(h^1), and as step size increases from 1e-1 to 4e-2, the error decreases until the order of convergence 
    # becomes exactly O(h^1), matching the expected theoretical convergence of the method. For regular Euler of [0,3] we see that the method
    # fits the exact solution fairly well on the interval, with the greatest divergence occuring from t=0.5 to t=2.25. As before, the smallest
    # step size diverged from the exact solution the least, while the largest step size diverged from the exact solution the most. The error 
    # remained steadily around O(h^1), regardless of the stepsize selected, matching the theoretical convergence of the method and reflecting
    # the decent accuracy visually observed in the plot. Finally, for regular Euler on [0,5] the method fits the exact solution almost exactly
    # until t=3.5 when the methods begin to diverge. As has been consistently observed, the method with the smallest step size diverged the least
    # whereas the largest step size diverged the most. The error of convergence consistently remained less than O(h^1), and increasing step size
    # from 1e-1 to 4e-2 only marginally decreased the error. This is consistent with the results observed in the plot, that each method experienced
    # significant divergence from the exact solution after t=3.5 and unperformed its theoretical convergence rate of O(h^1). This behavior can be
    # attributed to the lower order of convergence expected of this method, where the accumulation of errors affects the approximate solution 
    # earlier and more drastically than in the other higher order methods. Due to the theoretical order of convergence of this method being
    # O(h^1), it is only reasonable to expect an accurate approximate solution for a small interval around the initial condition, before errors
    # accumulate and the approximate solution diverges from the exact solution.
    # 
    # For midpoint Euler on [0,1] the method fit the exact solution nearly perfectly. As step size decreased, the order of convergence sat
    # between 1st and second order. As step size decreased from 1e-1 to 4e-2 the order of convergence grew closer to O(h^2) than O(h^1), which
    # we would expect with smaller timesteps and matches with the expected second order convergence we would expect from this method. For 
    # midpoint Euler on [0,3] the method fit the exact solution nearly perfectly until around t=2.75, where the various midpoint methods began
    # to diverge. As we would expect, the smaller time step methods diverged the least, due to their greater accuracy. The error remained roughly
    # O(h^2) for all timesteps ranging from 1e-1 to 4e-2, matching the theoretical convergence expected of the midpoint Euler method. Finally for
    # midpoint Euler on [0,5] the method fit the exact solution nearly perfectly until around t = 4.0, where the methods began to diverge. As 
    # per usual, the smallest timestep diverges the least and the largest time step by the most, but all methods diverged drastically beyond this 
    # timestep. This is most likely due to the  lower order of conergence attributed to the midpoint Euler method, O(h^2), when compared to
    # higher order methods like rk4. Similarly, small errors may accumulate over a sufficiently large number of timesteps, leading to good initial
    # results which gradually worsen until the accumulation of numerical errors causes serious divergence. The order of accuracy for the methods sat
    # between O(h^1) and O(h^2) and increasing step size between 1e-1 and 4e-2 did not significantly help convergence order. This makes sense because
    # after t=4.0 all methods experienced significant divergence, suggesting they were not matching their theoretically expected error of O(h^2).
    #  
    # For rk4 on [0,1] rk4 fit the exact solution nearly perfectly. As step size decreased the order of convergence increased significantly.
    # Starting from a step size of 1e-1 and continuing to 4e-2 the error remained between 3rd order and 4th order convergence the entire time, 
    # however as the step size decreased, the error became exactly O(h^4) as we could expect theoretically. On [0,3] rk4 again fit the exact
    # solution nearly perfectly. Similarly, as step size ranged from 1e-1 to 4e-2, the error maintained convergence of roughly O(h^4), and changes
    # in step size did not seem to drastically affect the convergence order. Finally, on [0,5] rk4 fits the exact solution nearly perfectly until
    # around t=4.5, where the various rk4 methods begin to diverge from the exact solution. There, we can clearly observe that the larger timesteps
    # diverge further from the exact solution than the smaller timesteps, agreeing with the convention that a higher degree of accuracy is achieved
    # with smaller timesteps. For all step sizes, the error is almost exactly O(h^4), which matches the theoretical convergence rate we would expect
    # as well as the empircal results we observed in the plot.
    """

    def f(t, y):
        return t * (y - t * np.sin(t))

    tf_values = [1, 3, 5]
    _, axs = plt.subplots(2, len(tf_values), figsize=(15, 10))

    for tf in tf_values:
        tspan = [0, tf]
        y0 = 1
        hs = np.array( [1e-1, 1e-1 * 10/11, 1e-1*4/5, 1e-1*0.72,
                         1e-1*0.64, 1e-1*1/2, 1e-1*2/5, 1e-1*0.32] )
        n_steps = np.array(  [int((tf - tspan[0]) / h) for h in hs] )
        error = np.zeros(len(n_steps))

        for i, n in enumerate(n_steps):
            t = np.linspace(tspan[0], tspan[1], n + 1)
            y_exact = t * np.sin(t) + np.cos(t)
            y_num = p1(f, y0, tspan, n, method) # Replaced 'rk4' with a general parameter
            error[i] = np.max(np.abs(y_num - y_exact))

            axs[0, tf_values.index(tf)].plot(t, y_num, label=f'h = {hs[i]:.2e}')

        axs[1, tf_values.index(tf)].loglog(hs, error, 'g-d', label='error vs. n_steps')
        axs[1, tf_values.index(tf)].loglog(hs, error[0] * hs **4/ hs[0] ** 4, 'ro--',
                            linewidth=1, markersize=2, label='4th order convergence')
        axs[1, tf_values.index(tf)].loglog(hs, error[0] * hs **3/ hs[0] ** 3, 'bo--',
                            linewidth=1, markersize=2,  label='3rd order convergence')
        axs[1, tf_values.index(tf)].loglog(hs, error[0] * hs **2/ hs[0] ** 2, 'mo--',
                            linewidth=1,  markersize=2, label='2nd order convergence')
        axs[1, tf_values.index(tf)].loglog(hs, error[0] * hs **1/ hs[0] ** 1, 'ko--',
                            linewidth=1,  markersize=2,  label='1st order convergence')
        axs[0, tf_values.index(tf)].plot(t, y_exact, '--', label='Exact Solution', linewidth=2)
        axs[0, tf_values.index(tf)].set_title(f'numerical and exact solutions on [0, {tf}]')
        axs[0, tf_values.index(tf)].legend()
        axs[0, tf_values.index(tf)].set_xlabel('t')
        axs[0, tf_values.index(tf)].set_ylabel('y')

        axs[1, tf_values.index(tf)].set_title(f'max error vs step sizes for solution on [0, {tf}]')
        axs[1, tf_values.index(tf)].legend()
        axs[1, tf_values.index(tf)].set_xlabel('h')
        axs[1, tf_values.index(tf)].set_ylabel('Error')

    plt.tight_layout()
    plt.show()

def p3():
    """
    # For 6630 ONLY
    # First implement the 3/8 rule for Runge Kutta method.
    # 
    # The implementation should be done in the function rk4_38_rule below. 
    # It is a subfunction which can only be called within p3 method.
    #
    # Then run p3() and compare the results with the 4th order Runge Kutta method. 
    # 
    # Write your observations in the comments. 
    #
    # Your comment here (e.g, how does the error change with step size and the time span, 
    # is there a clear difference in the running time and error
    #  (you may need to run a few times to conclude), etc.): 
    # 
    #
    #
    #
    #
    """
    def rk4_38_rule(func, y0, tspan, n_steps):
        h = (tspan[1] - tspan[0]) / n_steps
        t = np.linspace(tspan[0], tspan[1], n_steps + 1)
        y = np.zeros(n_steps + 1)
        y[0] = y0

        # your code here.

        return y

    def f(t, y):
        return t * (y - t * np.sin(t))

    tf_values = [3, 5, 7]
    _, axs = plt.subplots(2, len(tf_values), figsize=(15, 10))

    for tf in tf_values:
        t0 = 0
        y0 = 1
        hs = 0.1 / 2 ** np.array([1, 2, 3, 4, 5, 6, 7, 8])
        error_rk4 = np.zeros(len(hs))
        error_rk4_38 = np.zeros(len(hs))
        runtime_rk4 = np.zeros(len(hs))
        runtime_rk4_38 = np.zeros(len(hs))

        for i, h in enumerate(hs):
            n_steps = int((tf - t0) / h)
            t = np.linspace(t0, tf, n_steps + 1)
            y_exact = t * np.sin(t) + np.cos(t)

            time_start = time.time()

            for _ in range(20):
                y_rk4 = p1(f, y0, [t0, tf], n_steps, 'rk4')
            time_end = time.time()
            runtime_rk4[i] = (time_end - time_start) / 20

            time_start = time.time()
            for _ in range(20):
                y_rk4_38 = rk4_38_rule(f, y0, [t0, tf], n_steps)
            time_end = time.time()

            runtime_rk4_38[i] = (time_end - time_start) / 20

            error_rk4[i] = np.max(np.abs(y_exact - y_rk4))
            error_rk4_38[i] = np.max(np.abs(y_exact - y_rk4_38))

        axs[0, tf_values.index(tf)].loglog(hs, error_rk4, 'b-d',
                                label='Max error vs step size (RK4)',
                                linewidth=1, markersize=5)
        axs[0, tf_values.index(tf)].loglog(hs, error_rk4_38, 'g-o',
                                label='Max error vs step size (3/8 Rule)',
                                linewidth=1, markersize=5)
        axs[0, tf_values.index(tf)].set_title(f'Max error vs step size on [0, {tf}]')
        axs[0, tf_values.index(tf)].legend()
        axs[0, tf_values.index(tf)].set_xlabel('h')
        axs[0, tf_values.index(tf)].set_ylabel('Error')

        axs[1, tf_values.index(tf)].loglog(hs, runtime_rk4, 'b-d',
                                label='run time of RK4', linewidth=1, markersize=5)
        axs[1, tf_values.index(tf)].loglog(hs, runtime_rk4_38, 'g-o',
                                label='run time of 3/8 Rule', linewidth=1, markersize=5)

        axs[1, tf_values.index(tf)].set_title(f'Runtime vs. step size on [0, {tf}]')
        axs[1, tf_values.index(tf)].legend()
        axs[1, tf_values.index(tf)].set_xlabel('h')
        axs[1, tf_values.index(tf)].set_ylabel('Run time')

    plt.tight_layout()
    plt.show()

