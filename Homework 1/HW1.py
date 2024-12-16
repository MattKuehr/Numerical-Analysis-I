# Author: Matt Kuehr / mck0063@auburn.edu
# Date: 2024-09-17
# Assignment Name: hw01

import sys
import numpy as np

def p1():
    """
    This function only contains comments. Fill the following table. Do not write any code here.

    commands                                      |  results              | explanations
    ----------------------------------------------|-----------------------|----------------
    import sys;sys.float_info.epsilon             |2.220446049250313e-16  |The machine epsilon of python's floating point system; the upper bound on the relative error resultant from rounding.
    import sys;sys.float_info.max                 |1.7976931348623157e+308|The largest positive number that can be represented by python's double precision floating point system.
    import sys;sys.float_info.min                 |2.2250738585072014e-308|The smallest normalized positive number that can be represented by python's double precision floating point system.
    import sys;1 + sys.float_info.epsilon - 1     |2.220446049250313e-16  |The addition of the machine epsilon yields the smallest positive number larger than one, that, upon subtraction by 1 does not round down to zero.
    import sys;1 + sys.float_info.epsilon /2 - 1  |0.0                    |Dividing the machine epsilon unit by 2 rounds it down to zero, thus under double precision floating point rounding, this is computed as 1 + 0 - 1 = 0.
    import sys;sys.float_info.min/1e10            |2.225074e-318          |Dividing the min by a large quantity generates a subnormal number between 0 and the min, that has a leading 0 in the mantissa, leading to reduced precision.
    import sys;sys.float_info.min/1e16            |0.0                    |Dividing the min by 1e16 gives a result so small that even subnormal numbers cannot represent it, leading python's double precision floating point system to round down to 0.
    import sys;sys.float_info.max*10              |inf                    |Multiplying the max by 10 causes overflow; the number is too big to represent in the double precision floating point system. The system represents such a quantity as inf (infinity).
    """

def p2(n, choice):
    """
    This function computes the Archimedes' method for pi.
    @param n: the number of sides of the polygon
    @param choice: 1 or 2, the formula to use
    @return: s_n, the approximation of pi using Archimedes' method.

    
    Tabulate the error of |s_n - pi| for n = 0, 1, 2, ... 15 and choices n = 1, 2
    for both choices of formulas.
    
    n     | choice 1              | choice 2
    ------|-----------------------|-----------------------
    0     |0.32250896154796216    |0.32250896154796216
    1     |0.0737976555836819     |0.07379765558367968
    2     |0.01806728850771666    |0.018067288507708223
    3     |0.004493561541673685   |0.004493561541642599
    4     |0.0011219460557803096  |0.0011219460555764726
    5     |0.00028039639007282346 |0.00028039639003196726
    6     |7.009346527508953e-05  |7.009346705721953e-05
    7     |1.752300972857057e-05  |1.7523014898213063e-05
    8     |4.380733543918325e-06  |4.380731735142973e-06
    9     |1.0952270628195038e-06 |1.0951815614390625e-06
    10    |2.7428384008487683e-07 |2.7379530553872655e-07
    11    |7.203279839274046e-08  |6.844882260992335e-08
    12    |1.8151752101402963e-08 |1.7112206762703863e-08
    13    |3.468890685809356e-08  |4.278053022943595e-09
    14    |1.8151752101402963e-08 |1.0695151431150407e-09
    15    |7.177078202857956e-07  |2.673807841802045e-10
 

    Explanation of the results:
    After n = 0, choice 2 is almost always more accurate than choice 1, except for at n = 6,7. At these iterations, it is 
    likely that choice 1 is more accurate due to the rounding and error accumulation / cancellation for relatively small
    values of n. Overall, these instances are outliers because choice 2 is more numerically stable in general.

    Choice 2 performs better because that version of the algorithm is better optimized for floating point arithmetic. For example,
    choice 1 introduces catastrophic cancellation by subtracting two very close numbers from one another in the numerator. Choice 2,
    on the other hand, avoids subtraction entirely, thus sidestepping this error. Choice 1 also divides the already small numerator
    by a similarly small p_n, which could lead to lost precision from rounding errors. Choice 2, however, divides the small quantity
    p_n by a quantity at least as large as 1, thus dodging the loss of precision caused by dividing very small floating point numbers
    by each other. 

    Observe that in iteration 15, choice 1's error increases drastically, while choice 2's error continues to decrease steadily. This 
    is indicative of the safer and more optimal implementation of the algorithm in choice 2. It is likely that either catastrophic
    cancellation or some division error lead to the massive relative error in iteration 15 for choice 1. As described earlier, choice
    2 is more stable and won't suffer from such occurences.
    """

    # Write your code here

    if choice == 1:
        # Use the 1st formula 
        if n == 0:
            return 6 * (1 / np.sqrt(3))
        else:
            p_n = 1 / np.sqrt(3)
            for _ in range(n):
                p_n = (np.sqrt(1 + p_n**2) - 1) / p_n
            return (2**n)*6*p_n
    else:
        # Use the 2nd formula
        if n == 0:
            return 6 * (1 / np.sqrt(3))
        else:
            p_n = 1 / np.sqrt(3)
            for _ in range(n):
                p_n = p_n  / (1 + np.sqrt(1 + p_n**2))
            return (2**n)*6*p_n

def p3(a):
    """
    This function implements the Kahan summation algorithm. 

    @param a: a 1D numpy array of numbers
    @return: the Kahan sum of the array
    """
    
    # Indices are tracked differently in my implementation than in the algorithm on
    # the homework, but the underlying behavior is still the same

    j = 0
    e_j = 0
    s_j = 0
    
    while j < len(a):
        y_j = a[j] - e_j
        s_j_1 = s_j + y_j
        e_j = (s_j_1 - s_j) - y_j
        s_j = s_j_1
        j += 1

    # Return the accumulated value of s_j across all the iterations
    return s_j

def p4(a):
    """
    This function tests the performance of Kahan summation algorithm 
    against naive summation algorithm.

    @param a: a 1D numpy array of numbers
    @return: no return

    @task: Test this function with a = np.random.rand(n) with various size n multiple times. Summarize your findings below.

    @findings: With smaller arguments of n, 1 <= n <= 6, the kahan sum and naive sum attained the same results. However, if 
    I re-ran the program, the kahan sum would sometimes have a smaller error. In general, there are less operations for smaller
    n, so the accumulated error can sometimes be the same between the two methods. However, as I increased the value of n, the
    difference in error became more apparent, and there were no longer cases when the two errors matched. As I set n to be 
    larger and larger (7 <= n <= 250), the error from the naive sum was always several times larger than the kahan sum. This is 
    because the kahan summation mitigates accumulated rounding errors, while the naive sum does not. As a result, the aggregate 
    error in the naive sum becomes much larger than that of the kahan sum, as a larger n means more operations and therefore a 
    larger set of operations to induce error.
    """

    single_a = a.astype(np.float32) # Convert the input array to single precision
    s = p3(a) # Kahan sum of double precision as the ground truth
    single_kahan_s = p3(single_a) # Kahan sum of single precision
    single_naive_s = sum(single_a) # Naive sum of single precision

    print(f"Error of Kahan sum under single precision: {s - single_kahan_s}")
    print(f"Error of Naive sum under single precision: {s - single_naive_s}")

def p5(a):
    """
    For 6630. 

    This function computes summation of a vector using pairwise summation.
    @param a: a vector of numbers
    @return: the summation of the vector a using pairwise summation algorithm.

    @note: You may need to create a helper function if your code uses recursion.

    @task: Rewrite the p4 test function to test this summation method. Summarize your findings below.
    
    @findings: 
    
    
    
    
    
    """

    return 0 # Write your code here.

