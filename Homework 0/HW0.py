# Author: Matt Kuehr / mck0063@auburn.edu
# Date: 2024-09-03
# Assignment Name: hw00

import numpy as np
import time 

# The following class defines 3 functions for each problem respectively.
# Please follow the instruction inside each function. 

def p1(m):
    """
    This function takes an integer m and returns the term a_m in the sequence defined by 
    a_0 = 0, a_1 = 1, a_2 = 1, and a_n = a_{n-1} + a_{n-2} + a_{n-3} for n >= 3.
    :param m: an integer
    :return: the m-th term in the sequence
    """
    if m < 0:
        return None
    else:
        if m == 0:
            return 0
        elif m == 1 or m == 2:
            return 1
        else:
            # Temp variables to store previous values
            temp1 = 0
            temp2 = 1
            temp3 = 1

            # Iterate through sequence until mth term, remembering
            # previous 3 terms for future computation.
            for num in range(3, m+1):
                num = temp3 + temp2 + temp1
                temp1 = temp2
                temp2 = temp3
                temp3 = num
            
            # Return mth term in sequence 'num'
            return num


def p2(A):
    """
    This function takes a numpy matrix A of size n x n and returns the determinant of A.
    :param A: a numpy matrix of size n x n
    :return: the determinant of A
    """
    if A.shape[0] != A.shape[1]:
        return None
    else:
        
        # Identifying size of matrix
        n = A.shape[0]
        
        # Handling edge cases
        if n == 1:
            return A[0, 0]
        elif n == 2:
            return (A[0,0] *  A[1,1]) - (A[0, 1] * A[1,0])
        
        # Case when n >= 3
        else:

            # To handle cases where n >= 3, we will cofactor expand along the first row
            # I utilized recursion in the case that n is still greater than 2
            det = 0
        
            for j in range(n):
                
                # Calculate the minor of A by removing the first row and j-th column
                minor = np.delete(np.delete(A, 0, axis=0), j, axis=1)
            
                # Calculate the cofactor and apply sign by index
                cofactor = ((-1) ** j) * A[0, j] * p2(minor)
                det += cofactor

        # Return determinant after cofactor expansion
        return det
            

def p3():
    """
    This function should have a run time about 1 second.
    :return: no returns
    """
    
    duration = 1.0 
    start = time.time()

    # Run empty loop until duration has been exceeded
    while time.time() - start < duration:
        pass
