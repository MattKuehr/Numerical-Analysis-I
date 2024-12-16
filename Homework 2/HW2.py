# Author: Matt Kuehr / mck0063@auburn.edu
# Date: 2024-09-01
# Assignment Name: hw02

import sys
import numpy as np

def p1(f, a, b, epsilon, name, f_prime=None):
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
        
def p2():
    """
    summarize the iteration number for each method name in the table

    Note: For Newton and Steffensen methods, I chose the first iteration of 
    Regula Falsi as my initial guess, which led to results aligning with the
    known convergence behavior of these methods.

    |name          | iter | 
    |--------------|------|
    |bisection     |32    |
    |secant        |8     |
    |newton        |5     |
    |regula_falsi  |60    |
    |steffensen    |10    |
    """


def p3(f, a, b , epsilon):
    """
    For 6630 students only.

    Implement the Illinois algorithm to find the root of the function f in the interval [a, b]

    @param f: a function name
    @param a: real number, the left end of the interval
    @param b: real number, the right end of the interval
    @param epsilon: function tolerance

    @return: tuple (c, n), 
             c is the root of the function f in the interval [a, b]
             n is the number of iterations
    """
    # Write your code here
    pass

def p4(f, a, b , epsilon):
    """
    For 6630 students only.

    Implement the Pegasus algorithm to find the root of the function f in the interval [a, b]

    @param f: a function name
    @param a: real number, the left end of the interval
    @param b: real number, the right end of the interval
    @param epsilon: function tolerance
    
    @return: tuple (c, n), 
             c is the root of the function f in the interval [a, b]
             n is the number of iterations
    """
    # Write your code here
    pass

