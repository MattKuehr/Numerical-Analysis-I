# install pytest first: pip install pytest
# run with the following command: pytest hw_test.py

from math import sqrt

from hw02 import *


def test_p1():
    """
    test p1
    """
    assert(abs( p1(t1_func,1, 2, 1e-9, 'bisection')[0] -  sqrt(2)) < 1e-6) 
    assert(abs( p1(t1_func,1, 2, 1e-9, 'secant')[0] -  sqrt(2)) < 1e-6) 
    assert(abs( p1(t1_func,1, 2, 1e-9, 'newton', t1_func_prime)[0] -  sqrt(2)) < 1e-6) 
    assert(abs( p1(t1_func,1, 2, 1e-9, 'regula_falsi')[0] -  sqrt(2)) < 1e-6)
    assert(abs( p1(t1_func,1, 2, 1e-9, 'steffensen')[0] -  sqrt(2)) < 1e-6)

def test_p2():
    """
    test p2
    """
    x = 1.21
    assert(abs( p1(t2_func,1, 3, 1e-9, 'bisection')[0] - x) < 1e-6) 
    assert(abs( p1(t2_func,1, 3, 1e-9, 'secant')[0] - x) < 1e-6) 
    assert(abs( p1(t2_func,1, 3, 1e-9, 'newton', t2_func_prime)[0] -  x) < 1e-6) 
    assert(abs( p1(t2_func,1, 3, 1e-9, 'regula_falsi')[0] - x) < 1e-6)
    assert(abs( p1(t2_func,1, 3, 1e-9, 'steffensen')[0] - x) < 1e-6)


def t1_func(x):
    """
    test function for p1
    """
    return x**2 - 2
def t1_func_prime(x):
    """
    test function prime for p1
    """
    return 2*x

def t2_func(x):
    """
    test function for p2
    """
    return sqrt(x) - 1.1

def t2_func_prime(x):
    """
    test function prime for p2
    """
    return 0.5/sqrt(x)