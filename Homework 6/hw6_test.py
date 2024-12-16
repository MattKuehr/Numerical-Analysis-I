# install pytest first: pip install pytest
# run with the following command: pytest hw_test.py

import numpy as np
from hw06 import *

def test_p1_1():
    """
    test p1 option 1
    """
    assert np.isclose(p1(f1, 0, 1, 1, 1), 0.25, atol=1e-8)
    assert np.isclose(p1(f1, 0, 1, 4, 1), 0.328125, atol=1e-8)
    assert np.isclose(p1(f2, 0, 1, 1, 1), 1/8, atol=1e-8)
    assert np.isclose(p1(f2, 0, 1, 4, 1), 0.2421875, atol=1e-8)

    assert np.isclose(p1(f3, 0, 1, 1, 1), 1/16, atol=1e-8)
    assert np.isclose(p1(f3, 0, 1, 4, 1), 0.189697265625, atol=1e-8)
    assert np.isclose(p1(f4, 0, 1, 1, 1), 1/32, atol=1e-8)
    assert np.isclose(p1(f4, 0, 1, 4, 1), 0.15393066406250, atol=1e-8)

def test_p1_2():
    """
    test p1 option 2
    """
    assert np.isclose(p1(f1, 0, 1, 1, 2), 0.5, atol=1e-8)
    assert np.isclose(p1(f1, 0, 1, 4, 2), 0.34375, atol=1e-8)
    assert np.isclose(p1(f2, 0, 1, 1, 2), 1/2, atol=1e-8)
    assert np.isclose(p1(f2, 0, 1, 4, 2), 0.265625, atol=1e-8)

    assert np.isclose(p1(f3, 0, 1, 1, 2), 1/2, atol=1e-8)
    assert np.isclose(p1(f3, 0, 1, 4, 2), 0.220703125, atol=1e-8)
    assert np.isclose(p1(f4, 0, 1, 1, 2), 1/2, atol=1e-8)
    assert np.isclose(p1(f4, 0, 1, 4, 2), 0.1923828125, atol=1e-8)


def test_p1_3():
    """
    test p1 option 3
    """
    assert np.isclose(p1(f1, 0, 1, 16, 3), 1/3, atol=1e-4)
    assert np.isclose(p1(f2, 0, 1, 16, 3), 1/4, atol=1e-4)
    assert np.isclose(p1(f3, 0, 1, 16, 3), 1/5, atol=1e-4)
    assert np.isclose(p1(f4, 0, 1, 16, 3), 1/6, atol=1e-4)
    assert np.isclose(p1(f5, 0, 1, 16, 3), 1/7, atol=1e-4)
    assert np.isclose(p1(f6, 0, 1, 16, 3), (1 - np.cos(1)), atol=1e-4)
    assert np.isclose(p1(f7, 0, 1, 16, 3), (np.exp(1) - 1), atol=1e-4)
    assert np.isclose(p1(f8, 0, 1, 16, 3), np.pi/4, atol=1e-4)



def test_p3_1():
    """
    test p3 option 1
    """
    assert np.isclose(p3(f1, 0, 1, 4, 1), 1/3, atol=1e-8)
    assert np.isclose(p3(f2, 0, 1, 4, 1), 1/4, atol=1e-8)
    assert np.isclose(p3(f3, 0, 1, 4, 1), 1/5, atol=1e-8)
    assert np.isclose(p3(f4, 0, 1, 4, 1), 1/6, atol=1e-8)
    assert np.isclose(p3(f5, 0, 1, 4, 1), 1/7, atol=1e-8)
    assert np.isclose(p3(f6, 0, 1, 4, 1), (1 - np.cos(1)), atol=1e-8)
    assert np.isclose(p3(f7, 0, 1, 4, 1), (np.exp(1) - 1), atol=1e-8)
    assert np.isclose(p3(f8, 0, 1, 4, 1), np.pi/4, atol=1e-8)

def test_p3_2():
    """
    test p3 option 2
    """
    assert np.isclose(p3(f1, 0, 1, 4, 2), 1/3, atol=1e-8)
    assert np.isclose(p3(f2, 0, 1, 4, 2), 1/4, atol=1e-8)
    assert np.isclose(p3(f3, 0, 1, 4, 2), 1/5, atol=1e-8)
    assert np.isclose(p3(f4, 0, 1, 4, 2), 1/6, atol=1e-8)
    assert np.isclose(p3(f5, 0, 1, 4, 2), 1/7, atol=1e-8)
    assert np.isclose(p3(f6, 0, 1, 4, 2), (1 - np.cos(1)), atol=1e-8)
    assert np.isclose(p3(f7, 0, 1, 4, 2), (np.exp(1) - 1), atol=1e-8)
    assert np.isclose(p3(f8, 0, 1, 4, 2), np.pi/4, atol=1e-8)    

def test_p3_3():
    """
    test p3 option 3
    """
    assert np.isclose(p3(f1, 0, 1, 4, 3), 1/3, atol=1e-8)
    assert np.isclose(p3(f2, 0, 1, 4, 3), 1/4, atol=1e-8)
    assert np.isclose(p3(f3, 0, 1, 4, 3), 1/5, atol=1e-8)
    assert np.isclose(p3(f4, 0, 1, 4, 3), 1/6, atol=1e-8)
    assert np.isclose(p3(f5, 0, 1, 4, 3), 1/7, atol=1e-8)
    assert np.isclose(p3(f6, 0, 1, 4, 3), (1 - np.cos(1)), atol=1e-8)
    assert np.isclose(p3(f7, 0, 1, 4, 3), (np.exp(1) - 1), atol=1e-8)
    assert np.isclose(p3(f8, 0, 1, 4, 3), np.pi/4, atol=1e-8)

def test_p4():
    """
    test p4
    """
    assert run_p4(p4())

f1 = lambda x: x**2
f2 = lambda x: x**3
f3 = lambda x: x**4
f4 = lambda x: x**5
f5 = lambda x: x**6
f6 = lambda x: np.sin(x)
f7 = lambda x: np.exp(x)
f8 = lambda x: 1 / (1 + x**2)

def run_p4(xw):
    """
    run test for p4
    """
    x = xw[:, 0]
    w = xw[:, 1]
    ret = True
    for i in range(0, 11, 2):
        ret = ret and abs(np.dot(x ** i, w) - 2/(i+1)) < 1e-12
        ret = ret and abs(np.dot(x** (i+1), w)) < 1e-12
    return ret
