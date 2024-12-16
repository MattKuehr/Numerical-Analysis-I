# install pytest first: pip install pytest
# run with the following command: pytest hw_test.py

from hw00 import *

def test_p1():
    assert (p1(0) == 0)
    assert (p1(1) == 1)
    assert (p1(2) == 1)
    assert (p1(3) == 2)

def test_p2():
    A = np.array([[1]])
    assert (p2(A) == 1)
    A = np.array([[1, 2], [3, 4]])
    assert (p2(A) == -2)

def test_p3():
    start = time.time()
    p3()
    end = time.time()
    assert (end - start < 1.5)
    assert (end - start > 0.5) 