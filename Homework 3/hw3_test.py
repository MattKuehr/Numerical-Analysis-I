# install pytest first: pip install pytest
# run with the following command: pytest hw_test.py

import numpy as np
from hw03 import *

def test_p1():
    """
    test p1
    """
    assert(abs(p1([(-1, 1), (1, 1)], [0]) - np.array([1])) < 1e-8)
    assert(abs(p1([(-1, 1), (1, 2)], [0]) - np.array([1.5])) < 1e-8)
    assert(abs(p1([(-1, 1), (1, 2), (3,3)], [0]) - np.array([1.5])) < 1e-8)
    assert(abs(p1([(-1, 1), (1, 2), (3,7)], [0]) - np.array([1])) < 1e-8)
    assert(abs(p1([(-1, 1), (1, 2), (3,7), (4,2)], [0]) + np.array([0.8])) < 1e-8)
    assert(abs(p1([(-1, 1), (1, 2), (3,7), (4,2), (5, 9)], [0]) + np.array([6.25])) < 1e-8)
