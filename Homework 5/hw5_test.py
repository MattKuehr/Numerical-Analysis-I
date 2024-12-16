# install pytest first: pip install pytest
# run with the following command: pytest hw_test.py

import numpy as np
from hw05 import *

def test_p1():
    """
    test p1
    """
    assert(abs(p1(  [2, 1],    [1]              )         ) < 1e-8)
    assert(abs(p1(  [4, 2, 1], [1,2]            )         ) < 1e-8)
    assert(abs(p1(  [1, 2, 4], [1,2]            ) - 7.0   ) < 1e-8)
    assert(abs(p1(  [5, 4, 3], [1,3]            ) - 13/7  ) < 1e-8)
    assert(abs(p1(  [5, 4, 3, 2, 1], [1,2,3,4]  ) + 19/35 ) < 1e-8)
    assert(abs(p1(  [64, 8, 4, 1],   [1,2,3]    ) + 16/3  ) < 1e-8)


def test_p2():
    """
    test p2
    """
    assert(abs ( p2(1)   - 0.7853981633974483) < 1e-6)
    assert(abs ( p2(0.8) - 0.7436078366584389) < 1e-6)
    assert(abs ( p2(0.5) - 0.6676914571896092) < 1e-6)
    assert(abs ( p2(0.2) - 0.5737108471859466) < 1e-6)

def test_p3():
    """
    test p3
    """
    assert(abs ( p3([-1, 0, 1]) - np.array([-0.5, 0, 0.5])) < 1e-8).all()
    assert(abs ( p3([-2, -1, 1]) - np.array([0, -0.5, 0.5])) < 1e-8).all()
    assert(abs ( p3([-2, -3, 6]) - np.array([3/8, -4/9, 5/72])) < 1e-8).all()  