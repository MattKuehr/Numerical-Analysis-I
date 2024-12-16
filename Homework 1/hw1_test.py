# install pytest first: pip install pytest
# run with the following command: pytest hw_test.py

from hw01 import *

def test_p2():
    assert(abs( p2(0, 1)  - 2 * np.sqrt(3) ) < 1e-8) 
    assert(abs( p2(0, 2)  - 2 * np.sqrt(3) ) < 1e-8)
    assert(abs( p2(1, 1)  - 12/np.sqrt(3)/(1 + np.sqrt(4/3)) ) < 1e-8)
    assert(abs( p2(1, 2)  - 12/np.sqrt(3)/(1 + np.sqrt(4/3)) ) < 1e-8)

def test_p3():
    A = np.array([1,2,3])
    assert(p3(A) == 6)
    A = np.array([sys.float_info.epsilon, -sys.float_info.epsilon])
    assert(p3(A) == 0)