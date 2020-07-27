'''
Test the ODE model.
'''

import numpy as np
from ..ODE import solver


def test_ODErun():
    """ Test that we get a reasonable model output. """
    p = np.ones(7055) * 0.1
    output = solver(p, [0.0, 1.0])

    assert output.shape == (2, 83)
    assert np.all(np.isfinite(output))
