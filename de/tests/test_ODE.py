'''
Test the ODE model.
'''

import pytest
import numpy as np
from ..ODE import julia_solver, julia_sol_matrix


@pytest.mark.xfail(reason="Issue in NNlib.")
def test_ODErun():
    """ Test that we get a reasonable model output. """
    p = np.ones(7055) * 0.1
    output = julia_solver(p)

    assert output.shape == (83, )
    assert np.all(np.isfinite(output))


@pytest.mark.xfail(reason="Issue in NNlib.")
def test_ODErunMatrix():
    """ Test that we get a reasonable model output with knockdowns. """
    p = np.ones(7055) * 0.1
    output = julia_sol_matrix(p)

    assert output.shape == (83, 84)
    assert np.all(np.isfinite(output))
