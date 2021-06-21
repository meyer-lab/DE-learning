'''
Test the factorization model.
'''
import pytest
import numpy as np
from ..logistic import logisticF, invlogF


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("b", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("c", [0.5, 1.0, 2.0])
def test_factorizeEstimate(a, b, c):
    """ Test that this runs successfully with reasonable input. """
    data = np.random.normal(size=100)

    convD = logisticF([a, b, c], data)
    dataBack = invlogF([a, b, c], convD)

    np.testing.assert_allclose(data, dataBack)
