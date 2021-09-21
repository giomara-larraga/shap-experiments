from shapley_values.utilities import Normalizer
import numpy as np
import numpy.testing as npt

def test_scale():
    low_levels = np.array([-1, -1.5, 2])
    high_levels = np.array([2, -0.25, 10])

    normalizer = Normalizer(low_levels, high_levels)

    low = np.array([-1, -1.5, 2])
    low_scaled = normalizer.scale(low)

    middle = np.array([0.5, -0.875, 6])
    middle_scaled = normalizer.scale(middle)

    high = np.array([2, -0.25, 10])
    high_scaled = normalizer.scale(high)

    npt.assert_almost_equal(low_scaled, np.atleast_2d([0, 0, 0]))
    npt.assert_almost_equal(middle_scaled, np.atleast_2d([0.5, 0.5, 0.5]))
    npt.assert_almost_equal(high_scaled, np.atleast_2d([1, 1, 1]))

    low_inverse = normalizer.inverse_scale(low_scaled)
    middle_inverse = normalizer.inverse_scale(middle_scaled)
    high_inverse = normalizer.inverse_scale(high_scaled)

    npt.assert_almost_equal(low_inverse, np.atleast_2d(low))
    npt.assert_almost_equal(middle_inverse, np.atleast_2d(middle))
    npt.assert_almost_equal(high_inverse, np.atleast_2d(high))

    matrix = np.stack((low, middle, high))
    matrix_scaled = normalizer.scale(matrix)

    npt.assert_almost_equal(matrix_scaled, np.squeeze(np.stack((low_scaled, middle_scaled, high_scaled))))

    matrix_inverse = normalizer.inverse_scale(matrix_scaled)

    npt.assert_almost_equal(matrix_inverse, np.stack((low, middle, high)))