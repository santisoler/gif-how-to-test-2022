"""
Introduction to pytest
"""
import pytest
import numpy as np
import numpy.testing as npt


def add(x, y):
    """
    Add x to y
    """
    return x + y


def multiply(x, y):
    """
    Multiply x by y
    """
    return x * y


def divide(x, y):
    """
    Divide x by y
    """
    return x / y


def test_add():
    """
    Test add function
    """
    x = 20
    y = 3
    result = add(x, y)
    expected = 23
    npt.assert_allclose(result, expected)


def test_add_with_negative_floats():
    """
    Test add function with negative floats
    """
    x = 1.2
    y = -1.0
    result = add(x, y)
    expected = 0.2
    npt.assert_allclose(result, expected)


def test_add_with_arrays():
    """
    Test add function
    """
    x = np.array([20, 1.2, -5])
    y = np.array([3, -1.0, 15])
    expected = np.array([23, 0.2, 10])
    result = add(x, y)
    npt.assert_allclose(result, expected)


def test_multiply():
    """
    Test multiply function
    """
    x = 5
    y = 3
    result = multiply(x, y)
    expected = 15
    npt.assert_allclose(result, expected)


def test_multiply_by_zero():
    """
    Test multiply function when one of the factors is zero
    """
    x = 5
    y = 1e-32
    result = multiply(x, y)
    expected = 0
    npt.assert_allclose(result, expected, atol=1e-30)


def test_divide():
    """
    Test divide function
    """
    x = 35
    y = 5
    result = divide(x, y)
    expected = 7
    npt.assert_allclose(result, expected)


def test_divide_by_zero():
    """
    Test if we get a ZeroDivisionError
    """
    x = 1
    y = 0
    with pytest.raises(ZeroDivisionError):
        divide(x, y)
