"""
This file contains the mathematical functions

Author: Suhan Park (park94@kw.ac.kr)
"""
import numpy as np

def cubic_spline(t : float, t_0 : float, t_f : float, 
                       x_0 : np.ndarray, 
                       x_f : np.ndarray, 
                       x_dot_0 : np.ndarray, 
                       x_dot_f : np.ndarray):
    """
    Cubic polynomial spline interpolation.
    
    Args:
        t (float): Current time.
        t_0 (float): Start time.
        t_f (float): End time.
        x_0 (np.ndarray): Initial position.
        x_f (np.ndarray): Final position.
        x_dot_0 (np.ndarray): Initial velocity.
        x_dot_f (np.ndarray): Final velocity.
    
    Returns:
        np.ndarray: Position at time t.
    """

    if t < t_0:
        return x_0
    
    if t > t_f:
        return x_f

    t_e = t - t_0 # elapsed time
    t_s = t_f - t_0
    t_s2 = t_s ** 2
    t_s3 = t_s ** 3
    total_x = x_f - x_0

    term1 = x_dot_0 * t_e
    term2 = (3 * total_x / t_s2 - 2 * x_dot_0 / t_s - x_dot_f / t_s) * t_e ** 2
    term3 = (-2 * total_x / t_s3 + (x_dot_0 + x_dot_f) / t_s2) * t_e ** 3

    x_t = x_0 + term1 + term2 + term3
    return x_t

def quintic_spline(t : float, t_0 : float, t_f : float,
                   x_0 : float,
                   x_dot_0 : float,
                   x_ddot_0 : float,
                   x_f : float,
                   x_dot_f : float,
                   x_ddot_f : float):
    """
    Quintic polynomial spline interpolation.

    Args:
        t (float): Current time.
        t_0 (float): Start time.
        t_f (float): End time.
        x_0 (float): Initial position.
        x_dot_0 (float): Initial velocity.
        x_ddot_0 (float): Initial acceleration.
        x_f (float): Final position.
        x_dot_f (float): Final velocity.
        x_ddot_f (float): Final acceleration.

    Returns:
        np.ndarray: Position, velocity, and acceleration at time t.
    """
    if t < t_0:
        return np.array([x_0, x_dot_0, x_ddot_0])
    
    if t > t_f:
        return np.array([x_f, x_dot_f, x_ddot_f])
    
    t_s = t_f - t_0
    a1 = x_0
    a2 = x_dot_0
    a3 = x_ddot_0 / 2.0
    mat = np.array([
        [t_s**3, t_s**4, t_s**5],
        [3 * t_s**2, 4 * t_s**3, 5 * t_s**4],
        [6 * t_s, 12 * t_s**2, 20 * t_s**3]
    ])
    v = np.array([
        x_f - x_0 - x_dot_0 * t_s - 0.5 * x_ddot_0 * t_s**2,
        x_dot_f - x_dot_0 - x_ddot_0 * t_s,
        x_ddot_f - x_ddot_0
    ])
    res = np.linalg.solve(mat, v)
    a4, a5, a6 = res
    t_e = t - t_0
    position = a1 + a2 * t_e + a3 * t_e**2 + a4 * t_e**3 + a5 * t_e**4 + a6 * t_e**5
    velocity = a2 + 2 * a3 * t_e + 3 * a4 * t_e**2 + 4 * a5 * t_e**3 + 5 * a6 * t_e**4
    acceleration = 2 * a3 + 6 * a4 * t_e + 12 * a5 * t_e**2 + 20 * a6 * t_e**3

    return np.array([position, velocity, acceleration])

def quintic_spline_multi(t, t_0, t_f,
                        x_0 : np.ndarray, 
                        x_dot_0 : np.ndarray, 
                        x_ddot_0 : np.ndarray, 
                        x_f : np.ndarray, 
                        x_dot_f : np.ndarray, 
                        x_ddot_f : np.ndarray):
    if t < t_0:
        return np.array([x_0, x_dot_0, x_ddot_0])
    
    if t > t_f:
        return np.array([x_f, x_dot_f, x_ddot_f])
    
    dim = x_0.shape[0]

    results = np.zeros((3, dim))

    for d in range(dim):
        res = quintic_spline(t, t_0, t_f, x_0[d], x_dot_0[d], x_ddot_0[d], x_f[d], x_dot_f[d], x_ddot_f[d])
        results[:, d] = res

    return results

def finite_difference_derivative(arr : np.ndarray, order : int, dt : float, axis:int = 0):
    """
    Estimate the derivative using finite differences.

    Args:
        arr (np.ndarray): Array of values.
        order (int): Order of the derivative (1 or 2).
        dt (float): Time step.

    Returns:
        np.ndarray: Estimated derivative.
    """

    grad = np.gradient(arr, dt, axis=axis)
    for _ in range(order-1):
        grad = np.gradient(grad, dt, axis=axis)

    return grad
    
# Test cases for the cubic and quintic spline functions
def test_cubic_spline():
    t = 0.5
    t_0 = 0.0
    t_f = 1.0
    x_0 = 0.0
    x_f = 1.0
    x_dot_0 = 0.0
    x_dot_f = 1.0

    result = cubic_spline(t, t_0, t_f, x_0, x_f, x_dot_0, x_dot_f)
    
    assert np.isclose(result, 0.375, atol=1e-3), "Cubic spline test failed"

def test_cubic_spline_multi():
    t = 0.5
    t_0 = 0.0
    t_f = 1.0
    x_0 = np.array([0.0, 0.0])
    x_f = np.array([1.0, 1.0])
    x_dot_0 = np.array([0.0, 0.0])
    x_dot_f = np.array([1.0, 1.0])

    result = cubic_spline(t, t_0, t_f, x_0, x_f, x_dot_0, x_dot_f)
    print(f"Multi-variable Cubic Spline Result: {result}")
    assert np.allclose(result, np.array([0.375, 0.375]), atol=1e-3), "Multi-variable cubic spline test failed"

def test_quintic_spline():
    t = 0.5
    t_0 = 0.0
    t_f = 1.0
    x_0 = 0.0
    x_dot_0 = 0.0
    x_ddot_0 = 0.0
    x_f = 1.0
    x_dot_f = 1.0
    x_ddot_f = 1.0

    result = quintic_spline(t, t_0, t_f, x_0, x_dot_0, x_ddot_0, x_f, x_dot_f, x_ddot_f)
    print(f"Quintic Spline Result: {result}")
    assert np.isclose(result[0], 0.359375, atol=1e-3), "Quintic spline test failed"

def test_quintic_spline_multi():
    t = 0.5
    t_0 = 0.0
    t_f = 1.0
    x_0 = np.array([0.0, 0.0])
    x_dot_0 = np.array([0.0, 0.0])
    x_ddot_0 = np.array([0.0, 0.0])
    x_f = np.array([1.0, 1.0])
    x_dot_f = np.array([1.0, 1.0])
    x_ddot_f = np.array([1.0, 1.0])

    result = quintic_spline_multi(t, t_0, t_f, x_0, x_dot_0, x_ddot_0, x_f, x_dot_f, x_ddot_f)
    print(f"Multi-variable Quintic Spline Result: {result}")
    assert np.allclose(result[0], np.array([0.359375, 0.359375]), atol=1e-3), "Multi-variable quintic spline test failed"

def test_spline():
    test_cubic_spline()
    test_cubic_spline_multi()
    test_quintic_spline()
    test_quintic_spline_multi()

def test_finite_difference_derivative():
    arr = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    dt = 1.0
    order = 1

    result = finite_difference_derivative(arr, order, dt)
    expected_result = np.array([1.0, 1.5, 2.5, 3.5, 4.0])
    print(f"Finite Difference Derivative Result: {result}")
    
    assert np.allclose(result, expected_result), "Finite difference derivative test failed"

def test_second_derivative():
    arr = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    dt = 1.0
    order = 2

    result = finite_difference_derivative(arr, order, dt)
    expected_result = np.array([0.5,  0.75, 1.0, 0.75, 0.5  ])
    print(f"Finite Difference Second Derivative Result: {result}")
    
    assert np.allclose(result, expected_result), "Finite difference second derivative test failed"

if __name__ == "__main__":
    test_spline()
    test_finite_difference_derivative()
    test_second_derivative()
    print("All tests passed.")