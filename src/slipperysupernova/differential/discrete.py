"""
Series module containing functions for mathematical progressions.
"""
def diff(t_k,x):
    """
    Calculate the discrete derivative of x.

    Formula: v(t) = (x(t_k) - x(t_k-1))/(t_k - t_k-1)

    Args:
        t_k (array): time value where k is the index of time
        x (array): signal value

    Returns:
        array: v(t) discrete dertivative of x

    Raises:
        Index error: if the lengths of the arrays dont match.

    Examples:
        >>>diff([1,2],[3,5])
        [0,2]
        >>>diff([1,2,3,4],[6,4,9,-8]
        [0,-2,5,-17]
    """
    if(len(t_k) != len(x)):
        raise IndexError("The arrays must be the same length")
    v[0] = 0
    for i in range(1, len(x)):
        v[i] = (x[i] - x[i-1])/(t_k[i] - t_k[i-1])
    return v

