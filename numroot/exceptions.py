class SolverException(Exception):
    """Base exception for the solver module"""

class InvalidIntervalException(SolverException):
    """Exception raised when the given method is unable to compute a root for the function"""

class ConvergenceException(SolverException):
    """Exception raised when the given method is unable to compute a root for the function"""