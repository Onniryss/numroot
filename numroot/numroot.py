"""
This module implement 3 methods to solve equations of the form f(x) = 0
"""

from .exceptions import InvalidIntervalException, ConvergenceException

class NonlinearSolver:
    """Solver class to hold the different algorithms"""
    def __init__(self):
        pass

# pylint: disable=too-many-arguments
    def bisect(self, func, x_a, x_b, force = False, epsilon = 1E-3, maxiter = 100, n_iter=0):
        """Bisection algorithm
        
        Arguments:
        func -- real 1D function
        x_a -- first bound
        x_b -- second bound
        force -- wether or not to force the use of the algorithm.
                The value returned will have no meaning if the function
                doesn't change sign between x_a and x_b (default False)
        epsilon -- degree of precision for the algorithm to stop (default 1e-3)
        maxiter -- maximum number of iterations (default 100)
        iter -- current iteration (default 0)
        """

        f_a = func(x_a)
        f_b = func(x_b)

        if abs(f_a) < epsilon:
            return x_a, n_iter

        if abs(f_b) < epsilon:
            return x_b, n_iter

        if not force and f_a * f_b > 0:
            raise InvalidIntervalException("func(x_a) and func(x_b) should have opposite signs. \
                If not, this method doesn't guarantee a root.")

        if n_iter >= maxiter:
            raise ConvergenceException("The Bissection algorithm couldn't converge in the \
                given amount of iterations.")

        if abs(x_a-x_b) < epsilon:
            # If the algorithm converged, x_a ~ x_b so it doesn't matter
            return x_a, n_iter

        x_m = (x_a+x_b)/2
        f_m = func(x_m)

        if f_a * f_m < 0:
            return self.bisect(func, x_a, x_m, True, epsilon, maxiter, n_iter+1)

        return self.bisect(func, x_m, x_b, True, epsilon, maxiter, n_iter+1)

# pylint: disable=too-many-arguments
    def newton_ralphson(self, func, dfunc, x_0, epsilon=1E-3, maxiter=100, n_iter=0):
        """Newton-Ralphson algorithm
        
        Arguments:
        func -- real 1D function
        dfunc -- derivative of func
        x_0 -- starting value
        epsilon -- degree of precision for the algorithm to stop (default 1e-3)
        maxiter -- maximum number of iterations (default 100)
        iter -- current iteration (default 0)
        """
        x_1 = x_0 - func(x_0)/dfunc(x_0)

        if n_iter >= maxiter:
            raise ConvergenceException("The Newton-Ralphson algorithm couldn't converge in the \
                given amount of iterations.")

        if abs(x_1-x_0) < epsilon or func(x_1) < epsilon:
            return x_1, n_iter

        return self.newton_ralphson(func, dfunc, x_1, epsilon, maxiter, n_iter+1)

# pylint: disable=too-many-arguments
    def secant(self, func, x_0, x_1, epsilon=1E-3, maxiter=100, n_iter=0):
        """Secant algorithm
        
        Arguments:
        func -- real 1D function
        x_0 -- first intersection
        x_1 -- second intersection
        epsilon -- degree of precision for the algorithm to stop (default 1e-3)
        maxiter -- maximum number of iterations (default 100)
        iter -- current iteration (default 0)
        """

        if n_iter >= maxiter:
            raise ConvergenceException("The Secant algorithm couldn't converge in the \
                given amount of iterations.")

        if abs(x_1-x_0) < epsilon or func(x_1) < epsilon:
            return x_1, n_iter

        f_0 = func(x_0)
        f_1 = func(x_1)

        slope = (x_1-x_0)/(f_1-f_0)

        x_2 = x_1-slope*f_1

        return self.secant(func, x_1, x_2, epsilon, maxiter, n_iter+1)

    def compare_methods(self, func, dfunc, params):
        """function"""
