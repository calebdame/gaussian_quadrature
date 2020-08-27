# quassian_quadrature.py


import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy.stats import norm

class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """

    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        if polytype != "legendre" and polytype != "chebyshev":
            raise ValueError("polytype is not 'legendre' or 'chebyshev'") #if polytype is unknown
        self.n = n
        self.polytype = polytype
        if polytype == "legendre": # save attributes
            self.iwf = lambda x: 1
        else:
            self.iwf = lambda x: np.sqrt(1-x**2)
        self._xi, self._w = self.points_weights(self.n)
        

    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        
        if self.polytype == "chebyshev": # get Beta values
            b = np.ones(self.n)/2
            b[0] = 0.5**0.5
        else:
            b = np.sqrt(np.array([k**2 / (4 * k**2 - 1) for k in range(1, n)]))
        bUpper = np.diag(b, k=1) # build matrix
        bLower = np.diag(b, k=-1)
        J = bUpper + bLower
        val, vec = sp.linalg.eig(J[:n,:n]) # get eigenvecs and vals
        wi = np.zeros(self.n)
        if self.polytype == "chebyshev": # calculate weights
            wi = np.pi*vec[0]**2
        else:
            wi = 2*vec[0]**2
        return val, wi      


    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        return np.dot(f(self._xi)* self.iwf(self._xi), self._w) # take inner product of g and w   


    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        h = lambda x: f((b-a)/2*x+(a+b)/2)
        return self.basic(h) * (b-a) / 2 # integrate h 


    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        h = lambda x, y : f((b1-a1)/2*x+(b1+a1)/2, (b2-a2)/2*y+(a2+b2)/2) # 2-d h func 
        g = lambda x, y : h(x,y) * (self.iwf(x)*self.iwf(y)) # 2-d g func
        a = 0
        for i in range(self.n):
            for j in range(self.n):
                a += self._w[i] * self._w[j] * g(self._xi[i],self._xi[j]) # calc sum
        return a*(b1-a1)*(b2-a2)/4 # multiply by coeff



def example():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    f = lambda x : 1/(2*np.pi)**0.5*np.exp(-x**2/2) # Gauss dist func
    dom = 5*np.arange(1,11)
    i = norm.cdf(2) - norm.cdf(-3) # value to approx
    t = []
    for n in dom:
        g = GaussianQuadrature(n, "legendre") #initialize object legendre and integrate with n points
        t += [abs(i - g.integrate(f,-3,2))]
    plt.scatter(dom,t,label=g.polytype,c='red') # plot error points
    t = []
    for n in dom:
        g = GaussianQuadrature(n, "chebyshev") #initialize object cheb and integrate with n points
        t += [abs(i - g.integrate(f,-3,2))]
    plt.scatter(dom,t,label=g.polytype,c='blue') # plot error points
    plt.plot(dom,abs(sp.integrate.quad(f,-3,2)[0]-i)*np.ones(len(dom)),label="Scipy Error")
    plt.yscale("log") 
    plt.xlabel("Number of Quadrature Points")
    plt.ylabel("Abolute Error")
    plt.legend()
    plt.show()
