'''
This file was directly copied from the
codebase at https://github.com/sklus/d3s
'''

import math
import torch
from scipy.spatial import distance

def identity(x):
    '''
    Identity function.
    '''

    return x

class Monomials(object):
    '''
    Computation of monomials in d dimensions.
    '''

    def __init__(self, p):
        '''
        The parameter p defines the maximum order of the monomials.
        '''
        self.p = p

    def __call__(self, x):
        '''
        Evaluate all monomials of order up to p for all data points in x.
        '''
        [d, m] = x.shape
        c = allMonomialPowers(d, self.p)
        n = c.shape[1]
        y = torch.ones([n, m])
        for i in range(n):
            for j in range(d):
                y[i, :] = y[i, :] * torch.pow(x[j, :], c[j, i])
        return y

    def diff(self, x):
        '''
        Compute partial derivatives for all data points in x.
        '''
        [d, m] = x.shape
        c = allMonomialPowers(d, self.p)
        n = c.shape[1]
        y = torch.zeros([n, d, m])
        for i in range(n):
            for j in range(d):
                e = c[:, i].clone()
                a = e[j]
                e[j] = e[j] - 1

                if torch.any(e < 0):
                    continue

                y[i, j, :] = a * torch.ones([1, m])
                for k in range(d):
                    y[i, j, :] = y[i, j, :] * torch.pow(x[k, :], e[k])
        return y

    def ddiff(self, x):
        '''
        Compute second order derivatives for all data points in x.
        '''
        [d, m] = x.shape
        c = allMonomialPowers(d, self.p)
        n = c.shape[1]
        y = torch.zeros([n, d, d, m])
        for i in range(n):
            for j1 in range(d):
                for j2 in range(d):
                    e = c[:, i].clone()
                    a = e[j1]
                    e[j1] = e[j1] - 1
                    a *= e[j2]
                    e[j2] = e[j2] - 1

                    if torch.any(e < 0):
                        continue

                    y[i, j1, j2, :] = a * torch.ones([1, m])
                    for k in range(d):
                        y[i, j1, j2, :] = y[i, j1, j2, :] * torch.pow(x[k, :], e[k])
        return y

    def __repr__(self):
        return 'Monomials of order up to %d.' % self.p

    def display(self, alpha, d, name=None, eps=1e-6):
        c = allMonomialPowers(d, self.p)

        if name is not None:
            print(name + ' = ', end='')

        ind = torch.where(torch.abs(alpha) > eps)
        k = ind[0].shape[0]

        if k == 0:
            print('0')
            return

        for i in range(k):
            if i == 0:
                print('%.5f' % alpha[ind[0][i]], end='')
            else:
                if alpha[ind[0][i]] > 0:
                    print(' + %.5f' % alpha[ind[0][i]], end='')
                else:
                    print(' - %.5f' % -alpha[ind[0][i]], end='')

            self._displayMonomial(c[:, ind[0][i]])
        print('')

    def _displayMonomial(self, p):
        d = p.shape[0]
        if torch.all(p == 0):
            print('1', end='')
        else:
            for j in range(d):
                if p[j] == 0:
                    continue
                if p[j] == 1:
                    print(' x_%d' % (j + 1), end='')
                else:
                    print(' x_%d^%d' % (j + 1, p[j]), end='')

class Indicators(object):
    '''
    Indicator functions for box discretization Omega.
    '''

    def __init__(self, Omega):
        self.Omega = Omega

    def __call__(self, x):
        [d, m] = x.shape
        n = self.Omega.numBoxes()
        y = torch.zeros([n, m])
        for i in range(m):
            ind = self.Omega.index(x[:, i])
            if ind == -1:
                continue
            y[ind, i] = 1
        return y

    def __repr__(self):
        return 'Indicator functions for box discretization.'


# auxiliary functions
def nchoosek(n, k):
    '''
    Computes binomial coefficients.
    '''

    return math.factorial(n)//math.factorial(k)//math.factorial(n-k) # integer division operator


def nextMonomialPowers(x):
    '''
    Returns powers for the next monomial. Implementation based on John Burkardt's MONOMIAL toolbox, see
    http://people.sc.fsu.edu/~jburkardt/m_src/monomial/monomial.html.
    '''

    m = len(x)
    j = 0
    for i in range(1, m): # find the first index j > 1 s.t. x[j] > 0
        if x[i] > 0:
            j = i
            break
    if j == 0:
        t = x[0]
        x[0] = 0
        x[m - 1] = t + 1
    elif j < m - 1:
        x[j] = x[j] - 1
        t = x[0] + 1
        x[0] = 0
        x[j-1] = x[j-1] + t
    elif j == m - 1:
        t = x[0]
        x[0] = 0
        x[j - 1] = t + 1
        x[j] = x[j] - 1
    return x


def allMonomialPowers(d, p):
    '''
    All monomials in d dimensions of order up to p.
    '''

    # Example: For d = 3 and p = 2, we obtain
    # [[ 0  1  0  0  2  1  1  0  0  0]
    #  [ 0  0  1  0  0  1  0  2  1  0]
    #  [ 0  0  0  1  0  0  1  0  1  2]]
    n = nchoosek(p + d, p) # number of monomials
    x = np.zeros(d) # vector containing powers for the monomials, initially zero
    c = np.zeros([d, n]) # matrix containing all powers for the monomials
    for i in range(1, n):
        c[:, i] = nextMonomialPowers(x)
    c = np.flipud(c) # flip array in the up/down direction
    return c