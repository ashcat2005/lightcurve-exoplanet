##ESTA ES UNA FUNCIÓN DE PRUEBA PARA VER EN QUÉ ESTÁ FALLANDO LA NUESTRA!!!

import math
import numpy as np

import scipy.optimize, scipy.integrate

# def integrand(t, args):
#     w, p = args
#     return math.sin(t * w)/t + p

# def curve(x, p):
#     res = scipy.integrate.quad(integrand, 0.0, math.pi, [x, p])
#     return res[0]

def curve(x, a, b, c):
    return a*np.sqrt(x)+b*x*x+c/(x*x)

vcurve = np.vectorize(curve, excluded=set([1]))

truexdata = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
trueydata = curve(truexdata, 7.0, 8.0, 9.0)

xdata = truexdata + 0.1 * np.random.randn(7)
ydata = trueydata + 0.1 * np.random.randn(7)

popt, pcov = scipy.optimize.curve_fit(curve,
                                      xdata, ydata)
print(popt)