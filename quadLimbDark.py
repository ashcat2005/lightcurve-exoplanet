# Quadratic Limb Darkening Model

from scipy.misc import derivative
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

def L(r, P, Z):
  '''
  Obstruction function
  '''
  p = P/r
  z0 = Z/r
  
  # Reflects the information w.r.t. the y axis  
  if z0>0:
    z = z0
  if z0<0:
    z = -z0
  
  if z>1+p:
    return 0.
  elif (abs(1-p)<z and z<=1+p):
    k0 = np.arccos((p**2 + z**2 -1)/(2*p*z))
    k1 = np.arccos((1-p**2 + z**2)/(2*z))
    L0 = k0*p**2
    L2 = np.sqrt((4*z**2- (1+z**2-p**2)**2)/4) 
    return (L0 + k1 - L2)*r**2/np.pi
  elif (z<=1-p):
    return p**2*r**2
  elif z<= p-1: 
    return 1.*r**2


def I_function(r):
  '''
  Quadratic limb-darkening function
  '''
  mu = np.sqrt(1-r**2) 
  return 1. - gamma1*(1-mu) - gamma2*(1-mu)**2


def integrand_1(r, p, z):
  '''
  Integrand in the numerator
  '''
  T1 = derivative(L, r, dx=1e-6, args=(p,z))
  T2 = I_function(r)
  return T1*T2


def integrand_2(r):
  '''
  Integrand in the denominator
  '''
  return I_function(r)*2*r

# Parameters for the model
gamma1 = 0.296 
gamma2 = 0.34
p=0.1

# Discretization grid
N = 400
z_max = 3.
z_min = - z_max
z_range = np.linspace(z_min,z_max,N)
F = np.zeros(N)

# Main Loop to calculate the Flux
for i in range(N):
  Integral_1 = integrate.quad(integrand_1, 0.0, 1., args=(p,z_range[i]))[0]
  Integral_2 = integrate.quad(integrand_2, 0.0, 1.)[0]
  F[i] = 1. - Integral_1/Integral_2

# Plot the Flux as function of z
plt.figure(figsize=(7,7))
plt.plot(z_range, F)
plt.hlines(1., z_min, z_max,linestyle='dashed', alpha=0.3)
plt.xlabel(r'$z$')
plt.ylabel(r'$F(p,z)$')
plt.title('Exoplanet Transit Lightcurve')
plt.savefig('lightcurve.jpg')
plt.show()