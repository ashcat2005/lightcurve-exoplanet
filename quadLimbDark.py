# Exoplanet transit lightcurve with
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
  
  # Reflects the curve w.r.t. the y axis  
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


##############################################################
# Parameters for the model
p = 0.1 #radius ratio
b = 0.7 #impact. parameter
gamma1 = 0.296 #linear limb darkening 
gamma2 = 0.34 #quadratic limb darkening
a = 2. #semi-major axis
T = 10. #orbital period
delta = 0. #orbital phase
omega = 2*np.pi/T #angular velocity

# Grid definition
N = 1000
t = np.linspace(0.,30,N)
x = a*np.cos(omega*t+delta)

z_range = np.sqrt(x**2 + b**2)


# Flux
F1 = np.zeros(N)
F2 = np.zeros(N)


# Main Loop to calculate the Flux
for i in range(N):
  Integral_1 = integrate.quad(integrand_1, 0.0001, 1., args=(p,x[i]))[0]
  Integral_2 = integrate.quad(integrand_2, 0.0001, 1.)[0]
  F1[i] = 1. - Integral_1/Integral_2

  Integral_1 = integrate.quad(integrand_1, 0.0001, 1., args=(p,z_range[i]))[0]
  Integral_2 = integrate.quad(integrand_2, 0.0001, 1.)[0]
  F2[i] = 1. - Integral_1/Integral_2



# Plot the Flux as function of time
plt.figure(figsize=(10,7))
plt.plot(t, F1,color='cornflowerblue', label=f'b = 0.0')
plt.plot(t, F2,color='crimson', label=f'b = {b:.1f}')
plt.hlines(1., t[0], t[-1],linestyle='dashed', alpha=0.3)
plt.xlabel(r'$t$')
plt.ylabel(r'$F(p,z)$')
plt.title('Exoplanet Transit Lightcurve')
plt.legend()
plt.savefig('lightcurve.jpg')
plt.show()
