# -*- coding: utf-8 -*-
"""
@author: Simon Santacatterina

Simulate the different parameters to take into account for the AWG star coupler 
boundary design.

The first part, dealing with the length l_max of the taper. The second part 
computes the single mode condition given the dimensions of the rib waveguide.
Rib waveguide with : etching depth D and width W.

"""
import numpy as np
import matplotlib.pyplot as plt

"""
First part : taper length for adiabatic transition
"""
# Parameters
wavelength = 1.55*10**-6
alpha = 1
neff = 3.44
wavelength_eff = wavelength/neff
#print(wavelength_eff)
W_0 = 0.43*10**-6 # initial width of taper [m]
W_max = 1*10**-6 # final width of taper [m]
l_max = ((W_max)**2 - (W_0)**2)/(2*alpha*wavelength_eff) #length of taper

# Build array/vector:
z = np.linspace(0, l_max, 100)
#print (z)

# Build vector of the tapered waveguide using Okamoto's formula
W_up = np.sqrt(2*alpha*wavelength_eff*z + (W_0)**2)/2
#print (W_up)
W_down = -np.sqrt(2*alpha*wavelength_eff*z + (W_0)**2)/2
#print (W_down)

plt.plot(z, W_up)
plt.plot(z, W_down)
plt.title("Top-view of taper at AW/slab interface") #C band plot
plt.xlabel(r'Length of taper [m]')
plt.ylabel('Taper position relatively to the center [m]')
plt.show() # display plots

"""
Second part : boundary between Multi-mode and Single-mode consition for rib 
waveguide

The paper "Single-Mode Condition for Silicon Rib Waveguides" is taken as 
a reference. Fig. 1 of this paper is the scheme used for associating the 
parameters

W_guide refers to W in the paper, i.e. waveguide width limit for 
the single mode condition
The total height H = h + D where D is the depth and h is the height of the
slab. The only fixed value is H = 220nm
"""
# Parameters

H = 220*10**-9 # total height rib + slab
D_max = 110*10**-9
alpha_2 = 4 #arbitrary constant
D = np.linspace(D_max/100, D_max, 100)
h = H - D
W_wg =  (H*alpha_2) + (h)/(np.sqrt(1 - (h/H)**2))
Dplot = D*10**9
Wplot = W_wg*10**6

plt.figure(figsize=(10,8))
plt.plot(Dplot, Wplot)
plt.title("Single mode condition for Si rib waveguide") #C band plot
plt.xlabel(r'D - Etching depth [nm]')
plt.ylabel('W - Taper\'s width [$\mu$m]')
plt.show() # display plots