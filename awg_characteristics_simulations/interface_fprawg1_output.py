# -*- coding: utf-8 -*-

"""
Created on Mon Jul 19 06:55:18 2021

Computes fields and power patterns after interface of FPR1 and AWs, apply phase 
delays caused by the AWs, finally compute the input into FPR2 and the output of
FPR2.

Inputs are the diffracted field from the slab, computed in
slab_1_diffraction.py (float) and the simulated data using FDTD (DAT).
Ouputs are fields and powers at the beginning and end of the AWs and in FPR2 (PNG).

@author: Simon Santacatterina
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 

from input_slab1_diffraction import transit1
from input_slab1_diffraction import transit2
from filereading2 import file_read2
from fraunhofer import fraunhofer

"""
Take results form slab_1_diffraction.py
"""
a, b = transit1() # importing data
ex_int = transit2()
n = 24 # number of waveguides in grating

#def repeat(arr, count):
#    return np.stack([arr for _ in range(count)], axis=0)

#repeat the field for all the array
#ex_wg_in1 = np.tile(a,n)
#hy_wg_in1 = np.tile(b,n)

"""
Load the newly computed fields, i.e. FDTD data from interface simulations. 
FDTD using the optimized interface design at the input and a traditional linear 
design at the output.

Data are loaded and displayed, then field inside the AWs is displayed.
After, the field at the ouput of the waveguide grating is also displayed.
"""

"""
At first input side : power and ex and hy are plotted
"""
### Power
 
power_space_in_name = ".\\power_results\\fpr1awin_m3_t34100000_pow.dat"
power_space_in_dim = np.array([201,89])
power_space_in = file_read2(power_space_in_name, power_space_in_dim) # data for monitor10

# plot 
xin1 = np.linspace(-1, 1, 201) # x-domain monitor 2 at taper output
y = np.linspace(-0.44, 0.44, 89) # y-domain for monitor, slab or Si part height

# for contour plot need of a meshgrid
xxin1, yyin1 = np.meshgrid(xin1, y)

# for contour plot need of transposing the data, or x domain is in row, y in column for python it's opposite
# 2D contour

# power representation
contour_pw_in1 = plt.pcolor(xxin1, yyin1, np.transpose(power_space_in), cmap ='jet')
plt.title("Power profile for central wg in AWs")
plt.xlabel(r'[$\mu$m]')
plt.ylabel(r'[$\mu$m]')
plt.colorbar()
plt.savefig("Power profile for central wg in AWs", bbox_inches = 'tight')
plt.show()

### Ex and Hy

ex_in1_name = ".\\power_results\\fpr1awin_m3_t34100000_ex.dat"
hy_in1_name = ".\\power_results\\fpr1awin_m3_t34100000_hy.dat"

ex_in1_dim  = np.array([201,89])
hy_in1_dim  = np.array([201,89])

ex_in1 = file_read2(ex_in1_name, ex_in1_dim)
hy_in1 = file_read2(hy_in1_name, hy_in1_dim)

# plot 
ehx_in1 = np.linspace(-1, 1, 201) # x-domain monitor 2 at taper output
ehy_in1 = np.linspace(-0.44, 0.44, 89) # y-domain for monitor, slab or Si part height

# for contour plot need of a meshgrid
ehxx_in1, ehyy_in1 = np.meshgrid(ehx_in1, ehy_in1)

# launched (input wg)
#ex
contour_ex_in1 = plt.pcolor(ehxx_in1, ehyy_in1, np.transpose(ex_in1), cmap ='jet')
plt.title("Fundamental mode Ex profile in AWs")
plt.xlabel(r'x-axis [$\mu$m]')
plt.ylabel(r'y-axis [$\mu$m]')
plt.colorbar()
#plt.savefig("Power profile for central wg in AWs", bbox_inches = 'tight')
plt.show()

#hy
contour_hy_in1 = plt.pcolor(ehxx_in1, ehyy_in1, np.transpose(hy_in1), cmap ='jet')
plt.title("Fundamental mode Hy profile in AWs")
plt.xlabel(r'x-axis [$\mu$m]')
plt.ylabel(r'y-axis [$\mu$m]')
plt.colorbar()
plt.show()

"""
Then output side : power, ex and hy are plotted
"""
### Power

power_space_out_name = ".\\power_results\\awfpr2linear1_m2_t310000000_pow.dat"
power_space_out_dim = np.array([201,89])
power_space_out = file_read2(power_space_out_name, power_space_out_dim) # data for monitor10

#plot 
xout1 = np.linspace(-1, 1, 201) # x-domain monitor 2 at taper output
y = np.linspace(-0.44, 0.44, 89) # y-domain for monitor, slab or Si part height

# for contour plot need of a meshgrid
xxout1, yyout1 = np.meshgrid(xout1, y)

# for contour plot need of transposing the data, or x domain is in row, y in column for python it's opposite
#2D contour

# power representation
contour_pw_out1 = plt.pcolor(xxout1, yyout1, np.transpose(power_space_out), cmap ='jet')
plt.title("Spatial power profile for central wg out of interface FPR2/AW")
plt.xlabel(r'[$\mu$m]')
plt.ylabel(r'[$\mu$m]')
plt.colorbar()
plt.show()

### Ex and Hy

ex_out1_name = ".\\power_results\\awfpr2linear1_m2_t310000000_ex.dat"
hy_out1_name = ".\\power_results\\awfpr2linear1_m2_t310000000_hy.dat"

ex_out1_dim  = np.array([201,89])
hy_out1_dim  = np.array([201,89])

ex_out1 = file_read2(ex_out1_name, ex_out1_dim)
hy_out1 = file_read2(hy_out1_name, hy_out1_dim)

#plot 
ehx_out1 = np.linspace(-1, 1, 201) # x-domain monitor 2 at taper output
ehy_out1 = np.linspace(-0.44, 0.44, 89) # y-domain for monitor, slab or Si part height

# for contour plot need of a meshgrid
ehxx_out1, ehyy_out1 = np.meshgrid(ehx_out1, ehy_out1)

# launched (input wg)
#ex
contour_ex_out1 = plt.pcolor(ehxx_out1, ehyy_out1, np.transpose(ex_out1), cmap ='jet')
plt.title("Fundamental mode Ex profile out of AWs")
plt.xlabel(r'x-axis [$\mu$m]')
plt.ylabel(r'y-axis [$\mu$m]')
plt.colorbar()
plt.show()

#hy
contour_hy_out1 = plt.pcolor(ehxx_out1, ehyy_out1, np.transpose(hy_out1), cmap ='jet')
plt.title("Fundamental mode Hy profile out of AWs")
plt.xlabel(r'x-axis [$\mu$m]')
plt.ylabel(r'y-axis [$\mu$m]')
plt.colorbar()
plt.show()

"""
From here, total field in the AWs. In particular, affected by the pattern from  
the first slab diffraction.
"""
r = 30*10**-6 # slab dimension in z dirention

dwg = 2*10**-6 # space between the wgs
n = 24 # number of wgs
nwg_val = (n+1)*ex_out1.shape[0] # number of samples for field computation x-axis
#wg_num = np.arange(0,n//2+1) # array representing the ith wg

h = 0.11*10**-6 # semi-height of slab
nh = 10
nh_val = ex_out1.shape[1] # number of samples for field computation y-axis

x_aw1 = np.linspace(-(n//2)*dwg, (n//2)*dwg, nwg_val) # x-domain at FPR1/AWs interface
y_aw1 = np.linspace(-h, h, nh_val) # y-domain at FPR1/AWs interface

#phi_in =  np.zeros(shape=(n+1)*ex_out1.shape[0],ex_out1.shape[1]))

ex_all_in1 =  np.zeros(shape=(ex_in1.shape[0],ex_in1.shape[1]))
ex_all_in1 = ex_in1*a[0]
for i in np.arange(n):
    ex_all_in1 = np.concatenate((ex_all_in1, ex_in1*a[i+1,0]))

# for contour plot need of a meshgrid
xx_aw1, yy_aw1 = np.meshgrid(x_aw1, y_aw1)

# plot of profile at the interface 
#ex
contour_ex_fpr1 = plt.pcolor(xx_aw1, yy_aw1, np.transpose(ex_all_in1), cmap ='jet')
plt.title("Spatial profile of ex in the AW")
#plt.title("Spatial profile of for ex at FPR1/AWs interface")
plt.xlabel(r'x [$\mu$m]')
plt.ylabel(r'y [$\mu$m]')
plt.colorbar()
plt.show()

# plot of the x-axis profile in the array
#ex
plt.plot(x_aw1, ex_all_in1[:, nh_val//2], 'r--', label = "interface")
plt.title("Spatial profile of ex in the AW")
#plt.title("Spatial distribution of ex along x-axis at FPR1/AWs interface ")
plt.xlabel(r'x [$\mu$m]')
plt.ylabel("|Field value| [.]")
ax1 = plt.gca() #reference to the current plot 
ax1.legend()
#plt.savefig('Power flow in the z direction for interface WG/FPR1.png', bbox_inches = 'tight')
plt.show() # display plots
 
"""
Step of the AWG computation related to phase delay.
""" 
wavelength = 1.55*10**-6 # in m

n_si = 3.47
n_eff = 2.304 # effective refractive index computed by R-Soft

wavelength_si = 1.55*10**-6/n_si
wavelength_eff = 1.55*10**-6/n_eff
kneff = 2*np.pi/wavelength_eff

l_0 = 20*10**-6 # length of shortest waveguide 
delta_l = 25.4**10-6
wg_numb = np.linspace(-(n//2), (n//2), n+1)

l_phase = l_0 + delta_l*(wg_numb + (n//2)) # phasing brought by each waveguide in the array

phase_vec = np.exp(-1j*kneff*l_phase)

ex_all_out1 =  np.zeros(shape=(ex_out1.shape[0],ex_out1.shape[1]),dtype=np.complex_)
ex_all_out1 = ex_out1*a[0,0]*phase_vec[0]

for i in np.arange(n):
    ex_all_out1 = np.concatenate((ex_all_out1, ex_out1*a[i+1,0]*phase_vec[i+1]))

# for contour plot need of a meshgrid
# xx_aw1, yy_aw1 = np.meshgrid(x_aw1, y_aw1) #potentially other domain than AW one
    
"""
From Arrayed Waveguides until in the end of FPR2
"""
# plot of the profile at the entrance fpr2
#ex
contour_ex_fprboundary = plt.pcolor(xx_aw1, yy_aw1, np.transpose(np.abs(ex_all_out1)), cmap ='jet')
plt.title("Profile of ex at input of FPR2")
#plt.title("Spatial profile of for ex at FPR1/AWs interface")
plt.xlabel(r'x [$\mu$m]')
plt.ylabel(r'y [$\mu$m]')
plt.colorbar()
plt.show()

# plot of the x-axis profile at the entrance of fpr2
#ex
plt.plot(x_aw1, np.abs(ex_all_out1[:, nh_val//2]), 'r--', label = "interface")
plt.title("Ex profile along x-axis in the at input of FPR2")
#plt.title("Spatial distribution of ex along x-axis at FPR1/AWs interface ")
plt.xlabel(r'x [$\mu$m]')
plt.ylabel("|Field value| [.]")
ax11 = plt.gca() #reference to the current plot 
ax11.legend()
#plt.savefig('Power flow in the z direction for interface WG/FPR1.png', bbox_inches = 'tight')
plt.show() # display plots

"""
Output
"""
# resulting pattern for ex

dwg_out = 3.2*10**-6
n_out = 8
r_out = 30*10**-6
n_eff_out = 2.845

wavelength_si = 1.55*10**-6/n_si
wavelength_eff_out = 1.55*10**-6/n_eff
kneff_out = 2*np.pi/wavelength_eff

dx_out1 = 0.01
dy_out1 = 0.01

nwg_val_out = n_out*20+1 # number of samples for field computation x-axis

h_out = 0.11*10**-6 # semi-height of slab
nh_out = 10
nh_val_out = 2*nh+1

x_aw_fpr1 = np.linspace(-(n//2)*dwg, (n//2)*dwg, ex_all_in1.shape[0]) # x-domain at FPR1/AWs interface
y_aw_fpr1 = np.linspace(-h, h, ex_all_out1.shape[1]) # y-domain at FPR1/AWs interface

x_aw_out1 = np.linspace(-(n_out//2)*dwg_out, (n_out//2)*dwg_out, nwg_val_out) # x-domain at FPR1/AWs interface
y_aw_out1 = np.linspace(-h, h, nh_val_out) # y-domain at FPR1/AWs interface


ex_all_outputwg_complex = fraunhofer(dwg_out, n_out, r_out, wavelength_eff_out, kneff_out, ex_all_out1.shape[0], ex_all_out1.shape[1], dx_out1 , dy_out1, nwg_val_out, nh_val_out, x_aw_fpr1, y_aw_fpr1, x_aw_out1, y_aw_out1, ex_all_out1)
ex_all_outputwg = np.abs(ex_all_outputwg_complex)
#ex_all_outputwg = ex_all_outputwg_complex

# for contour plot need of a meshgrid
xx_aw_out1, yy_aw_out1 = np.meshgrid(x_aw_out1, y_aw_out1) # output plane representation

#plot of the profile at the interface with output wgs
#ex
contour_ex_fpr1 = plt.pcolor(xx_aw_out1, yy_aw_out1, np.transpose(ex_all_outputwg), cmap ='jet')
plt.title("Spatial profile of ex outof FPR2")
#plt.title("Spatial profile of for ex at FPR1/AWs interface")
plt.xlabel(r'x [$\mu$m]')
plt.ylabel(r'y [$\mu$m]')
plt.colorbar()
plt.show()

# plot of the x-axis profile at the interface with output wgs
#ex
plt.plot(x_aw_out1, ex_all_outputwg[:,nh_out+1], 'r--', label = "interface")
plt.title("Spatial profile of ex outof FPR2 along x-axis")
#plt.title("Spatial distribution of ex along x-axis at FPR1/AWs interface ")
plt.xlabel(r'x [$\mu$m]')
plt.ylabel("|Field value| [.]")
ax2 = plt.gca() # reference to the current plot 
ax2.legend()
#plt.savefig('Power flow in the z direction for interface WG/FPR1.png', bbox_inches = 'tight')
plt.show() # display plots
    
"""
Final step of the simulation : output waveguide interface
"""
#ex_all_in1 =  np.zeros(shape=(ex_out1.shape[0],ex_out1.shape[1]))

x_aw_out1_fft = np.linspace(-(n_out//2)*dwg_out, (n_out//2)*dwg_out, (n+1)*ex_out1.shape[0]) # x-domain at FPR1/AWs interface

#x_aw_out1_fft = np.linspace(-(n_out//2)*dwg_out, (n_out//2)*dwg_out, (n+1)*ex_out1.shape[0]) # x-domain at FPR1/AWs interface
fft_out = np.fft.fft(ex_all_out1[:, nh_val//2])
fft_out_final = np.abs(np.fft.fftshift(fft_out))
fft_out_dB = 10*np.log10(fft_out_final)

# plot of the x-axis profile at the interface
#ex
plt.plot(x_aw_out1_fft, fft_out_final, 'r--', label = "interface")
plt.title("Spatial profile of ex in FPR2 along x-axis")
#plt.title("Spatial distribution of ex along x-axis at FPR1/AWs interface ")
plt.xlabel(r'x [$\mu$m]')
plt.ylabel("|Field value| [.]")
ax2 = plt.gca() #reference to the current plot 
ax2.legend()
#plt.savefig('Power flow in the z direction for interface WG/FPR1.png', bbox_inches = 'tight')
plt.show() # display plots

#plot of the x-axis profile at the interface
#ex
plt.plot(x_aw_out1_fft, np.abs(fft_out_dB), 'r--', label = "interface")
plt.title("Spatial profile of ex in FPR2 along x-axis")
#plt.title("Spatial distribution of ex along x-axis at FPR1/AWs interface ")
plt.xlabel(r'x [$\mu$m]')
plt.ylabel("|Field value| [.]")
ax2 = plt.gca() # reference to the current plot 
ax2.legend()
#plt.savefig('Power flow in the z direction for interface WG/FPR1.png', bbox_inches = 'tight')
plt.show() # display plots
