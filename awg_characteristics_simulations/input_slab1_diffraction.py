# -*- coding: utf-8 -*-
"""
Created on Fri July 16 16:01:49 2021

Python code used for extracting data from Fullwave FDTD power data files.
The data handled here simulated the behavior of a slab array interface form input wgs to 1st slab. 
The data are presenting the power (or fields) transmitted at several point both spatailly and temporally.

@author: Simon Santacatterina

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  

#custom functions
from filereading import file_read
from filereading2 import file_read2
from fraunhofer import fraunhofer
#from filereading3 import file_read3

"""
Part1 : data processing and plots.

Code taking as input the file containing raw data simulated from FullWave FDTD
extracting those data in order to plot them. 

The input of this script is the data files
(or other) containing "string" version of the data. 

Calling function file_reading the data are read and the values are stored in
matrixes for manipulations and plots.
"""

"""
Temporal measurement of the power transmitted.
"""
power_temp_name = ".\\power_results\op1.dat"
power_temp_dim = np.array([80,4])
power_temp = file_read(power_temp_name, power_temp_dim)
power_temp[:,1:4] = np.square(power_temp[:,1:4])

# plot of raw data as measured by FDTD, normalized by input power
fig = plt.figure()
plt.plot(power_temp[:,0], power_temp[:,1], 'bx', label = "total power")
plt.plot(power_temp[:,0], power_temp[:,2], 'r--', label = "center wg")
plt.title("Power flow in the z direction for interface WG/FPR1")
plt.xlabel(r'cT [$\mu$m]')
plt.ylabel("|Normalized Power| [.]")
ax1 = plt.gca() # reference to the current plot 
ax1.legend()
fig.savefig("Comparison center wg and total power for WG-FPR1.png", bbox_inches = 'tight')
plt.show() # display plots

# plot of difference between Monitor 2 (Total P) and Monitor 1 (Central Wg P) 
power_diff = power_temp[:,2] - power_temp[:,1]
plt.plot(power_temp[:,0], power_diff, 'gD', label = "difference")
plt.title("Total Power-Central WG power at interface WG/FPR1")
plt.xlabel(r'cT [$\mu$m]')
plt.ylabel("Normalized Power Difference [.]")
ax2 = plt.gca() # reference to the current plot 
ax2.legend()
#plt.savefig('Total Power-Central WG power at interface WG/FPR1.png', bbox_inches = 'tight')
plt.show() # display plots

#plot of difference between Monitor 2 (Total P) and Monitor 1 (Central Wg P) in dB
power_db = 10*np.log10(power_diff[70:])
plt.plot(power_temp[70:,0], power_db, 'gD', label = "10*log10(total - central)")
plt.title("Log difference of central WG power with total power at interface WG/FPR1")
plt.xlabel(r'cT [$\mu$m]')
plt.ylabel("Pcentral-Ptot [dB]")
ax3 = plt.gca() # reference to the current plot 
ax3.legend()
#plt.savefig('Total Power-Central WG power at interface WG/FPR1.png', bbox_inches = 'tight')
plt.show() # display plots

"""
Spatial measurement of the power transmitted
"""
# power of monitor m2 = power of central wg
power_space_name_m2 = ".\\power_results\\op1_m2_t310000000_pow - Copie.dat"
power_space_dim_m2 = np.array([101,89]) 
power_space_m2 = file_read2(power_space_name_m2, power_space_dim_m2)# data for monitor2

# power of monitor m10 = 3wgs a bit further in the slab at a point where the beam starts diverging
power_space_name_m10 = ".\\power_results\\op1_m10_t310000000_pow - Copie.dat"
power_space_dim_m10 = np.array([301,89])
power_space_m10 = file_read2(power_space_name_m10, power_space_dim_m10)# data for monitor10

# plot 
x1 = np.linspace(-1, 1, 101) # x-domain monitor 2 at taper output
x2 = np.linspace(-3, 3, 301) # x-domain monitor 10 in the slab
y = np.linspace(-0.44, 0.44, 89) # y-domain for monitor, slab or Si part height

# for contour plot need of a meshgrid
xx1, yy1 = np.meshgrid(x1, y)
xx2, yy2 = np.meshgrid(x2, y)

# for contour plot need of transposing the data, or x domain is in row, y in column for python it's opposite
# 2D contour

# m2 - power representation
contour_m2 = plt.pcolor(xx1, yy1, np.transpose(power_space_m2), cmap ='jet')
plt.title("Spatial power profile for central wg at interface WG/FPR1")
plt.xlabel(r'x-axis [$\mu$m]')
plt.ylabel(r'y-axis [$\mu$m]')
plt.colorbar()
plt.savefig('Spatial power profile for central wg after WG-FPR1 .png', bbox_inches = 'tight')
plt.show()

# m10 - power representation
contour_m10 = plt.pcolor(xx2, yy2, np.transpose(power_space_m10), cmap ='jet')
plt.title(r'Diffracting spatial power profile for central wg at 1[$\mu$m] in slab')
plt.xlabel(r'x-axis [$\mu$m]')
plt.ylabel(r'y-axis [$\mu$m]')
plt.colorbar()
plt.xlim([-1.5, 1.5])
plt.savefig('Diffracting spatial power profile for central wg at 1[um] in slab.png', bbox_inches = 'tight')
plt.show()

# below some plot can be design for more fancy graphs
"""
#2D fancy plot Imshow
z_min, z_max = np.min(power_space_m10), np.max(power_space_m10)

c = plt.imshow(np.transpose(power_space_m2), cmap ='jet', vmin = z_min, vmax = z_max,
                 extent =[x1.min(), x1.max(), y.min(), y.max()],
                    interpolation ='nearest', origin ='lower')
plt.colorbar(c)
plt.title('interface value', fontweight ="bold")
plt.show()
"""
# one way for a 3D plot (surface plot)
"""
#3D plot
fig = plt.figure()
ax4 = fig.add_subplot(111, projection='3d')
ax4.plot_surface(xx1, yy1, np.transpose(power_space_m2))

ax4.set_xlabel('X Label')
ax4.set_ylabel('Y Label')
ax4.set_zlabel('Z Label')

plt.show()
"""
# 3D surface plot improved

# m2
fig, ax4 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,8))
surf = ax4.plot_surface(xx1, yy1, np.transpose(power_space_m2), cmap=cm.coolwarm,
                       linewidth=2, antialiased=False)
fig.colorbar(surf, shrink=0.75, aspect=5)
plt.title("Spatial power profile for central wg at interface WG/FPR1")
ax4.set_xlabel('x-axis [$\mu$m]')
ax4.set_ylabel('y-axis [$\mu$m]')
ax4.set_zlabel('Amplitude')
ax4.dist = 11
plt.savefig('Surface plot of spatial power profile for central wg after WG-FPR1 .png', bbox_inches = 'tight')
plt.show()

# m10, enlarged view using figsize
fig, ax5 = plt.subplots(subplot_kw={"projection": "3d"},figsize=(10,8))
surf = ax5.plot_surface(xx2, yy2, np.transpose(power_space_m10), cmap=cm.coolwarm,
                       linewidth=2, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title("Spatial power profile for central wg at 1[$\mu$m] in slab")
ax5.set_xlabel('x-axis [$\mu$m]')
ax5.set_ylabel('y-axis [$\mu$m]')
ax5.set_zlabel('Amplitude')
ax5.dist = 11
plt.show()

"""
Fundamental mode initially launched in the first middle input wg

power_space_name_fund = ".\\power_results\\Mode430_sz.dat"
power_space_dim_fund = np.array([1131,921]) 
power_space_fund = file_read2(power_space_name_fund, power_space_dim_fund)#data for monitor2

#plot 
xi = np.linspace(-0.565, 0.565, 1131) # x-domain input wg
yi = np.linspace(-0.46, 0.46, 921) # y-domain for monitor, slab or Si part height

# for contour plot need of a meshgrid
xxi, yyi = np.meshgrid(xi, yi)

#fundamental mode power 

contour_fund = plt.pcolor(xxi, yyi, np.transpose(power_space_fund), cmap ='jet')
plt.colorbar()
plt.show()

p_fund = sum(sum(power_space_fund))*0.001**2

"""

"""
This part is dedicated to the fields computation at the interface .
The flower is launched only in z direction and TE is the Ex while TM is Hy
"""
ex_launch_name = ".\\power_results\\op2_m11_t310000000_ex.dat"
hy_launch_name = ".\\power_results\\op2_m11_t310000000_hy.dat"

ex_int_name = ".\\power_results\\op2_m2_t310000000_ex.dat"
hy_int_name = ".\\power_results\\op2_m2_t310000000_hy.dat"

ex_launch_dim  = np.array([201,89])
hy_launch_dim  = np.array([201,89])

ex_int_dim  = np.array([201,89])
hy_int_dim  = np.array([201,89])


ex_launch = file_read2(ex_launch_name, ex_launch_dim)
hy_launch = file_read2(hy_launch_name, hy_launch_dim)

ex_int = -file_read2(ex_int_name, ex_int_dim)
ex_int[:,0] = -ex_int[:,0]
hy_int = -file_read2(hy_int_name, hy_int_dim)
hy_int[:,0] = -hy_int[:,0]

# plot 
ehx1 = np.linspace(-1, 1, 201) # x-domain monitor 2 at taper output
ehy1 = np.linspace(-0.44, 0.44, 89) # y-domain for monitor, slab or Si part height

# for contour plot need of a meshgrid
ehxx1, ehyy1 = np.meshgrid(ehx1, ehy1)

# launched (input wg)
#ex
contour_ex_launch = plt.pcolor(ehxx1, ehyy1, np.transpose(ex_launch), cmap ='jet')
plt.title("Fundamental mode Ex profile at input wg")
plt.xlabel(r'x-axis [$\mu$m]')
plt.ylabel(r'y-axis [$\mu$m]')
plt.colorbar()
plt.savefig('Ex component of the device input fundamental mode.png', bbox_inches = 'tight')
plt.show()

#hy
contour_hy_launch = plt.pcolor(ehxx1, ehyy1, np.transpose(hy_launch), cmap ='jet')
plt.title("Fundamental mode Hy profile at input wg")
plt.xlabel(r'x-axis [$\mu$m]')
plt.ylabel(r'y-axis [$\mu$m]')
plt.colorbar()
plt.savefig('Hy component of the device input fundamental mode.png', bbox_inches = 'tight')
plt.show()

# interface
#ex
contour_ex_int = plt.pcolor(ehxx1, ehyy1, np.transpose(ex_int), cmap ='jet')
plt.title("Spatial profile of Ex for central wg at interface WG/FPR1")
plt.xlabel(r'x-axis [$\mu$m]')
plt.ylabel(r'y-axis [$\mu$m]')
plt.colorbar()
plt.savefig('Ex profile for central wg after WG-FPR1.png', bbox_inches = 'tight')
plt.show()

#hy
contour_hy_int = plt.pcolor(ehxx1, ehyy1, np.transpose(hy_int), cmap ='jet')
plt.title("Spatial profile of for Hy for central wg at interface WG/FPR1")
plt.xlabel(r'x-axis [$\mu$m]')
plt.ylabel(r'y-axis [$\mu$m]')
plt.colorbar()
plt.savefig('Hy profile for central wg after WG-FPR1.png', bbox_inches = 'tight')
plt.show()

"""
All the data have been acquired and put into vectors. The next part deals with
computation of what happens when the beams diffract in the slab.
"""
#First Fourier Transform implementation : Fresnel diffraction, simplified only 
#by considering the axis x1, horizontal along the FPR/WGs interface and 
#successive jumps of theta = Asin(idwg/R) where R is length of FPR regions
"""
wavelength = 1.55*10**-6 #in m
n_si = 3.47
n_eff = 2.304
wavelength_si = 1.55*10**-6/n_si
wavelength_eff = 1.55*10**-6/n_eff

dwg = 2*10**-6 #space between the wgs
n = 24 #number of wgs
wg_num = np.arange(0,n//2+1) #array representing the ith wg
r = 30*10**-6 # slab dimension in m

#function for computing the contant of the diffraction function
k = lambda theta: (1j/wavelength_eff)*((1+np.cos(theta))/2*r)*np.exp(-1j*(2*np.pi/wavelength_eff)*r)

#intergral of the power
phi_p = sum(sum(ex_int))*(0.02*0.01)

#computing the field but at the interface of the FPR/WGs
phi_m = np.zeros(shape=(n//2+1,1)) 
for i in wg_num:
    theta = np.arcsin(i*dwg/r)
    phi_m[i]=  np.abs(k(theta)*phi_p)


#plot of profile of the field at the interface FPR1/WGs
plt.plot(wg_num[:], phi_m[:], 'bo', label = "progfile at interface")
plt.title("Field profile at interface FPR1/AWs")
plt.xlabel('ith waveguide')
#plt.xlabel(r'[$\mu$m]')
plt.ylabel("Normalized Field value [.]")
ax6 = plt.gca() #reference to the current plot 
ax6.legend()
#plt.savefig('Total Power-Central WG power at interface WG/FPR1.png', bbox_inches = 'tight')
plt.show() # display plots

"""

#First Fourier Transform implementation : Fraunhofer diffraction, simplified
# as considering only the far field approximation.
#By considering the axis x1, horizontal along the FPR/WGs interface and 
# we have the profile along this interface.
#The value for the center of the wgs are then computed representing the average 
#value on the surface and used as constant in the illumination.

"""
All the data have been acquired and put into vectors. The next part deals with
computation of what happens when the beams diffract in the slab.
"""
# physical constants
wavelength = 1.55*10**-6 #in m

n_si = 3.47
n_eff = 2.304

wavelength_si = 1.55*10**-6/n_si
wavelength_eff = 1.55*10**-6/n_eff
kneff = 2*np.pi/wavelength_eff

# devices shapes and paramter values

r = 30*10**-6 # slab dimension in z dirention

dwg = 2*10**-6 #space between the wgs
n = 24 #number of wgs
nwg_val = n*10+1 # number of samples for field computation x-axis

# wg_num = np.arange(0,n//2+1) #array representing the ith wg
h = 0.11*10**-6 # semi-height of slab
nh = 10
nh_val = 2*nh+1 # number of samples for field computation y-axis

xp1 = np.linspace(-(n//2)*dwg, (n//2)*dwg, nwg_val) # x-domain at FPR1/AWs interface
yp1 = np.linspace(-h, h, nh_val) # y-domain at FPR1/AWs interface

dx1 = 0.01*10**-6 #discretization of input field x-axis
dy1 = 0.01*10**-6 #discretization of input field y-axis

# below, the fraunhofer diffraction function for computing resulting field
# fraunhofer(dwg, n, r, wavelength, k, nx, ny, dx, dy, nwg_val, nh_val, x, y, xp, yp, phi_m)

# resulting pattern for ex
exp1_complex = fraunhofer(dwg, n, r, wavelength_eff, kneff, 201, 89, dx1, dy1, nwg_val, nh_val, ehx1, ehy1, xp1, yp1, ex_int)
exp1 = np.abs(exp1_complex)

hyp1_complex = fraunhofer(dwg, n, r, wavelength_eff, kneff, 201, 89, dx1, dy1, nwg_val, nh_val, ehx1, ehy1, xp1, yp1, hy_int)
hyp1 = np.abs(hyp1_complex)

# plot 
# for contour plot need of a meshgrid
xx_inter, yy_inter = np.meshgrid(xp1, yp1)

# plot of the profile at the interface
#ex
contour_ex_fpr1 = plt.pcolor(xx_inter, yy_inter, np.transpose(exp1), cmap ='jet')
plt.title("Spatial profile of Ex in FPR1")
#plt.title("Spatial profile of ex at FPR1/AWs interface")
plt.xlabel(r'x [$\mu$m]')
plt.ylabel(r'y [$\mu$m]')
plt.colorbar()
plt.savefig('Diffraction pattern for Ex profile in FPR1.png', bbox_inches = 'tight')
plt.show()

#hy
contour_ex_fpr1 = plt.pcolor(xx_inter, yy_inter, np.transpose(hyp1), cmap ='jet')
plt.title("Spatial profile of Hy in FPR1")
#plt.title("Spatial profile of hy at FPR1/AWs interface")
plt.xlabel(r'x [$\mu$m]')
plt.ylabel(r'y [$\mu$m]')
plt.colorbar()
plt.savefig('Diffraction pattern for Hy profile in FPR1.png', bbox_inches = 'tight')
plt.show()


# plot of the x-axis profile at the interface
#ex
plt.plot(xp1, exp1[:,nh+1], 'r--', label = "interface")
plt.title("Ex along x-axis at input of FPR1/AW interface")
#plt.title("Spatial distribution of ex along x-axis at FPR1/AWs interface ")
plt.xlabel(r'x [$\mu$m]')
plt.ylabel("|Field value| [.]")
ax6 = plt.gca() # reference to the current plot 
ax6.legend()
plt.savefig("Ex along x-axis  at input of WG-FPR1 interface", bbox_inches = 'tight')
#plt.savefig('Power flow in the z direction for interface WG/FPR1.png', bbox_inches = 'tight')
plt.show() # display plots

#hy
plt.plot(xp1, hyp1[:,nh+1], 'r--', label = "interface")
plt.title("Hy along x-axis at input of FPR1/AW interface")
#plt.title("Spatial distribution of ex along x-axis at FPR1/AWs interface ")
plt.xlabel(r'x [$\mu$m]')
plt.ylabel("|Field value| [.]")
ax62 = plt.gca() # reference to the current plot 
ax62.legend()
plt.savefig("Hy along x-axis at input of WG-FPR1 interface", bbox_inches = 'tight')
#plt.savefig('Power flow in the z direction for interface WG/FPR1.png', bbox_inches = 'tight')
plt.show() # display plots

xp_mid1 = np.linspace(-(n//2)*dwg, (n//2)*dwg, n+1)

exp_mid1 = np.zeros(shape=(n+1, 1))
hyp_mid1 = np.zeros(shape=(n+1, 1))

for j in np.arange(n+1):
    exp_mid1[j] = exp1[j*10,nh+1]
    hyp_mid1[j] = hyp1[j*10,nh+1]

# plot of the central profile at the interface
#ex
plt.plot(xp1, exp1[:,nh+1], 'r--', label = "interface values")
plt.plot(xp_mid1 , exp_mid1[:], 'gH', label = "center wgs values")
plt.title("Ex along x-axis of WG/FPR1 interface")
#plt.title("Spatial distribution of ex along x-axis at FPR1/AWs interface ")
plt.xlabel(r' x [$\mu$m]')
plt.ylabel("|ex(x)| [.]")
ax7 = plt.gca() # reference to the current plot 
ax7.legend()
plt.savefig("Ex along x-axis of FPR1_AW interface", bbox_inches = 'tight')
#plt.savefig('Power flow in the z direction for interface WG/FPR1.png', bbox_inches = 'tight')
plt.show() # display plots

#hy
plt.plot(xp1, hyp1[:,nh+1], 'r--', label = "interface values")
plt.plot(xp_mid1 , hyp_mid1[:], 'gH', label = "center wgs values")
plt.title("Hy along x-axis of WG/FPR1 interface")
#plt.title("Spatial distribution of ex along x-axis at FPR1/AWs interface ")
plt.xlabel(r' x [$\mu$m]')
plt.ylabel("|hy(x)| [.]")
ax72 = plt.gca() #reference to the current plot 
ax72.legend()
plt.savefig("Hy along x-axis of FPR1-AW interface", bbox_inches = 'tight')
#plt.savefig('Power flow in the z direction for interface WG/FPR1.png', bbox_inches = 'tight')
plt.show() # display plots

def transit1():
    return exp_mid1, hyp_mid1

def transit2():
    return ex_int
