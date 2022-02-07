# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 07:00:09 2020

Python code for extracting CSV file data, output from Fullwave.
Initial simulation computed the response of a simple waveguide to a pulse.
Data are overlap integral values while the etching dimension is varied.

@author: Simon Santacatterina

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

"""
First part of the code taking as input the raw data file "results\waveguidepulsed.wmn"
(or other) containing "string" version of data. 

The file is read and processed line by line. Each string separated by a space
(" ") correspond to one value and is converted into type "float" values.
"""
data_overlap_mon1 = open('results\waveguidepulsed.wmn', 'r')
X = np.zeros(shape=(70,2))

with data_overlap_mon1 :
    index = 0
    line = data_overlap_mon1.readline()
    
    while line != '':  # the EOF char is an empty string
         V = line.split(" ")
         for i in range(len(V)):
            X[index,i] = float(V[i])
         #print(line, end='')
         line = data_overlap_mon1.readline()
         index = index + 1         
"""
Reverse the order of the data in the matrix.
"""
X = np.flip(X, axis = 0)
size_row = X[0,:].shape
size_col = X[:,0].shape
size_row = size_row[0] # number of elements per row (number of columns)
size_col = size_col[0] # number of elements per column (number of rows)

"""
Then, take the data in the C band i.e. between 1.30 and 1.65 micrometers.
"""
X2 = X[(1.3 < X[:,0]) & (X[:,0] < 1.65),0] # C band boundaries (1.30-1.65 micrometers)
lx2 = X2.shape 
lx2 = lx2[0] # number of data concerned (number of rows)

i2 = np.where((1.3 < X[:,0]) & (X[:,0] < 1.65)) # condition to get the indexes
i2 = i2[0] # corresponding indexes
i2len = len(i2) # length of index array

X3 = np.zeros(shape=(lx2,size_row)) # X3 is the C band values matrix
X3[:,0] = X2 
X3[:,1:size_row] = X[range(i2[0], i2[i2len-1]+1),1:size_row]
"""
Max of each vector.
"""
ymaximum = [0 for x in range(size_row)]
xmaximum = [0 for x in range(size_row)]
argmaximum  = [0 for x in range(size_row)]

ymaximum3 = [0 for x in range(size_row)]
xmaximum3 = [0 for x in range(size_row)]
argmaximum3  = [0 for x in range(size_row)]

"""
Delete the outliers, i.e. the values being aberrations due to the calculations
algorthim.
"""
n_outliers = 20
for i in range(1,size_row): # for all the columns
    for j in range(0,n_outliers): # only for the first n_outliers rows
        if X[j,i] > 0.175:
            X[j,i] = 0
# The maximum is finally computed among the acceptable values and is casted into
# the vector called ymaximum, then the corresponding indexes (x axis values) are
# casted into the xmaximum value
for k in range(1,size_row):
    ymaximum[k] = np.amax(X[:,k])
    argmaximum[k] = np.argmax(X[:,k])
    xmaximum[k] = X[argmaximum[k],0]
    
    ymaximum3[k] = np.amax(X3[:,k])
    argmaximum3[k] = np.argmax(X3[:,k])
    xmaximum3[k] = X3[argmaximum3[k],0]

"""
The value of each C band vector at 1.55 micrometer.
"""
lambda_index = np.where(X3[:,0] == 1.55)[0]
lambda_val = X3[lambda_index,0]
xlambda_val = np.ones(1)*lambda_val
ylambda_val = X3[lambda_index,1]

"""
Spline interpolation of the data to increase the amount of point.
"""
def turn_spline(X3,s):
    
    xnew = np.linspace(X3[:,0].min(), X3[:,0].max(), 300) 
    sp = make_interp_spline(X3[:,0], X3[:,s], k=3)  # type: BSpline
    power = sp(xnew)
    return xnew, power

"""
Plot only some of the values, to better see the impact of the etching.
"""
vect_plot = [1]

dev1 = turn_spline(X3,vect_plot[0]) # dsp means "data subplot"

plt.plot(dev1[0], dev1[1],  label = "Si Waveguide")
plt.plot(xlambda_val[0], ylambda_val[0], 'p', color = (1,0,0), label = "value = "+str(ylambda_val[0]) )

current_handles, current_labels = plt.gca().get_legend_handles_labels()
reversed_handles = list(reversed(current_handles))
reversed_labels = list(reversed(current_labels))

plt.title("Overlap integral for the Si waveguide - C band") # C band plot
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel("|Overlap intergral| [.]")
#plot_namesb =  [Wetch_str[0], Wetch_str[1]]
plt.legend(reversed_handles,reversed_labels)
#plt.legend(reversed_handles,reversed_labels,loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('waveguide wavelength analysis.png', bbox_inches = 'tight')
plt.show()
