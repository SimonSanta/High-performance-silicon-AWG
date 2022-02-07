# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 03:41:49 2021

Python code used for extracting data from CSV file, taken from Fullwave.
The data handled here simulated the behavior of a slab array interface with 
etching in the slab.

Data are overlap integral values paramaters of the interface are varied. 
Those parameters are L, Lrib,  d, Wrib, Detch, Letch, Wetch.

@author: Simon Santacatterina

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from functions import turn_spline
"""
First part of the code taken as input the file containing raw data and
extracting those data in order to plot them. 

The input is the data file "results\Wetch_scan1_fw_mon_1_overlap.dat"
(or other) containing "string" version of the data. 

The file is read line by line. Then, each line is processed such that
each string separated by " " (a space) correspond to one data and is converted
into a value of type "float".
"""
#first simulation
#etch_init = 0.1 #initial value of the etching width
#etch_final = 1.3 #final value of the etching width
#n_scan = 11 #number of value taken by the parameter Wetch

#fourth simulation
etch_init = 0.1 #initial value of the etching width
etch_final = 1.3 #final value of the etching width
n_scan = 11 #number of value taken by the parameter Wetch

Wetch = np.linspace(etch_init, etch_final, n_scan) # the vector of value taken by the parameter Wetch
Wetch = np.round(Wetch,2) #rounded to a two digit precision as in Rsoft
Wetch_str = []
#Wetch = [0.1, 0.29, 0.48, 0.67, 0.86, 1.05, 1.24, 1.43, 1.62, 1.81, 2]
#Wetch_str = ['W = 0.1', 'W = 0.29', 'W = 0.48', 'W = 0.67', 'W = 0.86', 'W = 1.05', 'W = 1.24', 'W = 1.43', 'W = 1.62', 'W = 1.81', 'W = 2']
for s in range(1,n_scan):
    Wetch_str.append('W = ' +str(Wetch[s-1])) #loop for the etching names in the legends

#from this point, the particular file to analyze is being read

L01um08um_dwg20um_wmid10um_all = open('..\\rsoft_results\\L01um08um_dwg20um_wmid10um_work\\results\L01um08um_dwg20um_wmid10um_fw_mon_all_last.dat', 'r')
L1 = np.zeros(shape=(12,5)) #number of lines and columns in the file.

L01um08um_dwg20um_wmid08um_all = open('..\\rsoft_results\\L01um12um_dwg20um_wmid08um_work\\results\L01um12um_dwg20um_wmid08um_fw_mon_all_last.dat', 'r')
L2 = np.zeros(shape=(12,5)) #number of lines and columns in the file.

"""
L1 
"""
with L01um08um_dwg20um_wmid10um_all :
    index = 0
    line = L01um08um_dwg20um_wmid10um_all.readline()
    
    while line != '':  # The EOF char is an empty string
         V = line.split(" ")
         for i in range(len(V)-1):
            L1[index,i] = float(V[i])
         #print(line, end='')
         line = L01um08um_dwg20um_wmid10um_all.readline()
         index = index + 1         
"""
Reverse the order of the data in the matrix
"""
#L1 = np.flip(L1, axis = 0)
#The dimensions of the matrix are computed
size_row1 = L1[0,:].shape
size_col1 = L1[:,0].shape
size_row1 = size_row1[0] #The number of elements per row (number of columns)
size_col1 = size_col1[0] #The number of elements per column (number of rows)

"""
L2
"""
with L01um08um_dwg20um_wmid08um_all :
    index = 0
    line = L01um08um_dwg20um_wmid08um_all.readline()
    
    while line != '':  # The EOF char is an empty string
         V = line.split(" ")
         for i in range(len(V)-1):
            L2[index,i] = float(V[i])
         #print(line, end='')
         line = L01um08um_dwg20um_wmid08um_all.readline()
         index = index + 1         
"""
Reverse the order of the data in the matrix
"""
#L2 = np.flip(L2, axis = 0)
#The dimensions of the matrix are computed
size_row2 = L2[0,:].shape
size_col2 = L2[:,0].shape
size_row2 = size_row2[0] #The number of elements per row (number of columns)
size_col2 = size_col2[0] #The number of elements per column (number of rows)


"""
The max of each vector
"""
ymaximum1 = [0 for x in range(size_row1)] #using loops
xmaximum1 = [0 for x in range(size_row1)]
argmaximum1  = [0 for x in range(size_row1)]

ymaximum2 = [0 for x in range(size_row2)] #using loops
xmaximum2 = [0 for x in range(size_row2)]
argmaximum2  = [0 for x in range(size_row2)]

for k1 in range(1,size_row1):
    ymaximum1[k1] = np.amax(L1[:,k1])
    argmaximum1[k1] = np.argmax(L1[:,k1])
    xmaximum1[k1] = L1[argmaximum1[k1],0]
    

for k2 in range(1,size_row2):
    ymaximum2[k2] = np.amax(L2[:,k2])
    argmaximum2[k2] = np.argmax(L2[:,k2])
    xmaximum2[k2] = L2[argmaximum1[k2],0]

L1[1:size_row1,[2,4]] = np.sqrt(L1[1:size_row1,[2,4]])
L2[1:size_row2,[2,4]] = np.sqrt(L2[1:size_row2,[2,4]])

"""
Second part of the code. Below, the data previously extracted data are ploted. 
The first part is dedicated to the raw data of the global spectrum while the 
second part refers to the C band data only. The C band data are displayed after
a Bspline interpolation.
"""

figure1 = plt.figure(figsize=(14,7)) #fonction to modify the size of the plots
n = 5
for j in  [2,4]:    
#    xnew = np.linspace(X[:,0].min(), X[:,0].max(), 300) 
#    spl = make_interp_spline(X[:,0], X[:,j], k=3)  # type: BSpline
#    power_smooth = spl(xnew)
    dsp1 = turn_spline(L1,j)
    #dsp2 = turn_spline(L2,j)
    plt.plot(dsp1[0],dsp1[1]) #assign the names and links
    #plt.plot(dsp2[0],dsp2[1])
    #plt.plot(X[:,0], X[:,j])
    

plt.title("Overlap integral as a variation of the taper length, Wmid = 1um, Linear",  size= 20)
plt.xlabel(r'Length of the taper [$\mu$m]', size= 20)
plt.ylabel("|Overlap intergral| [.]", size= 20)
plt.xticks(size = 20)
plt.yticks(size = 20)
ax1 = plt.gca() #reference to the current plot 
ax1.legend()
plt.savefig('Overlap integral as a variation of the length for W = 1um.png', bbox_inches = 'tight')
plt.show() # display plots

figure2 = plt.figure(figsize=(14,7)) #fonction to modify the size of the plots
n = 5
for j in  [2,4]:    
#    xnew = np.linspace(X[:,0].min(), X[:,0].max(), 300) 
#    spl = make_interp_spline(X[:,0], X[:,j], k=3)  # type: BSpline
#    power_smooth = spl(xnew)
    dsp2 = turn_spline(L2,j)
    plt.plot(dsp2[0],dsp2[1])
    #plt.plot(X[:,0], X[:,j])
    

plt.title("Overlap integral as a variation of the taper length, Wmid = 0.8um, Linear", size= 20)
plt.xlabel(r'Length of the taper [$\mu$m]',size= 20)
plt.ylabel("|Overlap intergral| [.]", size= 20)
plt.xticks(size = 20)
plt.yticks(size = 20)
ax1 = plt.gca() #reference to the current plot 
ax1.legend()
plt.savefig('Overlap integral as a variation of the length for W = 0.8um.png', bbox_inches = 'tight')
plt.show() # display plots


"""
Parabolic taper
"""
#L2um4um_d011_dwg20um_wmid10um = open('..\\rsoft_results\\L2um4um_d011_dwg20um_wmid10um_parabolic_work\\results\L2um4um_d011_dwg20um_wmid10um_parabolic_fw_mon_1_overlap_last.dat', 'r')

"""
Variation of L and d
"""

d_init = 0.1 #initial value of the etching width
d_final = 1 #final value of the etching width
n_scan = 11 #number of value taken by the parameter Wetch

dscan = np.linspace(d_init, d_final, n_scan-1) # the vector of value taken by the parameter Wetch
dscan = np.round(dscan,2) #rounded to a two digit precision as in Rsoft
d_str = []
#Wetch = [0.1, 0.29, 0.48, 0.67, 0.86, 1.05, 1.24, 1.43, 1.62, 1.81, 2]
#Wetch_str = ['W = 0.1', 'W = 0.29', 'W = 0.48', 'W = 0.67', 'W = 0.86', 'W = 1.05', 'W = 1.24', 'W = 1.43', 'W = 1.62', 'W = 1.81', 'W = 2']
for s in range(1,n_scan):
    d_str.append('d = ' +str(dscan[s-1])) #loop for the etching names in the legends

#from this point, the particular file to analyze is being read

L2um4um_d011_dwg20um_wmid10um = open('..\\rsoft_results\\L2um4um_d011_dwg20um_wmid10um_parabolic_work\\results\L2um4um_d011_dwg20um_wmid10um_parabolic_fw_mon_1_overlap_last.dat', 'r')

X = np.zeros(shape=(3,12)) #number of lines and columns in the file. # third simulation from 0.3 to 0.9

with L2um4um_d011_dwg20um_wmid10um :
    index = 0
    line = L2um4um_d011_dwg20um_wmid10um.readline()
    
    while line != '':  # The EOF char is an empty string
         V = line.split(" ")
         for i in range(len(V)-1):
            X[index,i] = float(V[i])
         #print(line, end='')
         line = L2um4um_d011_dwg20um_wmid10um.readline()
         index = index + 1         
"""
Reverse the order of the data in the matrix
"""
#X = np.flip(X, axis = 0)
#The dimensions of the matrix are computed
size_row = X[0,:].shape
size_col = X[:,0].shape
size_row = size_row[0] #The number of elements per row (number of columns)
size_col = size_col[0] #The number of elements per column (number of rows)

"""
The max of each vector
"""
ymaximum = [0 for x in range(size_row)] #using loops
xmaximum = [0 for x in range(size_row)]
argmaximum  = [0 for x in range(size_row)]

#ymaximum3 = [0 for x in range(size_row)] #using loops
#xmaximum3 = [0 for x in range(size_row)]
#argmaximum3  = [0 for x in range(size_row)]

X[:,1:size_row] = np.sqrt(X[:,1:size_row])
#X3[:,1:size_row] = np.sqrt(X3[:,1:size_row])

"""
Delete the outliers, i.e. the values being aberrations due to the calculations
algorthim
"""
"""
n_outliers = 20
for i in range(1,size_row): #for all the columns
    for j in range(0,n_outliers): #only for the first n_outliers rows
        if X[j,i] > 0.3:
            X[j,i] = 0
"""          
#The maximum is finally computed among the acceptable values and is casted into
#the vector called ymaximum, then the corresponding indexes (x axis values) are
# casted into the xmaximum value
            
for k in range(1,size_row):
    ymaximum[k] = np.amax(X[:,k])
    argmaximum[k] = np.argmax(X[:,k])
    xmaximum[k] = X[argmaximum[k],0]
    
    #ymaximum3[k] = np.amax(X3[:,k])
    #argmaximum3[k] = np.argmax(X3[:,k])
    #xmaximum3[k] = X3[argmaximum3[k],0]

"""
The value of each C band vector at 1.55 micrometer.
"""
"""
lambda_index = np.where(X3[:,0] == 1.55)[0]
lambda_val = X3[lambda_index, 0:size_row]
xlambda_val = np.ones(size_row-2)*lambda_val[0,0]
ylambda_val = lambda_val[0,1:size_row-1]
"""
"""
Second part of the code. Below, the data previously extracted data are ploted. 
The first part is dedicated to the raw data of the global spectrum while the 
second part refers to the C band data only. The C band data are displayed after
a Bspline interpolation.
"""

#start of the first part

d_tab ={} # tab containing the plot objects and their names "W_i"
#we will assign the names and links them to the plots int the tab
#wetch_tab3 ={} #for the C band, same principle but finally not used

figure1 = plt.figure(figsize=(14,7)) #fonction to modify the size of the plots

#this curve can potentially be interpolated using Bsplines
for j in range(1,n_scan):    
#    xnew = np.linspace(X[:,0].min(), X[:,0].max(), 300) 
#    spl = make_interp_spline(X[:,0], X[:,j], k=3)  # type: BSpline
#    power_smooth = spl(xnew)
    dsp3 = turn_spline(X,j)
    #dsp2 = turn_spline(L2,j)
    
    d_tab["d_" +str(j)+" = "+str(dscan[j-1])] = plt.plot(dsp3[0],dsp3[1]) #assign the names and links
    #plt.plot(X[:,0], X[:,j])
    
plt.title("Overlap integral as a variation of the taper length and the curve of the taper's parabola",size= 18)
plt.xlabel(r'Length of the taper [$\mu$m]', size= 20)
plt.ylabel("|Overlap intergral| [.]", size= 20)
plt.xticks(size = 20)
plt.yticks(size = 20)
ax1 = plt.gca() #reference to the current plot 
plot_names = d_tab.keys()
ax1.legend(plot_names)
plt.savefig('Overlap integral as a variation of the taper length and the curve of the tapers parabola.png', bbox_inches = 'tight')
plt.show() # display plots

#Zoom on a certain range of the domain for both axis
"""
figurezoom = plt.figure(figsize=(20,10))
plt.plot(X[:,0], X[:,1:n_scan])
ax1zoom = plt.gca() #reference to the current plot 
plot_names = wetch_tab.keys()
ax1zoom.legend(plot_names)
ax1zoom.axis([1,5,0,0.3])

plt.title("Overlap integral as a variation of the etching width - zoomed in")
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel("|Overlap intergral| [.]")
plt.savefig('Overlap integral as a variation of the etching width - zoomed in.png', bbox_inches = 'tight')
plt.show() # display plots
"""

"""
Parabolic taper, optimized d and variation of the length, highest acuracy
"""

LP02um05um_dwg20um_wmid10um_all = open('..\\rsoft_results\\L2um5um_d01_dwg20um_wmid10um_parabolic_work\\results\L2um5um_d01_dwg20um_wmid10um_parabolic_fw_mon_all_last.dat', 'r')
LP1 = np.zeros(shape=(13,7)) #number of lines and columns in the file.

LP02um05um_dwg20um_wmid08um_all = open('..\\rsoft_results\\L2um5um_d01_dwg20um_wmid08um_parabolic_work\\results\L2um5um_d01_dwg20um_wmid08um_parabolic_fw_mon_all_last.dat', 'r')
LP2 = np.zeros(shape=(13,7)) #number of lines and columns in the file.

"""
LP1 
"""
with LP02um05um_dwg20um_wmid10um_all :
    index = 0
    line = LP02um05um_dwg20um_wmid10um_all.readline()
    
    while line != '':  # The EOF char is an empty string
         V = line.split(" ")
         for i in range(len(V)-1):
            LP1[index,i] = float(V[i])
         #print(line, end='')
         line = LP02um05um_dwg20um_wmid10um_all.readline()
         index = index + 1         
"""
Reverse the order of the data in the matrix
"""
#L1 = np.flip(L1, axis = 0)
#The dimensions of the matrix are computed
size_rowP1 = LP1[0,:].shape
size_colP1 = LP1[:,0].shape
size_rowP1 = size_rowP1[0] #The number of elements per row (number of columns)
size_colP1 = size_colP1[0] #The number of elements per column (number of rows)

"""
LP2
"""
with LP02um05um_dwg20um_wmid08um_all :
    index = 0
    line = LP02um05um_dwg20um_wmid08um_all.readline()
    
    while line != '':  # The EOF char is an empty string
         V = line.split(" ")
         for i in range(len(V)-1):
            LP2[index,i] = float(V[i])
         #print(line, end='')
         line = LP02um05um_dwg20um_wmid08um_all.readline()
         index = index + 1         
"""
Reverse the order of the data in the matrix
"""
#L2 = np.flip(L2, axis = 0)
#The dimensions of the matrix are computed
size_rowP2 = LP2[0,:].shape
size_colP2 = LP2[:,0].shape
size_rowP2 = size_rowP2[0] #The number of elements per row (number of columns)
size_colP2 = size_colP2[0] #The number of elements per column (number of rows)


"""
The max of each vector
"""
ymaximumP1 = [0 for x in range(size_rowP1)] #using loops
xmaximumP1 = [0 for x in range(size_rowP1)]
argmaximumP1  = [0 for x in range(size_rowP1)]

ymaximumP2 = [0 for x in range(size_rowP2)] #using loops
xmaximumP2 = [0 for x in range(size_rowP2)]
argmaximumP2  = [0 for x in range(size_rowP2)]

for k1 in range(1,size_rowP1):
    ymaximumP1[k1] = np.amax(LP1[:,k1])
    argmaximumP1[k1] = np.argmax(LP1[:,k1])
    xmaximumP1[k1] = LP1[argmaximumP1[k1],0]
    

for k2 in range(1,size_rowP2):
    ymaximumP2[k2] = np.amax(LP2[:,k2])
    argmaximumP2[k2] = np.argmax(LP2[:,k2])
    xmaximumP2[k2] = LP2[argmaximumP1[k2],0]

LP1[1:size_rowP1,[2,4]] = np.sqrt(LP1[1:size_rowP1,[2,4]])
LP2[1:size_rowP2,[2,4]] = np.sqrt(LP2[1:size_rowP2,[2,4]])

"""
Second part of the code. Below, the data previously extracted data are ploted. 
The first part is dedicated to the raw data of the global spectrum while the 
second part refers to the C band data only. The C band data are displayed after
a Bspline interpolation.
"""

figureP1 = plt.figure(figsize=(14,7)) #fonction to modify the size of the plots
n = 5
for j in  [2,4]:    
#    xnew = np.linspace(X[:,0].min(), X[:,0].max(), 300) 
#    spl = make_interp_spline(X[:,0], X[:,j], k=3)  # type: BSpline
#    power_smooth = spl(xnew)
    dspP1 = turn_spline(LP1,j)
    #dsp2 = turn_spline(L2,j)
    plt.plot(dspP1[0],dspP1[1]) #assign the names and links
    #plt.plot(dsp2[0],dsp2[1])
    #plt.plot(X[:,0], X[:,j])
    

plt.title("Overlap integral as a variation of the taper length, Wmid = 1um, Parabola", size= 20)
plt.xlabel(r'Length of the taper [$\mu$m]', size= 20)
plt.ylabel("|Overlap intergral| [.]", size= 20)
plt.xticks(size = 20)
plt.yticks(size = 20)
ax1 = plt.gca() #reference to the current plot 
ax1.legend()
plt.savefig('Overlap integral as a variation of the length for W = 1um, parabola.png', bbox_inches = 'tight')
plt.show() # display plots

figureP2 = plt.figure(figsize=(14,7)) #fonction to modify the size of the plots
n = 5
for j in  [2,4]:    
#    xnew = np.linspace(X[:,0].min(), X[:,0].max(), 300) 
#    spl = make_interp_spline(X[:,0], X[:,j], k=3)  # type: BSpline
#    power_smooth = spl(xnew)
    dspP2 = turn_spline(LP2,j)
    plt.plot(dspP2[0],dspP2[1])
    #plt.plot(X[:,0], X[:,j])
    

plt.title("Overlap integral as a variation of the taper length, Wmid = 0.8um, Parabola", size= 20)
plt.xlabel(r'Length of the taper [$\mu$m]', size= 20)
plt.ylabel("|Overlap intergral| [.]", size= 20)
plt.xticks(size = 20)
plt.yticks(size = 20)
ax1 = plt.gca() #reference to the current plot 
ax1.legend()
plt.savefig('Overlap integral as a variation of the length for W = 0.8um, parabola.png', bbox_inches = 'tight')
plt.show() # display plots

"""
Final desgin with 2 stages : the linear case for with W = 0.8 um and W = 1um and
the parabola case with W = 0.8 um and W = 1um.
The parameter scanned is this time Lrib.

Linear
"""

Lrib_2um7um_Wmid10um_Wrib16um_a0344_all = open('..\\rsoft_results\\Lrib_2um7um_Wmid10um_Wrib16um_a0344_work\\results\Lrib_2um7um_Wmid10um_Wrib16um_a0344_fw_mon_all_last.dat', 'r')
Lriblin1 = np.zeros(shape=(11,7)) #number of lines and columns in the file.

Lrib_0um8um_Wmid08um_Wrib16um_a035_last = open('..\\rsoft_results\\Lrib_0um8um_Wmid08um_Wrib16um_a035_work\\results\Lrib_0um8um_Wmid08um_Wrib16um_a035_fw_mon_all_last.dat', 'r')
Lriblin2 = np.zeros(shape=(17,5)) #number of lines and columns in the file.

"""
Lriblin1 
"""
with Lrib_2um7um_Wmid10um_Wrib16um_a0344_all :
    index = 0
    line = Lrib_2um7um_Wmid10um_Wrib16um_a0344_all.readline()
    
    while line != '':  # The EOF char is an empty string
         V = line.split(" ")
         for i in range(len(V)-1):
            Lriblin1[index,i] = float(V[i])
         #print(line, end='')
         line = Lrib_2um7um_Wmid10um_Wrib16um_a0344_all.readline()
         index = index + 1         
"""
Reverse the order of the data in the matrix
"""
#L1 = np.flip(L1, axis = 0)
#The dimensions of the matrix are computed
size_row_riblin1 = Lriblin1[0,:].shape
size_col_riblin1 = Lriblin1[:,0].shape
size_row_riblin1  = size_row_riblin1[0] #The number of elements per row (number of columns)
size_col_riblin1 = size_col_riblin1[0] #The number of elements per column (number of rows)

"""
Lriblin2 
"""
with Lrib_0um8um_Wmid08um_Wrib16um_a035_last :
    index = 0
    line = Lrib_0um8um_Wmid08um_Wrib16um_a035_last.readline()
    
    while line != '':  # The EOF char is an empty string
         V = line.split(" ")
         for i in range(len(V)-1):
            Lriblin2[index,i] = float(V[i])
         #print(line, end='')
         line = Lrib_0um8um_Wmid08um_Wrib16um_a035_last.readline()
         index = index + 1         
"""
Reverse the order of the data in the matrix
"""
#L1 = np.flip(L1, axis = 0)
#The dimensions of the matrix are computed
size_row_riblin2 = Lriblin2[0,:].shape
size_col_riblin2 = Lriblin2[:,0].shape
size_row_riblin2  = size_row_riblin2[0] #The number of elements per row (number of columns)
size_col_riblin2 = size_col_riblin2[0] #The number of elements per column (number of rows)

"""
The max of each vector
"""
ymaximumriblin1 = [0 for x in range(size_row_riblin1)] #using loops
xmaximumriblin1 = [0 for x in range(size_row_riblin1)]
argmaximumriblin1  = [0 for x in range(size_row_riblin1)]

ymaximumriblin2 = [0 for x in range(size_row_riblin2)] #using loops
xmaximumriblin2 = [0 for x in range(size_row_riblin2)]
argmaximumriblin2  = [0 for x in range(size_row_riblin2)]

for k1 in range(1,size_row_riblin1):
    ymaximumriblin1[k1] = np.amax(Lriblin1[:,k1])
    argmaximumriblin1[k1] = np.argmax(Lriblin1[:,k1])
    xmaximumriblin1[k1] = Lriblin1[argmaximumriblin1[k1],0]
    

for k2 in range(1,size_row_riblin2):
    ymaximumriblin2[k2] = np.amax(Lriblin2[:,k2])
    argmaximumriblin2[k2] = np.argmax(Lriblin2[:,k2])
    xmaximumriblin2[k2] = Lriblin2[argmaximumriblin2[k2],0]

Lriblin1[1:size_row_riblin1,[2,4]] = np.sqrt(Lriblin1[1:size_row_riblin1,[2,4]])
Lriblin2[1:size_row_riblin2,[2,4]] = np.sqrt(Lriblin2[1:size_row_riblin2,[2,4]])

"""
Second part of the code. Below, the data previously extracted data are ploted. 
The first part is dedicated to the raw data of the global spectrum while the 
second part refers to the C band data only. The C band data are displayed after
a Bspline interpolation.
"""

figureriblin1 = plt.figure(figsize=(14,7)) #fonction to modify the size of the plots
n = 5
for j in  [2,4]:    
#    xnew = np.linspace(X[:,0].min(), X[:,0].max(), 300) 
#    spl = make_interp_spline(X[:,0], X[:,j], k=3)  # type: BSpline
#    power_smooth = spl(xnew)
    dspriblin1 = turn_spline(Lriblin1,j)
    #dsp2 = turn_spline(L2,j)
    plt.plot(dspriblin1[0],dspriblin1[1]) #assign the names and links
    #plt.plot(dsp2[0],dsp2[1])
    #plt.plot(X[:,0], X[:,j])
    

plt.title("Overlap integral as a variation of the Rib length, Wmid = 1um, Linear", size= 20)
plt.xlabel(r'Length of the Rib waveguide  [$\mu$m]', size= 20)
plt.ylabel("|Overlap intergral| [.]", size= 20)
plt.xticks(size = 20)
plt.yticks(size = 20)
ax1 = plt.gca() #reference to the current plot 
ax1.legend()
plt.savefig('Overlap integral as a variation of theRib length, Wmid = 1um, Linear.png', bbox_inches = 'tight')
plt.show() # display plots

figureriblin2 = plt.figure(figsize=(14,7)) #fonction to modify the size of the plots
n = 5
for j in  [2,4]:    
#    xnew = np.linspace(X[:,0].min(), X[:,0].max(), 300) 
#    spl = make_interp_spline(X[:,0], X[:,j], k=3)  # type: BSpline
#    power_smooth = spl(xnew)
    dspriblin2 = turn_spline(Lriblin2,j)
    plt.plot(dspriblin2[0],dspriblin2[1])
    #plt.plot(X[:,0], X[:,j])
    

plt.title("Overlap integral as a variation of the Rib length, Wmid = 0.8um, Linear", size= 20)
plt.xlabel(r'Length of the Rib waveguide [$\mu$m]', size= 20)
plt.ylabel("|Overlap intergral| [.]", size= 20)
plt.xticks(size = 20)
plt.yticks(size = 20)
ax1 = plt.gca() #reference to the current plot 
ax1.legend()
plt.savefig('Overlap integral as a variation of the Rib length, Wmid = 0.8um, Linear.png', bbox_inches = 'tight')
plt.show() # display plots

"""
Final design with 2 stages : the linear case for with W = 0.8 um and W = 1um and
the parabola case with W = 0.8 um and W = 1um.
The parameter scanned is this time Lrib.

Parabolic
"""

Lrib_01um08um_d01_Wmid10um_Wrib16um_a035_all = open('..\\rsoft_results\\Lrib_01um08um_d01_Wmid10um_Wrib16um_a035_work\\results\Lrib_01um08um_d01_Wmid10um_Wrib16um_a035_fw_mon_all_last.dat', 'r')
Lribpara1 = np.zeros(shape=(15,5)) #number of lines and columns in the file.

Lrib_0um8um_Wmid08um_Wrib16um_a035_all = open('..\\rsoft_results\\Lrib_01um08um_d01_Wmid08um_Wrib16um_a0287_work\\results\Lrib_01um08um_d01_Wmid08um_Wrib16um_a0287_fw_mon_all_last.dat', 'r')
Lribpara2 = np.zeros(shape=(15,5)) #number of lines and columns in the file.

"""
Lribpara1 
"""
with Lrib_01um08um_d01_Wmid10um_Wrib16um_a035_all :
    index = 0
    line = Lrib_01um08um_d01_Wmid10um_Wrib16um_a035_all.readline()
    
    while line != '':  # The EOF char is an empty string
         V = line.split(" ")
         for i in range(len(V)-1):
            Lribpara1[index,i] = float(V[i])
         #print(line, end='')
         line = Lrib_01um08um_d01_Wmid10um_Wrib16um_a035_all.readline()
         index = index + 1         
"""
Reverse the order of the data in the matrix
"""
#L1 = np.flip(L1, axis = 0)
#The dimensions of the matrix are computed
size_row_ribpara1 = Lribpara1[0,:].shape
size_col_ribpara1 = Lribpara1[:,0].shape
size_row_ribpara1 = size_row_ribpara1[0] #The number of elements per row (number of columns)
size_col_ribpara1 = size_col_ribpara1[0] #The number of elements per column (number of rows)

"""
Lribpara2 
"""
with Lrib_0um8um_Wmid08um_Wrib16um_a035_all :
    index = 0
    line = Lrib_0um8um_Wmid08um_Wrib16um_a035_all.readline()
    
    while line != '':  # The EOF char is an empty string
         V = line.split(" ")
         for i in range(len(V)-1):
            Lribpara2[index,i] = float(V[i])
         #print(line, end='')
         line = Lrib_0um8um_Wmid08um_Wrib16um_a035_all.readline()
         index = index + 1         
"""
Reverse the order of the data in the matrix
"""
#L1 = np.flip(L1, axis = 0)
#The dimensions of the matrix are computed
size_row_ribpara2 = Lribpara2[0,:].shape
size_col_ribpara2 = Lribpara2[:,0].shape
size_row_ribpara2 = size_row_ribpara2[0] #The number of elements per row (number of columns)
size_col_ribpara2 = size_col_ribpara2[0] #The number of elements per column (number of rows)

"""
The max of each vector
"""
ymaximumribpara1 = [0 for x in range(size_row_ribpara1)] #using loops
xmaximumribpara1 = [0 for x in range(size_row_ribpara1)]
argmaximumribpara1  = [0 for x in range(size_row_ribpara1)]

ymaximumribpara2 = [0 for x in range(size_row_ribpara2)] #using loops
xmaximumribpara2 = [0 for x in range(size_row_ribpara2)]
argmaximumribpara2  = [0 for x in range(size_row_ribpara2)]

for k1 in range(1,size_row_ribpara1):
    ymaximumribpara1[k1] = np.amax(Lribpara1[:,k1])
    argmaximumribpara1[k1] = np.argmax(Lribpara1[:,k1])
    xmaximumribpara1[k1] = Lribpara1[argmaximumriblin1[k1],0]
    

for k2 in range(1,size_row_ribpara2):
    ymaximumribpara2[k2] = np.amax(Lribpara2[:,k2])
    argmaximumribpara2[k2] = np.argmax(Lribpara2[:,k2])
    xmaximumribpara2[k2] = Lriblin2[argmaximumriblin2[k2],0]

Lribpara1[1:size_row_ribpara1,[2,4]] = np.sqrt(Lribpara1[1:size_row_ribpara1,[2,4]])
Lribpara2[1:size_row_ribpara2,[2,4]] = np.sqrt(Lribpara2[1:size_row_ribpara2,[2,4]])

"""
Second part of the code. Below, the data previously extracted data are ploted. 
The first part is dedicated to the raw data of the global spectrum while the 
second part refers to the C band data only. The C band data are displayed after
a Bspline interpolation.
"""

figureribpara1 = plt.figure(figsize=(14,7)) #fonction to modify the size of the plots
n = 5
for j in  [2,4]:    
#    xnew = np.linspace(X[:,0].min(), X[:,0].max(), 300) 
#    spl = make_interp_spline(X[:,0], X[:,j], k=3)  # type: BSpline
#    power_smooth = spl(xnew)
    dspribpara1 = turn_spline(Lribpara1,j)
    #dsp2 = turn_spline(L2,j)
    plt.plot(dspribpara1[0],dspribpara1[1]) #assign the names and links
    #plt.plot(dsp2[0],dsp2[1])
    #plt.plot(X[:,0], X[:,j])
    

plt.title("Overlap integral as a variation of the Rib length, Wmid = 1um, Parabolic", size= 20)
plt.xlabel(r'Length of the Rib waveguide  [$\mu$m]', size= 20)
plt.ylabel("|Overlap intergral| [.]", size= 20)
plt.xticks(size = 20)
plt.yticks(size = 20)
ax1 = plt.gca() #reference to the current plot 
ax1.legend()
plt.savefig('Overlap integral as a variation of the Rib length, Wmid = 1um, parabolic.png', bbox_inches = 'tight')
plt.show() # display plots

figureribpara2 = plt.figure(figsize=(14,7)) #fonction to modify the size of the plots
n = 5
#for j in range(1,n): 
for j in [2,4]:    
#    xnew = np.linspace(X[:,0].min(), X[:,0].max(), 300) 
#    spl = make_interp_spline(X[:,0], X[:,j], k=3)  # type: BSpline
#    power_smooth = spl(xnew)
    dspribpara2 = turn_spline(Lribpara2,j)
    plt.plot(dspribpara2[0],dspribpara2[1])
    #plt.plot(X[:,0], X[:,j])
    

plt.title("Overlap integral as a variation of the Rib length, Wmid = 0.8um, Parabolic", size= 20)
plt.xlabel(r'Length of the Rib waveguide [$\mu$m]', size= 20)
plt.ylabel("|Overlap intergral| [.]", size= 20)
plt.xticks(size = 20)
plt.yticks(size = 20)
ax1 = plt.gca() #reference to the current plot 
ax1.legend()
plt.savefig('Overlap integral as a variation of the Rib length, Wmid = 0.8um, parabola.png', bbox_inches = 'tight')
plt.show() # display plots


"""
Case where taper and rib are both combined 
"""

d_initcomb = 0.1 #initial value of the etching width
d_finalcomb = 1 #final value of the etching width
n_scancomb = 2 #number of value taken by the parameter Wetch

dscancomb = np.linspace(d_initcomb, d_finalcomb, n_scancomb-1) # the vector of value taken by the parameter Wetch
dscancomb = np.round(dscancomb,2) #rounded to a two digit precision as in Rsoft
d_strcomb = []
#Wetch = [0.1, 0.29, 0.48, 0.67, 0.86, 1.05, 1.24, 1.43, 1.62, 1.81, 2]
#Wetch_str = ['W = 0.1', 'W = 0.29', 'W = 0.48', 'W = 0.67', 'W = 0.86', 'W = 1.05', 'W = 1.24', 'W = 1.43', 'W = 1.62', 'W = 1.81', 'W = 2']
for s in range(1,n_scancomb):
    d_strcomb.append('d = ' +str(dscancomb[s-1])) #loop for the etching names in the legends

#from this point, the particular file to analyze is being read


Ribtaper_L05um_a0365_Lrib011L_d011_overlap_last = open('..\\rsoft_results\\Ribtaper_L05um_a0365_Lrib011L_d011_work\\results\Ribtaper_L05um_a0365_Lrib011L_d011_fw_mon_1_overlap_last.dat', 'r')

Xcomb = np.zeros(shape=(6,5)) #number of lines and columns in the file. # third simulation from 0.3 to 0.9

with Ribtaper_L05um_a0365_Lrib011L_d011_overlap_last :
    index = 0
    line = Ribtaper_L05um_a0365_Lrib011L_d011_overlap_last.readline()
    
    while line != '':  # The EOF char is an empty string
         V = line.split(" ")
         for i in range(len(V)-1):
            Xcomb[index,i] = float(V[i])
         #print(line, end='')
         line = Ribtaper_L05um_a0365_Lrib011L_d011_overlap_last.readline()
         index = index + 1         
"""
Reverse the order of the data in the matrix
"""
#X = np.flip(X, axis = 0)
#The dimensions of the matrix are computed
size_rowcomb = Xcomb[0,:].shape
size_colcomb = Xcomb[:,0].shape
size_rowcomb = size_rowcomb[0] #The number of elements per row (number of columns)
size_colcomb = size_colcomb[0] #The number of elements per column (number of rows)

"""
The max of each vector
"""
ymaximumcomb = [0 for x in range(size_rowcomb)] #using loops
xmaximumcomb = [0 for x in range(size_rowcomb)]
argmaximumcomb  = [0 for x in range(size_rowcomb)]

#ymaximum3 = [0 for x in range(size_row)] #using loops
#xmaximum3 = [0 for x in range(size_row)]
#argmaximum3  = [0 for x in range(size_row)]

Xcomb[:,1:size_rowcomb] = np.sqrt(Xcomb[:,1:size_rowcomb])
#X3[:,1:size_row] = np.sqrt(X3[:,1:size_row])

"""
Delete the outliers, i.e. the values being aberrations due to the calculations
algorthim
"""
"""
n_outliers = 20
for i in range(1,size_row): #for all the columns
    for j in range(0,n_outliers): #only for the first n_outliers rows
        if X[j,i] > 0.3:
            X[j,i] = 0
"""          
#The maximum is finally computed among the acceptable values and is casted into
#the vector called ymaximum, then the corresponding indexes (x axis values) are
# casted into the xmaximum value
            
for k in range(1,size_rowcomb):
    ymaximumcomb[k] = np.amax(Xcomb[:,k])
    argmaximumcomb[k] = np.argmax(Xcomb[:,k])
    xmaximumcomb[k] = X[argmaximumcomb[k],0]
    
    #ymaximum3[k] = np.amax(X3[:,k])
    #argmaximum3[k] = np.argmax(X3[:,k])
    #xmaximum3[k] = X3[argmaximum3[k],0]

"""
The value of each C band vector at 1.55 micrometer.
"""
"""
lambda_index = np.where(X3[:,0] == 1.55)[0]
lambda_val = X3[lambda_index, 0:size_row]
xlambda_val = np.ones(size_row-2)*lambda_val[0,0]
ylambda_val = lambda_val[0,1:size_row-1]
"""
"""
Second part of the code. Below, the data previously extracted data are ploted. 
The first part is dedicated to the raw data of the global spectrum while the 
second part refers to the C band data only. The C band data are displayed after
a Bspline interpolation.
"""

#start of the first part

d_tabcomb ={} # tab containing the plot objects and their names "W_i"
#we will assign the names and links them to the plots int the tab
#wetch_tab3 ={} #for the C band, same principle but finally not used

figure1comb = plt.figure(figsize=(14,7)) #fonction to modify the size of the plots

#this curve can potentially be interpolated using Bsplines
for j in range(1,n_scancomb):    
#    xnew = np.linspace(X[:,0].min(), X[:,0].max(), 300) 
#    spl = make_interp_spline(X[:,0], X[:,j], k=3)  # type: BSpline
#    power_smooth = spl(xnew)
    dsp3comb = turn_spline(Xcomb,j)
    #dsp2 = turn_spline(L2,j)
    
    d_tabcomb["d_" +str(j)+" = "+str(dscancomb[j-1])] = plt.plot(dsp3comb[0],dsp3comb[1]) #assign the names and links
    #plt.plot(X[:,0], X[:,j])
    
plt.title("Overlap integral as a variation of the Rib length and the curve of the tapers parabola", size= 20)
plt.xlabel(r'Length of the Rib [$\mu$m]', size= 20)
plt.ylabel("|Overlap intergral| [.]", size= 20)
plt.xticks(size = 20)
plt.yticks(size = 20)
ax1 = plt.gca() #reference to the current plot 
plot_namescomb = d_tabcomb.keys()
ax1.legend(plot_names)
plt.savefig('Overlap integral as a variation of the Rib length and the curve of the tapers parabola.png', bbox_inches = 'tight')
plt.show() # display plots

"""
Graphs showing the optimized values
"""

figurermax1 = plt.figure(figsize=(12,10))
x = np.array([0,1,2,3])
y = np.array(np.sqrt([ymaximum1[2],ymaximum2[2],ymaximumP1[2],ymaximumP2[2]]))

my_xticks = [r'Linear taper, Wmid = 1[$\mu$m]','Linear taper, Wmid = 0.8[$\mu$m]','Parabolic taper, Wmid = 1[$\mu$m]','Parabolic taper Wmid = 0.8[$\mu$m]']
plt.xticks(x, my_xticks, size = 11, fontweight='bold')
plt.yticks(size = 15)
plt.plot(x, y,'o',color='r', markersize=20)
plt.title("Maximum values for each case when length of taper is varied", size= 20)
plt.ylabel("|Overlap intergral| [.]", size= 20)
plt.ylabel("|Maximum Overlap intergral| [.]", size= 20)
plt.savefig('firststageoptimal.png', bbox_inches = 'tight')
plt.show()

figurermax2 = plt.figure(figsize=(12,10))
plt.plot(dscan,ymaximum[1:11],'-o',color='c', markersize=10)
plt.ylabel("|Maximum Overlap intergral| [.]", size= 20)
plt.title("Maximum overlap intergral values as a variation of the parabola's parameter d", size= 15)
plt.xlabel('d []', size= 25)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.ylabel("|Overlap intergral| [.]")
plt.savefig('taperparameterd.png', bbox_inches = 'tight')
plt.show()

figurermax3 = plt.figure(figsize=(12,10))
x3 = np.array([0,1,2])
y3 = np.array([np.sqrt(ymaximumribpara1[2]),np.sqrt(ymaximumribpara2[2]),ymaximumcomb[2]])

my_xticks3 = [r'Optimal 2 stage design, Wmid = 1[$\mu$m]','Optimal 2 stage design, Wmid = 0.8[$\mu$m]','Optimal combined taper + rib']
plt.xticks(x3, my_xticks3, size = 13, fontweight='bold')
plt.yticks(size = 15)
plt.plot(x3, y3,'o',color='r', markersize=20)
plt.title("Final optimal values for each case", size= 20)
plt.ylabel("|Overlap intergral| [.]", size= 20)
plt.ylabel("|Maximum Overlap intergral| [.]", size= 20)
plt.savefig('finaloptimalvalues.png', bbox_inches = 'tight')
plt.show()