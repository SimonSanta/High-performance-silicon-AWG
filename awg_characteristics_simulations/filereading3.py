# -*- coding: utf-8 -*-
"""
Created on Fri July 16 16:01:49 2021


    Python code used for extracting data from power form data files, taken from Fullwave.
    The data handled here simulated the behavior of a slab array interface form input wgs to 1st slab. 
    The data are presenting the power transmitted at several point.

@author: Simon Santacatterina
"""
import numpy as np
"""
Code taken as input the file containing raw data and
extracting those data in order to plot them. 

The input is the data file
(or other) containing "string" version of the data. 

The file is read line by line. Then, each line is processed such that
each string separated by " " (a space) correspond to one data and is converted
into a value of type "float". After, the values are stored in a matrix for 
manipulations
"""

def file_read3(filename, dim):
    
    powerval = open(filename,'r')
    X = np.zeros(shape=(dim[0],dim[1])) #number of lines and columns in the file.
    
    """
    Files is read and data are extracted
    """
    with powerval :
        index = 0
        line = powerval.readline()
        line.strip()
        while line != '':  # The EOF char is an empty string
            V = line.replace('  ',' ').split(' ')
            for i in range(len(V)):
                X[index,i] = float(V[i])
                #print(line, end='')
            line = powerval.readline()
            line.strip()
            index = index + 1     
    powerval.close()
    return X