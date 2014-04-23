import numpy as np
import sys, time, os, inspect, datetime
#import h5py as h5
import matplotlib.pyplot as plt

#Load Data
times, avrRadius = np.loadtxt( "data/avrRadius_n2097152_t10000_f.dat" )


fig = plt.figure(0)
plt.plot( times, avrRadius/times )
plt.xscale("log")
plt.show()