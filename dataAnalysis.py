import numpy as np
import sys, time, os, inspect, datetime
#import h5py as h5
import matplotlib.pyplot as plt

def plotData( nParticles, collisionsPerRun, nIter, timesForRadius, radiusAll, timesOccupancy, device=True):
  #if plotting: plt.ioff()
  if device: timesOccupancy_h = timesOccupancy.get()
  else: timesOccupancy_h = timesOccupancy
  fig = plt.figure(0)
  fig.clf()
  #mngr = plt.get_current_fig_manager()
  # to put it into the upper left corner for example:
  #mngr.window.setGeometry(50,100,640, 545)
  plt.plot( timesForRadius, timesOccupancy_h )
  ax = plt.gca()
  ax.set_yscale('log')
  ax.set_ylabel(r"Time occupancy")
  ax.set_xlabel(r"Time")
  plt.title(r"nParticles={0}    nCollisions={1}".format(nParticles, nIter*collisionsPerRun))

  if device: radiusAll_h = radiusAll.get()
  else: radiusAll_h = radiusAll
  fig = plt.figure(1)
  fig.clf()
  plt.plot( timesForRadius, radiusAll_h )
  ax = plt.gca()
  ax.set_ylabel(r"$\overline{ r^2 } $", fontsize=20, rotation="horizontal")
  ax.set_xlabel(r"Time")
  plt.title(r"nParticles={0}    nCollisions={1}".format(nParticles, nIter*collisionsPerRun))
  #if plotting: plt.ion()
  plt.draw()
  
  fig = plt.figure(2)
  plt.plot( timesForRadius, radiusAll_h/timesForRadius )
  plt.xscale("log")
  plt.draw()

if __name__ == "__main__":
  #Load Data
  times, avrRadius = np.loadtxt( "data/avrRadius_n2097152_t10000_d.dat" )
  
  fig = plt.figure(0)
  plt.plot( times, avrRadius/times )
  plt.xscale("log")
  plt.show()