import numpy as np
import sys, time, os, inspect, datetime
import h5py as h5
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
  plt.title(r"nParticles={0}    time={1}".format(nParticles, timesForRadius[-1]))
  ax = plt.gca()
  ax.set_ylabel(r"$\overline{ r^2 }/t $", fontsize=20, rotation="horizontal")
  ax.set_xlabel(r"log(t)")
  plt.draw()


nParticles = 1024*1024*2
time = 1000

for option in sys.argv:
  if option.find(".hdf5")>=0 : 
    dataFileName = option
    nParticles = int(option[option.find("n")+1:option.find("_")])
    time = int(option[option.rfind("t")+1:option.rfind("_")])

if __name__ == "__main__":

  #Load Data
  print '\nLoading data... \n particles: {0}\n time: {1}\n'.format(nParticles, time)
  dataFile = h5.File( dataFileName ,'r')
  times = dataFile.get("timesForRadius")[...]
  avrRadius = dataFile.get("avrRadius")[...]
  dataFile.close()
  #times, avrRadius = np.loadtxt( "data/avrRadius_n{0}_t{1}_d.dat".format(nParticles, time) )
  
  fig = plt.figure(0)
  plt.plot( times, avrRadius/times )
  plt.xscale("log")
  plt.title(r"nParticles={0}    time={1}".format(nParticles, time))
  ax = plt.gca()
  ax.set_ylabel(r"$\overline{ r^2 }/t $", fontsize=20, rotation="horizontal")
  ax.set_xlabel(r"log(t)")
  plt.show()