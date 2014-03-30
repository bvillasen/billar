import numpy as np
import sys, time, os, inspect, datetime
#import h5py as h5
import matplotlib.pyplot as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel
from geometry import *
from plotting import *

currentDirectory = os.getcwd()
#Add Modules from other directories
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
#toolsDirectory = parentDirectory + "/tools"
#sys.path.append( toolsDirectory )
from cudaTools import setCudaDevice, getFreeMemory
from tools import printProgress


#Set Parameters

nParticles = 1024*1024
collisionsPerRun = 1e2
nRuns = 10

#For time sampling
maxTime = 2000.
deltaTime_anim = 5



#For final plotting
particlesForPlot = 1024
collisionsForPlot = 128
plotFinal = True

cudaP = "double"
devN = None
usingAnimation = False
plotting = False


#Read in-line parameters
for option in sys.argv:
  if option.find("part")>=0 : nParticles = int(option[option.find("=")+1:])
  if option.find("time")>=0 : maxTime = float(option[option.find("=")+1:])
  if option.find("anim")>=0: usingAnimation = True
  if option.find("plot") >=0: plotting = True
  if option == "double": cudaP = "double"
  if option == "float": cudaP = "float"
  if option.find("dev") >= 0 : devN = int(option[-1])


if usingAnimation: import points3D as pAnim #points3D Animation
precision  = {"float":np.float32, "double":np.float64} 
cudaPre = precision[cudaP]

#Set CUDA thread grid dimentions
block = ( 128, 1, 1 )
grid = ( (nParticles - 1)//block[0] + 1, 1, 1 )
maxTimeIndx = block[0]      #Don't change 
deltaTime_radius = maxTime/maxTimeIndx  


def configAnimation():
  global collisionsPerRun, deltaTime_anim
  collisionsPerRun = 10
  deltaTime_radius = 100
  deltaTime_anim = 3

###########################################################################
###########################################################################
#Initialize the frontier geometry 
radius = 0.4
geometry = Geometry()
#geometry.addSphere( (-0.5,  0.5,  0.5), 0.6 )
#geometry.addSphere( (-0.5,  0.5, -0.5), 0.6 )
#geometry.addSphere( ( 0.5,  0.5,  0.5), 0.6 )
#geometry.addSphere( ( 0.5,  0.5, -0.5), 0.6 )
#geometry.addSphere( ( 0.5, -0.5,  0.5), 0.6 )
#geometry.addSphere( ( 0.5, -0.5, -0.5), 0.6 )
#geometry.addSphere( (-0.5, -0.5,  0.5), 0.6 )
#geometry.addSphere( (-0.5, -0.5, -0.5), 0.6 )
geometry.addSphere( (   0.,  0.,   0. ), radius )
geometry.addWall( (-0.5,  0, 0), (-1,  0,  0), type=1 ) #left
geometry.addWall( ( 0,  0.5, 0), ( 0,  1,  0), type=1 ) #up
geometry.addWall( ( 0,  0, 0.5), ( 0,  0,  1), type=1 ) #top      
geometry.addWall( ( 0.5,  0, 0), ( 1,  0,  0), type=1 ) #rigth
geometry.addWall( ( 0, -0.5, 0), ( 0, -1,  0), type=1 ) #down
geometry.addWall( ( 0, 0, -0.5), ( 0,  0, -1), type=1 ) #bottom
#nCircles, circlesCaract_h, nLines, linesCaract_h = geometry.prepareCUDA( cudaP=cudaP )
nSpheres, spheresCaract_h, nWalls, wallsCaract_h = geometry.prepareCUDA_3D( cudaP=cudaP )

if usingAnimation:
  pAnim.nPoints = nParticles
  pAnim.viewXmin, pAnim.viewXmax = -10000., 10000.
  pAnim.viewYmin, pAnim.viewYmax = -10000., 10000.
  pAnim.viewZmin, pAnim.viewZmax = -10000000., 10000000. 
  pAnim.showGrid = False
  pAnim.nPointsPerCircle = 50
  pAnim.cirPos, pAnim.cirCol, pAnim.nCirclesGrid = geometry.circlesGrid( radius, -30., 30., -20., 20., nPoints=pAnim.nPointsPerCircle)

###########################################################################
###########################################################################
#Initialize and select CUDA device
if usingAnimation:
  pAnim.initGL()
  pAnim.CUDA_initialized = True
  configAnimation()

cudaDev = setCudaDevice( devN = devN, usingAnimation = usingAnimation )

#Read and compile CUDA code
print "\nCompiling CUDA code\n"
codeFiles = [ "vector3D.h", "sphere.h", "wall.h", "cudaBillar3D.cu"]
for fileName in codeFiles:
  codeString = open(fileName, "r").read().replace("cudaP", cudaP)
  outFile = open( fileName + "T", "w" )
  outFile.write( codeString )
  outFile.close()
  
cudaCodeStringTemp = open("cudaBillar3D.cuT", "r").read()
cudaCodeString = cudaCodeStringTemp % { "nSPHERES":nSpheres, "nWALLS":nWalls, "THREADS_PER_BLOCK":block[0], "TIME_INDEX_MAX":maxTimeIndx }
cudaCode = SourceModule(cudaCodeString, no_extern_c=True, include_dirs=[currentDirectory])
mainKernel = cudaCode.get_function("main_kernel" )

###########################################################################
###########################################################################
#Initialize Data
nData = particlesForPlot*collisionsForPlot
print "Initializing CUDA memory"
#np.random.seed(int(time.time()))  #Change numpy random seed
initialFreeMemory = getFreeMemory( show=True )
initialPosX_h = 0.45*np.ones(nParticles).astype(cudaPre)
initialPosY_h = 0.45*np.ones(nParticles).astype(cudaPre)
initialPosZ_h = 0.45*np.ones(nParticles).astype(cudaPre)
#Spherically uniform random distribution for initial velocities
initialTheta = 2*np.pi*np.random.rand(nParticles).astype(cudaPre) 
initialPhi = np.arccos(2*np.random.rand(nParticles).astype(cudaPre) - 1) 
initialVelX_h = np.cos(initialTheta)*np.sin(initialPhi)
initialVelY_h = np.sin(initialTheta)*np.sin(initialPhi)
initialVelZ_h = np.cos(initialPhi)
initialRegionX_h = np.zeros(nParticles).astype(np.int32)
initialRegionY_h = np.zeros(nParticles).astype(np.int32)
initialRegionZ_h = np.zeros(nParticles).astype(np.int32)
initialPosX_d = gpuarray.to_gpu( initialPosX_h )
initialPosY_d = gpuarray.to_gpu( initialPosY_h )
initialPosZ_d = gpuarray.to_gpu( initialPosZ_h )
initialVelX_d = gpuarray.to_gpu( initialVelX_h )
initialVelY_d = gpuarray.to_gpu( initialVelY_h )
initialVelZ_d = gpuarray.to_gpu( initialVelZ_h )
initialRegionX_d = gpuarray.to_gpu( initialRegionX_h )
initialRegionY_d = gpuarray.to_gpu( initialRegionY_h )
initialRegionZ_d = gpuarray.to_gpu( initialRegionZ_h )
spheresCaract_d = gpuarray.to_gpu(spheresCaract_h)
wallsCaract_d = gpuarray.to_gpu(wallsCaract_h)
outPosX_d = gpuarray.to_gpu( np.zeros( nData ).astype(cudaPre) )
outPosY_d = gpuarray.to_gpu( np.zeros( nData ).astype(cudaPre) )
outPosZ_d = gpuarray.to_gpu( np.zeros( nData ).astype(cudaPre) )
times_d = gpuarray.to_gpu( np.zeros(nParticles).astype(cudaPre) )
timesIdx_anim_d = gpuarray.to_gpu( np.zeros(nParticles).astype(np.int32) )
timesIdx_rad_d = gpuarray.to_gpu( np.ones(nParticles).astype(np.int32) )
timesOccupancy_h = np.zeros(maxTimeIndx).astype(np.int32)
timesOccupancy_h[0] = nParticles
timesOccupancy_d = gpuarray.to_gpu( timesOccupancy_h  )
timesForRadius = deltaTime_radius*np.arange(maxTimeIndx)
radiusAll_h = np.zeros(maxTimeIndx).astype(np.float32)
radiusAll_d = gpuarray.to_gpu( radiusAll_h )
#nRestarts_d = gpuarray.to_gpu(np.zeros(nRuns+1).astype(np.int32))
finalFreeMemory = getFreeMemory( show=False )
print  " Total global memory used: {0:0.0f} MB\n".format( float(initialFreeMemory - finalFreeMemory)/1e6 ) 
##########################################################################
##########################################################################
def runDynamics( nParticles=1024*1024, nRuns=0, collisionsPerRun=1e2,  plotFinal=True ):
  global usingAnimation
  

nAnimIter = 0
def animationUpdate():
  global nAnimIter
  mainKernel(np.uint8(True), np.int32(nParticles), np.int32(collisionsPerRun), np.int32(nSpheres), spheresCaract_d, np.int32(nWalls), wallsCaract_d,
	      initialPosX_d, initialPosY_d, initialPosZ_d,
	      initialVelX_d, initialVelY_d, initialVelZ_d,
	      initialRegionX_d, initialRegionY_d, initialRegionZ_d,
	      outPosX_d, outPosY_d, outPosZ_d,  
	      times_d, np.float32(deltaTime_anim), timesIdx_anim_d, 
	      np.float32(deltaTime_radius), timesIdx_rad_d, timesOccupancy_d, radiusAll_d,
	      np.int32(0), np.int32(0), np.int32(1),
	      np.intp(pAnim.cuda_VOB_ptr),  grid=grid, block=block)
  nAnimIter += 1
  if nAnimIter%50 == 0 and plotting: plotData(nAnimIter)

def plotData( nIter ):
  global radiusAll_h, timesOccupancy_h
  timesOccupancy_h = timesOccupancy_d.get()
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

  
  radiusAll_h = radiusAll_d.get()
  fig = plt.figure(1)
  fig.clf()
  #indx = timesOccupancy_h.nonzero()
  #radiusAll_h[indx] /= timesOccupancy_h[indx]
  plt.plot( timesForRadius, radiusAll_h )
  ax = plt.gca()
  ax.set_ylabel(r"$\overline{ r^2 } $", fontsize=20, rotation="horizontal")
  ax.set_xlabel(r"Time")
  plt.title(r"nParticles={0}    nCollisions={1}".format(nParticles, nIter*collisionsPerRun))
  plt.draw()



if usingAnimation:
  pAnim.updateFunc = animationUpdate
  if plotting: plt.ion(), plt.show()
  pAnim.startAnimation()


###########################################################################
###########################################################################
#Start Simulation
if plotting: plt.ion(), plt.show()

print "Starting simulation"
if cudaP == "double": print "Using double precision"
print " nParticles: ", nParticles
print " nRuns: ", nRuns
print " Collisions per run: ", collisionsPerRun
print " TOTAL Iterations per particle: ", collisionsPerRun*(nRuns) 
print "  Particles for final plot: ", particlesForPlot
print "  Collisions for final plot: ", collisionsForPlot
print ""

start = cuda.Event()
end = cuda.Event()
totalTime = 0
secs = 0

#Itererate for a particle bunch

changeInitial = True
savePos = False
start.record()
for runNumber in range(nRuns+1):
  if runNumber == nRuns:
    plt.ion(), plt.show()
    plotData( nRuns )
    savePos=True
    changeInitial = False
    collisionsPerRun = collisionsForPlot
  if runNumber%(1)==0:
    secs = start.time_till(end.record().synchronize())*1e-3
    totalTime += secs
    printProgress( runNumber, nRuns, 1, secs )
    start.record()
  mainKernel(np.uint8(usingAnimation), np.int32(nParticles), np.int32(collisionsPerRun), np.int32(nSpheres), spheresCaract_d, np.int32(nWalls), wallsCaract_d,
	    initialPosX_d, initialPosY_d, initialPosZ_d, 
	    initialVelX_d, initialVelY_d, initialVelZ_d,
	    initialRegionX_d, initialRegionY_d, initialRegionZ_d,
	    outPosX_d, outPosY_d, outPosZ_d,
	    times_d, np.float32(deltaTime_anim), timesIdx_anim_d,
	    np.float32( deltaTime_radius ), timesIdx_rad_d, timesOccupancy_d, radiusAll_d,
	    np.int32(savePos), np.int32(particlesForPlot), np.int32(changeInitial),
	    np.intp(0),  grid=grid, block=block)
  if runNumber%5==0  and plotting: plotData( runNumber )
print "\n\nFinished in : {0:.4f}  sec\n".format( float( totalTime ) ) 

#Get the results
outPosX = np.zeros(nData + particlesForPlot)
outPosY = np.zeros(nData + particlesForPlot)
outPosZ = np.zeros(nData + particlesForPlot)
#Add initial positions to the array
outPosX[:particlesForPlot] = initialPosX_d.get()[:particlesForPlot] + initialRegionX_d.get()[:particlesForPlot]
outPosY[:particlesForPlot] = initialPosY_d.get()[:particlesForPlot] + initialRegionY_d.get()[:particlesForPlot]
outPosZ[:particlesForPlot] = initialPosZ_d.get()[:particlesForPlot] + initialRegionZ_d.get()[:particlesForPlot]
outPosX[particlesForPlot:] = outPosX_d.get()
outPosY[particlesForPlot:] = outPosY_d.get()
outPosZ[particlesForPlot:] = outPosZ_d.get()
outPosX = outPosX.reshape(collisionsForPlot+1,particlesForPlot).transpose()
outPosY = outPosY.reshape(collisionsForPlot+1,particlesForPlot).transpose()
outPosZ = outPosZ.reshape(collisionsForPlot+1,particlesForPlot).transpose()
pos = (outPosX, outPosY, outPosZ )
rAvg = (np.sqrt(outPosX[:,-1]*outPosX[:,-1] + outPosY[:,-1]*outPosY[:,-1] + outPosZ[:,-1]*outPosZ[:,-1])).sum()/particlesForPlot
times = times_d.get()
if plotFinal:  plotPosGnuplot(pos)

raw_input( "\nPress ENTER to exit")
###########################################################################
###########################################################################











































