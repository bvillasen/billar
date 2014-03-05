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
toolsDirectory = parentDirectory + "/tools"
sys.path.append( toolsDirectory )
from cudaTools import setCudaDevice, getFreeMemory
from tools import printProgress
import points2D as pAnim #points3D Animation

precision  = {"float":np.float32, "double":np.float64} 
cudaP = "double"
devN = None
usingAnimation = False
#Read in-line parameters
for option in sys.argv:
  if option.find("anim") >=0: usingAnimation = True
  if option == "double": cudaP = "double"
  if option == "float": cudaP = "float"
  if option.find("dev") >= 0 : devN = int(option[-1])

cudaPre = precision[cudaP]
DIM = 2

#Set Parameters
factor = 1024*2
nParticles = 1024*factor
collisionsPerRun = 1e2
nRuns = 50

#For time sampling
deltaTime_anim = 5
deltaTime_radius = 500
maxTimeIndx = 128

#Set CUDA thread grid dimentions
block = ( 128, 1, 1 )
grid = ( (nParticles - 1)//block[0] + 1, 1, 1 )

###########################################################################
###########################################################################
#Initialize the frontier geometry 
radius = 0.3
geometry = Geometry()
#geometry.addCircle( (-0.5,  0.5), 0.25  )
#geometry.addCircle( ( 0.5,  0.5), 0.25  )
#geometry.addCircle( ( 0.5, -0.5), 0.25  )
#geometry.addCircle( (-0.5, -0.5), 0.25  )
geometry.addCircle( (  0.,   0.), radius )
geometry.addLine( (-0.5,  0), (-1, 0), type=1 ) #type: 1->Periodic, 0->Real
geometry.addLine( ( 0,  0.5), ( 0, 1), type=1 )
geometry.addLine( ( 0.5,  0), ( 1, 0), type=1 )
geometry.addLine( ( 0, -0.5), ( 0,-1), type=1 )
nCircles, circlesCaract_h, nLines, linesCaract_h = geometry.prepareCUDA( cudaP=cudaP )

pAnim.nPoints = nParticles
pAnim.viewXmin, pAnim.viewXmax = -2500., 2500.
pAnim.viewYmin, pAnim.viewYmax = -2500., 2500.
pAnim.viewZmin, pAnim.viewZmax = -1, 1

pAnim.showGrid = True
pAnim.nPointsPerCircle = 50
pAnim.cirPos, pAnim.cirCol, pAnim.nCirclesGrid = geometry.circlesGrid( radius, -30., 30., -20., 20., nPoints=pAnim.nPointsPerCircle)
#print pAnim.nCirclesGrid
###########################################################################
###########################################################################
#Initialize and select CUDA device
if usingAnimation: pAnim.initGL()
cudaDev = setCudaDevice( devN = devN, usingAnimation = usingAnimation )
if usingAnimation: pAnim.CUDA_initialized = True


#Read and compile CUDA code
print "Compiling CUDA code"
codeFiles = [ "vector2D.h", "circle.h", "line.h", "cudaBillar.cu"]
for fileName in codeFiles:
  codeString = open(fileName, "r").read().replace("cudaP", cudaP)
  outFile = open( fileName + "T", "w" )
  outFile.write( codeString )
  outFile.close()
  
cudaCodeStringTemp = open("cudaBillar.cuT", "r").read()
cudaCodeString = cudaCodeStringTemp % { "nCIRCLES":nCircles, "nLINES":nLines, "THREADS_PER_BLOCK":block[0], "TIME_INDEX_MAX":maxTimeIndx }
cudaCode = SourceModule(cudaCodeString, no_extern_c=True, include_dirs=[currentDirectory, toolsDirectory])
mainKernel = cudaCode.get_function("main_kernel" )

###########################################################################
###########################################################################
#Initialize Data
  #For final plotting
particlesForPlot = 1024
collisionsForPlot = 128
nData = particlesForPlot*collisionsForPlot
print "Initializing CUDA memory"
#np.random.seed(int(time.time()))  #Change numpy random seed
initialFreeMemory = getFreeMemory( show=True )
initialPosX_h = 0.495*np.ones(nParticles).astype(cudaPre)
initialPosY_h = 0.495*np.ones(nParticles).astype(cudaPre)
#initialVelX_h = np.ones(nParticles).astype(cudaPre)
#initialVelY_h = 0*np.ones(nParticles).astype(cudaPre)
initialTheta = 2*np.pi*np.random.rand(nParticles).astype(cudaPre) - np.pi
initialVelX_h = np.cos(initialTheta)
initialVelY_h = np.sin(initialTheta)
initialRegionX_h = np.zeros(nParticles).astype(np.int32)
initialRegionY_h = np.zeros(nParticles).astype(np.int32)
initialPosX_d = gpuarray.to_gpu( initialPosX_h )
initialPosY_d = gpuarray.to_gpu( initialPosY_h )
initialVelX_d = gpuarray.to_gpu( initialVelX_h )
initialVelY_d = gpuarray.to_gpu( initialVelY_h )
initialRegionX_d = gpuarray.to_gpu( initialRegionX_h )
initialRegionY_d = gpuarray.to_gpu( initialRegionY_h )
circlesCaract_d = gpuarray.to_gpu(circlesCaract_h)
linesCaract_d = gpuarray.to_gpu(linesCaract_h)
output_h = np.zeros( nData ).astype(cudaPre)
outPosX_d = gpuarray.to_gpu( np.zeros_like(output_h) )
outPosY_d = gpuarray.to_gpu( np.zeros_like(output_h) )
#outVelX_d = gpuarray.to_gpu( np.zeros_like(output_h) )
#outVelY_d = gpuarray.to_gpu( np.zeros_like(output_h) )
#outRegionX_d = gpuarray.to_gpu( np.zeros( nData ).astype(np.int32) )
#outRegionY_d = gpuarray.to_gpu( np.zeros( nData ).astype(np.int32) )

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
print  " Total global memory used: {0:0.0f} MB".format( float(initialFreeMemory - finalFreeMemory)/1e6 ) 
###########################################################################
###########################################################################
##Prepare cuda kernels   
#mainKernel.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, 
		    #np.intp, np.intp, np.intp, np.intp, np.intp, np.intp, 
		    #np.intp, np.intp, np.intp, 
		    #np.intp, cudaPre, np.int32, np.int32, np.int32,
		    #np.int32, np.intp])


###########################################################################
###########################################################################
def runDynamics( nParticles=1024*1024, nRuns=50, collisionsPerRun=1e2,  plotFinal=True ):
  global usingAnimation 
  #Start Simulation
  print ""
  print "Starting simulation"
  if cudaP == "double": print "Using double precision"
  print " nParticles: ", nParticles
  print " nRuns: ", nRuns
  print " Collisions per run: ", collisionsPerRun
  print " TOTAL Iterations per particle: ", collisionsPerRun*(nRuns) 
  print "  Particles for plot: ", particlesForPlot
  print "  Collisions for plot: ", collisionsForPlot
  print ""

  start = cuda.Event()
  end = cuda.Event()
  totalTime = 0
  secs = 0

  #Itererate for a particle bunch
  usingAnimation = False
  changeInitial = True
  savePos = False
  start.record()
  for runNumber in range(nRuns+1):
    if runNumber == nRuns:
      savePos=True
      changeInitial = False
      collisionsPerRun = collisionsForPlot
    if runNumber%(1)==0:
      secs = start.time_till(end.record().synchronize())*1e-3
      totalTime += secs
      printProgress( runNumber, nRuns, 1, secs )
      start.record()
    mainKernel(np.uint8(usingAnimation), np.int32(nParticles), np.int32(collisionsPerRun), np.int32(nCircles), circlesCaract_d, np.int32(nLines), linesCaract_d,
	      initialPosX_d, initialPosY_d, initialVelX_d, initialVelY_d, initialRegionX_d, initialRegionY_d,
	      outPosX_d, outPosY_d, times_d,
	      np.float32(deltaTime_anim), timesIdx_anim_d,
	      np.float32( deltaTime_radius ), timesIdx_rad_d, timesOccupancy_d, radiusAll_d,
	      np.int32(savePos), np.int32(particlesForPlot), np.int32(changeInitial),
	      np.intp(0),  grid=grid, block=block)
  print "\n\nFinished in : {0:.4f}  sec\n".format( float( totalTime ) ) 

  #Get the results
  outPosX = np.zeros(nData + particlesForPlot)
  outPosY = np.zeros(nData + particlesForPlot)
  #Add initial positions to the array
  outPosX[:particlesForPlot] = initialPosX_d.get()[:particlesForPlot] + initialRegionX_d.get()[:particlesForPlot]
  outPosY[:particlesForPlot] = initialPosY_d.get()[:particlesForPlot] + initialRegionY_d.get()[:particlesForPlot]
  outPosX[particlesForPlot:] = outPosX_d.get()
  outPosY[particlesForPlot:] = outPosY_d.get()
  outPosX = outPosX.reshape(collisionsForPlot+1,particlesForPlot).transpose()
  outPosY = outPosY.reshape(collisionsForPlot+1,particlesForPlot).transpose()
  pos = (outPosX, outPosY)
  rAvg = (np.sqrt(outPosX[:,-1]*outPosX[:,-1] + outPosY[:,-1]*outPosY[:,-1])).sum()/particlesForPlot
  times = times_d.get()
  if plotFinal:  plotPosGnuplot(pos)
  #return pos, rAvg, times
###########################################################################
###########################################################################

##Get mean radius over time
#meanRadius = np.array([cudaPre( gpuarray.sum(radiusAll_d[i*nParticles:(i+1)*nParticles], dtype = cudaPre).get() ) for i in range(maxTimeIndx)])
#meanRadius /= nParticles

def configAnimation():
  global collisionsPerRun, deltaTime_anim
  collisionsPerRun = 10
  deltaTime_anim = 3

nAnimIter = 0
def animationUpdate():
  global nAnimIter
  mainKernel(np.uint8(True), np.int32(nParticles), np.int32(collisionsPerRun), np.int32(nCircles), circlesCaract_d, np.int32(nLines), linesCaract_d,
	      initialPosX_d, initialPosY_d, initialVelX_d, initialVelY_d, initialRegionX_d, initialRegionY_d,
	      outPosX_d, outPosY_d, times_d,
	      np.float32(deltaTime_anim), timesIdx_anim_d, 
	      np.float32(deltaTime_radius), timesIdx_rad_d, timesOccupancy_d, radiusAll_d,
	      np.int32(0), np.int32(0), np.int32(1),
	      np.intp(pAnim.cuda_VOB_ptr),  grid=grid, block=block)
  nAnimIter += 1
  if nAnimIter%50 == 0: plotData()

def plotData():
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

  
  radiusAll_h = radiusAll_d.get()
  fig = plt.figure(1)
  fig.clf()
  plt.plot( timesForRadius, radiusAll_h )
  ax = plt.gca()
  ax.set_ylabel(r"$\overline{ r^2 } $", fontsize=20, rotation="horizontal")
  ax.set_xlabel(r"Time")
  plt.draw()



if usingAnimation:
  configAnimation()
  pAnim.updateFunc = animationUpdate
  plt.ion()
  plt.show()
  pAnim.startAnimation()

runDynamics()














































