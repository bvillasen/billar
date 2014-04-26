import numpy as np
import sys, time, os, inspect, datetime
#import h5py as h5
import matplotlib.pyplot as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel
from geometry import Geometry
from plotting import plotPosGnuplot
from dataAnalysis import plotData

currentDirectory = os.getcwd()
#Add Modules from other directories
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
#sys.path.append( toolsDirectory )
from cudaTools import setCudaDevice, getFreeMemory, kernelMemoryInfo
from tools import printProgressTime, ensureDirectory


#Set Parameters for Simulation
nParticles = 1024*1024*2
collisionsPerRun = 1e2              #Number of collitions for each kernel launch
#For time sampling
maxTime = 1000.

#For final plotting
plotFinal = False
particlesForPlot = 1024
collisionsForPlot = 128

cudaP = "double"
devN = None
usingAnimation = False
plotting = False
showKernelMemInfo = False

#Read in-line parameters
for option in sys.argv:
  if option.find("part")>=0 : nParticles = int(option[option.find("=")+1:])
  if option.find("time")>=0 : maxTime = float(option[option.find("=")+1:])
  if option.find("iter")>=0 : collisionsPerRun = int(option[option.find("=")+1:])
  if option.find("anim") >=0: usingAnimation = True
  if option.find("mem") >=0: showKernelMemInfo = True
  if option.find("plot") >=0: plotting = True
  if option == "double": cudaP = "double"
  if option == "float": cudaP = "float"
  if option.find("dev") >= 0 : devN = int(option[-1])

if usingAnimation: import points2D as pAnim #points2D Animation
precision  = {"float":np.float32, "double":np.float64} 
cudaPre = precision[cudaP]

if int(np.log2(nParticles)) != np.log2(nParticles):
  choice = int( raw_input( "\nERROR: Number of particles is not a 2-power.\nChose: \n 1) {0}\n 2) {1}\nChoice: ".format(2**int(np.log2(nParticles)), 2**int(1+np.log2(nParticles)) )) )
  if choice == 1: nParticles = 2**int(np.log2(nParticles))
  if choice == 2: nParticles = 2**int(np.log2(nParticles)+1)

#Set CUDA thread grid dimentions
block = ( 128, 1, 1 )
maxThreads = 1024*1024*2
if nParticles <= maxThreads: grid = ( (nParticles - 1)//block[0] + 1, 1, 1 )
else: grid = ( (maxThreads - 1)//block[0] + 1, 1, 1 )
maxTimeIndx = block[0]*4   #Number of points for avrRadius sampling
deltaTime_radius = float( maxTime )/maxTimeIndx
deltaTime_anim = 5

def configAnimation():
  global collisionsPerRun, deltaTime_radius, deltaTime_anim
  collisionsPerRun = 10
  deltaTime_radius = 100
  deltaTime_anim = 3
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

if usingAnimation:
  pAnim.nPoints = nParticles
  pAnim.viewXmin, pAnim.viewXmax = -2500., 2500.
  pAnim.viewYmin, pAnim.viewYmax = -2500., 2500.
  pAnim.viewZmin, pAnim.viewZmax = -1, 1
  pAnim.showGrid = True
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
if showKernelMemInfo: 
  kernelMemoryInfo(mainKernel, 'mainKernel')
  sys.exit()
###########################################################################
###########################################################################
#Initialize Data
nData = particlesForPlot*collisionsForPlot
if not plotFinal: nData = 1
print "Initializing CUDA memory"
#np.random.seed(int(time.time()))  #Change numpy random seed
initialFreeMemory = getFreeMemory( show=True )
initialPosX_h = 0.49*np.ones(nParticles).astype(cudaPre)
initialPosY_h = 0.49*np.ones(nParticles).astype(cudaPre)
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
times_d = gpuarray.to_gpu( np.zeros(nParticles).astype(cudaPre) )
timesIdx_anim_d = gpuarray.to_gpu( np.zeros(nParticles, dtype=np.int32) )
timesIdx_rad_d = gpuarray.to_gpu( np.ones(nParticles, dtype=np.int32) )
timesOccupancy_h = np.zeros(maxTimeIndx).astype(np.int32)
#timesOccupancy_h[0] = nParticles
timesOccupancy_d = gpuarray.to_gpu( timesOccupancy_h  )
timesForRadius = deltaTime_radius*( np.arange(maxTimeIndx) + 1 )
radiusAll_h = np.zeros(maxTimeIndx).astype(np.float32)
radiusAll_d = gpuarray.to_gpu( radiusAll_h )
#output_h = np.zeros( nData ).astype(cudaPre)
outPosX_d = gpuarray.to_gpu(  np.zeros( nData ).astype(cudaPre) )
outPosY_d = gpuarray.to_gpu(  np.zeros( nData ).astype(cudaPre) )
#nRestarts_d = gpuarray.to_gpu(np.zeros(nRuns+1).astype(np.int32))
finalFreeMemory = getFreeMemory( show=False )
print  " Total global memory used: {0:0.0f} MB\n".format( float(initialFreeMemory - finalFreeMemory)/1e6 ) 
###########################################################################
###########################################################################
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
  if nAnimIter%50 == 0 and plotting: plotData( nParticles, collisionsPerRun, nAnimIter, timesForRadius, radiusAll_d, timesOccupancy_d )

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
#print " nRuns: ", nRuns
print " time: ", maxTime
print " Collisions per kernel-launch: ", collisionsPerRun
#print " TOTAL Iterations per particle: ", collisionsPerRun*(nRuns)
if plotFinal:
  print "  Particles for final plot: ", particlesForPlot
  print "  Collisions for final plot: ", collisionsForPlot
print ""

changeInitial = True
savePos = False
start = cuda.Event()
end = cuda.Event()
occupancyOld, occupancy = 0, 0
launchCouter = 0
start.record()
while occupancyOld < maxTimeIndx:
  #print "############################################################"
  occupancy = sum(timesOccupancy_d.get()>=nParticles)
  if occupancy>occupancyOld or occupancy==0: printProgressTime( occupancy, maxTimeIndx, start.time_till(end.record().synchronize())*1e-3 )
  occupancyOld = occupancy
  mainKernel(np.uint8(usingAnimation), np.int32(nParticles), np.int32(collisionsPerRun), np.int32(nCircles), circlesCaract_d, np.int32(nLines), linesCaract_d,
	    initialPosX_d, initialPosY_d, initialVelX_d, initialVelY_d, initialRegionX_d, initialRegionY_d,
	    outPosX_d, outPosY_d, times_d,
	    np.float32(deltaTime_anim), timesIdx_anim_d,
	    np.float32( deltaTime_radius ), timesIdx_rad_d, timesOccupancy_d, radiusAll_d,
	    np.uint8(savePos), np.int32(particlesForPlot), np.uint8(changeInitial),
	    np.intp(0),  grid=grid, block=block)
  launchCouter += 1
  #if runNumber%5==0  and plotting: plotData( runNumber )
print "\n\nFinished in : {0:.4f}  sec\n".format( float( start.time_till(end.record().synchronize())*1e-3 ) ) 

#Get the results
if plotFinal:
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
  plotPosGnuplot(pos)

#Save data
dataDir = currentDirectory + "/data"
ensureDirectory( dataDir )
data= np.array([ timesForRadius, radiusAll_d.get() ])
dataFile = dataDir + "/avrRadius_n{0}_t{1:.0f}_{2}.dat".format(nParticles, float(maxTime), cudaP[0])
np.savetxt( dataFile, data )
print "Data Saved: ", dataFile, "\n"

#cuda.stop_profiler()
if plotting: 
  plotData( nParticles, collisionsPerRun, launchCouter, timesForRadius, radiusAll_d, timesOccupancy_d )
  raw_input( "Press ENTER to exit\n")
###########################################################################
###########################################################################


