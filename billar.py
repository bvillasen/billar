import numpy as np
import sys, time, os, inspect, datetime
#import h5py as h5
import matplotlib.pylab as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel
from geometry import *
from plotting import *

currentDirectory = os.getcwd()
##Add Modules from other directories
#parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
#myToolsDirectory = parentDirectory + "/myTools"
#sys.path.append( myToolsDirectory )
from cudaTools import *
from tools import *

precision  = {"float":np.float32, "double":np.float64} 
cudaP = "float"
devN = None
#Read in-line parameters
for option in sys.argv:
  if option == "double": cudaP = "double"
  if option.find("dev") >= 0 : devN = int(option[-1])

cudaPre = precision[cudaP]

#Set Parameters
factor = 1024
nParticles = 1024*factor
collisionsPerRun = 1e3
nRuns = 10


particlesForPlot = 1024
collisionsForPlot = 128
#nWrites = 1000
nData = particlesForPlot*collisionsForPlot
endTime = 100000
timeSteps = 100
#writePos = True
deltaTime = float(endTime)/(timeSteps-1)
maxTimeIndx = 10
timeSampling = np.linspace(0, endTime, timeSteps ).astype(cudaPre)



###########################################################################
###########################################################################
#Initialize the frontier geometry 
geometry = Geometry()
#geometry.addCircle( (-0.5,  0.5), 0.25  )
#geometry.addCircle( ( 0.5,  0.5), 0.25  )
#geometry.addCircle( ( 0.5, -0.5), 0.25  )
#geometry.addCircle( (-0.5, -0.5), 0.25  )
geometry.addCircle( (  0.,   0.), 0.45 )
geometry.addLine( (-0.5,  0), (-1, 0), type=1 ) #type: 1->Periodic, 0->Real
geometry.addLine( ( 0,  0.5), ( 0, 1), type=1 )
geometry.addLine( ( 0.5,  0), ( 1, 0), type=1 )
geometry.addLine( ( 0, -0.5), ( 0,-1), type=1 )
nCircles, circlesCaract_h, nLines, linesCaract_h = geometry.prepareCUDA( cudaP=cudaP )



###########################################################################
###########################################################################
#Initialize and select CUDA device
cudaDev = setCudaDevice( devN = devN )

#Set thread grid dimentions
block = ( 256, 1, 1 )
grid = ( (nParticles - 1)//block[0] + 1, 1, 1 )

#Read and compile CUDA code
print "Compiling CUDA code"
codeFiles = [ "vector2D.h", "circle.h", "line.h", "cudaBillar.cu"]
for fileName in codeFiles:
  codeString = open(fileName, "r").read().replace("float", cudaP)
  outFile = open( fileName + "T", "w" )
  outFile.write( codeString )
  outFile.close()
  
cudaCodeStringTemp = open("cudaBillar.cuT", "r").read()
cudaCodeString = cudaCodeStringTemp % { "nCIRCLES":nCircles, "nLINES":nLines, "THREADS_PER_BLOCK":block[0] }
cudaCode = SourceModule(cudaCodeString, no_extern_c=True, include_dirs=[currentDirectory])
mainKernel = cudaCode.get_function("main_kernel" )

###########################################################################
###########################################################################
#Initialize Data
print "Initializing CUDA memory"
#np.random.seed(int(time.time()))  #Change numpy random seed
initialFreeMemory = getFreeMemory( show=True )
initialPosX_h = 0.4*np.ones(nParticles).astype(cudaPre)
initialPosY_h = 0.4*np.ones(nParticles).astype(cudaPre)
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
timeSampling_d = gpuarray.to_gpu( timeSampling  )
times_d = gpuarray.to_gpu( np.zeros(nParticles).astype(np.int32) )
nRestarts_d = gpuarray.to_gpu(np.zeros(nRuns+1).astype(np.int32))
finalFreeMemory = getFreeMemory( show=False )
print  " Total global memory used: {0:0.0f} MB".format( float(initialFreeMemory - finalFreeMemory)/1e6 ) 
###########################################################################
###########################################################################
#Prepare cuda kernels   
mainKernel.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, 
		    np.intp, np.intp, np.intp, np.intp, np.intp, np.intp, 
		    np.intp, np.intp, np.intp, 
		    np.intp, cudaPre, np.int32, np.int32, np.int32,
		    np.int32, np.intp])
#####################################################################
#Satart Simulation
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

#####################################################################
#Itererate for a particle bunch
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
  mainKernel(np.int32(collisionsPerRun), np.int32(nCircles), circlesCaract_d, np.int32(nLines), linesCaract_d,
	    initialPosX_d, initialPosY_d, initialVelX_d, initialVelY_d, initialRegionX_d, initialRegionY_d,
	    outPosX_d, outPosY_d, times_d,
	    timeSampling_d, cudaPre(deltaTime), np.int32(savePos), np.int32(particlesForPlot), np.int32(changeInitial),
	    np.int32(runNumber), nRestarts_d, grid=grid, block=block)

print "\n\nFinished in : {0:.4f}  sec\n".format( float( totalTime ) ) 
#######################################################################################
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
#nRestarts = nRestarts_d.get().sum()
times = times_d.get()

#END CUDA
#cuda.Context.pop()  #Disable previus CUDA context

plotPosGnuplot(pos)




























#print "Compiling CUDA code"

#codeFiles = [ "testDouble.cu"]
##if cudaP == "double":
#for fileName in codeFiles:
  #codeString = open(fileName, "r").read().replace("cudaP", cudaP)
#cudaCodeStringTemp = open("testDouble.cu", "r").read()
#cudaCodeString = cudaCodeStringTemp.replace("cudaP", cudaP)
##cudaCodeString = cudaCodeStringTemp % { "nCIRCLES":nCircles, "nLINES":nLines, "THREADS_PER_BLOCK":block[0] }
#cudaCode = SourceModule(cudaCodeString, no_extern_c=True, include_dirs=[currentDirectory])
#testDoubleKernel = cudaCode.get_function("testDouble_kernel" )
#nThreads = 1024*1024
#block = (256, 1, 1)
#grid = ((nThreads-1)/block[0]+1, 1, 1)

#input_h = np.ones(nThreads).astype(cudaPre)
#input_d = gpuarray.to_gpu( input_h ) 
#output_d = gpuarray.to_gpu( np.zeros_like(input_h) )
#testDoubleKernel( input_d, output_d, block=block, grid=grid )
#r = output_d.get()







































#############################################################
##Using matplotlib
##particleNumber = 0
#fig = plt.figure(0)
#plt.clf()
#plt.axes().set_aspect('equal', 'datalim')
#for particleNumber in range(nParticles/32):
  #plt.plot(outPosX_h[particleNumber], outPosY_h[particleNumber])
#geometry.plot()
#plt.show()



#print "Writting results to disk"
#outPut = np.nan*(np.empty([(nWrites+2)*nParticles,3]).astype(cudaPre))
#for i in range(nParticles):
  #outPut[i*(nWrites+2):(i+1)*(nWrites+2)-2,0] = i
  #outPut[i*(nWrites+2):(i+1)*(nWrites+2)-2,1] = outPosX_h[i]
  #outPut[i*(nWrites+2):(i+1)*(nWrites+2)-2,2] = outPosY_h[i]
#outString = str(outPut).replace("[)


