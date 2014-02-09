import numpy as np

def plotPosGnuplot( pos ):
  import subprocess
  print "Plotting results "
  outPosX, outPosY = pos
  nParticles, nWrites = outPosX.shape
  rAvg = (np.sqrt(outPosX[:,-1]*outPosX[:,-1] + outPosY[:,-1]*outPosY[:,-1])).sum()/nParticles
  gnuplot = subprocess.Popen(["/usr/bin/gnuplot", "-persist"], stdin=subprocess.PIPE)
  #gnuplot.stdin.write("unset autoscale \n")
  #print rAvg
  #gnuplot.stdin.write("set yrange [-{0:.1f}:{0:.1f}]\n".format(float(rAvg*2)))
  #gnuplot.stdin.write("set xrange [-{0:.1f}:{0:.1f}]\n".format(float(rAvg*2)))
  gnuplot.stdin.write("set size ratio -1\n")
  gnuplot.stdin.write("plot '-' u 2:3:1 w l pal\n")
  
  particleOutput = np.zeros([nWrites,3])
  outputString = ""
  for particleNumber in range(nParticles):
    particleOutput[:,0] = particleNumber
    particleOutput[:,1] = outPosX[particleNumber]
    particleOutput[:,2] = outPosY[particleNumber]
    particleOutputList = particleOutput.tolist()
    particleString = str(particleOutputList).replace("[","").replace("], ","\n").replace(",","")
    particleString += "\n\n"
    outputString += particleString
  #text_file = open("Output.txt", "w")
  #text_file.write(outputString)
  #text_file.close()
  gnuplot.stdin.write(outputString)
  gnuplot.stdin.write("e\n")
  gnuplot.stdin.flush()