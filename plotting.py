import numpy as np

def plotPosGnuplot( pos, ):
  import subprocess
  _3D = False
  if len(pos) == 3: _3D = True
  print "Plotting results "
  if _3D: "3D lines"
  if _3D: outPosX, outPosY, outPosZ = pos
  else: outPosX, outPosY = pos
  nParticles, nWrites = outPosX.shape
  rAvg = (np.sqrt(outPosX[:,-1]*outPosX[:,-1] + outPosY[:,-1]*outPosY[:,-1])).sum()/nParticles
  gnuplot = subprocess.Popen(["/usr/bin/gnuplot", "-persist"], stdin=subprocess.PIPE)
  #gnuplot.stdin.write("unset autoscale \n")
  #print rAvg
  #gnuplot.stdin.write("set yrange [-{0:.1f}:{0:.1f}]\n".format(float(rAvg*2)))
  #gnuplot.stdin.write("set xrange [-{0:.1f}:{0:.1f}]\n".format(float(rAvg*2)))
  gnuplot.stdin.write("set size ratio -1\n")
  if not _3D: gnuplot.stdin.write("plot '-' u 2:3:1 w l pal\n")
  else: gnuplot.stdin.write("splot '-' u 2:3:4:1 w l pal\n")
  particleOutput = np.zeros([nWrites,3])
  if _3D: particleOutput = np.zeros([nWrites,4])
  outputString = ""
  for particleNumber in range(nParticles):
    particleOutput[:,0] = particleNumber
    particleOutput[:,1] = outPosX[particleNumber]
    particleOutput[:,2] = outPosY[particleNumber]
    if _3D: particleOutput[:,3] = outPosZ[particleNumber] 
    particleOutputList = particleOutput.tolist()
    particleString = str(particleOutputList).replace("[","").replace("], ","\n").replace(",","").replace("]]","")
    particleString += "\n\n\n"
    outputString += particleString
  #text_file = open("Output.txt", "w")
  #text_file.write(outputString)
  #text_file.close()
  #print outputString
  gnuplot.stdin.write(outputString)
  gnuplot.stdin.write("e\n")
  gnuplot.stdin.flush()
  

def plotPoints3d( points, colors = None ):
  import subprocess
  print "Plotting results "
  nPoints = points.shape[0]
  gnuplot = subprocess.Popen(["/usr/bin/gnuplot", "-persist"], stdin=subprocess.PIPE)
  if colors != None:
    data = np.zeros([nPoints,4])
    data[:,:3] = points
    data[:,3] = colors
    gnuplot.stdin.write("splot '-' u 1:2:3:4 w p pal\n")
    outputString = str(data.tolist()).replace("[","").replace("], ","\n").replace(",","").replace("]]","")
  else:  
    gnuplot.stdin.write("splot '-' u 1:2:3 w p\n")
    outputString = str(points.tolist()).replace("[","").replace("], ","\n").replace(",","").replace("]]","")
  gnuplot.stdin.write(outputString)
  gnuplot.stdin.write("e\n")
  gnuplot.stdin.flush()










def plotPoints3d( points, colors = None ):
  import subprocess
  print "Plotting results "
  nPoints = points.shape[0]
  gnuplot = subprocess.Popen(["/usr/bin/gnuplot", "-persist"], stdin=subprocess.PIPE)
  if colors != None:
    data = np.zeros([nPoints,4])
    data[:,:3] = points
    data[:,3] = colors
    gnuplot.stdin.write("splot '-' u 1:2:3:4 w p pal\n")
    outputString = str(data.tolist()).replace("[","").replace("], ","\n").replace(",","").replace("]]","")
  else:  
    gnuplot.stdin.write("splot '-' u 1:2:3 w p\n")
    outputString = str(points.tolist()).replace("[","").replace("], ","\n").replace(",","").replace("]]","")
  gnuplot.stdin.write(outputString)
  gnuplot.stdin.write("e\n")
  gnuplot.stdin.flush()