import pylab as plt
import numpy as np

class Geometry:
  
  def __init__(self):
    self.circles = {}
    self.lines = {}
  
  def addCircle(self, center=(0,0), radius=1, type=0, key=None):
    if not key: key = len(self.circles)
    self.circles[key] = { "c":center, "r":radius , "t":type }
  
  def addLine(self, center=(0,0), normal=(1,0), length=1, type=0, key=None):
    if not key: key = len(self.lines)
    self.lines[key] = { "c":center, "n":normal , "t":type }
  
  def prepareCUDA(self, cudaP="float"):
    circlesPar = []
    for (key,circle) in self.circles.items():
      par = [ circle["c"][0], circle["c"][1], circle["r"], circle["t"]  ]
      circlesPar.append(par)
    linesPar = []
    for (key,line) in self.lines.items():
      par = [ line["c"][0], line["c"][1], line["n"][0], line["n"][1], line["t"]  ]
      linesPar.append(par)
    cudaPrec = np.float32
    if cudaP == "double": cudaPrec = np.float64
    circlesPar = np.array( circlesPar ).astype(cudaPrec)
    linesPar = np.array( linesPar ).astype(cudaPrec)
    return len(self.circles), circlesPar, len(self.lines), linesPar
  
  def plot(self, show=False):
    fig = plt.figure(0)
    #plt.clf()
    plt.axes().set_aspect('equal', 'datalim')
    for (key,circle) in self.circles.items():
      c = plt.Circle( circle["c"], circle["r"] )  
      fig.gca().add_artist(c)
    plt.axis([-1, 1, -1, 1] )
    if show: plt.show()
      
    
    