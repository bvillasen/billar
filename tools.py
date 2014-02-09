import sys

def printProgress( current, total, deltaIter,  deltaTime ):
  terminalString = "\rProgress: "
  if total==0: total+=1
  percent = 100.*current/total
  nDots = int(percent/5)
  dotsString = "[" + nDots*"." + (20-nDots)*" " + "]"
  percentString = "{0:.0f}%".format(percent)
  ETR = deltaTime*(total - current)/float(deltaIter)
  hours = int(ETR/3600)
  minutes = int(ETR - 3600*hours)/60
  seconds = int(ETR - 3600*hours - 60*minutes)
  ETRstring = "  ETR= {0}:{1:02}:{2:02}    ".format(hours, minutes, seconds)
  if deltaTime < 0.0001: ETRstring = "  ETR=    "
  terminalString  += dotsString + percentString + ETRstring
  sys.stdout. write(terminalString)
  sys.stdout.flush() 