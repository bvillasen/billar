import pycuda.driver as cuda

######################################################################
def setCudaDevice( devN = None, usingAnimation=False  ):
  import pycuda.autoinit
  nDevices = cuda.Device.count()
  print "Available Devices:"
  for i in range(nDevices):
    dev = cuda.Device( i )
    print "  Device {0}: {1}".format( i, dev.name() )
  devNumber = 0
  if nDevices > 1:
    if devN == None: 
      devNumber = int(raw_input("Select device number: "))  
    else:
      devNumber = devN 
  dev = cuda.Device( devNumber)
  cuda.Context.pop()  #Disable previus CUDA context
  if usingAnimation:
    cuda_gl.make_context(dev)
  else:
    dev.make_context()
  print "Using device {0}: {1}".format( devNumber, dev.name() ) 
  return dev

#####################################################################
def getFreeMemory( show=True):
  Mbytes = float(cuda.mem_get_info()[0])/1e6
  if show:
    print " Free Global Memory: {0:.0f} MB".format(float(Mbytes))
  return cuda.mem_get_info()[0]
#####################################################################
def kernelMemoryInfo(kernel):
  shared=kernel.shared_size_bytes
  regs=kernel.num_regs
  local=kernel.local_size_bytes
  const=kernel.const_size_bytes
  mbpt=kernel.max_threads_per_block
  print("""=MEM=\nLocal:%d,\nShared:%d,\nRegisters:%d,\nConst:%d,\nMax Threads/B:%d"""%(local,shared,regs,const,mbpt))
 