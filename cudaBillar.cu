#include "vector2D.hT"
#include "circle.hT"
#include "line.hT"


__device__ void move( Vector2D &pos, Vector2D &vel, float time ){ 
  Vector2D deltaPos = vel/time;
  pos = pos + deltaPos ;
}

__device__ void initCircles( int tid, Circle *obstaclesCircle, float *cCrt ){
  obstaclesCircle[tid] = Circle( Vector2D(cCrt[4*tid+0], cCrt[4*tid+1] ), cCrt[4*tid+2], int(cCrt[4*tid+3]) );
}

__device__ void initLines( int tid, Line *obstaclesLine, float *lCrt ){
  obstaclesLine[tid] = Line( Vector2D(lCrt[5*tid+0], lCrt[5*tid+1] ), Vector2D(lCrt[5*tid+2], lCrt[5*tid+3] ), int(lCrt[5*tid+4]) );
}

__device__ void checkPos( Vector2D &pos, Vector2D &vel, bool &restart){
//   bool changed = false;
  if (pos.x<-0.5){ pos.x = -0.5f;  }
  if (pos.x> 0.5){ pos.x =  0.5f;  }
  if (pos.y<-0.5){ pos.y = -0.5f;  }
  if (pos.y> 0.5){ pos.y =  0.5f;  }
//   if ( pos.x==pos.y) restart = true;
}
  
__device__ void restartConditions( int tid, Vector2D &pos, Vector2D &vel, Vector2D &region,
				   int &collideWith, float &particleTime, float &samplingTime,
				   float *initPosX, float*initPosY, float *initVelX, float *initVelY,
				   float deltaTime, bool &restart, int runNumber, int *nRestarts){
  pos.x = initPosX[tid];
  pos.y = initPosY[tid];
  vel.x = initVelX[tid-1];
  vel.y = initVelY[tid+1];
  region.redefine(0.f,0.f);
  particleTime = 0.0f;
  samplingTime = deltaTime;
  collideWith = -1;
//   nRestarts[runNumber] += 1;
  atomicAdd( &nRestarts[runNumber], 1);
  restart = false;
}
  
extern "C"{
  
__global__ void main_kernel( const int collisionsPerRun, const int nCircles, float *circlesCaract, const int nLines, float *linesCaract,
			     float *initPosX, float *initPosY, float *initVelX, float *initVelY, int *initRegionX, int *initRegionY,
			     float *outPosX, float *outPosY, float *times,
			     float *timeSampling, float deltaTime, int savePos, int particlesForSave, int changeInitial, int runNumber, int *nRestarts  ){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
//   int nThreads = blockDim.x * gridDim.x;
  
  //Initialize particle position and velocity
  Vector2D pos( initPosX[tid], initPosY[tid] );
//   Vector2D posAbsolte( initPosX[tid], initPosY[tid] );
  Vector2D vel( initVelX[tid], initVelY[tid] );
  vel.normalize();
  Vector2D region( initRegionX[tid], initRegionY[tid] );
  float particleTime = times[tid];
//   float samplingTime = deltaTime;
//   bool insideCircle = false;
//   bool restart = false;
  
  
//   //Initialize Obstacles in shared memory
//   __shared__ Circle obstaclesCircle[ %(nCIRCLES)s ];
//   __shared__ Line obstaclesLine[ %(nLINES)s ];
//   if ( threadIdx.x < nCircles ) initCircles( threadIdx.x, obstaclesCircle, circlesCaract);
//   if ( threadIdx.x >= nCircles and threadIdx.x < nCircles+nLines) initLines( threadIdx.x - nCircles, obstaclesLine, linesCaract );
//   __syncthreads();
  
  //Initialize Obstacles
  Circle obstaclesCircle[ %(nCIRCLES)s ];
  Line obstaclesLine[ %(nLINES)s ];
  for (int i=0; i<nCircles; i++)
    obstaclesCircle[i] = Circle( Vector2D(circlesCaract[4*i+0], circlesCaract[4*i+1] ), circlesCaract[4*i+2], int(circlesCaract[4*i+3]) );
  for (int i=0; i<nLines; i++)
    obstaclesLine[i] = Line( Vector2D(linesCaract[5*i+0], linesCaract[5*i+1] ), Vector2D(linesCaract[5*i+2], linesCaract[5*i+3] ), int(linesCaract[5*i+4]) );

//   //Initialize shared array for radiusSampling
//   __shared__ float radiusArray[ %(THREADS_PER_BLOCK)s ];
//   radiusArray[threadIdx.x] = -1.f;
//   __syncthreads();
//   
//   
  
  int collideWith = -1;
  float timeMin, timeTemp;
  for (int collisionNumber=0; collisionNumber<collisionsPerRun; collisionNumber++){
    timeMin = 1e20;
    for (int i=0; i<nCircles; i++){
      if ( i == collideWith ) continue;
      timeTemp = obstaclesCircle[i].collideTime( pos, vel );
      if (timeTemp < timeMin and timeTemp > 0){
	timeMin = timeTemp;
	collideWith = i;
      }
    }
    for (int i=0; i<nLines; i++){
      if ( i+nCircles == collideWith ) continue;
      timeTemp = obstaclesLine[i].collideTime( pos, vel );
      if (timeTemp < timeMin and timeTemp > 0){
	timeMin = timeTemp;
	collideWith = i+nCircles;
      }
    }

    move( pos, vel, timeMin );
    particleTime += timeMin;
//     checkPos( pos, vel, restart );
//     if (restart){
//       restartConditions( tid, pos, vel, region, collideWith, particleTime, samplingTime,
// 			 initPosX, initPosY, initVelX, initVelY, deltaTime, restart, runNumber, nRestarts);
//       restart = false;
//       continue;
//     }
//     
    if ( collideWith < nCircles ) obstaclesCircle[collideWith].bounce(pos, vel);
    else{
      if (obstaclesLine[collideWith-nCircles].isPeriodic()){
	obstaclesLine[collideWith-nCircles].bouncePeriodic(pos, vel, region);
	collideWith = (collideWith-nCircles + 2 )%%4 + nCircles; // Only for square geometry
      }
      else obstaclesLine[collideWith-nCircles].bounce(pos, vel);
    }
    
//     posAbsolte = pos + region;
    if (savePos==1){
      if (tid < particlesForSave){
      outPosX[particlesForSave*collisionNumber + tid] = pos.x + region.x;
      outPosY[particlesForSave*collisionNumber + tid] = pos.y + region.y;
    //     outVelX[particlesForSave*writeNumber + tid] = vel.x;
    //     outVelY[particlesForSave*writeNumber + tid] = vel.y;
      }
    }
  }
 
  //Save final states
  if (changeInitial==1){
    initPosX[tid] = pos.x;
    initPosY[tid] = pos.y;
    initVelX[tid] = vel.x;
    initVelY[tid] = vel.y;
    initRegionX[tid] = region.x;
    initRegionY[tid] = region.y;
    times[tid] = particleTime;
  }
//   for (int i=0; i<nCircles; i++)
//     delete[] &obstaclesCircle[i];
//   for (int i=0; i<nLines; i++)
//     delete[] &obstaclesLine[i]; 
//   delete[] &pos;
//   delete[] &posAbsolte;
//   delete[] &vel;
//   delete[] &region;
}
  
}//Extern C end
  