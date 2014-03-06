#include "vector2D.hT"
#include "circle.hT"
#include "line.hT"


__device__ void move( Vector2D &pos, Vector2D &vel, cudaP time ){ 
  Vector2D deltaPos = vel/time;
  pos = pos + deltaPos ;
}

// __device__ void initCircles( int tid, Circle *obstaclesCircle, cudaP *cCrt ){
//   obstaclesCircle[tid] = Circle( Vector2D(cCrt[4*tid+0], cCrt[4*tid+1] ), cCrt[4*tid+2], int(cCrt[4*tid+3]) );
// }
// 
// __device__ void initLines( int tid, Line *obstaclesLine, cudaP *lCrt ){
//   obstaclesLine[tid] = Line( Vector2D(lCrt[5*tid+0], lCrt[5*tid+1] ), Vector2D(lCrt[5*tid+2], lCrt[5*tid+3] ), int(lCrt[5*tid+4]) );
// }

// __device__ void checkPos( Vector2D &pos, Vector2D &vel, bool &restart){
// //   bool changed = false;
//   if (pos.x<-0.5){ pos.x = -0.5f;  }
//   if (pos.x> 0.5){ pos.x =  0.5f;  }
//   if (pos.y<-0.5){ pos.y = -0.5f;  }
//   if (pos.y> 0.5){ pos.y =  0.5f;  }
// //   if ( pos.x==pos.y) restart = true;
// }
//   
// __device__ void restartConditions( int tid, Vector2D &pos, Vector2D &vel, Vector2D &region,
// 				   int &collideWith, cudaP &particleTime, cudaP &samplingTime,
// 				   cudaP *initPosX, cudaP*initPosY, cudaP *initVelX, cudaP *initVelY,
// 				   cudaP deltaTime, bool &restart, int runNumber, int *nRestarts){
//   pos.x = initPosX[tid];
//   pos.y = initPosY[tid];
//   vel.x = initVelX[tid-1];
//   vel.y = initVelY[tid+1];
//   region.redefine(0.f,0.f);
//   particleTime = 0.0f;
//   samplingTime = deltaTime;
//   collideWith = -1;
// //   nRestarts[runNumber] += 1;
//   atomicAdd( &nRestarts[runNumber], 1);
//   restart = false;
// }
  
extern "C"{
  
__global__ void main_kernel( const unsigned char usingAnimation, const int nParticles, const int collisionsPerRun, 
			     const int nCircles, cudaP *circlesCaract, const int nLines, cudaP *linesCaract,
			     cudaP *initPosX, cudaP *initPosY, cudaP *initVelX, cudaP *initVelY, int *initRegionX, int *initRegionY,
			     cudaP *outPosX, cudaP *outPosY, cudaP *times,
			     float deltaTime_anim, int *timesIdx_anim,  
			     float deltaTime_rad, int *timesIdx_rad, int *timesOccupancy, float *radiusAll,
			     int savePosForPlot, int particlesForSave, int changeInitial, 
			     float *cuda_VOB){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
//   int nThreads = blockDim.x * gridDim.x;
  
  while (tid < nParticles){
    
    //Initialize particle position and velocity
    Vector2D pos( initPosX[tid], initPosY[tid] );
    Vector2D vel( initVelX[tid], initVelY[tid] );
    vel.normalize();
    int region[2] = { initRegionX[tid], initRegionY[tid] };// initRegionX[tid], initRegionY[tid] );
    cudaP particleTime = times[tid];
    int timeIdx_anim = timesIdx_anim[tid];
    int timeIdx_rad = timesIdx_rad[tid];
    
//     //Initialize Obstacles in shared memory
//     __shared__ Circle obstaclesCircle[ %(nCIRCLES)s ];
//     __shared__ Line obstaclesLine[ %(nLINES)s ];
//     if ( threadIdx.x < nCircles ) initCircles( threadIdx.x, obstaclesCircle, circlesCaract);
//     if ( threadIdx.x < nLines) initLines( threadIdx.x, obstaclesLine, linesCaract );
//     __syncthreads();
    
    //Initialize Obstacles
    Circle obstaclesCircle[ %(nCIRCLES)s ];
    Line obstaclesLine[ %(nLINES)s ];
    for (int i=0; i<nCircles; i++)
      obstaclesCircle[i] = Circle( Vector2D(circlesCaract[4*i+0], circlesCaract[4*i+1] ), circlesCaract[4*i+2], int(circlesCaract[4*i+3]) );
    for (int i=0; i<nLines; i++)
      obstaclesLine[i] = Line( Vector2D(linesCaract[5*i+0], linesCaract[5*i+1] ), Vector2D(linesCaract[5*i+2], linesCaract[5*i+3] ), int(linesCaract[5*i+4]) );

    //Initialize shared array for position sampling
    __shared__ float posX_sh[ %(THREADS_PER_BLOCK)s ];
    __shared__ float posY_sh[ %(THREADS_PER_BLOCK)s ];
//     for (int i=0; i<timeIdxMax; i++){
    posX_sh[threadIdx.x] = float(pos.x + region[0]);
    posY_sh[threadIdx.x] = float(pos.y + region[1]);
    __shared__ int timesOccupancy_sh[ %(TIME_INDEX_MAX)s ];
    __shared__ float radiusAll_sh[ %(TIME_INDEX_MAX)s ];
    timesOccupancy_sh[threadIdx.x] = 0;
    radiusAll_sh[threadIdx.x] = 0.f;
    __syncthreads();
     
    int collideWith = -1;
    cudaP timeMin, timeTemp;
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
      particleTime += timeMin;
      move( pos, vel, timeMin );
      if (usingAnimation){
	if (particleTime >= timeIdx_anim*deltaTime_anim ){
	  if (timeIdx_anim != 0) move( pos, vel, timeIdx_anim*deltaTime_anim-particleTime );
	  posX_sh[threadIdx.x] = pos.x + region[0];
	  posY_sh[threadIdx.x] = pos.y + region[1];
	  if (timeIdx_anim != 0) move( pos, vel, particleTime-timeIdx_anim*deltaTime_anim );
	  timeIdx_anim +=1;
	}
      }
    
      
      if (particleTime >= timeIdx_rad*deltaTime_rad and timeIdx_rad< %(TIME_INDEX_MAX)s ){
	move( pos, vel, timeIdx_rad*deltaTime_rad-particleTime );
	atomicAdd( &(timesOccupancy_sh[timeIdx_rad]) , 1);
	atomicAdd( &(radiusAll_sh[timeIdx_rad]) , float((pos.x+region[0])*(pos.x+region[0]) + (pos.y+region[1])*(pos.y+region[1])) );
	move( pos, vel, particleTime-timeIdx_rad*deltaTime_rad );
	timeIdx_rad +=1;
      }
         
      if ( collideWith < nCircles ) obstaclesCircle[collideWith].bounce(pos, vel);
      else{
	if (obstaclesLine[collideWith-nCircles].isPeriodic()){
	  obstaclesLine[collideWith-nCircles].bouncePeriodic(pos, vel, region);
	  collideWith = (collideWith-nCircles + 2 )%%4 + nCircles; // Only for square geometry
	}
	else obstaclesLine[collideWith-nCircles].bounce(pos, vel);
      }
      
      if (savePosForPlot==1){
	if (tid < particlesForSave){
	outPosX[particlesForSave*collisionNumber + tid] = pos.x + region[0];
	outPosY[particlesForSave*collisionNumber + tid] = pos.y + region[1];
	}
      }
    }
    
    //Save data in animation buffer
    if (usingAnimation){
      cuda_VOB[2*tid] = posX_sh[threadIdx.x];
      cuda_VOB[2*tid + 1] = posY_sh[threadIdx.x];
    }
    //Save final states
    if (changeInitial==1){
      initPosX[tid] = pos.x;
      initPosY[tid] = pos.y;
      initVelX[tid] = vel.x;
      initVelY[tid] = vel.y;
      initRegionX[tid] = region[0];
      initRegionY[tid] = region[1];
      times[tid] = particleTime;
      timesIdx_anim[tid] = timeIdx_anim;
      timesIdx_rad[tid] = timeIdx_rad;
    }
    __syncthreads();
    atomicAdd( &(timesOccupancy[threadIdx.x]), timesOccupancy_sh[threadIdx.x] );
    atomicAdd( &(radiusAll[threadIdx.x]), radiusAll_sh[threadIdx.x]/nParticles );
    
    tid += blockDim.x * gridDim.x;
  }
  
}
  
}//Extern C end
  