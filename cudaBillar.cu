#include "vector2D.hT"
#include "circle.hT"
#include "line.hT"


__device__ void move( Vector2D &pos, Vector2D &vel, cudaP time ){ 
  Vector2D deltaPos = vel/time;
  pos = pos + deltaPos ;
}

__device__ void initCircles( int tid, Circle *obstaclesCircle, cudaP *cCrt ){
  obstaclesCircle[tid] = Circle( Vector2D(cCrt[4*tid+0], cCrt[4*tid+1] ), cCrt[4*tid+2], int(cCrt[4*tid+3]) );
}

__device__ void initLines( int tid, Line *obstaclesLine, cudaP *lCrt ){
  obstaclesLine[tid] = Line( Vector2D(lCrt[5*tid+0], lCrt[5*tid+1] ), Vector2D(lCrt[5*tid+2], lCrt[5*tid+3] ), int(lCrt[5*tid+4]) );
}
  
extern "C"{
  
__global__ void main_kernel( const unsigned char usingAnimation, const int nParticles, const int collisionsPerRun, 
			     const int nCircles, cudaP *circlesCaract, const int nLines, cudaP *linesCaract,
			     cudaP *initPosX, cudaP *initPosY, cudaP *initVelX, cudaP *initVelY, int *initRegionX, int *initRegionY,
			     cudaP *outPosX, cudaP *outPosY, cudaP *times,
			     const float deltaTime_anim, int *timesIdx_anim,  
			     const float deltaTime_rad, int *timesIdx_rad, int *timesOccupancy, float *radiusAll,
			     const unsigned char savePosForPlot, const int particlesForSave, const unsigned char changeInitial, 
			     float *cuda_VOB){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int tid_b = threadIdx.x;
  
  //Initialize Obstacles
  Circle obstaclesCircle[ %(nCIRCLES)s ];
  Line obstaclesLine[ %(nLINES)s ];
  for (int i=0; i<nCircles; i++)
    obstaclesCircle[i] = Circle( Vector2D(circlesCaract[4*i+0], circlesCaract[4*i+1] ), circlesCaract[4*i+2], int(circlesCaract[4*i+3]) );
  for (int i=0; i<nLines; i++)
    obstaclesLine[i] = Line( Vector2D(linesCaract[5*i+0], linesCaract[5*i+1] ), Vector2D(linesCaract[5*i+2], linesCaract[5*i+3] ), int(linesCaract[5*i+4]) );

//   //Initialize Obstacles in shared memory
//   __shared__ Circle obstaclesCircle[ %(nCIRCLES)s ];
//   __shared__ Line obstaclesLine[ %(nLINES)s ];
//   if ( threadIdx.x < nCircles ) initCircles( threadIdx.x, obstaclesCircle, circlesCaract);
//   if ( threadIdx.x < nLines) initLines( threadIdx.x, obstaclesLine, linesCaract );
  //Initialize shared array for position sampling
  __shared__ float posY_sh[ %(THREADS_PER_BLOCK)s ];
  __shared__ float posX_sh[ %(THREADS_PER_BLOCK)s ];
  __shared__ int timesOccupancy_sh[ %(TIME_INDEX_MAX)s ];
  __shared__ float radiusAll_sh[ %(TIME_INDEX_MAX)s ];  
  while ( tid_b < %(TIME_INDEX_MAX)s ){
    timesOccupancy_sh[ tid_b ] = 0;
    radiusAll_sh[ tid_b ] = 0.f;
    tid_b += blockDim.x;
  }
  tid_b = threadIdx.x;
  __syncthreads();
  
  while (tid < nParticles){
    int timeIdx_rad = timesIdx_rad[tid];
      
    //Initialize particle position and velocity
    Vector2D pos( initPosX[tid], initPosY[tid] );
    Vector2D vel( initVelX[tid], initVelY[tid] );
    vel.normalize();
    int region[2] = { initRegionX[tid], initRegionY[tid] };// initRegionX[tid], initRegionY[tid] );
    cudaP particleTime = times[tid];
    int timeIdx_anim = timesIdx_anim[tid];

    posX_sh[threadIdx.x] = float(pos.x + region[0]);
    posY_sh[threadIdx.x] = float(pos.y + region[1]);

    int collideWith = -1;
    cudaP timeMin, timeTemp;
    if ( timeIdx_rad <= %(TIME_INDEX_MAX)s or usingAnimation ){
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
	if (usingAnimation and particleTime >= timeIdx_anim*deltaTime_anim){
	  if (timeIdx_anim != 0) move( pos, vel, timeIdx_anim*deltaTime_anim-particleTime );
	  posX_sh[threadIdx.x] = pos.x + region[0];
	  posY_sh[threadIdx.x] = pos.y + region[1];
	  if (timeIdx_anim != 0) move( pos, vel, particleTime-timeIdx_anim*deltaTime_anim );
	  timeIdx_anim +=1;
	}
      
	if (particleTime >= (timeIdx_rad*deltaTime_rad) and ( timeIdx_rad <= %(TIME_INDEX_MAX)s ) ){
	  move( pos, vel, (timeIdx_rad*deltaTime_rad)-particleTime );
	  atomicAdd( &(timesOccupancy_sh[timeIdx_rad-1]) , 1);
	  atomicAdd( &(radiusAll_sh[timeIdx_rad-1]) , float( ( (pos.x+region[0])*(pos.x+region[0]) + (pos.y+region[1])*(pos.y+region[1]) ) ) );
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
    }
    
    //Save data in animation buffer
    if (usingAnimation){
      timesIdx_anim[tid] = timeIdx_anim;
      cuda_VOB[2*tid] = posX_sh[threadIdx.x];
      cuda_VOB[2*tid + 1] = posY_sh[threadIdx.x];
    }
    //Save final states
    if (changeInitial==1 ){
      initPosX[tid] = pos.x;
      initPosY[tid] = pos.y;
      initVelX[tid] = vel.x;
      initVelY[tid] = vel.y;
      initRegionX[tid] = region[0];
      initRegionY[tid] = region[1];
      times[tid] = particleTime;
      timesIdx_rad[tid] = timeIdx_rad;
    }
    tid += blockDim.x * gridDim.x;
  }
  
  __syncthreads();
  while ( tid_b < %(TIME_INDEX_MAX)s ){
    atomicAdd( &(timesOccupancy[tid_b]), timesOccupancy_sh[tid_b] );
    atomicAdd( &(radiusAll[tid_b]), radiusAll_sh[tid_b]/nParticles );
    tid_b += blockDim.x;
  }
}
 
__global__ void mainSimple_kernel(  const int nParticles, const int collisionsPerRun, 
			     const int nCircles, cudaP *circlesCaract, const int nLines, cudaP *linesCaract,
			     cudaP *initPosX, cudaP *initPosY, cudaP *initVelX, cudaP *initVelY, int *initRegionX, int *initRegionY,
			     cudaP *times, 
			     const float deltaTime_rad, int *timesIdx_rad, int *timesOccupancy, float *radiusAll,
			     const unsigned char changeInitial){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int tid_b = threadIdx.x;
  int i;
  
//   Initialize Obstacles
  Circle obstaclesCircle[ %(nCIRCLES)s ];
  Line obstaclesLine[ %(nLINES)s ];
  for ( i=0; i<nCircles; i++)
    obstaclesCircle[i] = Circle( Vector2D(circlesCaract[4*i+0], circlesCaract[4*i+1] ), circlesCaract[4*i+2], int(circlesCaract[4*i+3]) );
  for ( i=0; i<nLines; i++)
    obstaclesLine[i] = Line( Vector2D(linesCaract[5*i+0], linesCaract[5*i+1] ), Vector2D(linesCaract[5*i+2], linesCaract[5*i+3] ), int(linesCaract[5*i+4]) );

//   //Initialize Obstacles in shared memory
//   __shared__ Circle obstaclesCircle[ %(nCIRCLES)s ];
//   __shared__ Line obstaclesLine[ %(nLINES)s ];
//   if ( threadIdx.x < nCircles ) initCircles( threadIdx.x, obstaclesCircle, circlesCaract);
//   if ( threadIdx.x < nLines) initLines( threadIdx.x, obstaclesLine, linesCaract );

  __shared__ int timesOccupancy_sh[ %(TIME_INDEX_MAX)s ];
  __shared__ float radiusAll_sh[ %(TIME_INDEX_MAX)s ];  
  while ( tid_b < %(TIME_INDEX_MAX)s ){
    timesOccupancy_sh[ tid_b ] = 0;
    radiusAll_sh[ tid_b ] = 0.f;
    tid_b += blockDim.x;
  }
  tid_b = threadIdx.x;
//   __shared__ Vector2D pos_sh[ %(THREADS_PER_BLOCK)s ];
//   __shared__ Vector2D vel_sh[ %(THREADS_PER_BLOCK)s ];
  __syncthreads();
  

  
  int timeIdx_rad, collideWith, collisionNumber; 
  cudaP timeMin, timeTemp;
  while (tid < nParticles){
    timeIdx_rad = timesIdx_rad[tid];
      
    //Initialize particle position and velocity
    Vector2D pos( initPosX[tid], initPosY[tid] );
//     pos_sh[tid_b].redefine( initPosX[tid], initPosY[tid] );
    Vector2D vel( initVelX[tid], initVelY[tid] );
    vel.normalize();
    int region[2] = { initRegionX[tid], initRegionY[tid] };// initRegionX[tid], initRegionY[tid] );
    cudaP particleTime = times[tid];

    collideWith = -1;
    if ( timeIdx_rad <= %(TIME_INDEX_MAX)s  ){
      for (collisionNumber=0; collisionNumber<collisionsPerRun; collisionNumber++){
	timeMin = 1e20;
	for (i=0; i<nCircles; i++){
	  if ( i == collideWith ) continue;
	  timeTemp = obstaclesCircle[i].collideTime( pos, vel );
	  if (timeTemp < timeMin and timeTemp > 0){
	    timeMin = timeTemp;
	    collideWith = i;
	  }
	}
	for ( i=0; i<nLines; i++){
	  if ( i+nCircles == collideWith ) continue;
	  timeTemp = obstaclesLine[i].collideTime( pos, vel );
	  if (timeTemp < timeMin and timeTemp > 0){
	    timeMin = timeTemp;
	    collideWith = i+nCircles;
	  }
	}
	particleTime += timeMin;
	move( pos, vel, timeMin );
      
	if (particleTime >= (timeIdx_rad*deltaTime_rad) and ( timeIdx_rad <= %(TIME_INDEX_MAX)s ) ){
	  move( pos, vel, (timeIdx_rad*deltaTime_rad)-particleTime );
	  atomicAdd( &(timesOccupancy_sh[timeIdx_rad-1]) , 1);
	  atomicAdd( &(radiusAll_sh[timeIdx_rad-1]) , float( ( (pos.x+region[0])*(pos.x+region[0]) + (pos.y+region[1])*(pos.y+region[1]) ) ) );
	  move( pos, vel, particleTime-timeIdx_rad*deltaTime_rad );
	  timeIdx_rad +=1;
	}
	  
	if ( collideWith < nCircles ) obstaclesCircle[collideWith].bounce(pos, vel);
	else{
	  obstaclesLine[collideWith-nCircles].bouncePeriodic(pos, vel, region);
	  collideWith = (collideWith-nCircles + 2 )%%4 + nCircles; // Only for square geometry  
	}
      }
    }
    //Save final states
    if (changeInitial==1 ){
      initPosX[tid] = pos.x;
      initPosY[tid] = pos.y;
      initVelX[tid] = vel.x;
      initVelY[tid] = vel.y;
      initRegionX[tid] = region[0];
      initRegionY[tid] = region[1];
      times[tid] = particleTime;
      timesIdx_rad[tid] = timeIdx_rad;
    }
    tid += blockDim.x * gridDim.x;
  }
  
  __syncthreads();
  while ( tid_b < %(TIME_INDEX_MAX)s ){
    atomicAdd( &(timesOccupancy[tid_b]), timesOccupancy_sh[tid_b] );
    atomicAdd( &(radiusAll[tid_b]), radiusAll_sh[tid_b]/nParticles );
    tid_b += blockDim.x;
  }
} 
}//Extern C end
  