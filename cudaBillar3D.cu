#include "vector3D.hT"
#include "sphere.hT"
#include "wall.hT"


__device__ void move( Vector3D &pos, Vector3D &vel, cudaP time ){ 
  Vector3D deltaPos = vel/time;
  pos = pos + deltaPos ;
}

// __device__ void initSpheres( int tid, Sphere *obstaclesSphere, cudaP *cCrt ){
//   obstaclesSphere[tid] = Sphere( Vector3D(cCrt[4*tid+0], cCrt[4*tid+1] ), cCrt[4*tid+2], int(cCrt[4*tid+3]) );
// }
// 
// __device__ void initWalls( int tid, Wall *obstaclesWall, cudaP *lCrt ){
//   obstaclesWall[tid] = Wall( Vector3D(lCrt[5*tid+0], lCrt[5*tid+1] ), Vector3D(lCrt[5*tid+2], lCrt[5*tid+3] ), int(lCrt[5*tid+4]) );
// }
  
extern "C"{
  
__global__ void main_kernel( const unsigned char usingAnimation, const int nParticles, const int collisionsPerRun, 
			     const int nSpheres, cudaP *spheresCaract, const int nWalls, cudaP *wallsCaract,
			     cudaP *initPosX, cudaP *initPosY, cudaP *initPosZ, 
			     cudaP *initVelX, cudaP *initVelY, cudaP *initVelZ,
			     int *initRegionX, int *initRegionY, int *initRegionZ,
			     cudaP *outPosX, cudaP *outPosY, cudaP *outPosZ,
			     cudaP *times,
			     float deltaTime_anim, int *timesIdx_anim,  
			     float deltaTime_rad, int *timesIdx_rad, int *timesOccupancy, float *radiusAll, 
			     int savePos, int particlesForSave, int changeInitial, float *cuda_VOB){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
//   int nThreads = blockDim.x * gridDim.x;
  
  if (tid < nParticles){
    
    //Initialize particle position and velocity
    Vector3D pos( initPosX[tid], initPosY[tid], initPosZ[tid] );
    Vector3D vel( initVelX[tid], initVelY[tid], initVelZ[tid] );
    vel.normalize();
    int region[3] = { initRegionX[tid], initRegionY[tid], initRegionZ[tid] };// initRegionX[tid], initRegionY[tid] );
    cudaP particleTime = times[tid];
    int timeIdx_anim = timesIdx_anim[tid];
    int timeIdx_rad = timesIdx_rad[tid];
    
//     //Initialize Obstacles in shared memory
//     __shared__ Sphere obstaclesSphere[ %(nSPHERES)s ];
//     __shared__ Wall obstaclesWall[ %(nWALLS)s ];
//     if ( threadIdx.x < nSpheres ) initSpheres( threadIdx.x, obstaclesSphere, spheresCaract);
//     if ( threadIdx.x < nWalls) initWalls( threadIdx.x, obstaclesWall, wallsCaract );
//     __syncthreads();
    
    //Initialize Obstacles
    Sphere obstaclesSphere[ %(nSPHERES)s ];
    Wall obstaclesWall[ %(nWALLS)s ];
    for (int i=0; i<nSpheres; i++)
      obstaclesSphere[i] = Sphere( Vector3D(spheresCaract[5*i+0], spheresCaract[5*i+1], spheresCaract[5*i+2] ), spheresCaract[5*i+3], int(spheresCaract[5*i+4]) );
    for (int i=0; i<nWalls; i++)
      obstaclesWall[i] = Wall( Vector3D(wallsCaract[7*i+0], wallsCaract[7*i+1], wallsCaract[7*i+2] ), Vector3D(wallsCaract[7*i+3], wallsCaract[7*i+4], wallsCaract[7*i+5] ), int(wallsCaract[7*i+6]) );

    //Initialize shared array for position sampling
    __shared__ float posX_sh[ %(THREADS_PER_BLOCK)s ];
    __shared__ float posY_sh[ %(THREADS_PER_BLOCK)s ];
    __shared__ float posZ_sh[ %(THREADS_PER_BLOCK)s ];
//     for (int i=0; i<timeIdxMax; i++){
    posX_sh[threadIdx.x] = float(pos.x + region[0]);
    posY_sh[threadIdx.x] = float(pos.y + region[1]);
    posZ_sh[threadIdx.x] = float(pos.z + region[2]);
    __shared__ int timesOccupancy_sh[ %(TIME_INDEX_MAX)s ];
    __shared__ float radiusAll_sh[ %(TIME_INDEX_MAX)s ];
    timesOccupancy_sh[threadIdx.x] = 0;
    radiusAll_sh[threadIdx.x] = 0.f;
    __syncthreads();
     
    int collideWith = -1;
    cudaP timeMin, timeTemp;
    for (int collisionNumber=0; collisionNumber<collisionsPerRun; collisionNumber++){
      timeMin = 1e20;
      for (int i=0; i<nSpheres; i++){
	if ( i == collideWith ) continue;
	timeTemp = obstaclesSphere[i].collideTime( pos, vel );
	if (timeTemp < timeMin and timeTemp > 0){
	  timeMin = timeTemp;
	  collideWith = i;
	}
      }
      for (int i=0; i<nWalls; i++){
	if ( i+nSpheres == collideWith ) continue;
	timeTemp = obstaclesWall[i].collideTime( pos, vel );
	if (timeTemp < timeMin and timeTemp > 0){
	  timeMin = timeTemp;
	  collideWith = i+nSpheres;
	}
      }
      particleTime += timeMin;
      move( pos, vel, timeMin );
      if (usingAnimation){
	if (particleTime >= timeIdx_anim*deltaTime_anim ){
	  if (timeIdx_anim != 0) move( pos, vel, timeIdx_anim*deltaTime_anim-particleTime );
	  posX_sh[threadIdx.x] = pos.x + region[0];
	  posY_sh[threadIdx.x] = pos.y + region[1];
	  posZ_sh[threadIdx.x] = pos.z + region[2];
	  if (timeIdx_anim != 0) move( pos, vel, particleTime-timeIdx_anim*deltaTime_anim );
	  timeIdx_anim +=1;
	}
      }
      
      if (particleTime >= timeIdx_rad*deltaTime_rad and timeIdx_rad< %(TIME_INDEX_MAX)s ){
	move( pos, vel, timeIdx_rad*deltaTime_rad-particleTime );
	atomicAdd( &(timesOccupancy_sh[timeIdx_rad]) , 1);
	atomicAdd( &(radiusAll_sh[timeIdx_rad]) , float((pos.x+region[0])*(pos.x+region[0]) + (pos.y+region[1])*(pos.y+region[1]) + (pos.z+region[2])*(pos.z+region[2])) );
	move( pos, vel, particleTime-timeIdx_rad*deltaTime_rad );
	timeIdx_rad +=1;
      }
         
      if ( collideWith < nSpheres ) obstaclesSphere[collideWith].bounce(pos, vel);
      else{
	if (obstaclesWall[collideWith-nSpheres].isPeriodic()){
	  obstaclesWall[collideWith-nSpheres].bouncePeriodic(pos, vel, region);
	  collideWith = (collideWith-nSpheres + 3 )%%6 + nSpheres; // Only for square geometry
	}
	else obstaclesWall[collideWith-nSpheres].bounce(pos, vel);
      }
      
      if (savePos==1){
	if (tid < particlesForSave){
	outPosX[particlesForSave*collisionNumber + tid] = pos.x + region[0];
	outPosY[particlesForSave*collisionNumber + tid] = pos.y + region[1];
	outPosZ[particlesForSave*collisionNumber + tid] = pos.z + region[2];
	  
	}
      }
    }
    
    //Save data in animation buffer
    if (usingAnimation){
      cuda_VOB[3*tid + 0] = posX_sh[threadIdx.x];
      cuda_VOB[3*tid + 1] = posY_sh[threadIdx.x];
      cuda_VOB[3*tid + 2] = posZ_sh[threadIdx.x];
    }
    //Save final states
    if (changeInitial==1){
      initPosX[tid] = pos.x;
      initPosY[tid] = pos.y;
      initPosZ[tid] = pos.z;
      initVelX[tid] = vel.x;
      initVelY[tid] = vel.y;
      initVelZ[tid] = vel.z;
      initRegionX[tid] = region[0];
      initRegionY[tid] = region[1];
      initRegionZ[tid] = region[2];
      times[tid] = particleTime;
      timesIdx_anim[tid] = timeIdx_anim;
      timesIdx_rad[tid] = timeIdx_rad;
    }
    __syncthreads();
    atomicAdd( &(timesOccupancy[threadIdx.x]), timesOccupancy_sh[threadIdx.x] );
    atomicAdd( &(radiusAll[threadIdx.x]), radiusAll_sh[threadIdx.x]/nParticles );
  }
}
  
}//Extern C end
  